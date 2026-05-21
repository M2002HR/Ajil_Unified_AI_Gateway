from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..config import Settings
from ..observability import get_request_id, log_event
from ..providers.base import ProviderResult
from ..providers.registry import ProviderRegistry
from ..schemas import RouterOptions
from ..state.event_tracker import EventTracker
from ..state.rate_limit import RateLimitGuard
from ..state.usage_tracker import UsageTracker


@dataclass
class ProviderScore:
    avg_latency_ms: float = 900.0
    failures: int = 0
    rate_limited: int = 0
    total_calls: int = 0


@dataclass
class Candidate:
    provider: str
    model: str
    priority: int = 0


class RoutingEngine:
    def __init__(
        self,
        settings: Settings,
        registry: ProviderRegistry,
        guard: RateLimitGuard,
        usage_tracker: Optional[UsageTracker] = None,
        event_tracker: Optional[EventTracker] = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.guard = guard
        self.usage_tracker = usage_tracker
        self.event_tracker = event_tracker
        self.logger = logging.getLogger("uag.router")
        self.scores: Dict[Tuple[str, str], ProviderScore] = {}

    def _get_score(self, provider: str, model: str) -> ProviderScore:
        key = (provider, model)
        if key not in self.scores:
            self.scores[key] = ProviderScore()
        return self.scores[key]

    def _update_score(self, result: ProviderResult) -> None:
        score = self._get_score(result.provider, result.model or "default")
        score.total_calls += 1
        score.avg_latency_ms = (score.avg_latency_ms * 0.8) + (result.latency_ms * 0.2)
        if not result.ok:
            score.failures += 1
        if result.status_code == 429:
            score.rate_limited += 1

    @staticmethod
    def _parse_model_hint(model_hint: str) -> Tuple[Optional[str], str]:
        if "/" in model_hint:
            p, m = model_hint.split("/", 1)
            return p.strip().lower(), m.strip()
        if ":" in model_hint and model_hint.split(":", 1)[0].lower() in {"groq", "gemini", "pollinations"}:
            p, m = model_hint.split(":", 1)
            return p.strip().lower(), m.strip()
        return None, model_hint

    @staticmethod
    def _to_priority(value: Any, *, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _expand_model_entry(
        self,
        *,
        providers: List[str],
        available: set[str],
        provider_hint: Optional[str],
        model_name: str,
        priority: int,
    ) -> List[Candidate]:
        out: List[Candidate] = []
        if provider_hint:
            provider_key = provider_hint.strip().lower()
            if provider_key in available and provider_key in providers:
                out.append(Candidate(provider=provider_key, model=model_name, priority=priority))
            return out

        for provider_key in providers:
            if provider_key in available:
                out.append(Candidate(provider=provider_key, model=model_name, priority=priority))
        return out

    def _extract_payload_model_candidates(
        self,
        *,
        payload_model: Any,
        providers: List[str],
        available: set[str],
    ) -> List[Candidate]:
        out: List[Candidate] = []
        if payload_model is None:
            return out

        if isinstance(payload_model, list):
            for entry in payload_model:
                if isinstance(entry, dict):
                    raw_model = str(entry.get("model") or entry.get("name") or "").strip()
                    if not raw_model:
                        continue
                    provider_hint = str(entry.get("provider") or "").strip() or None
                    priority = self._to_priority(entry.get("priority"), default=0)
                    out.extend(
                        self._expand_model_entry(
                            providers=providers,
                            available=available,
                            provider_hint=provider_hint,
                            model_name=raw_model,
                            priority=priority,
                        )
                    )
                    continue

                if isinstance(entry, str):
                    p_hint, model = self._parse_model_hint(entry)
                    out.extend(
                        self._expand_model_entry(
                            providers=providers,
                            available=available,
                            provider_hint=p_hint,
                            model_name=model,
                            priority=0,
                        )
                    )
            return out

        if isinstance(payload_model, dict):
            raw_model = str(payload_model.get("model") or payload_model.get("name") or "").strip()
            if not raw_model:
                return out
            provider_hint = str(payload_model.get("provider") or "").strip() or None
            priority = self._to_priority(payload_model.get("priority"), default=0)
            out.extend(
                self._expand_model_entry(
                    providers=providers,
                    available=available,
                    provider_hint=provider_hint,
                    model_name=raw_model,
                    priority=priority,
                )
            )
            return out

        if isinstance(payload_model, str):
            p_hint, model = self._parse_model_hint(payload_model)
            out.extend(
                self._expand_model_entry(
                    providers=providers,
                    available=available,
                    provider_hint=p_hint,
                    model_name=model,
                    priority=0,
                )
            )
        return out

    def _build_candidates(self, payload: Dict[str, Any], options: RouterOptions, capability: str) -> List[Candidate]:
        available = set(self.registry.names())
        providers = [p.lower() for p in options.providers if p.lower() in available]
        if not providers:
            providers = sorted(list(available))

        candidates: List[Candidate] = []
        if options.model_preferences:
            for pref in options.model_preferences:
                model_name = str(pref.model or "").strip()
                if not model_name:
                    continue
                provider_hint = str(pref.provider or "").strip() or None
                candidates.extend(
                    self._expand_model_entry(
                        providers=providers,
                        available=available,
                        provider_hint=provider_hint,
                        model_name=model_name,
                        priority=self._to_priority(pref.priority, default=0),
                    )
                )

        payload_candidates = self._extract_payload_model_candidates(
            payload_model=payload.get("model"),
            providers=providers,
            available=available,
        )
        if payload_candidates:
            candidates.extend(payload_candidates)

        if options.models:
            for hint in options.models:
                p_hint, model = self._parse_model_hint(hint)
                candidates.extend(
                    self._expand_model_entry(
                        providers=providers,
                        available=available,
                        provider_hint=p_hint,
                        model_name=model,
                        priority=0,
                    )
                )

        if not candidates:
            defaults_by_capability = {
                "chat.completions": {
                    "gemini": self.settings.gemini.default_model,
                    "groq": "llama-3.3-70b-versatile",
                },
                "responses": {
                    "gemini": self.settings.gemini.default_model,
                    "groq": "llama-3.3-70b-versatile",
                },
                "embeddings": {
                    "gemini": "gemini-embedding-001",
                },
                "images.generations": {
                    "pollinations": self.settings.pollinations.default_image_model,
                },
                "audio.transcriptions": {
                    "groq": self.settings.groq.stt_primary_model,
                },
                "audio.speech": {
                    "groq": self.settings.groq.tts_default_model,
                },
            }
            defaults = defaults_by_capability.get(capability, {})
            for p in providers:
                default_model = str(defaults.get(p, "")).strip()
                if default_model:
                    candidates.append(Candidate(provider=p, model=default_model, priority=0))

        # capability filtering
        capability_map = {
            "chat.completions": {"gemini", "groq"},
            "responses": {"gemini", "groq"},
            "embeddings": {"gemini", "groq"},
            "images.generations": {"pollinations"},
            "audio.transcriptions": {"groq"},
            "audio.speech": {"groq"},
        }
        allowed_providers = capability_map.get(capability, {"gemini", "groq", "pollinations"})
        filtered: List[Candidate] = []
        for c in candidates:
            if c.provider not in allowed_providers:
                continue
            filtered.append(c)

        # Deduplicate preserving order.
        seen: set[tuple[str, str]] = set()
        out: List[Candidate] = []
        for c in filtered:
            key = (c.provider, c.model)
            if key in seen:
                continue
            seen.add(key)
            out.append(c)

        return out[: max(1, self.settings.router.max_candidates)]

    def _sort_candidates(self, candidates: List[Candidate], mode: str) -> List[Candidate]:
        def _priority_key(c: Candidate) -> int:
            return c.priority

        if mode == "quality_first":
            quality_rank = {"gemini": 0, "groq": 1, "pollinations": 2}
            return sorted(candidates, key=lambda c: (_priority_key(c), quality_rank.get(c.provider, 99)))
        if mode == "limit_safe":
            return sorted(
                candidates,
                key=lambda c: (
                    _priority_key(c),
                    self._get_score(c.provider, c.model).rate_limited,
                    self._get_score(c.provider, c.model).failures,
                ),
            )
        # latency_first
        return sorted(
            candidates,
            key=lambda c: (
                _priority_key(c),
                self._get_score(c.provider, c.model).avg_latency_ms,
            ),
        )

    def _is_retryable_status(self, status_code: int) -> bool:
        return int(status_code) in set(self.settings.router.retryable_status_codes)

    def _summarize_failure(self, results: List[ProviderResult]) -> Dict[str, Any]:
        if not results:
            return {
                "status_code": 429 if self.settings.router.strict_rate_limit_errors_only else 503,
                "error_type": "no_attempt",
                "all_rate_limited": False,
            }

        status_codes = [int(r.status_code) for r in results]
        all_rate_limited = all(code == 429 for code in status_codes)
        if all_rate_limited:
            return {"status_code": 429, "error_type": "all_rate_limited", "all_rate_limited": True}

        if self.settings.router.strict_rate_limit_errors_only:
            return {"status_code": 429, "error_type": "strict_rate_limited_policy", "all_rate_limited": False}

        if any(code in {408, 504} for code in status_codes):
            return {"status_code": 504, "error_type": "upstream_timeout", "all_rate_limited": False}
        if any(code >= 500 for code in status_codes):
            return {"status_code": 503, "error_type": "upstream_unavailable", "all_rate_limited": False}
        if any(code == 429 for code in status_codes):
            return {"status_code": 429, "error_type": "rate_limited", "all_rate_limited": False}
        return {"status_code": 502, "error_type": "upstream_failed", "all_rate_limited": False}

    async def _call_provider(
        self,
        capability: str,
        candidate: Candidate,
        payload: Dict[str, Any],
        *,
        timeout_sec: float | None = None,
    ) -> ProviderResult:
        adapter = self.registry.providers[candidate.provider]

        ok_to_call = await self.guard.allow(
            scope=f"{candidate.provider}:{candidate.model}:{capability}",
            limit=self.settings.redis.default_limit_per_minute,
            window_sec=60,
        )
        if not ok_to_call:
            return ProviderResult(
                provider=candidate.provider,
                capability=capability,
                ok=False,
                status_code=429,
                latency_ms=0.0,
                payload=None,
                error="Global gateway rate limit reached for candidate",
                model=candidate.model,
            )

        try:
            if capability == "chat.completions":
                coro = adapter.chat_completions(payload, model=candidate.model)
            elif capability == "responses":
                coro = adapter.responses(payload, model=candidate.model)
            elif capability == "embeddings":
                coro = adapter.embeddings(payload, model=candidate.model)
            elif capability == "images.generations":
                coro = adapter.image_generations(payload, model=candidate.model)
            elif capability == "audio.speech":
                coro = adapter.tts(payload, model=candidate.model)
            else:
                raise ValueError(f"Unsupported capability for _call_provider: {capability}")

            if timeout_sec and float(timeout_sec) > 0:
                return await asyncio.wait_for(coro, timeout=max(1.0, float(timeout_sec)))
            return await coro
        except asyncio.TimeoutError:
            return ProviderResult(
                provider=candidate.provider,
                capability=capability,
                ok=False,
                status_code=504,
                latency_ms=0.0,
                payload=None,
                error=f"provider timeout after {timeout_sec}s",
                model=candidate.model,
            )

    async def _call_provider_with_candidate(
        self,
        capability: str,
        candidate: Candidate,
        payload: Dict[str, Any],
        *,
        timeout_sec: float | None = None,
    ) -> Tuple[Candidate, ProviderResult]:
        result = await self._call_provider(capability, candidate, payload, timeout_sec=timeout_sec)
        return candidate, result

    async def _record_usage(self, capability: str, candidate: Candidate, result: ProviderResult) -> None:
        if self.usage_tracker is None:
            return
        await self.usage_tracker.record(
            capability=capability,
            priority=candidate.priority,
            result=result,
        )

    async def _record_event(self, *, event_type: str, level: str, message: str, data: Dict[str, Any]) -> None:
        if self.event_tracker is None:
            return
        await self.event_tracker.record(
            event_type=event_type,
            level=level,
            message=message,
            request_id=get_request_id(),
            data=data,
        )

    async def _fallback_chain(
        self,
        capability: str,
        candidates: List[Candidate],
        payload: Dict[str, Any],
        *,
        timeout_sec: float | None = None,
        max_attempts: int | None = None,
        mode: str = "latency_first",
    ) -> Tuple[Optional[ProviderResult], List[ProviderResult]]:
        results: List[ProviderResult] = []
        attempt_budget = max(len(candidates), int(max_attempts or self.settings.router.fallback_max_attempts))
        attempts_done = 0
        deadline_ts: float | None = None
        if timeout_sec and float(timeout_sec) > 0:
            deadline_ts = time.monotonic() + max(1.0, float(timeout_sec))

        while attempts_done < attempt_budget:
            round_candidates = self._sort_candidates(list(candidates), mode)
            round_results: List[ProviderResult] = []
            for c in round_candidates:
                if attempts_done >= attempt_budget:
                    break
                call_timeout = timeout_sec
                if deadline_ts is not None:
                    remaining = deadline_ts - time.monotonic()
                    if remaining <= 0:
                        break
                    call_timeout = min(float(timeout_sec or remaining), remaining)
                per_attempt_cap = max(1.0, float(self.settings.router.fallback_attempt_timeout_sec))
                if call_timeout is None:
                    call_timeout = per_attempt_cap
                else:
                    call_timeout = min(float(call_timeout), per_attempt_cap)
                res = await self._call_provider(capability, c, payload, timeout_sec=call_timeout)
                attempts_done += 1
                self._update_score(res)
                await self._record_usage(capability, c, res)
                if self.settings.logging.enabled and self.settings.logging.log_provider_attempts:
                    log_event(
                        self.logger,
                        event="router.provider.attempt",
                        level=logging.INFO if res.ok else logging.WARNING,
                        message="provider attempt completed",
                        capability=capability,
                        provider=c.provider,
                        model=c.model,
                        priority=c.priority,
                        ok=res.ok,
                        status_code=res.status_code,
                        latency_ms=round(res.latency_ms, 3),
                    )
                await self._record_event(
                    event_type="router.provider.attempt",
                    level="INFO" if res.ok else "WARNING",
                    message="provider attempt completed",
                    data={
                        "capability": capability,
                        "provider": c.provider,
                        "model": c.model,
                        "priority": c.priority,
                        "ok": bool(res.ok),
                        "status_code": int(res.status_code),
                        "latency_ms": round(res.latency_ms, 3),
                    },
                )
                results.append(res)
                round_results.append(res)
                if res.ok:
                    return res, results

            if not round_results:
                break
            if all(not self._is_retryable_status(int(item.status_code)) for item in round_results):
                break
            if attempts_done >= attempt_budget:
                break
            backoff = max(0.0, float(self.settings.router.fallback_round_backoff_sec))
            if backoff > 0:
                await asyncio.sleep(backoff)
        return None, results

    async def _parallel_race(self, capability: str, candidates: List[Candidate], payload: Dict[str, Any], timeout_sec: float) -> Tuple[Optional[ProviderResult], List[ProviderResult]]:
        tasks = [asyncio.create_task(self._call_provider_with_candidate(capability, c, payload, timeout_sec=timeout_sec)) for c in candidates]
        results: List[ProviderResult] = []
        winner: Optional[ProviderResult] = None
        try:
            for coro in asyncio.as_completed(tasks, timeout=max(1.0, timeout_sec)):
                c, res = await coro
                self._update_score(res)
                await self._record_usage(capability, c, res)
                if self.settings.logging.enabled and self.settings.logging.log_provider_attempts:
                    log_event(
                        self.logger,
                        event="router.provider.attempt",
                        level=logging.INFO if res.ok else logging.WARNING,
                        message="provider attempt completed",
                        capability=capability,
                        provider=c.provider,
                        model=c.model,
                        priority=c.priority,
                        ok=res.ok,
                        status_code=res.status_code,
                        latency_ms=round(res.latency_ms, 3),
                    )
                await self._record_event(
                    event_type="router.provider.attempt",
                    level="INFO" if res.ok else "WARNING",
                    message="provider attempt completed",
                    data={
                        "capability": capability,
                        "provider": c.provider,
                        "model": c.model,
                        "priority": c.priority,
                        "ok": bool(res.ok),
                        "status_code": int(res.status_code),
                        "latency_ms": round(res.latency_ms, 3),
                    },
                )
                results.append(res)
                if winner is None and res.ok:
                    winner = res
                    break
        except TimeoutError:
            pass
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            for t in tasks:
                if t.done() and not t.cancelled():
                    try:
                        c, r = t.result()
                        if r not in results:
                            self._update_score(r)
                            await self._record_usage(capability, c, r)
                            if self.settings.logging.enabled and self.settings.logging.log_provider_attempts:
                                log_event(
                                    self.logger,
                                    event="router.provider.attempt",
                                    level=logging.INFO if r.ok else logging.WARNING,
                                    message="provider attempt completed",
                                    capability=capability,
                                    provider=c.provider,
                                    model=c.model,
                                    priority=c.priority,
                                    ok=r.ok,
                                    status_code=r.status_code,
                                    latency_ms=round(r.latency_ms, 3),
                                )
                            await self._record_event(
                                event_type="router.provider.attempt",
                                level="INFO" if r.ok else "WARNING",
                                message="provider attempt completed",
                                data={
                                    "capability": capability,
                                    "provider": c.provider,
                                    "model": c.model,
                                    "priority": c.priority,
                                    "ok": bool(r.ok),
                                    "status_code": int(r.status_code),
                                    "latency_ms": round(r.latency_ms, 3),
                                },
                            )
                            results.append(r)
                    except Exception:
                        pass
        return winner, results

    async def _aggregate(self, capability: str, candidates: List[Candidate], payload: Dict[str, Any], timeout_sec: float) -> List[ProviderResult]:
        tasks = [asyncio.create_task(self._call_provider_with_candidate(capability, c, payload, timeout_sec=timeout_sec)) for c in candidates]
        task_map = {task: candidate for task, candidate in zip(tasks, candidates)}
        results: List[ProviderResult] = []
        done, pending = await asyncio.wait(tasks, timeout=max(1.0, timeout_sec))
        for task in done:
            candidate = task_map.get(task)
            try:
                candidate, res = task.result()
            except Exception as exc:  # noqa: BLE001
                if candidate is None:
                    candidate = Candidate(provider="unknown", model="", priority=0)
                res = ProviderResult(provider="unknown", capability=capability, ok=False, status_code=500, latency_ms=0.0, error=str(exc))
            self._update_score(res)
            await self._record_usage(capability, candidate, res)
            if self.settings.logging.enabled and self.settings.logging.log_provider_attempts:
                log_event(
                    self.logger,
                    event="router.provider.attempt",
                    level=logging.INFO if res.ok else logging.WARNING,
                    message="provider attempt completed",
                    capability=capability,
                    provider=candidate.provider,
                    model=candidate.model,
                    priority=candidate.priority,
                    ok=res.ok,
                    status_code=res.status_code,
                    latency_ms=round(res.latency_ms, 3),
                )
            await self._record_event(
                event_type="router.provider.attempt",
                level="INFO" if res.ok else "WARNING",
                message="provider attempt completed",
                data={
                    "capability": capability,
                    "provider": candidate.provider,
                    "model": candidate.model,
                    "priority": candidate.priority,
                    "ok": bool(res.ok),
                    "status_code": int(res.status_code),
                    "latency_ms": round(res.latency_ms, 3),
                },
            )
            results.append(res)
        for task in pending:
            task.cancel()
        return results

    async def dispatch(self, capability: str, payload: Dict[str, Any], options: Optional[RouterOptions] = None) -> Dict[str, Any]:
        options = options or RouterOptions(
            strategy=self.settings.router.default_strategy,
            mode=self.settings.router.default_mode,
            timeout_sec=self.settings.router.parallel_timeout_sec,
        )
        started = time.monotonic()
        candidates = self._build_candidates(payload, options, capability)
        candidates = self._sort_candidates(candidates, options.mode)
        if self.settings.logging.enabled and self.settings.logging.log_router_dispatch:
            log_event(
                self.logger,
                event="router.dispatch.start",
                level=logging.INFO,
                message="router dispatch started",
                capability=capability,
                strategy=options.strategy,
                mode=options.mode,
                candidates=[{"provider": c.provider, "model": c.model, "priority": c.priority} for c in candidates],
            )
        await self._record_event(
            event_type="router.dispatch.start",
            level="INFO",
            message="router dispatch started",
            data={
                "capability": capability,
                "strategy": options.strategy,
                "mode": options.mode,
                "candidates_count": len(candidates),
            },
        )

        if not candidates:
            await self._record_event(
                event_type="router.dispatch.completed",
                level="ERROR",
                message="router dispatch failed: no candidate",
                data={
                    "capability": capability,
                    "strategy": options.strategy,
                    "mode": options.mode,
                    "ok": False,
                    "latency_ms": round((time.monotonic() - started) * 1000.0, 2),
                },
            )
            return {
                "ok": False,
                "error": "No provider candidate available for this capability",
                "strategy": options.strategy,
                "capability": capability,
                "results": [],
            }

        if options.strategy == "aggregate":
            results = await self._aggregate(capability, candidates, payload, timeout_sec=options.timeout_sec)
            winner_result = next((r for r in results if r.ok), None)
            failure_meta = self._summarize_failure(results) if winner_result is None else {}
            out = {
                "ok": any(r.ok for r in results),
                "strategy": "aggregate",
                "mode": options.mode,
                "capability": capability,
                "latency_ms": round((time.monotonic() - started) * 1000.0, 2),
                "winner": winner_result.__dict__ if winner_result else None,
                "results": [r.__dict__ for r in results],
                "status_code": int((winner_result.status_code if winner_result else failure_meta.get("status_code", 502))),
                "error_type": str(failure_meta.get("error_type", "")) if winner_result is None else "",
                "all_rate_limited": bool(failure_meta.get("all_rate_limited", False)) if winner_result is None else False,
            }
            if self.settings.logging.enabled and self.settings.logging.log_router_dispatch:
                log_event(
                    self.logger,
                    event="router.dispatch.completed",
                    level=logging.INFO if out["ok"] else logging.WARNING,
                    message="router dispatch completed",
                    capability=capability,
                    strategy=out["strategy"],
                    mode=out["mode"],
                    ok=out["ok"],
                    latency_ms=out["latency_ms"],
                    attempts=len(results),
                    winner_provider=(out.get("winner") or {}).get("provider"),
                    winner_model=(out.get("winner") or {}).get("model"),
                )
            await self._record_event(
                event_type="router.dispatch.completed",
                level="INFO" if out["ok"] else "WARNING",
                message="router dispatch completed",
                data={
                    "capability": capability,
                    "strategy": out["strategy"],
                    "mode": out["mode"],
                    "ok": bool(out["ok"]),
                    "latency_ms": out["latency_ms"],
                    "attempts": len(results),
                    "winner_provider": ((out.get("winner") or {}).get("provider") or ""),
                    "winner_model": ((out.get("winner") or {}).get("model") or ""),
                },
            )
            return out

        if options.strategy == "parallel_race":
            winner, results = await self._parallel_race(capability, candidates, payload, timeout_sec=options.timeout_sec)
            failure_meta = self._summarize_failure(results) if winner is None else {}
            out = {
                "ok": winner is not None,
                "strategy": "parallel_race",
                "mode": options.mode,
                "capability": capability,
                "latency_ms": round((time.monotonic() - started) * 1000.0, 2),
                "winner": winner.__dict__ if winner else None,
                "results": [r.__dict__ for r in results],
                "status_code": int((winner.status_code if winner else failure_meta.get("status_code", 502))),
                "error_type": str(failure_meta.get("error_type", "")) if winner is None else "",
                "all_rate_limited": bool(failure_meta.get("all_rate_limited", False)) if winner is None else False,
            }
            if self.settings.logging.enabled and self.settings.logging.log_router_dispatch:
                log_event(
                    self.logger,
                    event="router.dispatch.completed",
                    level=logging.INFO if out["ok"] else logging.WARNING,
                    message="router dispatch completed",
                    capability=capability,
                    strategy=out["strategy"],
                    mode=out["mode"],
                    ok=out["ok"],
                    latency_ms=out["latency_ms"],
                    attempts=len(results),
                    winner_provider=(out.get("winner") or {}).get("provider"),
                    winner_model=(out.get("winner") or {}).get("model"),
                )
            await self._record_event(
                event_type="router.dispatch.completed",
                level="INFO" if out["ok"] else "WARNING",
                message="router dispatch completed",
                data={
                    "capability": capability,
                    "strategy": out["strategy"],
                    "mode": out["mode"],
                    "ok": bool(out["ok"]),
                    "latency_ms": out["latency_ms"],
                    "attempts": len(results),
                    "winner_provider": ((out.get("winner") or {}).get("provider") or ""),
                    "winner_model": ((out.get("winner") or {}).get("model") or ""),
                },
            )
            return out

        winner, results = await self._fallback_chain(
            capability,
            candidates,
            payload,
            timeout_sec=options.timeout_sec,
            max_attempts=max(int(options.max_attempts or 0), int(self.settings.router.fallback_max_attempts)),
            mode=options.mode,
        )
        failure_meta = self._summarize_failure(results) if winner is None else {}
        out = {
            "ok": winner is not None,
            "strategy": "fallback_chain",
            "mode": options.mode,
            "capability": capability,
            "latency_ms": round((time.monotonic() - started) * 1000.0, 2),
            "winner": winner.__dict__ if winner else None,
            "results": [r.__dict__ for r in results],
            "status_code": int((winner.status_code if winner else failure_meta.get("status_code", 502))),
            "error_type": str(failure_meta.get("error_type", "")) if winner is None else "",
            "all_rate_limited": bool(failure_meta.get("all_rate_limited", False)) if winner is None else False,
        }
        if self.settings.logging.enabled and self.settings.logging.log_router_dispatch:
            log_event(
                self.logger,
                event="router.dispatch.completed",
                level=logging.INFO if out["ok"] else logging.WARNING,
                message="router dispatch completed",
                capability=capability,
                strategy=out["strategy"],
                mode=out["mode"],
                ok=out["ok"],
                latency_ms=out["latency_ms"],
                attempts=len(results),
                winner_provider=(out.get("winner") or {}).get("provider"),
                winner_model=(out.get("winner") or {}).get("model"),
            )
        await self._record_event(
            event_type="router.dispatch.completed",
            level="INFO" if out["ok"] else "WARNING",
            message="router dispatch completed",
            data={
                "capability": capability,
                "strategy": out["strategy"],
                "mode": out["mode"],
                "ok": bool(out["ok"]),
                "latency_ms": out["latency_ms"],
                "attempts": len(results),
                "winner_provider": ((out.get("winner") or {}).get("provider") or ""),
                "winner_model": ((out.get("winner") or {}).get("model") or ""),
            },
        )
        return out

    def stats(self) -> Dict[str, Any]:
        out = []
        for (provider, model), score in self.scores.items():
            out.append(
                {
                    "provider": provider,
                    "model": model,
                    "avg_latency_ms": round(score.avg_latency_ms, 2),
                    "failures": score.failures,
                    "rate_limited": score.rate_limited,
                    "total_calls": score.total_calls,
                }
            )
        return {"items": sorted(out, key=lambda x: (x["provider"], x["model"]))}
