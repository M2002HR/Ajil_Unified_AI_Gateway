from __future__ import annotations

import asyncio
import json
import logging
import time
from uuid import uuid4
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from ..config import Settings
from ..observability import get_request_id, log_event
from ..providers.base import ProviderResult, ProviderStreamResult
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

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    if item.strip():
                        parts.append(item.strip())
                    continue
                if isinstance(item, dict):
                    if str(item.get("type") or "").lower() in {"text", "input_text"}:
                        txt = str(item.get("text") or "").strip()
                        if txt:
                            parts.append(txt)
            return "\n".join(parts).strip()
        if isinstance(content, dict):
            if str(content.get("type") or "").lower() in {"text", "input_text"}:
                return str(content.get("text") or "").strip()
        return ""

    @staticmethod
    def _extract_prompt_hint(payload: Dict[str, Any]) -> str:
        messages = payload.get("messages")
        if isinstance(messages, list):
            for row in reversed(messages):
                if not isinstance(row, dict):
                    continue
                if str(row.get("role") or "").lower() != "user":
                    continue
                txt = RoutingEngine._extract_text_content(row.get("content"))
                if txt:
                    return txt
        input_value = payload.get("input")
        if isinstance(input_value, str):
            return input_value.strip()
        if isinstance(input_value, list):
            parts: List[str] = []
            for row in input_value:
                if isinstance(row, str):
                    if row.strip():
                        parts.append(row.strip())
                elif isinstance(row, dict):
                    txt = str(row.get("text") or row.get("content") or "").strip()
                    if txt:
                        parts.append(txt)
            return "\n".join(parts).strip()
        return ""

    @staticmethod
    def _is_farsi_text(text: str) -> bool:
        return any("\u0600" <= ch <= "\u06ff" for ch in str(text or ""))

    def _build_local_fallback_text(self, payload: Dict[str, Any], reason: str) -> str:
        hint = self._extract_prompt_hint(payload)
        short_hint = (hint[:160] + "...") if len(hint) > 160 else hint
        if self._is_farsi_text(hint):
            base = "در حال حاضر ارتباط با مدل‌های خارجی پایدار نیست. پاسخ موقت از لایه پایداری داخلی ارائه شد."
            if short_hint:
                return f"{base}\n\nخلاصه درخواست شما: {short_hint}"
            return base
        base = "Upstream providers are temporarily unstable. Returning a stable local fallback response."
        if short_hint:
            return f"{base}\n\nRequest summary: {short_hint}"
        return base

    def _build_local_fallback_result(self, capability: str, payload: Dict[str, Any], reason: str) -> ProviderResult:
        text = self._build_local_fallback_text(payload, reason)
        now = int(time.time())
        if capability == "responses":
            response_payload = {
                "id": f"resp_local_{uuid4().hex[:12]}",
                "object": "response",
                "created_at": now,
                "status": "completed",
                "output_text": text,
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                ],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }
        else:
            response_payload = {
                "id": f"chatcmpl_local_{uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": now,
                "model": "local/fallback",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        return ProviderResult(
            provider="local-fallback",
            capability=capability,
            ok=True,
            status_code=200,
            latency_ms=0.0,
            payload=response_payload,
            error="",
            model="local/fallback",
            headers={"x-uag-fallback": f"local-{reason or 'upstream-failure'}"},
        )

    @staticmethod
    def _should_local_fallback(capability: str, failure_meta: Dict[str, Any]) -> bool:
        if capability not in {"chat.completions", "responses"}:
            return False
        if bool(failure_meta.get("all_rate_limited", False)):
            return False
        return True

    @staticmethod
    def _requires_structured_output(payload: Dict[str, Any]) -> bool:
        """A prose fallback is unsafe when a caller explicitly requested JSON."""
        response_format = payload.get("response_format")
        if not isinstance(response_format, dict):
            return False
        kind = str(response_format.get("type") or "").strip().lower()
        return kind in {"json_object", "json_schema"}

    @staticmethod
    def _candidate_supports_capability(capability: str, candidate: Candidate) -> bool:
        model = str(candidate.model or "").strip().lower()
        provider = str(candidate.provider or "").strip().lower()
        if not model:
            return False

        if capability in {"chat.completions", "responses"}:
            if provider == "groq":
                blocked_tokens = ["prompt-guard", "safeguard", "whisper", "embedding", "playai", "orpheus"]
                return not any(token in model for token in blocked_tokens)
            if provider == "gemini":
                blocked_tokens = ["embedding"]
                return not any(token in model for token in blocked_tokens)
            return True

        if capability == "embeddings":
            if provider == "gemini":
                return "embedding" in model
            if provider == "groq":
                return "embedding" in model
            return False

        if capability == "audio.speech":
            if provider != "groq":
                return False
            return any(token in model for token in ["playai", "orpheus", "tts"])

        if capability == "images.generations":
            return provider == "pollinations"

        return True

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
            if not self._candidate_supports_capability(capability, c):
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

        if out:
            return out[: max(1, self.settings.router.max_candidates)]

        # If explicit candidates were present but all were filtered out as incompatible,
        # fall back to stable defaults instead of returning no candidate.
        fallback_defaults = {
            "chat.completions": [
                Candidate(provider="groq", model="llama-3.3-70b-versatile", priority=999),
                Candidate(provider="gemini", model=self.settings.gemini.default_model, priority=999),
            ],
            "responses": [
                Candidate(provider="groq", model="llama-3.3-70b-versatile", priority=999),
                Candidate(provider="gemini", model=self.settings.gemini.default_model, priority=999),
            ],
            "embeddings": [
                Candidate(provider="gemini", model="gemini-embedding-001", priority=999),
            ],
            "images.generations": [
                Candidate(provider="pollinations", model=self.settings.pollinations.default_image_model, priority=999),
            ],
            "audio.speech": [
                Candidate(provider="groq", model=self.settings.groq.tts_default_model, priority=999),
            ],
        }
        available = set(self.registry.names())
        safe: List[Candidate] = []
        for c in fallback_defaults.get(capability, []):
            if c.provider not in available:
                continue
            if not self._candidate_supports_capability(capability, c):
                continue
            safe.append(c)
        return safe[: max(1, self.settings.router.max_candidates)]

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

    @staticmethod
    def _is_hard_terminal_status(status_code: int) -> bool:
        code = int(status_code)
        # Model not found / bad request / auth/permission issues are terminal for same candidate.
        return code in {400, 401, 403, 404, 405, 410, 422}

    def _summarize_failure(self, results: List[ProviderResult]) -> Dict[str, Any]:
        if not results:
            return {
                "status_code": 503,
                "error_type": "no_attempt",
                "all_rate_limited": False,
            }

        status_codes = [int(r.status_code) for r in results]
        all_rate_limited = all(code == 429 for code in status_codes)
        if all_rate_limited:
            return {"status_code": 429, "error_type": "all_rate_limited", "all_rate_limited": True}

        if any(code in {408, 504} for code in status_codes):
            return {"status_code": 504, "error_type": "upstream_timeout", "all_rate_limited": False}
        if any(code >= 500 for code in status_codes):
            return {"status_code": 503, "error_type": "upstream_unavailable", "all_rate_limited": False}
        if any(code == 429 for code in status_codes):
            return {"status_code": 429, "error_type": "rate_limited", "all_rate_limited": False}
        return {"status_code": 502, "error_type": "upstream_failed", "all_rate_limited": False}

    @staticmethod
    def _timeout_result(capability: str, candidate: Candidate, timeout_sec: float) -> ProviderResult:
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
        configured_budget = int(max_attempts or self.settings.router.fallback_max_attempts)
        attempt_budget = max(1, configured_budget)
        attempts_done = 0
        deadline_ts: float | None = None
        if timeout_sec and float(timeout_sec) > 0:
            deadline_ts = time.monotonic() + max(1.0, float(timeout_sec))

        # First round: try each unique candidate once, in routing order.
        round_candidates = self._sort_candidates(list(candidates), mode)
        retry_pool: List[Candidate] = []
        used_in_round: set[tuple[str, str]] = set()
        while attempts_done < attempt_budget and round_candidates:
            round_results: List[ProviderResult] = []
            for c in round_candidates:
                if attempts_done >= attempt_budget:
                    break
                key = (c.provider, c.model)
                if key in used_in_round:
                    continue
                used_in_round.add(key)
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
                if self._is_retryable_status(int(res.status_code)) and not self._is_hard_terminal_status(int(res.status_code)):
                    retry_pool.append(c)

            if not round_results:
                break
            if all(not self._is_retryable_status(int(item.status_code)) for item in round_results):
                break
            if attempts_done >= attempt_budget:
                break
            if not retry_pool:
                break

            # Retry round: only retry previously retryable failures once more, not full candidate set loops.
            round_candidates = self._sort_candidates(list(retry_pool), mode)
            retry_pool = []
            backoff = max(0.0, float(self.settings.router.fallback_round_backoff_sec))
            if backoff > 0:
                await asyncio.sleep(backoff)
        return None, results

    def _rescue_candidates(self, capability: str, attempted: List[ProviderResult]) -> List[Candidate]:
        if capability not in {"chat.completions", "responses"}:
            return []
        attempted_keys = {(r.provider, str(r.model or "").strip()) for r in attempted}
        available = set(self.registry.names())
        defaults: List[Candidate] = []
        gemini_model = str(self.settings.gemini.default_model or "").strip()
        if "gemini" in available and gemini_model:
            defaults.append(Candidate(provider="gemini", model=gemini_model, priority=999))
        if "groq" in available:
            defaults.append(Candidate(provider="groq", model="llama-3.1-8b-instant", priority=999))
            defaults.append(Candidate(provider="groq", model="llama-3.3-70b-versatile", priority=1000))

        out: List[Candidate] = []
        for c in defaults:
            key = (c.provider, c.model)
            if key in attempted_keys:
                continue
            out.append(c)
        return out

    def _build_rescue_payload(self, capability: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(payload)
        if capability in {"chat.completions", "responses"}:
            current_max = out.get("max_tokens")
            try:
                parsed = int(current_max)
            except (TypeError, ValueError):
                parsed = 0
            if parsed <= 0 or parsed > 512:
                out["max_tokens"] = 256
            if "temperature" not in out:
                out["temperature"] = 0.2
        return out

    async def _parallel_race(self, capability: str, candidates: List[Candidate], payload: Dict[str, Any], timeout_sec: float) -> Tuple[Optional[ProviderResult], List[ProviderResult]]:
        tasks = [asyncio.create_task(self._call_provider_with_candidate(capability, c, payload, timeout_sec=timeout_sec)) for c in candidates]
        task_map = {task: candidate for task, candidate in zip(tasks, candidates)}
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
        except asyncio.TimeoutError:
            # Ensure we account for timed-out attempts as explicit provider failures.
            for task in tasks:
                if task.done():
                    continue
                candidate = task_map.get(task)
                if candidate is None:
                    continue
                timeout_res = self._timeout_result(capability, candidate, float(timeout_sec))
                results.append(timeout_res)
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
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
            candidate = task_map.get(task)
            if candidate is not None:
                timeout_res = self._timeout_result(capability, candidate, float(timeout_sec))
                self._update_score(timeout_res)
                await self._record_usage(capability, candidate, timeout_res)
                await self._record_event(
                    event_type="router.provider.attempt",
                    level="WARNING",
                    message="provider attempt completed",
                    data={
                        "capability": capability,
                        "provider": candidate.provider,
                        "model": candidate.model,
                        "priority": candidate.priority,
                        "ok": False,
                        "status_code": 504,
                        "latency_ms": 0.0,
                    },
                )
                results.append(timeout_res)
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
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
            if winner_result is None and self._should_local_fallback(capability, failure_meta):
                fallback_result = self._build_local_fallback_result(
                    capability,
                    payload,
                    str(failure_meta.get("error_type") or "upstream_failed"),
                )
                local_candidate = Candidate(provider="local-fallback", model="local/fallback", priority=9999)
                self._update_score(fallback_result)
                await self._record_usage(capability, local_candidate, fallback_result)
                await self._record_event(
                    event_type="router.provider.attempt",
                    level="WARNING",
                    message="provider attempt completed",
                    data={
                        "capability": capability,
                        "provider": local_candidate.provider,
                        "model": local_candidate.model,
                        "priority": local_candidate.priority,
                        "ok": True,
                        "status_code": 200,
                        "latency_ms": 0.0,
                        "fallback": True,
                    },
                )
                results.append(fallback_result)
                winner_result = fallback_result
                failure_meta = {}
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
            if winner is None and self._should_local_fallback(capability, failure_meta):
                fallback_result = self._build_local_fallback_result(
                    capability,
                    payload,
                    str(failure_meta.get("error_type") or "upstream_failed"),
                )
                local_candidate = Candidate(provider="local-fallback", model="local/fallback", priority=9999)
                self._update_score(fallback_result)
                await self._record_usage(capability, local_candidate, fallback_result)
                await self._record_event(
                    event_type="router.provider.attempt",
                    level="WARNING",
                    message="provider attempt completed",
                    data={
                        "capability": capability,
                        "provider": local_candidate.provider,
                        "model": local_candidate.model,
                        "priority": local_candidate.priority,
                        "ok": True,
                        "status_code": 200,
                        "latency_ms": 0.0,
                        "fallback": True,
                    },
                )
                results.append(fallback_result)
                winner = fallback_result
                failure_meta = {}
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
            max_attempts=int(options.max_attempts or self.settings.router.fallback_max_attempts),
            mode=options.mode,
        )
        if winner is None:
            rescue = self._rescue_candidates(capability, results)
            if rescue:
                rescue_timeout = min(
                    max(2.0, float(self.settings.router.fallback_attempt_timeout_sec)),
                    max(2.0, float(options.timeout_sec or self.settings.router.parallel_timeout_sec)),
                )
                rescue_payload = self._build_rescue_payload(capability, payload)
                rescue_winner, rescue_results = await self._fallback_chain(
                    capability,
                    rescue,
                    rescue_payload,
                    timeout_sec=rescue_timeout,
                    max_attempts=len(rescue),
                    mode=options.mode,
                )
                results.extend(rescue_results)
                if rescue_winner is not None:
                    winner = rescue_winner
        failure_meta = self._summarize_failure(results) if winner is None else {}
        if (
            winner is None
            and not self._requires_structured_output(payload)
            and self._should_local_fallback(capability, failure_meta)
        ):
            fallback_result = self._build_local_fallback_result(
                capability,
                payload,
                str(failure_meta.get("error_type") or "upstream_failed"),
            )
            local_candidate = Candidate(provider="local-fallback", model="local/fallback", priority=9999)
            self._update_score(fallback_result)
            await self._record_usage(capability, local_candidate, fallback_result)
            await self._record_event(
                event_type="router.provider.attempt",
                level="WARNING",
                message="provider attempt completed",
                data={
                    "capability": capability,
                    "provider": local_candidate.provider,
                    "model": local_candidate.model,
                    "priority": local_candidate.priority,
                    "ok": True,
                    "status_code": 200,
                    "latency_ms": 0.0,
                    "fallback": True,
                },
            )
            results.append(fallback_result)
            winner = fallback_result
            failure_meta = {}
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

    async def dispatch_chat_stream(self, payload: Dict[str, Any], options: Optional[RouterOptions] = None) -> Dict[str, Any]:
        options = options or RouterOptions(
            strategy=self.settings.router.default_strategy,
            mode=self.settings.router.default_mode,
            timeout_sec=self.settings.router.parallel_timeout_sec,
        )
        started = time.monotonic()
        candidates = self._build_candidates(payload, options, "chat.completions")
        candidates = self._sort_candidates(candidates, options.mode)

        attempt_results: List[ProviderResult] = []
        winner_stream: Optional[AsyncIterator[bytes]] = None
        winner_candidate: Optional[Candidate] = None
        winner_headers: Dict[str, str] = {}
        winner_latency_ms = 0.0
        deadline_ts: float | None = None
        if options.timeout_sec and float(options.timeout_sec) > 0:
            deadline_ts = time.monotonic() + max(1.0, float(options.timeout_sec))

        max_attempts = max(1, int(options.max_attempts or self.settings.router.fallback_max_attempts))
        for idx, candidate in enumerate(candidates):
            if idx >= max_attempts:
                break
            # For streaming, allow full request timeout budget per attempt (bounded by global remaining deadline).
            # A hard 12s cap frequently aborts valid streams before first token.
            call_timeout: Optional[float] = max(
                1.0,
                float(options.timeout_sec or self.settings.router.parallel_timeout_sec),
            )
            if deadline_ts is not None:
                remaining = deadline_ts - time.monotonic()
                if remaining <= 0:
                    break
                call_timeout = min(call_timeout, remaining)

            adapter = self.registry.providers.get(candidate.provider)
            if adapter is None or not hasattr(adapter, "chat_completions_stream"):
                fail = ProviderResult(
                    provider=candidate.provider,
                    capability="chat.completions",
                    ok=False,
                    status_code=501,
                    latency_ms=0.0,
                    payload=None,
                    error="streaming not supported by adapter",
                    model=candidate.model,
                )
                attempt_results.append(fail)
                self._update_score(fail)
                await self._record_usage("chat.completions", candidate, fail)
                continue

            ok_to_call = await self.guard.allow(
                scope=f"{candidate.provider}:{candidate.model}:chat.completions",
                limit=self.settings.redis.default_limit_per_minute,
                window_sec=60,
            )
            if not ok_to_call:
                fail = ProviderResult(
                    provider=candidate.provider,
                    capability="chat.completions",
                    ok=False,
                    status_code=429,
                    latency_ms=0.0,
                    payload=None,
                    error="Global gateway rate limit reached for candidate",
                    model=candidate.model,
                )
                attempt_results.append(fail)
                self._update_score(fail)
                await self._record_usage("chat.completions", candidate, fail)
                continue

            stream_res: ProviderStreamResult
            try:
                stream_res = await asyncio.wait_for(
                    adapter.chat_completions_stream(payload, model=candidate.model, timeout_sec=call_timeout),  # type: ignore[attr-defined]
                    timeout=max(1.0, float(call_timeout or 1.0)),
                )
            except asyncio.TimeoutError:
                stream_res = ProviderStreamResult(
                    provider=candidate.provider,
                    capability="chat.completions",
                    ok=False,
                    status_code=504,
                    latency_ms=0.0,
                    model=candidate.model,
                    error=f"provider timeout after {call_timeout}s",
                )
            except Exception as exc:  # noqa: BLE001
                stream_res = ProviderStreamResult(
                    provider=candidate.provider,
                    capability="chat.completions",
                    ok=False,
                    status_code=502,
                    latency_ms=0.0,
                    model=candidate.model,
                    error=str(exc),
                )

            attempt = ProviderResult(
                provider=stream_res.provider,
                capability="chat.completions",
                ok=stream_res.ok,
                status_code=stream_res.status_code,
                latency_ms=stream_res.latency_ms,
                payload=None,
                error=stream_res.error,
                model=stream_res.model or candidate.model,
                headers=dict(stream_res.headers or {}),
            )
            attempt_results.append(attempt)
            self._update_score(attempt)
            await self._record_usage("chat.completions", candidate, attempt)

            if stream_res.ok and stream_res.stream is not None:
                winner_stream = stream_res.stream
                winner_candidate = candidate
                winner_headers = dict(stream_res.headers or {})
                winner_latency_ms = float(stream_res.latency_ms or 0.0)
                break

            if not self._is_retryable_status(int(stream_res.status_code)):
                continue

        if winner_stream is not None and winner_candidate is not None:
            return {
                "ok": True,
                "status_code": 200,
                "strategy": options.strategy,
                "mode": options.mode,
                "capability": "chat.completions",
                "latency_ms": round((time.monotonic() - started) * 1000.0, 2),
                "winner": {
                    "provider": winner_candidate.provider,
                    "model": winner_candidate.model,
                    "latency_ms": winner_latency_ms,
                    "headers": winner_headers,
                },
                "results": [r.__dict__ for r in attempt_results],
                "stream": winner_stream,
            }

        failure_meta = self._summarize_failure(attempt_results)
        if self._should_local_fallback("chat.completions", failure_meta):
            fallback_result = self._build_local_fallback_result("chat.completions", payload, str(failure_meta.get("error_type") or "upstream_failed"))
            fallback_payload = fallback_result.payload or {}
            content = ""
            try:
                content = str((fallback_payload.get("choices") or [{}])[0].get("message", {}).get("content", ""))
            except Exception:
                content = ""

            async def _fallback_stream() -> AsyncIterator[bytes]:
                chunk = {
                    "id": f"chatcmpl_local_{uuid4().hex[:12]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "local/fallback",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                terminal = {
                    "id": chunk["id"],
                    "object": "chat.completion.chunk",
                    "created": chunk["created"],
                    "model": "local/fallback",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(terminal, ensure_ascii=False)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

            attempt_results.append(fallback_result)
            return {
                "ok": True,
                "status_code": 200,
                "strategy": options.strategy,
                "mode": options.mode,
                "capability": "chat.completions",
                "latency_ms": round((time.monotonic() - started) * 1000.0, 2),
                "winner": {"provider": "local-fallback", "model": "local/fallback", "latency_ms": 0.0, "headers": dict(fallback_result.headers or {})},
                "results": [r.__dict__ for r in attempt_results],
                "stream": _fallback_stream(),
            }

        return {
            "ok": False,
            "status_code": int(failure_meta.get("status_code", 502)),
            "error_type": str(failure_meta.get("error_type", "upstream_failed")),
            "all_rate_limited": bool(failure_meta.get("all_rate_limited", False)),
            "strategy": options.strategy,
            "mode": options.mode,
            "capability": "chat.completions",
            "latency_ms": round((time.monotonic() - started) * 1000.0, 2),
            "winner": None,
            "results": [r.__dict__ for r in attempt_results],
            "stream": None,
        }

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
