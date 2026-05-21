from __future__ import annotations

import asyncio
import math
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

from ..providers.base import ProviderResult


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_token_usage(payload: Any) -> Dict[str, Optional[int]]:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    if isinstance(payload, dict):
        usage_meta = payload.get("usageMetadata")
        if isinstance(usage_meta, dict):
            prompt_tokens = _parse_int(usage_meta.get("promptTokenCount"))
            completion_tokens = _parse_int(usage_meta.get("candidatesTokenCount"))
            total_tokens = _parse_int(usage_meta.get("totalTokenCount"))

        usage = payload.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = prompt_tokens if prompt_tokens is not None else _parse_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
            completion_tokens = completion_tokens if completion_tokens is not None else _parse_int(usage.get("completion_tokens") or usage.get("output_tokens"))
            total_tokens = total_tokens if total_tokens is not None else _parse_int(usage.get("total_tokens"))

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _extract_rate_headers(headers: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in headers.items():
        lk = k.lower()
        if lk.startswith("x-ratelimit-") or lk.startswith("ratelimit-") or lk == "retry-after":
            out[lk] = str(v)
    return out


def _build_key_identity(provider: str, headers: Dict[str, str]) -> Tuple[str, Optional[int], Optional[str]]:
    key_slot = _parse_int(headers.get("x-proxy-key-slot") or headers.get("X-Proxy-Key-Slot"))
    key_mask = headers.get("x-proxy-key-mask") or headers.get("X-Proxy-Key-Mask")

    if key_mask:
        key_id = f"{provider}:{key_mask}"
    elif key_slot is not None:
        key_id = f"{provider}:slot:{key_slot}"
    else:
        key_id = f"{provider}:unknown"
    return key_id, key_slot, key_mask


class UsageTracker:
    def __init__(self, max_events: int = 20000) -> None:
        self._events: Deque[Dict[str, Any]] = deque(maxlen=max(1000, int(max_events)))
        self._latest_rate_by_key: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def record(
        self,
        *,
        capability: str,
        priority: int,
        result: ProviderResult,
    ) -> None:
        ts = _now()
        headers = dict(result.headers or {})
        rate_headers = _extract_rate_headers(headers)
        key_id, key_slot, key_mask = _build_key_identity(result.provider, headers)
        tokens = _extract_token_usage(result.payload)

        item = {
            "ts": ts.isoformat(),
            "provider": result.provider,
            "model": result.model or "",
            "capability": capability,
            "priority": int(priority),
            "ok": bool(result.ok),
            "status_code": int(result.status_code),
            "latency_ms": round(_safe_float(result.latency_ms), 3),
            "error": str(result.error or ""),
            "key": {
                "id": key_id,
                "slot": key_slot,
                "mask": key_mask,
            },
            "tokens": tokens,
            "rate_headers": rate_headers,
        }

        async with self._lock:
            self._events.append(item)
            if rate_headers:
                self._latest_rate_by_key[key_id] = {
                    "provider": result.provider,
                    "model": result.model or "",
                    "captured_at": ts.isoformat(),
                    "key": item["key"],
                    "headers": rate_headers,
                    "status_code": int(result.status_code),
                }

    @staticmethod
    def _within_since(ts_iso: str, since_minutes: int) -> bool:
        if since_minutes <= 0:
            return True
        try:
            ts = datetime.fromisoformat(ts_iso)
        except Exception:  # noqa: BLE001
            return False
        return ts >= (_now() - timedelta(minutes=since_minutes))

    def _filter(
        self,
        *,
        provider: Optional[str],
        model: Optional[str],
        key_id: Optional[str],
        capability: Optional[str],
        since_minutes: int,
    ) -> List[Dict[str, Any]]:
        provider_l = (provider or "").strip().lower()
        model_l = (model or "").strip().lower()
        key_l = (key_id or "").strip().lower()
        cap_l = (capability or "").strip().lower()

        items: List[Dict[str, Any]] = []
        for e in list(self._events):
            if since_minutes > 0 and not self._within_since(e.get("ts", ""), since_minutes):
                continue
            if provider_l and str(e.get("provider", "")).lower() != provider_l:
                continue
            if model_l and str(e.get("model", "")).lower() != model_l:
                continue
            if key_l and str((e.get("key") or {}).get("id", "")).lower() != key_l:
                continue
            if cap_l and str(e.get("capability", "")).lower() != cap_l:
                continue
            items.append(e)
        return items

    @staticmethod
    def _percentile(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        sorted_vals = sorted(values)
        idx = max(0, min(len(sorted_vals) - 1, int(math.ceil(q * len(sorted_vals)) - 1)))
        return sorted_vals[idx]

    def _aggregate_group(self, rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        rows_list = list(rows)
        count = len(rows_list)
        successes = sum(1 for r in rows_list if bool(r.get("ok")))
        status_codes = [int(r.get("status_code", 0)) for r in rows_list]
        latencies = [float(r.get("latency_ms", 0.0)) for r in rows_list]

        prompt_sum = 0
        completion_sum = 0
        total_sum = 0

        last_seen = None
        last_error = ""
        for r in rows_list:
            ts = r.get("ts")
            if isinstance(ts, str):
                if last_seen is None or ts > last_seen:
                    last_seen = ts
            if r.get("error"):
                last_error = str(r.get("error"))

            tokens = r.get("tokens") or {}
            pt = tokens.get("prompt_tokens")
            ct = tokens.get("completion_tokens")
            tt = tokens.get("total_tokens")
            if isinstance(pt, int):
                prompt_sum += pt
            if isinstance(ct, int):
                completion_sum += ct
            if isinstance(tt, int):
                total_sum += tt

        return {
            "requests_total": count,
            "success_total": successes,
            "error_total": count - successes,
            "success_rate": round((successes / count), 6) if count else 0.0,
            "status_2xx": sum(1 for s in status_codes if 200 <= s < 300),
            "status_4xx": sum(1 for s in status_codes if 400 <= s < 500),
            "status_5xx": sum(1 for s in status_codes if s >= 500),
            "status_429": sum(1 for s in status_codes if s == 429),
            "latency_avg_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "latency_p95_ms": round(self._percentile(latencies, 0.95), 3) if latencies else 0.0,
            "tokens_prompt_total": prompt_sum,
            "tokens_completion_total": completion_sum,
            "tokens_total": total_sum,
            "last_seen": last_seen,
            "last_error": last_error,
        }

    async def events(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        key_id: Optional[str] = None,
        capability: Optional[str] = None,
        since_minutes: int = 60,
        limit: int = 200,
    ) -> Dict[str, Any]:
        async with self._lock:
            filtered = self._filter(
                provider=provider,
                model=model,
                key_id=key_id,
                capability=capability,
                since_minutes=since_minutes,
            )
            filtered.sort(key=lambda x: x.get("ts", ""), reverse=True)
            out = filtered[: max(1, min(limit, 5000))]
            return {
                "generated_at": _now().isoformat(),
                "since_minutes": since_minutes,
                "count": len(out),
                "items": out,
            }

    async def aggregate(
        self,
        *,
        group_by: str,
        since_minutes: int = 60,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        capability: Optional[str] = None,
    ) -> Dict[str, Any]:
        valid_groups = {"provider", "model", "key", "provider_model", "provider_model_key", "capability"}
        if group_by not in valid_groups:
            raise ValueError(f"Unsupported group_by: {group_by}")

        async with self._lock:
            rows = self._filter(
                provider=provider,
                model=model,
                key_id=None,
                capability=capability,
                since_minutes=since_minutes,
            )

            buckets: Dict[str, List[Dict[str, Any]]] = {}
            for r in rows:
                p = str(r.get("provider", ""))
                m = str(r.get("model", ""))
                k = str((r.get("key") or {}).get("id", ""))
                c = str(r.get("capability", ""))

                if group_by == "provider":
                    g = p
                elif group_by == "model":
                    g = m
                elif group_by == "key":
                    g = k
                elif group_by == "provider_model":
                    g = f"{p}::{m}"
                elif group_by == "provider_model_key":
                    g = f"{p}::{m}::{k}"
                else:  # capability
                    g = c
                buckets.setdefault(g, []).append(r)

            items: List[Dict[str, Any]] = []
            for g, records in buckets.items():
                stats = self._aggregate_group(records)
                row = {
                    "group": g,
                    **stats,
                }

                # attach latest seen rate headers for key group
                if group_by == "key":
                    latest = self._latest_rate_by_key.get(g)
                    if latest:
                        row["latest_rate_headers"] = latest
                items.append(row)

            items.sort(key=lambda x: (x.get("requests_total", 0), x.get("group", "")), reverse=True)
            return {
                "generated_at": _now().isoformat(),
                "group_by": group_by,
                "since_minutes": since_minutes,
                "count": len(items),
                "items": items,
            }

    async def key_limits_latest(self, *, provider: Optional[str] = None) -> Dict[str, Any]:
        provider_l = (provider or "").strip().lower()
        async with self._lock:
            items = []
            for key_id, value in self._latest_rate_by_key.items():
                p = str(value.get("provider", "")).lower()
                if provider_l and p != provider_l:
                    continue
                items.append({"key_id": key_id, **value})
            items.sort(key=lambda x: x.get("captured_at", ""), reverse=True)
            return {
                "generated_at": _now().isoformat(),
                "count": len(items),
                "items": items,
            }

    async def overview(self, *, since_minutes: int = 60) -> Dict[str, Any]:
        async with self._lock:
            rows = self._filter(
                provider=None,
                model=None,
                key_id=None,
                capability=None,
                since_minutes=since_minutes,
            )
            return {
                "generated_at": _now().isoformat(),
                "since_minutes": since_minutes,
                "overall": self._aggregate_group(rows),
                "providers_count": len({str(r.get("provider", "")) for r in rows}),
                "models_count": len({str(r.get("model", "")) for r in rows}),
                "keys_count": len({str((r.get("key") or {}).get("id", "")) for r in rows}),
            }
