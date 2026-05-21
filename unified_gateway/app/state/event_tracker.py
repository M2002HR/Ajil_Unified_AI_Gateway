from __future__ import annotations

import asyncio
import math
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, Iterable, List, Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class EventTracker:
    def __init__(self, max_events: int = 50000) -> None:
        self._events: Deque[Dict[str, Any]] = deque(maxlen=max(1000, int(max_events)))
        self._seq: int = 0
        self._lock = asyncio.Lock()

    @staticmethod
    def _within_since(ts_iso: str, since_minutes: int) -> bool:
        if since_minutes <= 0:
            return True
        try:
            ts = datetime.fromisoformat(ts_iso)
        except Exception:  # noqa: BLE001
            return False
        return ts >= (_now() - timedelta(minutes=since_minutes))

    async def record(
        self,
        *,
        event_type: str,
        level: str,
        message: str,
        request_id: str = "",
        data: Optional[Dict[str, Any]] = None,
    ) -> int:
        seq = 0
        item = {
            "ts": _now().isoformat(),
            "event_type": str(event_type),
            "level": str(level).upper(),
            "message": str(message),
            "request_id": str(request_id or ""),
            "data": dict(data or {}),
        }
        async with self._lock:
            self._seq += 1
            seq = self._seq
            item["seq"] = seq
            self._events.append(item)
        return seq

    def _filter(
        self,
        *,
        since_minutes: int,
        level: Optional[str] = None,
        event_type: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        lv = (level or "").strip().upper()
        et = (event_type or "").strip()
        rid = (request_id or "").strip()
        out: List[Dict[str, Any]] = []
        for e in list(self._events):
            if since_minutes > 0 and not self._within_since(e.get("ts", ""), since_minutes):
                continue
            if lv and str(e.get("level", "")).upper() != lv:
                continue
            if et and str(e.get("event_type", "")) != et:
                continue
            if rid and str(e.get("request_id", "")) != rid:
                continue
            out.append(e)
        return out

    @staticmethod
    def _percentile(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        vals = sorted(values)
        idx = max(0, min(len(vals) - 1, int(math.ceil(q * len(vals)) - 1)))
        return vals[idx]

    async def events(
        self,
        *,
        since_minutes: int = 60,
        level: Optional[str] = None,
        event_type: Optional[str] = None,
        request_id: Optional[str] = None,
        after_seq: int = 0,
        limit: int = 200,
    ) -> Dict[str, Any]:
        async with self._lock:
            rows = self._filter(
                since_minutes=since_minutes,
                level=level,
                event_type=event_type,
                request_id=request_id,
            )
            if int(after_seq or 0) > 0:
                rows = [r for r in rows if _to_int(r.get("seq", 0), 0) > int(after_seq)]
            rows.sort(key=lambda x: x.get("ts", ""), reverse=True)
            items = rows[: max(1, min(limit, 5000))]
            return {
                "generated_at": _now().isoformat(),
                "since_minutes": since_minutes,
                "latest_seq": int(self._seq),
                "count": len(items),
                "items": items,
            }

    def _summarize_http_rows(self, rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        http_rows = [r for r in rows if str(r.get("event_type")) == "http.request.completed"]
        status_codes = [_to_int((r.get("data") or {}).get("status_code")) for r in http_rows]
        latencies = [_to_float((r.get("data") or {}).get("latency_ms")) for r in http_rows]
        total = len(http_rows)
        ok = sum(1 for s in status_codes if 200 <= s < 400)
        return {
            "requests_total": total,
            "success_total": ok,
            "error_total": total - ok,
            "success_rate": round((ok / total), 6) if total else 0.0,
            "status_2xx": sum(1 for s in status_codes if 200 <= s < 300),
            "status_3xx": sum(1 for s in status_codes if 300 <= s < 400),
            "status_4xx": sum(1 for s in status_codes if 400 <= s < 500),
            "status_5xx": sum(1 for s in status_codes if s >= 500),
            "latency_avg_ms": round((sum(latencies) / len(latencies)), 3) if latencies else 0.0,
            "latency_p95_ms": round(self._percentile(latencies, 0.95), 3) if latencies else 0.0,
        }

    async def summary(self, *, since_minutes: int = 60) -> Dict[str, Any]:
        async with self._lock:
            rows = self._filter(since_minutes=since_minutes)
            by_level: Dict[str, int] = {}
            by_type: Dict[str, int] = {}
            for r in rows:
                by_level[str(r.get("level", "INFO"))] = by_level.get(str(r.get("level", "INFO")), 0) + 1
                by_type[str(r.get("event_type", "unknown"))] = by_type.get(str(r.get("event_type", "unknown")), 0) + 1
            return {
                "generated_at": _now().isoformat(),
                "since_minutes": since_minutes,
                "latest_seq": int(self._seq),
                "events_total": len(rows),
                "by_level": by_level,
                "by_event_type": by_type,
                "http": self._summarize_http_rows(rows),
            }

    async def http_request_stats(self, *, group_by: str = "path", since_minutes: int = 60) -> Dict[str, Any]:
        valid = {"path", "method", "status_code", "status_class", "path_method"}
        if group_by not in valid:
            raise ValueError(f"Unsupported group_by: {group_by}")

        async with self._lock:
            rows = self._filter(since_minutes=since_minutes)
            http_rows = [r for r in rows if str(r.get("event_type")) == "http.request.completed"]
            buckets: Dict[str, List[Dict[str, Any]]] = {}
            for r in http_rows:
                d = r.get("data") or {}
                path = str(d.get("path", ""))
                method = str(d.get("method", ""))
                status_code = _to_int(d.get("status_code"))
                if group_by == "path":
                    key = path
                elif group_by == "method":
                    key = method
                elif group_by == "status_code":
                    key = str(status_code)
                elif group_by == "status_class":
                    key = f"{max(0, status_code // 100)}xx"
                else:
                    key = f"{method} {path}"
                buckets.setdefault(key, []).append(r)

            items: List[Dict[str, Any]] = []
            for key, records in buckets.items():
                stats = self._summarize_http_rows(records)
                items.append({"group": key, **stats})
            items.sort(key=lambda x: (x.get("requests_total", 0), x.get("group", "")), reverse=True)
            return {
                "generated_at": _now().isoformat(),
                "since_minutes": since_minutes,
                "latest_seq": int(self._seq),
                "group_by": group_by,
                "count": len(items),
                "items": items,
            }
