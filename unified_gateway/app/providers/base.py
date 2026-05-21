from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional


@dataclass
class ProviderResult:
    provider: str
    capability: str
    ok: bool
    status_code: int
    latency_ms: float
    payload: Any = None
    error: str = ""
    model: str = ""
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProviderStreamResult:
    provider: str
    capability: str
    ok: bool
    status_code: int
    latency_ms: float
    model: str = ""
    error: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    stream: Optional[AsyncIterator[bytes]] = None


class ProviderAdapter:
    name: str = "unknown"

    @staticmethod
    def started() -> float:
        return time.monotonic()

    @staticmethod
    def done(started_at: float) -> float:
        return (time.monotonic() - started_at) * 1000.0
