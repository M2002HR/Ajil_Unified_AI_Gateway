from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


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


class ProviderAdapter:
    name: str = "unknown"

    @staticmethod
    def started() -> float:
        return time.monotonic()

    @staticmethod
    def done(started_at: float) -> float:
        return (time.monotonic() - started_at) * 1000.0
