from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict


try:
    import redis.asyncio as redis  # type: ignore
except Exception:  # noqa: BLE001
    redis = None


@dataclass
class WindowCounter:
    count: int
    reset_at: float


class RateLimitGuard:
    def __init__(self, *, redis_url: str, key_prefix: str, required: bool) -> None:
        self.key_prefix = key_prefix
        self._required = required
        self._memory: Dict[str, WindowCounter] = {}
        self._lock = asyncio.Lock()
        self._redis = None

        if redis is not None:
            self._redis = redis.from_url(redis_url, decode_responses=True)
        elif required:
            raise RuntimeError("Redis dependency missing but UAG_REDIS_REQUIRED=true")

    async def ping(self) -> bool:
        if self._redis is None:
            return not self._required
        try:
            pong = await self._redis.ping()
            return bool(pong)
        except Exception:  # noqa: BLE001
            if self._required:
                raise
            return False

    async def allow(self, scope: str, limit: int, window_sec: int = 60) -> bool:
        limit = max(1, int(limit))
        window_sec = max(1, int(window_sec))
        key = f"{self.key_prefix}:rl:{scope}:{int(time.time() // window_sec)}"

        if self._redis is not None:
            try:
                val = await self._redis.incr(key)
                if val == 1:
                    await self._redis.expire(key, window_sec + 1)
                return int(val) <= limit
            except Exception:  # noqa: BLE001
                if self._required:
                    raise

        now = time.monotonic()
        async with self._lock:
            current = self._memory.get(scope)
            if current is None or now >= current.reset_at:
                current = WindowCounter(count=0, reset_at=now + window_sec)
                self._memory[scope] = current
            current.count += 1
            return current.count <= limit

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
