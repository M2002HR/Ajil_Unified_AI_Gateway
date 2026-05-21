from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..config import Settings
from .gemini_adapter import GeminiAdapter
from .groq_adapter import GroqAdapter
from .pollinations_adapter import PollinationsAdapter


@dataclass
class ProviderRegistry:
    providers: Dict[str, object]

    @classmethod
    def build(cls, settings: Settings) -> "ProviderRegistry":
        providers: Dict[str, object] = {}
        if settings.gemini.enabled:
            providers["gemini"] = GeminiAdapter(settings.gemini)
        if settings.groq.enabled:
            providers["groq"] = GroqAdapter(settings.groq)
        if settings.pollinations.enabled:
            providers["pollinations"] = PollinationsAdapter(settings.pollinations)
        return cls(providers=providers)

    def names(self) -> List[str]:
        return list(self.providers.keys())

    async def close(self) -> None:
        for adapter in self.providers.values():
            aclose = getattr(adapter, "aclose", None)
            if callable(aclose):
                await aclose()
