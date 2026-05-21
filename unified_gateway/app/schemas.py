from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RouterOptions(BaseModel):
    providers: List[str] = Field(default_factory=list)
    models: List[str] = Field(default_factory=list)
    strategy: Literal["fallback_chain", "parallel_race", "aggregate"] = "fallback_chain"
    mode: Literal["latency_first", "limit_safe", "quality_first"] = "latency_first"
    max_attempts: int = 6
    timeout_sec: float = 25.0
    result_policy: Literal["envelope_with_candidates", "best_only", "raw"] = "envelope_with_candidates"


class OrchestrateRequest(BaseModel):
    capability: Literal[
        "chat.completions",
        "responses",
        "embeddings",
        "images.generations",
        "audio.transcriptions",
        "audio.speech",
    ]
    payload: Dict[str, Any] = Field(default_factory=dict)
    x_router: Optional[RouterOptions] = None


class HealthResponse(BaseModel):
    status: str
    providers: Dict[str, bool]
    redis_ok: bool
