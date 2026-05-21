from __future__ import annotations

import contextvars
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict


request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_request_id(value: str) -> contextvars.Token[str]:
    return request_id_ctx.set(value)


def reset_request_id(token: contextvars.Token[str]) -> None:
    request_id_ctx.reset(token)


def get_request_id() -> str:
    return request_id_ctx.get()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": now_iso(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = get_request_id()
        if request_id:
            payload["request_id"] = request_id
        event = getattr(record, "event", None)
        if event:
            payload["event"] = event
        extra = getattr(record, "extra_data", None)
        if isinstance(extra, dict) and extra:
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(*, level: str = "INFO", use_json: bool = True, enabled: bool = True) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    if not enabled:
        root.setLevel(logging.CRITICAL)
        return

    root.setLevel(level.upper())

    handler = logging.StreamHandler()
    if use_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    root.addHandler(handler)


def log_event(
    logger: logging.Logger,
    *,
    event: str,
    level: int = logging.INFO,
    message: str = "",
    **fields: Any,
) -> None:
    logger.log(level, message or event, extra={"event": event, "extra_data": fields})
