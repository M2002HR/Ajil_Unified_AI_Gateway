from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator, Dict

from modules.groq_proxy.api.app.config import (  # type: ignore
    AdminSection,
    AppSection,
    GroqSection,
    ProxySection,
    Settings as GroqSettings,
)
from modules.groq_proxy.api.app.services import GroqProxyService  # type: ignore

from ..config import GroqProviderConfig
from .base import ProviderAdapter, ProviderResult, ProviderStreamResult


class GroqAdapter(ProviderAdapter):
    name = "groq"

    def __init__(self, cfg: GroqProviderConfig) -> None:
        keys = list(cfg.api_keys)
        if not keys:
            keys = ["__missing__"]

        settings = GroqSettings(
            app=AppSection(request_timeout_sec=90),
            proxy=ProxySection(
                retry_on_429=cfg.retry_on_429,
                retry_on_5xx=cfg.retry_on_5xx,
                max_retries_per_key=cfg.max_retries_per_key,
                max_retries_on_5xx=cfg.max_retries_on_5xx,
                retry_backoff_sec=cfg.retry_backoff_sec,
                cooloff_sec=cfg.cooloff_sec,
                min_interval_sec=cfg.min_interval_sec,
            ),
            groq=GroqSection(
                base_url=cfg.base_url,
                api_key=keys[0],
                api_keys=keys,
                stt_primary_model=cfg.stt_primary_model,
                stt_fallback_model=cfg.stt_fallback_model,
                stt_language=cfg.stt_language,
                stt_temperature=cfg.stt_temperature,
                stt_response_format=cfg.stt_response_format,
                stt_prompt=cfg.stt_prompt,
                tts_default_model=cfg.tts_default_model,
                tts_default_voice=cfg.tts_default_voice,
                tts_default_response_format=cfg.tts_default_response_format,
            ),
            admin=AdminSection(enabled=False),
        )
        self.tts_default_model = str(cfg.tts_default_model or "canopylabs/orpheus-v1-english")
        self.tts_default_voice = str(cfg.tts_default_voice or "diana")
        self.tts_default_response_format = str(cfg.tts_default_response_format or "wav")
        self.service = GroqProxyService(settings)

    async def aclose(self) -> None:
        await self.service.aclose()

    @staticmethod
    def _parse(resp: Any) -> Any:
        ctype = str(resp.headers.get("content-type", "application/json")).lower()
        if "application/json" in ctype:
            try:
                return resp.json()
            except json.JSONDecodeError:
                return {"raw": resp.text}
        return resp.content

    @staticmethod
    def _extract_error_payload(parsed: Any) -> Dict[str, Any]:
        if isinstance(parsed, dict):
            err = parsed.get("error")
            if isinstance(err, dict):
                return err
        return {}

    def _tts_fallback_candidates(self, requested_model: str) -> list[str]:
        req = str(requested_model or "").strip().lower()
        preferred = (
            "canopylabs/orpheus-arabic-saudi"
            if "arabic" in req
            else "canopylabs/orpheus-v1-english"
        )
        ordered = [
            preferred,
            self.tts_default_model,
            "canopylabs/orpheus-v1-english",
            "canopylabs/orpheus-arabic-saudi",
        ]
        out: list[str] = []
        for model_name in ordered:
            m = str(model_name or "").strip()
            if m and m not in out:
                out.append(m)
        return out

    @staticmethod
    def _extract_allowed_voices(message: str) -> list[str]:
        msg = str(message or "")
        match = re.search(r"voices?\s*:\s*\[([^\]]+)\]", msg, flags=re.IGNORECASE)
        if not match:
            return []
        values = [v.strip().strip(",").lower() for v in match.group(1).split() if v.strip().strip(",")]
        out: list[str] = []
        for value in values:
            if value and value not in out:
                out.append(value)
        return out

    async def _tts_request_with_auto_fix(self, body: Dict[str, Any]) -> tuple[Any, Any, Dict[str, Any]]:
        current = dict(body)
        for _ in range(3):
            resp = await self.service.forward_json(path="/audio/speech", payload=current)
            parsed = self._parse(resp)
            if resp.status_code < 400:
                return resp, parsed, current

            err = self._extract_error_payload(parsed)
            err_message = str(err.get("message") or "")
            low = err_message.lower()
            changed = False

            if "response_format must be one of [wav]" in low and str(current.get("response_format", "")).lower() != "wav":
                current["response_format"] = "wav"
                changed = True

            allowed_voices = self._extract_allowed_voices(err_message)
            if allowed_voices:
                current_voice = str(current.get("voice", "")).strip().lower()
                if current_voice not in allowed_voices:
                    preferred_voice = self.tts_default_voice.strip().lower()
                    current["voice"] = preferred_voice if preferred_voice in allowed_voices else allowed_voices[0]
                    changed = True

            if not changed:
                return resp, parsed, current

        resp = await self.service.forward_json(path="/audio/speech", payload=current)
        return resp, self._parse(resp), current

    async def chat_completions(self, payload: Dict[str, Any], model: str | None = None) -> ProviderResult:
        start = self.started()
        body = dict(payload)
        if model:
            body["model"] = model
        try:
            resp = await self.service.forward_json(path="/chat/completions", payload=body)
            parsed = self._parse(resp)
            return ProviderResult(self.name, "chat.completions", resp.status_code < 400, resp.status_code, self.done(start), parsed, "" if resp.status_code < 400 else str(parsed), str(body.get("model", "")), dict(resp.headers))
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(self.name, "chat.completions", False, 502, self.done(start), None, str(exc), str(body.get("model", "")))

    async def chat_completions_stream(
        self,
        payload: Dict[str, Any],
        model: str | None = None,
        *,
        timeout_sec: float | None = None,
    ) -> ProviderStreamResult:
        start = self.started()
        body = dict(payload)
        if model:
            body["model"] = model
        body["stream"] = True

        retries = max(0, int(self.service.settings.proxy.max_retries_per_key) - 1)
        for _ in range(retries + 1):
            await self.service.limiter.wait()
            key = self.service.key_rr.active
            headers = self.service._build_headers(key=key, extra_headers={"accept": "text/event-stream"})
            url = self.service._join_url(self.service.settings.groq.base_url, "/chat/completions")
            request = self.service.client.build_request("POST", url, headers=headers, json=body)
            try:
                # httpx.AsyncClient.send() in 0.28.x does not accept a timeout kwarg.
                # Per-attempt timeout is enforced by router asyncio.wait_for() wrapping this call.
                resp = await self.service.client.send(request, stream=True)
            except Exception as exc:  # noqa: BLE001
                self.service.key_rr.next()
                return ProviderStreamResult(
                    provider=self.name,
                    capability="chat.completions",
                    ok=False,
                    status_code=502,
                    latency_ms=self.done(start),
                    error=str(exc),
                    model=str(body.get("model", "")),
                )

            key_slot = self.service.key_rr.slot
            key_mask = self.service._mask_key(key)
            resp.headers["x-proxy-key-slot"] = str(key_slot)
            resp.headers["x-proxy-key-mask"] = key_mask
            resp.headers["x-proxy-key-pool-size"] = str(len(self.service.key_rr.values))
            resp.headers["x-proxy-attempts"] = "1"
            resp.headers["x-proxy-key-rotated"] = "false"

            if resp.status_code < 400:
                async def _stream() -> AsyncIterator[bytes]:
                    try:
                        async for line in resp.aiter_lines():
                            if line is None:
                                continue
                            if line.strip() == "":
                                continue
                            yield f"{line}\n\n".encode("utf-8")
                    finally:
                        await resp.aclose()

                return ProviderStreamResult(
                    provider=self.name,
                    capability="chat.completions",
                    ok=True,
                    status_code=200,
                    latency_ms=self.done(start),
                    model=str(body.get("model", "")),
                    headers=dict(resp.headers),
                    stream=_stream(),
                )

            error_text = ""
            try:
                parsed = await resp.aread()
                error_text = parsed.decode("utf-8", errors="ignore")
            finally:
                await resp.aclose()
            should_retry = (
                resp.status_code == 429
                or (
                    bool(self.service.settings.proxy.retry_on_5xx)
                    and resp.status_code >= 500
                )
            )
            if should_retry:
                self.service.key_rr.next()
                if retries > 0:
                    await asyncio.sleep(max(0.0, float(self.service.settings.proxy.retry_backoff_sec)))
                continue
            return ProviderStreamResult(
                provider=self.name,
                capability="chat.completions",
                ok=False,
                status_code=int(resp.status_code),
                latency_ms=self.done(start),
                error=error_text or f"upstream status={resp.status_code}",
                model=str(body.get("model", "")),
                headers=dict(resp.headers),
            )

        return ProviderStreamResult(
            provider=self.name,
            capability="chat.completions",
            ok=False,
            status_code=429,
            latency_ms=self.done(start),
            error="all stream retries exhausted",
            model=str(body.get("model", "")),
        )

    async def responses(self, payload: Dict[str, Any], model: str | None = None) -> ProviderResult:
        start = self.started()
        body = dict(payload)
        if model:
            body["model"] = model
        try:
            resp = await self.service.forward_json(path="/responses", payload=body)
            parsed = self._parse(resp)
            if resp.status_code < 400:
                return ProviderResult(self.name, "responses", True, resp.status_code, self.done(start), parsed, "", str(body.get("model", "")), dict(resp.headers))

            # Stability fallback: map OpenAI-style responses payload to chat/completions.
            if int(resp.status_code) in {400, 404, 405, 422}:
                mapped = dict(body)
                if "messages" not in mapped:
                    input_value = mapped.get("input")
                    if isinstance(input_value, list):
                        text_parts = []
                        for item in input_value:
                            if isinstance(item, dict):
                                txt = item.get("content") or item.get("text")
                                if txt is not None:
                                    text_parts.append(str(txt))
                            elif item is not None:
                                text_parts.append(str(item))
                        mapped["messages"] = [{"role": "user", "content": "\n".join([t for t in text_parts if t]).strip()}]
                    else:
                        mapped["messages"] = [{"role": "user", "content": str(input_value or "")}]
                mapped.pop("input", None)
                mapped.pop("instructions", None)
                fallback_resp = await self.service.forward_json(path="/chat/completions", payload=mapped)
                fallback_parsed = self._parse(fallback_resp)
                headers = dict(fallback_resp.headers)
                headers["x-uag-fallback"] = "groq.responses->chat.completions"
                return ProviderResult(
                    self.name,
                    "responses",
                    fallback_resp.status_code < 400,
                    fallback_resp.status_code,
                    self.done(start),
                    fallback_parsed,
                    "" if fallback_resp.status_code < 400 else str(fallback_parsed),
                    str(mapped.get("model", "")),
                    headers,
                )

            return ProviderResult(self.name, "responses", False, resp.status_code, self.done(start), parsed, str(parsed), str(body.get("model", "")), dict(resp.headers))
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(self.name, "responses", False, 502, self.done(start), None, str(exc), str(body.get("model", "")))

    async def embeddings(self, payload: Dict[str, Any], model: str | None = None) -> ProviderResult:
        start = self.started()
        body = dict(payload)
        if model:
            body["model"] = model
        try:
            resp = await self.service.forward_json(path="/embeddings", payload=body)
            parsed = self._parse(resp)
            return ProviderResult(self.name, "embeddings", resp.status_code < 400, resp.status_code, self.done(start), parsed, "" if resp.status_code < 400 else str(parsed), str(body.get("model", "")), dict(resp.headers))
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(self.name, "embeddings", False, 502, self.done(start), None, str(exc), str(body.get("model", "")))

    async def tts(self, payload: Dict[str, Any], model: str | None = None) -> ProviderResult:
        start = self.started()
        body = dict(payload)
        if model:
            body["model"] = model
        body.setdefault("voice", self.tts_default_voice)
        body.setdefault("response_format", self.tts_default_response_format)
        try:
            resp, parsed, used_body = await self._tts_request_with_auto_fix(body)
            if resp.status_code < 400:
                return ProviderResult(self.name, "audio.speech", True, resp.status_code, self.done(start), parsed, "", str(used_body.get("model", "")), dict(resp.headers))

            err = self._extract_error_payload(parsed)
            err_code = str(err.get("code") or "").strip().lower()
            err_message = str(err.get("message") or "").strip().lower()
            requested_model = str(used_body.get("model", "")).strip()
            should_retry = err_code == "model_decommissioned" or "decommissioned" in err_message

            if should_retry:
                for fallback_model in self._tts_fallback_candidates(requested_model):
                    if fallback_model == requested_model:
                        continue
                    retry_body = dict(used_body)
                    retry_body["model"] = fallback_model
                    retry_resp, retry_parsed, retry_used_body = await self._tts_request_with_auto_fix(retry_body)
                    if retry_resp.status_code < 400:
                        headers = dict(retry_resp.headers)
                        headers["x-uag-tts-fallback-from"] = requested_model
                        headers["x-uag-tts-fallback-to"] = str(retry_used_body.get("model", fallback_model))
                        return ProviderResult(
                            self.name,
                            "audio.speech",
                            True,
                            retry_resp.status_code,
                            self.done(start),
                            retry_parsed,
                            "",
                            str(retry_used_body.get("model", fallback_model)),
                            headers,
                        )
            return ProviderResult(self.name, "audio.speech", False, resp.status_code, self.done(start), parsed, str(parsed), str(used_body.get("model", "")), dict(resp.headers))
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(self.name, "audio.speech", False, 502, self.done(start), None, str(exc), str(body.get("model", "")))

    async def stt(self, *, audio_bytes: bytes, filename: str, content_type: str, language: str | None) -> ProviderResult:
        start = self.started()
        try:
            parsed = await self.service.transcribe(
                audio_bytes=audio_bytes,
                filename=filename,
                content_type=content_type,
                language=language,
                prompt=None,
                response_format=None,
                temperature=None,
            )
            model_used = ""
            if isinstance(parsed, dict):
                model_used = str(parsed.get("model_used") or "")
            return ProviderResult(self.name, "audio.transcriptions", True, 200, self.done(start), parsed, "", model_used)
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(self.name, "audio.transcriptions", False, 502, self.done(start), None, str(exc))

    async def list_models(self) -> ProviderResult:
        start = self.started()
        try:
            resp = await self.service.forward_models(path="/models")
            parsed = self._parse(resp)
            return ProviderResult(self.name, "models", resp.status_code < 400, resp.status_code, self.done(start), parsed, "" if resp.status_code < 400 else str(parsed), headers=dict(resp.headers))
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(self.name, "models", False, 502, self.done(start), None, str(exc))
