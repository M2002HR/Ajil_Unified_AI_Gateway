#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime
import json
import mimetypes
import os
from pathlib import Path
import random
import statistics
import string
import sys
import time
import wave
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import httpx


TEXT_PROMPTS = [
    "در یک پاراگراف کوتاه توضیح بده این موضوع چیست.",
    "این موضوع را در دو بولت مهم خلاصه کن.",
    "یک چک لیست عملی کوتاه برای اجرای این کار بنویس.",
    "یک پاسخ کوتاه با یک مثال کاربردی بده.",
    "این متن را به فارسی روان و طبیعی بازنویسی کن.",
    "دو گزینه را کوتاه مقایسه کن و بهترین را پیشنهاد بده.",
    "یک راهنمای عیب یابی مرحله ای برای این مشکل بده.",
    "خروجی را به فرمت JSON با نکات کلیدی برگردان.",
]

EMBED_TEXTS = [
    "تست بار گیت وی هوش مصنوعی",
    "مسیردهی درخواست و فالبک در API",
    "مدیریت ریت لیمیت با چندین کلید",
    "پایش، لاگ و رهگیری درخواست ها",
    "امبدینگ متنی برای جستجوی معنایی",
    "مانیتورینگ FastAPI و وب سوکت",
]

IMAGE_PROMPTS = [
    "یک آیکن مینیمال نارنجی برای سرویس API روی پس زمینه سفید",
    "یک دیاگرام ساده از مسیردهی درخواست ها در گیت وی",
    "یک تصویر اینفوگرافیک تمیز درباره لود بالانس مدل ها",
    "یک لوگوی کوچک و مدرن با تم نارنجی برای سامانه AI",
]

TTS_TEXTS = [
    "سلام، این یک نمونه برای تست بار تبدیل متن به گفتار است.",
    "وضعیت سامانه پایدار است و تاخیر در محدوده قابل قبول قرار دارد.",
    "این درخواست برای بررسی توان عملیاتی تبدیل متن به صدا ارسال می شود.",
    "این رویداد باید در داشبورد مانیتورینگ به صورت زنده نمایش داده شود.",
]


@dataclass
class RequestOutcome:
    request_id: int
    scenario: str
    endpoint: str
    status_code: int
    ok: bool
    latency_ms: float
    error: str = ""
    artifact_dir: str = ""


@dataclass
class ScenarioPlan:
    name: str
    weight: int


class LoadTester:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.random = random.Random(args.seed)
        self.total = int(args.total)
        self.done = 0
        self.success = 0
        self.fail = 0
        self.started_at = 0.0

        self.image_sent = 0
        self.speech_sent = 0
        self.stt_sent = 0

        self.outcomes: List[RequestOutcome] = []
        self.http_status_counts: Dict[int, int] = {}
        self.scenario_counts: Dict[str, int] = {}

        self.print_lock = asyncio.Lock()
        self.result_lock = asyncio.Lock()

        self.chat_models_by_provider: Dict[str, List[str]] = {
            "gemini": self._split_csv(args.gemini_models),
            "groq": self._split_csv(args.groq_models),
        }
        self.embedding_models_by_provider: Dict[str, List[str]] = {
            "gemini": self._split_csv(args.embedding_models),
        }
        self.image_models: List[str] = self._split_csv(args.image_models)
        self._stt_sample_cache: Optional[Tuple[str, bytes, str]] = None

        self.run_stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(str(args.results_dir)).expanduser() / f"{args.report_prefix}_{self.run_stamp}"
        self.requests_dir = self.run_dir / "requests"
        self.requests_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _split_csv(raw: str) -> List[str]:
        return [item.strip() for item in str(raw or "").split(",") if item.strip()]

    @staticmethod
    def _normalize_proxy_url(url: str) -> str:
        out = str(url or "").strip()
        if out.startswith("socks://"):
            out = "socks5://" + out[len("socks://") :]
        return out

    def _sanitize_proxy_env(self) -> None:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            raw = os.getenv(key)
            if not raw:
                continue
            fixed = self._normalize_proxy_url(raw)
            if fixed != raw:
                os.environ[key] = fixed

    @staticmethod
    def _now_ms() -> float:
        return time.perf_counter() * 1000.0

    def _auth_headers(self) -> Dict[str, str]:
        return {self.args.token_header: self.args.token}

    def _admin_headers(self) -> Dict[str, str]:
        if not self.args.admin_token:
            return {}
        return {self.args.admin_token_header: self.args.admin_token}

    async def _log(self, line: str) -> None:
        async with self.print_lock:
            print(line, flush=True)

    def _random_text(self) -> str:
        return self.random.choice(TEXT_PROMPTS)

    def _choose(self, items: List[str], fallback: str) -> str:
        if not items:
            return fallback
        return self.random.choice(items)

    def _normalize_model_name(self, provider: str, raw: str) -> str:
        out = str(raw or "").strip()
        if out.startswith("models/"):
            out = out[len("models/") :]
        if "/" in out:
            prefix, rest = out.split("/", 1)
            if prefix.strip().lower() == provider.lower():
                out = rest.strip()
        return out

    def _extract_model_names(self, value: Any) -> List[str]:
        found: set[str] = set()

        def walk(obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    key = str(k).lower()
                    if key in {"id", "name", "model", "base_model"} and isinstance(v, str):
                        vv = v.strip()
                        if vv and len(vv) > 2:
                            found.add(vv)
                    walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(value)
        return sorted(found)

    @staticmethod
    def _extract_pollinations_image_models(payload: Any) -> List[str]:
        if not isinstance(payload, dict):
            return []
        rows = payload.get("data")
        if not isinstance(rows, list):
            return []

        out: List[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            model_name = str(row.get("name") or row.get("id") or "").strip()
            if not model_name:
                continue
            output_modalities = [str(v).lower() for v in (row.get("output_modalities") or [])]
            capability = str(row.get("capability") or "").lower()
            if "image" in output_modalities or "image" in capability:
                if model_name not in out:
                    out.append(model_name)
        return out

    async def discover_models(self, client: httpx.AsyncClient) -> None:
        if not self.args.discover_models:
            await self._log("[discover] model discovery disabled")
            return

        try:
            resp = await client.get(
                f"{self.args.base_url.rstrip('/')}/v1/models",
                headers=self._auth_headers(),
                timeout=self.args.timeout_sec,
            )
        except Exception as exc:  # noqa: BLE001
            await self._log(f"[discover] failed to query /v1/models: {exc}")
            return

        if resp.status_code >= 400:
            await self._log(f"[discover] /v1/models returned {resp.status_code}; using fallback model lists")
            return

        payload = resp.json()
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            await self._log("[discover] unexpected /v1/models payload; using fallback model lists")
            return

        discovered: Dict[str, List[str]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            provider = str(row.get("provider") or "").strip().lower()
            if not provider:
                continue
            payload_block = row.get("payload")
            if provider == "pollinations":
                names = self._extract_pollinations_image_models(payload_block)
            else:
                names = self._extract_model_names(payload_block)
            normalized = [self._normalize_model_name(provider, name) for name in names]
            cleaned = [name for name in normalized if name]
            if cleaned:
                discovered[provider] = cleaned

        gemini_discovered = discovered.get("gemini", [])
        if gemini_discovered:
            gemma4 = [m for m in gemini_discovered if "gemma-4" in m.lower()]
            gemma = [m for m in gemini_discovered if "gemma" in m.lower() and m not in gemma4]
            flash = [m for m in gemini_discovered if "flash" in m.lower()]
            ordered = gemma4 + flash + gemma
            if ordered:
                self.chat_models_by_provider["gemini"] = ordered[: max(1, self.args.max_models_per_provider)]

        groq_discovered = discovered.get("groq", [])
        if groq_discovered:
            preferred = [m for m in groq_discovered if any(tag in m.lower() for tag in ["llama", "gpt-oss", "qwen", "mixtral"])]
            fallback = preferred or groq_discovered
            self.chat_models_by_provider["groq"] = fallback[: max(1, self.args.max_models_per_provider)]

        emb_discovered = discovered.get("gemini", [])
        emb = [m for m in emb_discovered if "embedding" in m.lower()]
        if emb:
            self.embedding_models_by_provider["gemini"] = emb[: max(1, self.args.max_models_per_provider)]

        poll_discovered = discovered.get("pollinations", [])
        if poll_discovered:
            self.image_models = poll_discovered[: max(1, self.args.max_models_per_provider)]

        await self._log(
            "[discover] selected models | "
            + f"gemini={self.chat_models_by_provider.get('gemini', [])} "
            + f"groq={self.chat_models_by_provider.get('groq', [])} "
            + f"emb={self.embedding_models_by_provider.get('gemini', [])} "
            + f"img={self.image_models}"
        )

    def _candidate_pair(self) -> List[Dict[str, Any]]:
        g_model = self._choose(self.chat_models_by_provider.get("gemini", []), "gemma-3-27b-it")
        q_model = self._choose(self.chat_models_by_provider.get("groq", []), "llama-3.3-70b-versatile")

        entries = [
            {"provider": "gemini", "model": g_model, "priority": 0},
            {"provider": "groq", "model": q_model, "priority": 1},
        ]
        if self.random.random() < 0.35:
            entries[1]["priority"] = 0
        if self.random.random() < 0.5:
            self.random.shuffle(entries)
        return entries

    def _routing_block(self) -> Dict[str, Any]:
        strategies = ["fallback_chain", "parallel_race", "aggregate"]
        modes = ["latency_first", "limit_safe", "quality_first"]
        strategy = self.random.choices(strategies, weights=[70, 20, 10], k=1)[0]
        mode = self.random.choice(modes)
        return {
            "strategy": strategy,
            "mode": mode,
            "timeout_sec": self.args.router_timeout_sec,
            "max_attempts": self.args.router_max_attempts,
        }

    def _scenarios(self) -> List[ScenarioPlan]:
        return [
            ScenarioPlan("chat", self.args.weight_chat),
            ScenarioPlan("responses", self.args.weight_responses),
            ScenarioPlan("embeddings", self.args.weight_embeddings),
            ScenarioPlan("orchestrate", self.args.weight_orchestrate),
            ScenarioPlan("image", self.args.weight_images),
            ScenarioPlan("speech", self.args.weight_speech),
            ScenarioPlan("stt", self.args.weight_stt),
        ]

    def _select_scenario(self) -> str:
        plans = self._scenarios()
        if self.image_sent >= self.args.max_image_requests:
            plans = [p for p in plans if p.name != "image"]
        if self.speech_sent >= self.args.max_speech_requests:
            plans = [p for p in plans if p.name != "speech"]
        if self.stt_sent >= self.args.max_stt_requests:
            plans = [p for p in plans if p.name != "stt"]

        if not plans:
            return "chat"

        names = [p.name for p in plans]
        weights = [max(0, p.weight) for p in plans]
        if sum(weights) <= 0:
            return self.random.choice(names)
        return self.random.choices(names, weights=weights, k=1)[0]

    @staticmethod
    def _silence_wav_bytes(duration_ms: int = 800, sample_rate: int = 16000) -> bytes:
        frame_count = int(max(100, sample_rate * (duration_ms / 1000.0)))
        raw = BytesIO()
        with wave.open(raw, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(b"\x00\x00" * frame_count)
        return raw.getvalue()

    def _stt_sample(self) -> Tuple[str, bytes, str]:
        if self._stt_sample_cache is not None:
            return self._stt_sample_cache

        sample_path = Path(str(self.args.stt_sample_file or "")).expanduser()
        if sample_path.is_file():
            payload_bytes = sample_path.read_bytes()
            ctype = mimetypes.guess_type(str(sample_path))[0] or "application/octet-stream"
            self._stt_sample_cache = (sample_path.name, payload_bytes, ctype)
            return self._stt_sample_cache

        fallback_name = "sample.wav"
        fallback_bytes = self._silence_wav_bytes(duration_ms=self.random.randint(600, 1200))
        self._stt_sample_cache = (fallback_name, fallback_bytes, "audio/wav")
        return self._stt_sample_cache

    def _request_payload(self, scenario: str) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Tuple[str, bytes, str]]], Optional[Dict[str, str]]]:
        rid = "".join(self.random.choices(string.ascii_lowercase + string.digits, k=8))

        if scenario == "chat":
            body = {
                "model": self._candidate_pair(),
                "messages": [{"role": "user", "content": f"[{rid}] {self._random_text()}"}],
                "temperature": round(self.random.uniform(0.1, 0.8), 2),
                "x_router": self._routing_block(),
            }
            return "/v1/chat/completions", body, None, None

        if scenario == "responses":
            body = {
                "model": self._candidate_pair(),
                "input": f"[{rid}] {self._random_text()}",
                "x_router": self._routing_block(),
            }
            return "/v1/responses", body, None, None

        if scenario == "embeddings":
            emb_model = self._choose(self.embedding_models_by_provider.get("gemini", []), "gemini-embedding-001")
            body = {
                "model": [{"provider": "gemini", "model": emb_model, "priority": 0}],
                "input": self.random.choice(EMBED_TEXTS),
                "x_router": {
                    "providers": ["gemini"],
                    "strategy": "fallback_chain",
                    "mode": "latency_first",
                },
            }
            return "/v1/embeddings", body, None, None

        if scenario == "orchestrate":
            body = {
                "capability": "chat.completions",
                "payload": {
                    "model": self._candidate_pair(),
                    "messages": [{"role": "user", "content": f"[{rid}] {self._random_text()}"}],
                    "temperature": round(self.random.uniform(0.1, 0.8), 2),
                },
                "x_router": self._routing_block(),
            }
            return "/v1/orchestrate", body, None, None

        if scenario == "image":
            model = self._choose(self.image_models, "flux")
            body = {
                "model": [{"provider": "pollinations", "model": model, "priority": 0}],
                "prompt": self.random.choice(IMAGE_PROMPTS),
                "size": self.random.choice(["512x512", "768x768"]),
                "quality": self.random.choice(["low", "medium"]),
                "response_format": "b64_json",
                "x_router": {"providers": ["pollinations"], "strategy": "fallback_chain", "mode": "latency_first"},
            }
            return "/v1/images/generations", body, None, None

        if scenario == "speech":
            body = {
                "model": [{"provider": "groq", "model": "canopylabs/orpheus-v1-english", "priority": 0}],
                "voice": self.random.choice(self._split_csv(self.args.tts_voices)),
                "input": self.random.choice(TTS_TEXTS),
                "response_format": "wav",
                "x_router": {"providers": ["groq"], "strategy": "fallback_chain", "mode": "latency_first"},
            }
            return "/v1/audio/speech", body, None, None

        if scenario == "stt":
            data = {"language": "fa"}
            filename, sample_bytes, content_type = self._stt_sample()
            files = {"file": (filename, sample_bytes, content_type)}
            return "/v1/audio/transcriptions", {}, files, data

        body = {
            "model": self._candidate_pair(),
            "messages": [{"role": "user", "content": f"[{rid}] {self._random_text()}"}],
            "x_router": self._routing_block(),
        }
        return "/v1/chat/completions", body, None, None

    def _extract_texts(self, payload: Any) -> List[str]:
        out: List[str] = []

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    low = str(key).lower()
                    if low in {"content", "text", "output_text", "transcript"} and isinstance(value, str):
                        vv = value.strip()
                        if vv:
                            out.append(vv)
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(payload)
        uniq: List[str] = []
        for item in out:
            if item not in uniq:
                uniq.append(item)
        return uniq

    async def _persist_request_artifacts(
        self,
        *,
        client: httpx.AsyncClient,
        request_id: int,
        scenario: str,
        endpoint: str,
        request_body: Dict[str, Any],
        request_data: Optional[Dict[str, str]],
        request_files: Optional[Dict[str, Tuple[str, bytes, str]]],
        response: Optional[httpx.Response],
        latency_ms: float,
        error: str,
    ) -> str:
        req_dir = self.requests_dir / f"{request_id:05d}_{scenario}"
        req_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "request_id": request_id,
            "scenario": scenario,
            "endpoint": endpoint,
            "latency_ms": round(latency_ms, 3),
            "error": error,
        }
        if response is not None:
            meta["status_code"] = response.status_code
            meta["content_type"] = str(response.headers.get("content-type", ""))
        (req_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        req_snapshot: Dict[str, Any] = {"endpoint": endpoint}
        if request_body:
            req_snapshot["json"] = request_body
        if request_data:
            req_snapshot["form"] = request_data
        if request_files:
            req_snapshot["files"] = {
                name: {"filename": tup[0], "content_type": tup[2], "size": len(tup[1])}
                for name, tup in request_files.items()
            }
        (req_dir / "request.json").write_text(json.dumps(req_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

        if response is None:
            return str(req_dir)

        ctype = str(response.headers.get("content-type", "")).lower()
        if "application/json" in ctype:
            try:
                payload = response.json()
            except Exception:
                payload = {"raw_text": response.text}

            (req_dir / "response.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            texts = self._extract_texts(payload)
            if texts:
                (req_dir / "generated_text.txt").write_text("\n\n---\n\n".join(texts), encoding="utf-8")

            image_count = 0

            async def save_image_url(url: str, out_path: Path) -> None:
                try:
                    resp = await client.get(url, timeout=self.args.timeout_sec)
                    if resp.status_code < 400:
                        out_path.write_bytes(resp.content)
                except Exception:
                    pass

            url_tasks: List[asyncio.Task[Any]] = []

            def walk(node: Any) -> None:
                nonlocal image_count
                if isinstance(node, dict):
                    b64_data = node.get("b64_json")
                    if isinstance(b64_data, str) and b64_data.strip():
                        try:
                            img = base64.b64decode(b64_data)
                            outp = req_dir / f"generated_image_{image_count:02d}.png"
                            outp.write_bytes(img)
                            image_count += 1
                        except Exception:
                            pass
                    url = node.get("url")
                    if isinstance(url, str) and url.startswith("http"):
                        outp = req_dir / f"generated_image_{image_count:02d}.bin"
                        image_count += 1
                        url_tasks.append(asyncio.create_task(save_image_url(url, outp)))
                    for value in node.values():
                        walk(value)
                elif isinstance(node, list):
                    for item in node:
                        walk(item)

            walk(payload)
            if url_tasks:
                await asyncio.gather(*url_tasks, return_exceptions=True)

        elif ctype.startswith("audio/"):
            ext = ".wav" if "wav" in ctype else ".bin"
            (req_dir / f"generated_audio{ext}").write_bytes(response.content)
        elif ctype.startswith("image/"):
            ext = ".png" if "png" in ctype else ".jpg" if "jpeg" in ctype else ".bin"
            (req_dir / f"generated_image{ext}").write_bytes(response.content)
        else:
            (req_dir / "response.bin").write_bytes(response.content)

        return str(req_dir)

    async def _post_request(self, client: httpx.AsyncClient, request_id: int, scenario: str) -> RequestOutcome:
        endpoint, body, files, data = self._request_payload(scenario)
        url = f"{self.args.base_url.rstrip('/')}{endpoint}"

        start_ms = self._now_ms()
        error = ""
        status_code = 0
        ok = False
        response: Optional[httpx.Response] = None

        headers = self._auth_headers().copy()
        headers["x-request-id"] = f"load-{request_id:05d}"

        try:
            if files is not None:
                response = await client.post(
                    url,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=self.args.timeout_sec,
                )
            else:
                response = await client.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=self.args.timeout_sec,
                )
            status_code = response.status_code
            ok = 200 <= status_code < 300
            if not ok:
                sample = response.text[:220].replace("\n", " ")
                error = f"http_{status_code}: {sample}"
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc!s}".strip()
            if error.endswith(":"):
                error = type(exc).__name__
            status_code = 0
            ok = False

        latency_ms = self._now_ms() - start_ms
        artifact_dir = await self._persist_request_artifacts(
            client=client,
            request_id=request_id,
            scenario=scenario,
            endpoint=endpoint,
            request_body=body,
            request_data=data,
            request_files=files,
            response=response,
            latency_ms=latency_ms,
            error=error,
        )

        return RequestOutcome(
            request_id=request_id,
            scenario=scenario,
            endpoint=endpoint,
            status_code=status_code,
            ok=ok,
            latency_ms=latency_ms,
            error=error,
            artifact_dir=artifact_dir,
        )

    async def _record(self, outcome: RequestOutcome) -> None:
        async with self.result_lock:
            self.outcomes.append(outcome)
            self.done += 1
            self.http_status_counts[outcome.status_code] = self.http_status_counts.get(outcome.status_code, 0) + 1
            self.scenario_counts[outcome.scenario] = self.scenario_counts.get(outcome.scenario, 0) + 1
            if outcome.ok:
                self.success += 1
            else:
                self.fail += 1

    async def _print_progress(self, outcome: RequestOutcome) -> None:
        if not self.args.verbose and (self.done % self.args.log_every != 0) and outcome.ok:
            return

        elapsed = time.perf_counter() - self.started_at
        rps = self.done / elapsed if elapsed > 0 else 0.0
        status_label = "OK" if outcome.ok else "ERR"

        msg = (
            f"[{self.done:04d}/{self.total}] {status_label:<3} "
            f"{outcome.scenario:<11} {outcome.status_code:<4} {outcome.latency_ms:7.1f}ms "
            f"succ={self.success} fail={self.fail} rps={rps:5.2f}"
        )
        if outcome.error and self.args.verbose:
            msg += f" | {outcome.error[:180]}"
        await self._log(msg)

    async def _worker(self, worker_id: int, queue: "asyncio.Queue[int]", client: httpx.AsyncClient) -> None:
        while True:
            request_id = await queue.get()
            if request_id <= 0:
                queue.task_done()
                return

            scenario = self._select_scenario()
            if scenario == "image":
                self.image_sent += 1
            elif scenario == "speech":
                self.speech_sent += 1
            elif scenario == "stt":
                self.stt_sent += 1

            outcome = await self._post_request(client, request_id, scenario)
            await self._record(outcome)
            await self._print_progress(outcome)
            queue.task_done()

    def _latency_stats(self) -> Dict[str, float]:
        values = [o.latency_ms for o in self.outcomes]
        if not values:
            return {"min": 0.0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

        sorted_vals = sorted(values)

        def pct(p: float) -> float:
            idx = int(round((len(sorted_vals) - 1) * p))
            idx = max(0, min(idx, len(sorted_vals) - 1))
            return sorted_vals[idx]

        return {
            "min": min(values),
            "avg": statistics.mean(values),
            "p50": pct(0.50),
            "p95": pct(0.95),
            "max": max(values),
        }

    async def _print_summary(self) -> None:
        elapsed = time.perf_counter() - self.started_at
        stats = self._latency_stats()
        rps = self.done / elapsed if elapsed > 0 else 0.0

        await self._log("\n========== Load Test Summary ==========")
        await self._log(f"total={self.done} success={self.success} fail={self.fail} success_rate={(self.success / max(1, self.done)) * 100:.2f}%")
        await self._log(f"elapsed={elapsed:.2f}s rps={rps:.2f}")
        await self._log(
            "latency_ms "
            + f"min={stats['min']:.2f} avg={stats['avg']:.2f} p50={stats['p50']:.2f} p95={stats['p95']:.2f} max={stats['max']:.2f}"
        )
        await self._log("scenario_counts=" + json.dumps(self.scenario_counts, ensure_ascii=True, sort_keys=True))
        await self._log("status_counts=" + json.dumps(self.http_status_counts, ensure_ascii=True, sort_keys=True))

        if self.fail > 0:
            samples = [o for o in self.outcomes if not o.ok][: min(8, self.fail)]
            await self._log("sample_errors:")
            for item in samples:
                await self._log(
                    f"  - req={item.request_id} scenario={item.scenario} status={item.status_code} latency={item.latency_ms:.1f}ms error={item.error[:220]}"
                )

    async def _print_admin_snapshot(self, client: httpx.AsyncClient) -> None:
        if not self.args.admin_token:
            await self._log("[admin] skipped (admin token not provided)")
            return

        headers = self._admin_headers()
        base = self.args.base_url.rstrip("/")

        async def get_json(path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            try:
                resp = await client.get(f"{base}{path}", headers=headers, params=params, timeout=self.args.timeout_sec)
                if resp.status_code >= 400:
                    await self._log(f"[admin] {path} -> {resp.status_code}")
                    return None
                return resp.json()
            except Exception as exc:  # noqa: BLE001
                await self._log(f"[admin] {path} failed: {exc}")
                return None

        summary = await get_json("/admin/logs/summary", {"since_minutes": self.args.admin_since_minutes})
        if summary:
            events = summary.get("events", {})
            http = events.get("http", {}) if isinstance(events, dict) else {}
            usage = summary.get("usage_overview", {}).get("overall", {})
            await self._log("\n----- Admin Snapshot (for dashboard sanity-check) -----")
            await self._log(
                "http: "
                + f"requests={http.get('requests_total', 0)} success_rate={http.get('success_rate', 0)} "
                + f"latency_avg_ms={http.get('latency_avg_ms', 0)} p95_ms={http.get('latency_p95_ms', 0)}"
            )
            await self._log(
                "usage: "
                + f"requests={usage.get('requests_total', 0)} success_rate={usage.get('success_rate', 0)} "
                + f"status_429={usage.get('status_429', 0)}"
            )

        usage_by_provider = await get_json("/admin/usage/aggregate", {"group_by": "provider", "since_minutes": self.args.admin_since_minutes})
        if usage_by_provider:
            top = (usage_by_provider.get("items") or [])[:5]
            await self._log("top_providers=" + json.dumps(top, ensure_ascii=True))

        usage_by_model = await get_json("/admin/usage/aggregate", {"group_by": "model", "since_minutes": self.args.admin_since_minutes})
        if usage_by_model:
            top = (usage_by_model.get("items") or [])[:5]
            await self._log("top_models=" + json.dumps(top, ensure_ascii=True))

    def _summary_payload(self) -> Dict[str, Any]:
        elapsed = max(0.001, time.perf_counter() - self.started_at)
        latency = self._latency_stats()
        error_samples = [
            {
                "request_id": row.request_id,
                "scenario": row.scenario,
                "endpoint": row.endpoint,
                "status_code": row.status_code,
                "latency_ms": round(row.latency_ms, 3),
                "error": row.error,
                "artifact_dir": row.artifact_dir,
            }
            for row in self.outcomes
            if not row.ok
        ][:80]

        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "run_dir": str(self.run_dir),
            "config": {
                "base_url": self.args.base_url,
                "total": self.total,
                "concurrency": self.args.concurrency,
                "seed": self.args.seed,
                "stt_sample_file": str(self.args.stt_sample_file),
            },
            "totals": {
                "done": self.done,
                "success": self.success,
                "fail": self.fail,
                "success_rate": round((self.success / max(1, self.done)) * 100, 4),
                "elapsed_sec": round(elapsed, 4),
                "rps": round(self.done / elapsed, 4),
            },
            "latency_ms": {k: round(v, 4) for k, v in latency.items()},
            "scenario_counts": dict(sorted(self.scenario_counts.items(), key=lambda kv: kv[0])),
            "status_counts": {str(k): v for k, v in sorted(self.http_status_counts.items(), key=lambda kv: kv[0])},
            "sample_errors": error_samples,
        }

    async def _write_report(self) -> None:
        if not self.args.save_report:
            return
        report_path = self.run_dir / "report.json"
        report_path.write_text(json.dumps(self._summary_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
        await self._log(f"[report] {report_path}")

    async def run(self) -> int:
        if not self.args.token:
            await self._log("ERROR: auth token is empty. pass --token or set UAG_AUTH_TOKEN in environment.")
            return 2

        limits = httpx.Limits(max_connections=max(32, self.args.concurrency * 4), max_keepalive_connections=max(16, self.args.concurrency * 2))
        timeout = httpx.Timeout(connect=self.args.timeout_sec, read=self.args.timeout_sec, write=self.args.timeout_sec, pool=self.args.timeout_sec)

        self._sanitize_proxy_env()
        async with httpx.AsyncClient(limits=limits, timeout=timeout, trust_env=self.args.trust_env) as client:
            await self.discover_models(client)

            queue: asyncio.Queue[int] = asyncio.Queue()
            for i in range(1, self.total + 1):
                queue.put_nowait(i)
            for _ in range(self.args.concurrency):
                queue.put_nowait(0)

            self.started_at = time.perf_counter()
            await self._log(
                "[start] "
                + f"base={self.args.base_url} total={self.total} concurrency={self.args.concurrency} "
                + f"image_max={self.args.max_image_requests} speech_max={self.args.max_speech_requests} stt_max={self.args.max_stt_requests}"
            )
            await self._log(f"[artifacts] {self.run_dir}")

            workers = [asyncio.create_task(self._worker(i + 1, queue, client)) for i in range(self.args.concurrency)]
            await queue.join()
            await asyncio.gather(*workers)

            await self._print_summary()
            await self._print_admin_snapshot(client)
            await self._write_report()

        return 0 if self.fail == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Randomized load test runner for Unified AI Gateway")

    parser.add_argument("--base-url", default=os.getenv("UAG_LOAD_BASE_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--token", default=os.getenv("UAG_AUTH_TOKEN", ""))
    parser.add_argument("--token-header", default=os.getenv("UAG_AUTH_HEADER_NAME", "x-api-token"))
    parser.add_argument("--admin-token", default=os.getenv("UAG_ADMIN_TOKEN", ""))
    parser.add_argument("--admin-token-header", default=os.getenv("UAG_ADMIN_HEADER_NAME", "x-admin-token"))

    parser.add_argument("--total", type=int, default=100, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=12, help="Concurrent worker count")
    parser.add_argument("--timeout-sec", type=float, default=75.0)
    parser.add_argument("--trust-env", action="store_true", default=True, help="Allow HTTP proxy/env settings")
    parser.add_argument("--no-trust-env", action="store_false", dest="trust_env", help="Ignore HTTP proxy/env settings")
    parser.add_argument("--seed", type=int, default=int(time.time()), help="Random seed")
    parser.add_argument("--log-every", type=int, default=5, help="Print progress every N requests (unless --verbose)")
    parser.add_argument("--verbose", action="store_true", help="Print every request line")

    parser.add_argument("--discover-models", action="store_true", default=True)
    parser.add_argument("--no-discover-models", action="store_false", dest="discover_models")
    parser.add_argument("--max-models-per-provider", type=int, default=8)

    parser.add_argument("--gemini-models", default="gemma-4-27b-it,gemma-3-27b-it,gemini-2.5-flash")
    parser.add_argument("--groq-models", default="llama-3.3-70b-versatile,llama-3.1-8b-instant")
    parser.add_argument("--embedding-models", default="gemini-embedding-001")
    parser.add_argument("--image-models", default="flux")
    parser.add_argument("--tts-voices", default="autumn,diana,hannah,austin,daniel,troy")

    parser.add_argument("--max-image-requests", type=int, default=4)
    parser.add_argument("--max-speech-requests", type=int, default=12)
    parser.add_argument("--max-stt-requests", type=int, default=10)

    parser.add_argument("--weight-chat", type=int, default=34)
    parser.add_argument("--weight-responses", type=int, default=20)
    parser.add_argument("--weight-embeddings", type=int, default=18)
    parser.add_argument("--weight-orchestrate", type=int, default=14)
    parser.add_argument("--weight-images", type=int, default=4)
    parser.add_argument("--weight-speech", type=int, default=6)
    parser.add_argument("--weight-stt", type=int, default=4)

    parser.add_argument("--router-timeout-sec", type=float, default=20.0)
    parser.add_argument("--router-max-attempts", type=int, default=6)
    parser.add_argument("--admin-since-minutes", type=int, default=120)
    parser.add_argument("--stt-sample-file", default="assets/samples/audio/06-01_Audio_out.mp3")
    parser.add_argument("--results-dir", default="artifacts/load_tests")
    parser.add_argument("--report-prefix", default="load_test")
    parser.add_argument("--save-report", action="store_true", default=True)
    parser.add_argument("--no-save-report", action="store_false", dest="save_report")

    return parser


async def main_async() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.total <= 0:
        print("--total must be > 0", file=sys.stderr)
        return 2
    if args.concurrency <= 0:
        print("--concurrency must be > 0", file=sys.stderr)
        return 2

    tester = LoadTester(args)
    return await tester.run()


def main() -> int:
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
