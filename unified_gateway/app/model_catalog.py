from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_provider_model_id(provider: str, model_id: str) -> str:
    out = _to_str(model_id)
    if out.startswith("models/"):
        out = out[len("models/") :]
    if "/" in out:
        p, rest = out.split("/", 1)
        if p.strip().lower() == provider.lower():
            out = rest.strip()
    return out


def _family_from_model_id(model_id: str) -> str:
    value = model_id.lower()
    candidates = [
        "gemma",
        "gemini",
        "llama",
        "mixtral",
        "qwen",
        "gpt",
        "whisper",
        "flux",
        "seedream",
        "kontext",
        "grok",
    ]
    for cand in candidates:
        if cand in value:
            return cand
    return "other"


def _capabilities_from_gemini_methods(methods: Iterable[str]) -> List[str]:
    out: List[str] = []
    method_set = {str(m) for m in methods}
    if any(m in method_set for m in ["generateContent", "streamGenerateContent", "bidiGenerateContent"]):
        out.extend(["chat.completions", "responses"])
    if "embedContent" in method_set:
        out.append("embeddings")
    return sorted(set(out))


def _capabilities_from_groq_model_id(model_id: str) -> List[str]:
    lid = model_id.lower()
    if "whisper" in lid:
        return ["audio.transcriptions"]
    if any(x in lid for x in ["tts", "playai", "orpheus"]):
        return ["audio.speech"]
    if "embedding" in lid:
        return ["embeddings"]
    return ["chat.completions", "responses"]


def _modalities_from_capabilities(capabilities: Iterable[str]) -> tuple[list[str], list[str]]:
    caps = set(capabilities)
    inputs: set[str] = set()
    outputs: set[str] = set()

    if any(c in caps for c in ["chat.completions", "responses"]):
        inputs.add("text")
        outputs.add("text")
    if "embeddings" in caps:
        inputs.add("text")
        outputs.add("embedding")
    if "images.generations" in caps:
        inputs.add("text")
        outputs.add("image")
    if "audio.speech" in caps:
        inputs.add("text")
        outputs.add("audio")
    if "audio.transcriptions" in caps:
        inputs.add("audio")
        outputs.add("text")

    return sorted(inputs), sorted(outputs)


def _model_type_from_capabilities(capabilities: Iterable[str]) -> str:
    caps = set(capabilities)
    kinds: list[str] = []
    if any(c in caps for c in ["chat.completions", "responses"]):
        kinds.append("llm")
    if "embeddings" in caps:
        kinds.append("embedding")
    if "images.generations" in caps:
        kinds.append("image")
    if "audio.speech" in caps:
        kinds.append("audio_tts")
    if "audio.transcriptions" in caps:
        kinds.append("audio_stt")
    if not kinds:
        return "other"
    if len(kinds) == 1:
        return kinds[0]
    return "multi"


def _normalize_gemini_models(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return []

    out: List[Dict[str, Any]] = []
    for row in models:
        if not isinstance(row, dict):
            continue
        raw_name = _to_str(row.get("name"))
        model_id = _normalize_provider_model_id("gemini", raw_name)
        caps = _capabilities_from_gemini_methods(row.get("supported_generation_methods") or [])
        in_mod, out_mod = _modalities_from_capabilities(caps)
        out.append(
            {
                "provider": "gemini",
                "id": model_id,
                "name": model_id,
                "label": _to_str(row.get("display_name") or model_id),
                "family": _family_from_model_id(model_id),
                "capabilities": caps,
                "model_type": _model_type_from_capabilities(caps),
                "input_modalities": in_mod,
                "output_modalities": out_mod,
                "preview": bool(row.get("is_preview")),
                "paid_only": None,
                "context_window": row.get("input_token_limit"),
                "max_output_tokens": row.get("output_token_limit"),
                "raw": row,
            }
        )
    return out


def _normalize_groq_models(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []

    out: List[Dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        model_id = _normalize_provider_model_id("groq", _to_str(row.get("id") or row.get("name")))
        if not model_id:
            continue
        caps = _capabilities_from_groq_model_id(model_id)
        in_mod, out_mod = _modalities_from_capabilities(caps)
        out.append(
            {
                "provider": "groq",
                "id": model_id,
                "name": model_id,
                "label": model_id,
                "family": _family_from_model_id(model_id),
                "capabilities": caps,
                "model_type": _model_type_from_capabilities(caps),
                "input_modalities": in_mod,
                "output_modalities": out_mod,
                "preview": False,
                "paid_only": None,
                "context_window": row.get("context_window") or row.get("context_length"),
                "max_output_tokens": row.get("max_output_tokens"),
                "raw": row,
            }
        )
    return out


def _normalize_pollinations_models(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []

    out: List[Dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        model_id = _normalize_provider_model_id("pollinations", _to_str(row.get("name") or row.get("id")))
        if not model_id:
            continue
        output_modalities = row.get("output_modalities") or []
        has_image = "image" in [str(v) for v in output_modalities]
        caps = ["images.generations"] if has_image or not output_modalities else []
        in_mod = sorted({str(v) for v in (row.get("input_modalities") or ["text"]) if str(v)})
        out_mod = sorted({str(v) for v in output_modalities if str(v)}) or ["image"]

        out.append(
            {
                "provider": "pollinations",
                "id": model_id,
                "name": model_id,
                "label": _to_str(row.get("display_name") or model_id),
                "family": _family_from_model_id(model_id),
                "capabilities": caps,
                "model_type": _model_type_from_capabilities(caps),
                "input_modalities": in_mod,
                "output_modalities": out_mod,
                "preview": bool(row.get("preview") or False),
                "paid_only": bool(row.get("paid_only", False)) if "paid_only" in row else None,
                "context_window": None,
                "max_output_tokens": None,
                "raw": row,
            }
        )
    return out


def normalize_models(provider: str, payload: Any) -> List[Dict[str, Any]]:
    provider_key = _to_str(provider).lower()
    root = payload if isinstance(payload, dict) else {}

    if provider_key == "gemini":
        return _normalize_gemini_models(root)
    if provider_key == "groq":
        return _normalize_groq_models(root)
    if provider_key == "pollinations":
        return _normalize_pollinations_models(root)
    return []


def filter_models(
    rows: List[Dict[str, Any]],
    *,
    providers: Optional[List[str]] = None,
    capability: Optional[str] = None,
    model_type: Optional[str] = None,
    modality: Optional[str] = None,
    include_preview: bool = True,
    include_paid: bool = True,
    search: Optional[str] = None,
) -> List[Dict[str, Any]]:
    provider_set = {p.strip().lower() for p in (providers or []) if p.strip()}
    cap = _to_str(capability)
    kind = _to_str(model_type).lower()
    mod = _to_str(modality).lower()
    qry = _to_str(search).lower()

    out: List[Dict[str, Any]] = []
    for row in rows:
        if provider_set and str(row.get("provider", "")).lower() not in provider_set:
            continue
        if not include_preview and bool(row.get("preview", False)):
            continue
        if not include_paid and bool(row.get("paid_only", False)):
            continue
        if cap and cap not in (row.get("capabilities") or []):
            continue
        if kind and str(row.get("model_type") or "").lower() != kind:
            continue
        if mod:
            in_mod = {str(v).lower() for v in (row.get("input_modalities") or [])}
            out_mod = {str(v).lower() for v in (row.get("output_modalities") or [])}
            if mod not in in_mod and mod not in out_mod:
                continue
        if qry:
            text = " ".join(
                [
                    str(row.get("provider") or ""),
                    str(row.get("id") or ""),
                    str(row.get("label") or ""),
                    str(row.get("family") or ""),
                    " ".join(row.get("capabilities") or []),
                ]
            ).lower()
            if qry not in text:
                continue
        out.append(row)
    return out


def summarize_models(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_provider: Dict[str, int] = {}
    by_capability: Dict[str, int] = {}
    by_family: Dict[str, int] = {}
    by_model_type: Dict[str, int] = {}
    by_modality: Dict[str, int] = {}

    for row in rows:
        provider = str(row.get("provider") or "unknown")
        by_provider[provider] = by_provider.get(provider, 0) + 1

        family = str(row.get("family") or "other")
        by_family[family] = by_family.get(family, 0) + 1
        model_type = str(row.get("model_type") or "other")
        by_model_type[model_type] = by_model_type.get(model_type, 0) + 1

        for cap in row.get("capabilities") or []:
            by_capability[str(cap)] = by_capability.get(str(cap), 0) + 1

        for mod in list(row.get("input_modalities") or []) + list(row.get("output_modalities") or []):
            m = str(mod)
            by_modality[m] = by_modality.get(m, 0) + 1

    return {
        "total": len(rows),
        "by_provider": by_provider,
        "by_capability": by_capability,
        "by_family": by_family,
        "by_model_type": by_model_type,
        "by_modality": by_modality,
    }
