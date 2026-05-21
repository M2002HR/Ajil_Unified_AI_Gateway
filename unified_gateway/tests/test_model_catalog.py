from __future__ import annotations

from unified_gateway.app.model_catalog import filter_models, normalize_models, summarize_models


def test_normalize_models_gemini_and_groq_and_pollinations():
    gemini_payload = {
        "models": [
            {
                "name": "models/gemma-4-27b-it",
                "display_name": "Gemma 4",
                "supported_generation_methods": ["generateContent"],
                "is_preview": False,
            },
            {
                "name": "models/gemini-embedding-001",
                "supported_generation_methods": ["embedContent"],
                "is_preview": False,
            },
        ]
    }
    groq_payload = {
        "data": [
            {"id": "llama-3.3-70b-versatile", "context_window": 131072},
            {"id": "whisper-large-v3-turbo"},
        ]
    }
    pollinations_payload = {
        "data": [
            {"name": "flux", "output_modalities": ["image"], "paid_only": False},
            {"name": "gpt-image-2", "output_modalities": ["image"], "paid_only": True},
        ]
    }

    gm = normalize_models("gemini", gemini_payload)
    gr = normalize_models("groq", groq_payload)
    pl = normalize_models("pollinations", pollinations_payload)

    assert any("chat.completions" in item["capabilities"] for item in gm)
    assert any("embeddings" in item["capabilities"] for item in gm)
    assert any("audio.transcriptions" in item["capabilities"] for item in gr)
    assert any(item["paid_only"] is True for item in pl)


def test_filter_and_summary_models():
    items = [
        {
            "provider": "gemini",
            "id": "gemma-4-27b-it",
            "label": "Gemma 4",
            "family": "gemma",
            "capabilities": ["chat.completions", "responses"],
            "model_type": "llm",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "preview": False,
            "paid_only": False,
        },
        {
            "provider": "pollinations",
            "id": "gpt-image-2",
            "label": "GPT Image 2",
            "family": "gpt",
            "capabilities": ["images.generations"],
            "model_type": "image",
            "input_modalities": ["text"],
            "output_modalities": ["image"],
            "preview": False,
            "paid_only": True,
        },
    ]

    filtered = filter_models(
        items,
        providers=["pollinations"],
        capability="images.generations",
        modality="image",
        include_preview=True,
        include_paid=True,
        search="gpt",
    )
    assert len(filtered) == 1
    assert filtered[0]["provider"] == "pollinations"

    free_only = filter_models(items, include_paid=False)
    assert len(free_only) == 1
    assert free_only[0]["provider"] == "gemini"

    images_only = filter_models(items, model_type="image")
    assert len(images_only) == 1
    assert images_only[0]["id"] == "gpt-image-2"

    summary = summarize_models(items)
    assert summary["total"] == 2
    assert summary["by_provider"]["gemini"] == 1
    assert summary["by_provider"]["pollinations"] == 1
    assert summary["by_model_type"]["llm"] == 1
    assert summary["by_model_type"]["image"] == 1
