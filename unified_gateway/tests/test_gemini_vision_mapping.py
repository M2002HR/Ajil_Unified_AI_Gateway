from __future__ import annotations

from unified_gateway.app.providers.gemini_adapter import _openai_messages_to_gemini_contents


def test_openai_messages_with_data_url_image_are_mapped_to_gemini_inline_data():
    tiny_png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIW2P8z8DwHwAFgwJ/lF0xNQAAAABJRU5ErkJggg=="
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{tiny_png_base64}"}},
            ],
        }
    ]

    contents = _openai_messages_to_gemini_contents(messages)

    assert len(contents) == 1
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "Describe this image"
    assert contents[0]["parts"][1]["inlineData"]["mimeType"] == "image/png"
    assert contents[0]["parts"][1]["inlineData"]["data"] == tiny_png_base64
