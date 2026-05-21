from __future__ import annotations

from fastapi.testclient import TestClient

from unified_gateway.app.config import get_settings
from unified_gateway.app.main import app


def _setup_env(monkeypatch):
    monkeypatch.setenv("UAG_AUTH_ENABLED", "false")
    monkeypatch.setenv("UAG_ADMIN_ENABLED", "false")
    monkeypatch.setenv("UAG_LOG_ENABLED", "true")
    get_settings.cache_clear()


def test_studio_ui_files_are_served(monkeypatch):
    _setup_env(monkeypatch)
    with TestClient(app) as client:
        index_resp = client.get("/studio")
        assert index_resp.status_code == 200
        assert "UAG Studio" in index_resp.text

        css_resp = client.get("/studio/styles.css")
        assert css_resp.status_code == 200
        assert "--brand" in css_resp.text

        js_resp = client.get("/studio/app.js")
        assert js_resp.status_code == 200
        assert "refreshModels" in js_resp.text
        assert "applyChatMaxTokensDefault" in js_resp.text
        assert "resolveChatMaxTokens" in js_resp.text
        assert "/v1/images/options" in js_resp.text
        assert "imgRefreshOpts" in js_resp.text
        assert 'fillModelSelect(providerEl, modelEl, "chat.completions", { visionOnly: true }));' in js_resp.text
        assert "apiStreamChat" in js_resp.text
        assert "chatStream" in js_resp.text
        assert "visionStream" in js_resp.text


def test_studio_unknown_path_spa_fallback(monkeypatch):
    _setup_env(monkeypatch)
    with TestClient(app) as client:
        resp = client.get("/studio/some/deep/path")
        assert resp.status_code == 200
        assert "UAG Studio" in resp.text
