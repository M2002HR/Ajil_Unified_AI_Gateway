from __future__ import annotations

from fastapi.testclient import TestClient

from unified_gateway.app.config import get_settings
from unified_gateway.app.main import app


def _setup_env(monkeypatch):
    monkeypatch.setenv("UAG_ADMIN_ENABLED", "true")
    monkeypatch.setenv("UAG_ADMIN_TOKEN", "test-admin-token")
    monkeypatch.setenv("UAG_ADMIN_HEADER_NAME", "x-admin-token")
    monkeypatch.setenv("UAG_AUTH_ENABLED", "false")
    monkeypatch.setenv("UAG_LOG_ENABLED", "true")
    get_settings.cache_clear()


def test_admin_ui_files_are_served(monkeypatch):
    _setup_env(monkeypatch)
    with TestClient(app) as client:
        index_resp = client.get("/admin")
        assert index_resp.status_code == 200
        assert "UAG Control" in index_resp.text

        css_resp = client.get("/admin/styles.css")
        assert css_resp.status_code == 200
        assert "--brand" in css_resp.text

        js_resp = client.get("/admin/app.js")
        assert js_resp.status_code == 200
        assert "connectWs" in js_resp.text


def test_admin_logs_endpoint_requires_auth(monkeypatch):
    _setup_env(monkeypatch)
    with TestClient(app) as client:
        unauthorized = client.get("/admin/logs/summary")
        assert unauthorized.status_code == 401

        authorized = client.get("/admin/logs/summary", headers={"x-admin-token": "test-admin-token"})
        assert authorized.status_code == 200
        data = authorized.json()
        assert "events" in data
        assert "usage_overview" in data


def test_admin_websocket_hello_ping_and_filters(monkeypatch):
    _setup_env(monkeypatch)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/admin?token=test-admin-token&since_minutes=5&limit=50") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"
            assert "summary" in hello
            assert "events" in hello
            assert "usage_overview" in hello

            ws.send_json({"type": "ping"})
            got_pong = False
            for _ in range(4):
                msg = ws.receive_json()
                if msg.get("type") == "pong":
                    got_pong = True
                    break
            assert got_pong is True

            ws.send_json(
                {
                    "type": "filters",
                    "since_minutes": 15,
                    "limit": 120,
                    "level": "WARNING",
                    "event_type": "router.provider.attempt",
                    "http_group_by": "method",
                    "after_seq": 0,
                }
            )
            applied = ws.receive_json()
            assert applied["type"] == "filters.applied"
            assert applied["since_minutes"] == 15
            assert applied["limit"] == 120
            assert applied["level"] == "WARNING"
            assert applied["event_type"] == "router.provider.attempt"
            assert applied["http_group_by"] == "method"
