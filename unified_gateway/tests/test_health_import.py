from unified_gateway.app.main import app


def test_app_created():
    assert app.title == "Unified AI Gateway"
