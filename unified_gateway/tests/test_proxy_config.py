from unified_gateway.app.config import ProxyConfig


def test_proxy_normalize_socks_scheme():
    assert ProxyConfig.normalize('socks://127.0.0.1:2080') == 'socks5://127.0.0.1:2080'
