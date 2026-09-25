import pytest

from agent.tools import web_search as ws


def test_network_failure_returns_readable_string(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(ws.httpx, "get", boom)
    out = ws.web_search("transformer")
    assert isinstance(out, str)
    assert "失败" in out or "不可用" in out


def test_timeout_returns_readable_string(monkeypatch):
    def boom(*a, **k):
        raise ws.httpx.TimeoutException("timed out")

    monkeypatch.setattr(ws.httpx, "get", boom)
    assert isinstance(ws.web_search("q"), str)


def test_parses_results_from_html(monkeypatch):
    html = ('<a class="result__a" href="https://x.com">标题一</a>'
            '<a class="result__a" href="https://y.com">标题二</a>')
    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: type("R", (), {"text": html, "status_code": 200})())
    out = ws.web_search("q", max_results=2)
    assert "标题一" in out and "https://x.com" in out


def test_no_results_returns_readable_message(monkeypatch):
    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: type("R", (), {"text": "<html></html>", "status_code": 200})())
    out = ws.web_search("q")
    assert isinstance(out, str) and len(out) > 0


def test_empty_query_is_rejected_gracefully():
    assert "为空" in ws.web_search("")


def test_respects_max_results(monkeypatch):
    html = "".join(f'<a class="result__a" href="https://s{i}.com">T{i}</a>' for i in range(10))
    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: type("R", (), {"text": html, "status_code": 200})())
    out = ws.web_search("q", max_results=2)
    assert out.count("https://") <= 2
