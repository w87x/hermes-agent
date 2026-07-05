"""Tests for the AnySearch web search + extract provider.

Covers:
- AnySearchWebSearchProvider.is_available() env var gating
- AnySearchWebSearchProvider.search() — happy path, HTTP error, request error,
  missing key
- Result normalization (title, url, description via snippet/content fallback,
  position)
- AnySearchWebSearchProvider.extract() — MCP JSON-RPC happy path, per-URL
  MCP-failure -> direct-HTTP fallback, missing key
- _is_backend_available("anysearch") integration
- _get_backend() recognizes "anysearch" as a valid configured backend
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.tools.conftest import register_all_web_providers


# ---------------------------------------------------------------------------
# AnySearchWebSearchProvider unit tests
# ---------------------------------------------------------------------------


class TestAnySearchProviderIsConfigured:
    def test_configured_when_key_set(self, monkeypatch):
        monkeypatch.setenv("ANYSEARCH_API_KEY", "as-key-123")
        from plugins.web.anysearch.provider import AnySearchWebSearchProvider
        assert AnySearchWebSearchProvider().is_available() is True

    def test_not_configured_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("ANYSEARCH_API_KEY", raising=False)
        from plugins.web.anysearch.provider import AnySearchWebSearchProvider
        assert AnySearchWebSearchProvider().is_available() is False

    def test_not_configured_when_key_whitespace(self, monkeypatch):
        monkeypatch.setenv("ANYSEARCH_API_KEY", "   ")
        from plugins.web.anysearch.provider import AnySearchWebSearchProvider
        assert AnySearchWebSearchProvider().is_available() is False

    def test_provider_name(self):
        from plugins.web.anysearch.provider import AnySearchWebSearchProvider
        assert AnySearchWebSearchProvider().name == "anysearch"

    def test_implements_web_search_provider(self):
        from agent.web_search_provider import WebSearchProvider
        from plugins.web.anysearch.provider import AnySearchWebSearchProvider
        assert issubclass(AnySearchWebSearchProvider, WebSearchProvider)

    def test_supports_both_search_and_extract(self):
        from plugins.web.anysearch.provider import AnySearchWebSearchProvider
        p = AnySearchWebSearchProvider()
        assert p.supports_search() is True
        assert p.supports_extract() is True


class TestAnySearchProviderSearch:
    _SAMPLE_RESPONSE = {
        "results": [
            {"title": "A", "url": "https://a.example.com", "snippet": "snip A", "content": "full A"},
            {"title": "B", "url": "https://b.example.com", "content": "full B"},
            {"title": "C", "url": "https://c.example.com", "snippet": "", "content": ""},
        ]
    }

    @staticmethod
    def _mock_resp(json_data, status_code=200):
        m = MagicMock()
        m.status_code = status_code
        m.json.return_value = json_data
        m.raise_for_status = MagicMock()
        return m

    def test_happy_path_normalizes_results(self, monkeypatch):
        monkeypatch.setenv("ANYSEARCH_API_KEY", "as-key-123")
        from plugins.web.anysearch.provider import AnySearchWebSearchProvider

        with patch("httpx.post", return_value=self._mock_resp(self._SAMPLE_RESPONSE)):
            result = AnySearchWebSearchProvider().search("test query", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 3
        assert web[0] == {
            "title": "A", "url": "https://a.example.com", "description": "snip A", "position": 1,
        }
        # No snippet -> falls back to content.
        assert web[1]["description"] == "full B"
        # Empty snippet and content -> empty description, not a crash.
        assert web[2]["description"] == ""

    def test_sends_bearer_header_and_max_results(self, monkeypatch):
        monkeypatch.setenv("ANYSEARCH_API_KEY", "as-key-123")
        from plugins.web.anysearch.provider import AnySearchWebSearchProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            captured["json"] = kwargs.get("json", {})
            return self._mock_resp({"results": []})

        with patch("httpx.post", side_effect=fake_post):
            AnySearchWebSearchProvider().search("q", limit=5)

        assert captured["url"] == "https://api.anysearch.com/v1/search"
        assert captured["headers"].get("Authorization") == "Bearer as-key-123"
        assert captured["json"]["query"] == "q"
        assert captured["json"]["max_results"] == 5

    def test_max_results_is_capped_at_10(self, monkeypatch):
        monkeypatch.setenv("ANYSEARCH_API_KEY", "as-key-123")
        from plugins.web.anysearch.provider import AnySearchWebSearchProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json", {})
            return self._mock_resp({"results": []})

        with patch("httpx.post", side_effect=fake_post):
            AnySearchWebSearchProvider().search("q", limit=100)

        assert captured["json"]["max_results"] == 10

    def test_base_url_override(self, monkeypatch):
        monkeypatch.setenv("ANYSEARCH_API_KEY", "as-key-123")
        monkeypatch.setenv("ANYSEARCH_BASE_URL", "https://anysearch.internal.example.com")
        from plugins.web.anysearch.provider import AnySearchWebSearchProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            return self._mock_resp({"results": []})

        with patch("httpx.post", side_effect=fake_post):
            AnySearchWebSearchProvider().search("q", limit=5)

        assert captured["url"] == "https://anysearch.internal.example.com/v1/search"

    def test_http_error_returns_failure(self, monkeypatch):
        import httpx
        monkeypatch.setenv("ANYSEARCH_API_KEY", "as-key-123")
        from plugins.web.anysearch.provider import AnySearchWebSearchProvider

        bad = MagicMock()
        bad.status_code = 429
        err = httpx.HTTPStatusError("429", request=MagicMock(), response=bad)

        with patch("httpx.post", side_effect=err):
            result = AnySearchWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "429" in result["error"] or "AnySearch" in result["error"]

    def test_request_error_returns_failure(self, monkeypatch):
        import httpx
        monkeypatch.setenv("ANYSEARCH_API_KEY", "as-key-123")
        from plugins.web.anysearch.provider import AnySearchWebSearchProvider

        with patch("httpx.post", side_effect=httpx.RequestError("boom")):
            result = AnySearchWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "boom" in result["error"] or "AnySearch" in result["error"]

    def test_missing_key_returns_failure(self, monkeypatch):
        monkeypatch.delenv("ANYSEARCH_API_KEY", raising=False)
        from plugins.web.anysearch.provider import AnySearchWebSearchProvider

        result = AnySearchWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "ANYSEARCH_API_KEY" in result["error"]


class TestAnySearchProviderExtract:
    @staticmethod
    def _mock_resp(json_data, status_code=200):
        m = MagicMock()
        m.status_code = status_code
        m.json.return_value = json_data
        m.raise_for_status = MagicMock()
        m.text = json_data if isinstance(json_data, str) else ""
        return m

    def test_extract_via_mcp_happy_path(self, monkeypatch):
        monkeypatch.setenv("ANYSEARCH_API_KEY", "as-key-123")
        from plugins.web.anysearch.provider import AnySearchWebSearchProvider

        mcp_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": "# Page content\n\nHello world."}]},
        }

        with patch("httpx.post", return_value=self._mock_resp(mcp_response)):
            docs = AnySearchWebSearchProvider().extract(["https://example.com/page"])

        assert len(docs) == 1
        assert docs[0]["url"] == "https://example.com/page"
        assert docs[0]["content"] == "# Page content\n\nHello world."
        assert docs[0]["raw_content"] == docs[0]["content"]
        assert "error" not in docs[0]

    def test_extract_calls_mcp_endpoint_with_jsonrpc_payload(self, monkeypatch):
        monkeypatch.setenv("ANYSEARCH_API_KEY", "as-key-123")
        from plugins.web.anysearch.provider import AnySearchWebSearchProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json", {})
            captured["headers"] = kwargs.get("headers", {})
            return self._mock_resp(
                {"result": {"content": [{"type": "text", "text": "content"}]}}
            )

        with patch("httpx.post", side_effect=fake_post):
            AnySearchWebSearchProvider().extract(["https://example.com/page"])

        assert captured["url"] == "https://api.anysearch.com/mcp"
        assert captured["json"]["method"] == "tools/call"
        assert captured["json"]["params"]["name"] == "extract"
        assert captured["json"]["params"]["arguments"]["url"] == "https://example.com/page"
        assert captured["headers"].get("Authorization") == "Bearer as-key-123"

    def test_extract_falls_back_to_direct_fetch_on_mcp_failure(self, monkeypatch):
        import httpx
        monkeypatch.setenv("ANYSEARCH_API_KEY", "as-key-123")
        from plugins.web.anysearch.provider import AnySearchWebSearchProvider

        def fake_post(url, **kwargs):
            raise httpx.RequestError("mcp unreachable")

        def fake_get(url, **kwargs):
            m = MagicMock()
            m.raise_for_status = MagicMock()
            m.text = "<html>raw fallback content</html>"
            return m

        with patch("httpx.post", side_effect=fake_post), patch("httpx.get", side_effect=fake_get):
            docs = AnySearchWebSearchProvider().extract(["https://example.com/page"])

        assert len(docs) == 1
        assert docs[0]["content"] == "<html>raw fallback content</html>"
        assert "error" not in docs[0]

    def test_extract_reports_per_url_error_when_both_paths_fail(self, monkeypatch):
        import httpx
        monkeypatch.setenv("ANYSEARCH_API_KEY", "as-key-123")
        from plugins.web.anysearch.provider import AnySearchWebSearchProvider

        with patch("httpx.post", side_effect=httpx.RequestError("mcp down")), \
             patch("httpx.get", side_effect=httpx.RequestError("fetch down")):
            docs = AnySearchWebSearchProvider().extract(["https://example.com/page"])

        assert len(docs) == 1
        assert docs[0]["content"] == ""
        assert "error" in docs[0]

    def test_extract_multiple_urls_independent_results(self, monkeypatch):
        monkeypatch.setenv("ANYSEARCH_API_KEY", "as-key-123")
        from plugins.web.anysearch.provider import AnySearchWebSearchProvider

        with patch(
            "httpx.post",
            return_value=self._mock_resp({"result": {"content": [{"type": "text", "text": "ok"}]}}),
        ):
            docs = AnySearchWebSearchProvider().extract(
                ["https://a.example.com", "https://b.example.com"]
            )

        assert [d["url"] for d in docs] == ["https://a.example.com", "https://b.example.com"]
        assert all(d["content"] == "ok" for d in docs)

    def test_extract_missing_key_returns_error_per_url(self, monkeypatch):
        monkeypatch.delenv("ANYSEARCH_API_KEY", raising=False)
        from plugins.web.anysearch.provider import AnySearchWebSearchProvider

        docs = AnySearchWebSearchProvider().extract(["https://example.com"])
        assert len(docs) == 1
        assert "ANYSEARCH_API_KEY" in docs[0]["error"]

    def test_mcp_jsonrpc_error_falls_back_to_direct_fetch(self, monkeypatch):
        monkeypatch.setenv("ANYSEARCH_API_KEY", "as-key-123")
        from plugins.web.anysearch.provider import AnySearchWebSearchProvider

        def fake_get(url, **kwargs):
            m = MagicMock()
            m.raise_for_status = MagicMock()
            m.text = "fallback text"
            return m

        with patch(
            "httpx.post",
            return_value=self._mock_resp({"error": {"code": -32000, "message": "tool failed"}}),
        ), patch("httpx.get", side_effect=fake_get):
            docs = AnySearchWebSearchProvider().extract(["https://example.com"])

        assert docs[0]["content"] == "fallback text"


# ---------------------------------------------------------------------------
# Integration: _is_backend_available / _get_backend
# ---------------------------------------------------------------------------


class TestAnySearchBackendWiring:
    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        # "anysearch" isn't in _LEGACY_WEB_BACKENDS, so availability
        # resolution goes through the registry — needs the plugin registered.
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

    def test_is_backend_available_true_when_key_set(self, monkeypatch):
        monkeypatch.setenv("ANYSEARCH_API_KEY", "as-key-123")
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("anysearch") is True

    def test_is_backend_available_false_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("ANYSEARCH_API_KEY", raising=False)
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("anysearch") is False

    def test_configured_backend_accepted(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "anysearch"})
        monkeypatch.setenv("ANYSEARCH_API_KEY", "as-key-123")
        assert web_tools._get_backend() == "anysearch"

    def test_check_web_api_key_true_when_anysearch_configured(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "anysearch"})
        monkeypatch.setenv("ANYSEARCH_API_KEY", "as-key-123")
        assert web_tools.check_web_api_key() is True
