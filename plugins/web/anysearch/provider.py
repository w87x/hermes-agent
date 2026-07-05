"""AnySearch web search + content extraction — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. AnySearch is
search infrastructure purpose-built for AI agents — 17 vertical domains, and
search+extract available in one call (we still surface them as separate
``search()``/``extract()`` methods to match the Hermes provider contract).

Two capabilities:

- ``supports_search()``  -> True — ``POST /v1/search`` (plain REST, JSON in/out)
- ``supports_extract()`` -> True — no dedicated REST extract endpoint; goes
  through AnySearch's MCP server (``POST /mcp``, JSON-RPC 2.0
  ``tools/call`` for the ``extract`` tool), falling back to a direct HTTP
  GET of the URL if the MCP call fails for any reason.

Config keys this provider responds to::

    web:
      search_backend: "anysearch"     # explicit per-capability
      extract_backend: "anysearch"    # explicit per-capability
      backend: "anysearch"            # shared fallback for both

Env vars::

    ANYSEARCH_API_KEY=...    # https://www.anysearch.com/console/api-keys (required)
    ANYSEARCH_BASE_URL=...   # optional override of https://api.anysearch.com
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


def _anysearch_base_url() -> str:
    return os.getenv("ANYSEARCH_BASE_URL", "https://api.anysearch.com").rstrip("/")


def _anysearch_api_key() -> str:
    return os.getenv("ANYSEARCH_API_KEY", "").strip()


def _anysearch_request(path: str, payload: Dict[str, Any], *, timeout: float) -> Dict[str, Any]:
    """POST to the AnySearch REST API and return the parsed JSON response.

    Raises ``ValueError`` when ``ANYSEARCH_API_KEY`` is unset; the caller
    catches and surfaces as a typed error response.
    """
    import httpx

    api_key = _anysearch_api_key()
    if not api_key:
        raise ValueError(
            "ANYSEARCH_API_KEY environment variable not set. "
            "Get a free API key at https://www.anysearch.com/console/api-keys"
        )

    url = f"{_anysearch_base_url()}/{path.lstrip('/')}"
    resp = httpx.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _normalize_anysearch_search_results(response: Dict[str, Any]) -> Dict[str, Any]:
    """Map AnySearch ``/v1/search`` response to ``{success, data: {web: [...]}}``."""
    web_results = []
    for i, result in enumerate(response.get("results", [])):
        web_results.append(
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "description": result.get("snippet") or result.get("content", ""),
                "position": i + 1,
            }
        )
    return {"success": True, "data": {"web": web_results}}


class AnySearchWebSearchProvider(WebSearchProvider):
    """AnySearch search + extract provider."""

    @property
    def name(self) -> str:
        return "anysearch"

    @property
    def display_name(self) -> str:
        return "AnySearch"

    def is_available(self) -> bool:
        """Return True when ``ANYSEARCH_API_KEY`` is set to a non-empty value."""
        return bool(_anysearch_api_key())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute an AnySearch general web search."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            logger.info("AnySearch search: '%s' (limit=%d)", query, limit)
            raw = _anysearch_request(
                "v1/search",
                {"query": query, "max_results": max(1, min(int(limit), 10))},
                timeout=30,
            )
            return _normalize_anysearch_search_results(raw)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — including httpx errors
            logger.warning("AnySearch search error: %s", exc)
            return {"success": False, "error": f"AnySearch search failed: {exc}"}

    def _extract_via_mcp(self, url: str) -> str:
        """Fetch page content via AnySearch's MCP ``extract`` tool (JSON-RPC 2.0)."""
        import httpx

        api_key = _anysearch_api_key()
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "extract", "arguments": {"url": url}},
        }
        resp = httpx.post(
            f"{_anysearch_base_url()}/mcp",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            error = data["error"]
            message = error.get("message", "unknown error") if isinstance(error, dict) else str(error)
            raise ValueError(f"AnySearch MCP extract error: {message}")

        content_blocks = data.get("result", {}).get("content", [])
        text_parts = [
            block.get("text", "")
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in text_parts if part)

    def _extract_via_http_fallback(self, url: str) -> str:
        """Best-effort direct fetch used when the MCP extract call fails.

        AnySearch has no dedicated REST extract endpoint, so this is the
        documented fallback path — a plain GET returning raw page text
        rather than the cleaned markdown the MCP tool produces.
        """
        import httpx

        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        return resp.text

    def _extract_one(self, url: str) -> str:
        if not _anysearch_api_key():
            raise ValueError("ANYSEARCH_API_KEY is not set")
        try:
            return self._extract_via_mcp(url)
        except Exception as exc:  # noqa: BLE001 — fall back to direct fetch
            logger.warning(
                "AnySearch MCP extract failed for %s (%s); falling back to direct fetch",
                url, exc,
            )
            return self._extract_via_http_fallback(url)

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via AnySearch.

        Sync — AnySearch's ``extract`` tool takes one URL per call, so this
        loops per-URL (no batch extract endpoint exists). Per-URL failures
        become items with an ``error`` field rather than raising.
        """
        from tools.interrupt import is_interrupted

        if is_interrupted():
            return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]

        documents: List[Dict[str, Any]] = []
        for url in urls:
            if is_interrupted():
                documents.append(
                    {"url": url, "title": "", "content": "", "raw_content": "", "error": "Interrupted"}
                )
                continue
            try:
                logger.info("AnySearch extract: %s", url)
                content = self._extract_one(url)
                documents.append(
                    {
                        "url": url,
                        "title": "",
                        "content": content,
                        "raw_content": content,
                        "metadata": {"sourceURL": url},
                    }
                )
            except ValueError as exc:
                documents.append(
                    {"url": url, "title": "", "content": "", "raw_content": "", "error": str(exc)}
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("AnySearch extract error for %s: %s", url, exc)
                documents.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": f"AnySearch extract failed: {exc}",
                    }
                )
        return documents

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "AnySearch",
            "badge": "free",
            "tag": "Agent-native search + extract — 17 vertical domains, 1k free requests/day.",
            "env_vars": [
                {
                    "key": "ANYSEARCH_API_KEY",
                    "prompt": "AnySearch API key",
                    "url": "https://www.anysearch.com/console/api-keys",
                },
            ],
        }
