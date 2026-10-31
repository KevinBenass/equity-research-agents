"""Search 10-K text through the filings-copilot service (hybrid search + reranking).

The agent system reuses the RAG pipeline from the first project as a tool over HTTP:
POST {FILINGS_API_URL}/retrieve -> ranked passages with company, year and section.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx

from equity_agents.tools.base import Tool, ToolError


class FilingsSearch:
    def __init__(
        self, base_url: str, timeout: float = 60.0, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def search(self, query: str, top_k: int = 6) -> str:
        try:
            resp = self._client.post("/retrieve", json={"question": query, "top_k": min(top_k, 10)})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ToolError(
                f"Filings search unavailable ({exc.__class__.__name__}). Is filings-copilot running?"
            ) from exc
        passages = resp.json()["passages"]
        if not passages:
            return "No passages found. Try naming the company and fiscal year explicitly."
        blocks = []
        for p in passages:
            blocks.append(
                f'<passage ref="filing:{p["chunk_id"]}" company="{p["company"]}" fiscal_year="{p["fiscal_year"]}" '
                f'section="{p["section"]}">\n{p["text"].strip()}\n</passage>'
            )
        return "\n\n".join(blocks)

    def tool(self) -> Tool:
        return Tool(
            "search_filings",
            "Search the text of 10-K annual reports (business description, risk factors, MD&A, notes). "
            "Name the company and fiscal year in the query, e.g. 'NVIDIA fiscal 2025 export control risks China'. "
            "Each passage has a `ref` to cite.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
            self.search,
        )


def static_filings_tool(passages: dict[str, str]) -> Tool:
    """Offline stand-in: returns canned passages whose key appears in the query (tests, demos)."""

    def search(query: str, top_k: int = 6) -> str:
        hits = [text for key, text in passages.items() if key.lower() in query.lower()]
        return "\n\n".join(hits[:top_k]) or "No passages found."

    fn: Callable[..., str] = search
    return Tool(
        "search_filings",
        "Search 10-K text.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        fn,
    )
