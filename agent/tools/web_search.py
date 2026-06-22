from typing import Any, List, Dict
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
from agent.tools.base import BaseTool


class WebSearchTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Searches the live web for macroeconomic indicators, market updates, and external financial context. Input should be a specific search query string."
        )

    def _run(self, query: str, max_results: int = 4) -> List[Dict[str, str]]:
        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_results))

            formatted_results = []
            for r in raw_results:
                if isinstance(r, dict):
                    formatted_results.append({
                        "title": r.get("title", "") or r.get("t", ""),
                        "snippet": r.get("body", "") or r.get("b", "") or r.get("snippet", ""),
                        "url": r.get("href", "") or r.get("link", "") or r.get("u", "")
                    })
                elif isinstance(r, tuple) and len(r) >= 3:
                    formatted_results.append({
                        "title": str(r[0]), "url": str(r[1]), "snippet": str(r[2])
                    })
                elif isinstance(r, tuple) and len(r) == 2:
                    formatted_results.append({
                        "title": str(r[0]), "url": str(r[1]), "snippet": ""
                    })

            return formatted_results
        except Exception as e:
            raise RuntimeError(f"External search engine execution failure: {str(e)}")