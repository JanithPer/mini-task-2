from __future__ import annotations

import os
from typing import Any

import requests

from agent.models import ToolResult


class WebSearchTool:
    name = "web_search"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

    async def __call__(self, payload: dict[str, Any]) -> ToolResult:
        query = str(payload.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, output=None, error="query is required.")
        if not self.api_key:
            return ToolResult(ok=False, output=None, error="TAVILY_API_KEY is not configured.")

        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": payload.get("search_depth", "advanced"),
                    "max_results": int(payload.get("max_results", 5)),
                    "include_answer": False,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            results = [
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "content": item.get("content"),
                    "score": item.get("score"),
                }
                for item in data.get("results", [])
            ]
            return ToolResult(ok=True, output={"results": results})
        except Exception as exc:
            return ToolResult(ok=False, output=None, error=str(exc))

