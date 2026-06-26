from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agent.models import ToolResult
from tools.bash_executor import BashExecutorTool
from tools.file_tools import ReadFileTool, WriteFileTool
from tools.python_executor import PythonExecutorTool
from tools.web_search import WebSearchTool

Tool = Callable[[dict[str, Any]], Awaitable[ToolResult]]


class ToolRegistry:
    def __init__(self, workspace: Path) -> None:
        self._tools: dict[str, Tool] = {
            "web_search": WebSearchTool(),
            "read_file": ReadFileTool(workspace),
            "write_file": WriteFileTool(workspace),
            "execute_python": PythonExecutorTool(workspace),
            "execute_bash": BashExecutorTool(workspace),
        }

    def has(self, name: str) -> bool:
        return name in self._tools

    async def execute(self, name: str, payload: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, output=None, error=f"Unknown tool: {name}")
        return await tool(payload)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

