from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.models import ToolResult
from sandbox.runner import DockerSandbox


class BashExecutorTool:
    name = "execute_bash"

    def __init__(self, workspace: Path) -> None:
        self.sandbox = DockerSandbox(workspace)

    async def __call__(self, payload: dict[str, Any]) -> ToolResult:
        command = payload.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolResult(ok=False, output=None, error="command is required.")
        try:
            result = self.sandbox.run_bash(command, timeout=int(payload.get("timeout", 30)))
            return ToolResult(ok=result["exit_code"] == 0, output=result, error=str(result["stderr"]) or None)
        except Exception as exc:
            return ToolResult(ok=False, output=None, error=str(exc))

