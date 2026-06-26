from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.models import ToolResult


def resolve_workspace_path(workspace: Path, relative_path: str) -> Path:
    if not relative_path:
        raise ValueError("Path is required.")
    workspace = workspace.resolve()
    target = (workspace / relative_path).resolve()
    if target != workspace and workspace not in target.parents:
        raise ValueError(f"Path escapes workspace: {relative_path}")
    return target


class ReadFileTool:
    name = "read_file"

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    async def __call__(self, payload: dict[str, Any]) -> ToolResult:
        try:
            path = resolve_workspace_path(self.workspace, str(payload.get("path", "")))
            return ToolResult(ok=True, output={"path": str(path), "content": path.read_text(encoding="utf-8")})
        except Exception as exc:
            return ToolResult(ok=False, output=None, error=str(exc))


class WriteFileTool:
    name = "write_file"

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    async def __call__(self, payload: dict[str, Any]) -> ToolResult:
        try:
            path = resolve_workspace_path(self.workspace, str(payload.get("path", "")))
            content = payload.get("content")
            if not isinstance(content, str):
                raise ValueError("content must be a string.")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ToolResult(ok=True, output={"path": str(path), "bytes": len(content.encode("utf-8"))})
        except Exception as exc:
            return ToolResult(ok=False, output=None, error=str(exc))

