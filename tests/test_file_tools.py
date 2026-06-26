from __future__ import annotations

from pathlib import Path

import pytest

from tools.file_tools import resolve_workspace_path


def test_resolve_workspace_path_allows_nested_path(tmp_path: Path) -> None:
    target = resolve_workspace_path(tmp_path, "reports/report.md")
    assert target == (tmp_path / "reports" / "report.md").resolve()


def test_resolve_workspace_path_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        resolve_workspace_path(tmp_path, "../outside.txt")

