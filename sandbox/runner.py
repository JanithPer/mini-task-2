from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


class DockerSandbox:
    def __init__(self, workspace: Path, image: str | None = None) -> None:
        self.workspace = workspace.resolve()
        self.image = image or os.getenv("SANDBOX_IMAGE", "research-agent-sandbox")

    def run_python(self, code: str, timeout: int = 30) -> dict[str, object]:
        with tempfile.NamedTemporaryFile("w", suffix=".py", dir=self.workspace, delete=False, encoding="utf-8") as handle:
            handle.write(code)
            script_name = Path(handle.name).name
        try:
            return self._run(["python", f"/workspace/{script_name}"], timeout=timeout)
        finally:
            Path(self.workspace / script_name).unlink(missing_ok=True)

    def run_bash(self, command: str, timeout: int = 30) -> dict[str, object]:
        return self._run(["bash", "-lc", command], timeout=timeout)

    def _run(self, command: list[str], timeout: int) -> dict[str, object]:
        docker_command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--memory",
            "512m",
            "--cpus",
            "1",
            "-v",
            f"{self.workspace}:/workspace",
            "-w",
            "/workspace",
            self.image,
            *command,
        ]
        completed = subprocess.run(
            docker_command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "exit_code": completed.returncode,
        }

