from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4


class DockerSandbox:
    def __init__(self, workspace: Path, image: str | None = None) -> None:
        self.workspace = workspace.resolve()
        self.image = image or os.getenv("SANDBOX_IMAGE", "research-agent-sandbox")

    def run_python(self, code: str, timeout: int = 30) -> dict[str, object]:
        with tempfile.NamedTemporaryFile("w", suffix=".py", dir=self.workspace, delete=False, encoding="utf-8") as handle:
            handle.write(code)
            script_name = Path(handle.name).name
        try:
            combined = (
                f"python /workspace/{script_name}; "
                f"echo '===GENERATED_FILES==='; "
                f"find /workspace -type f ! -name '{script_name}' ! -name '.*' | sort"
            )
            result = self._run(["bash", "-lc", combined], timeout=timeout)
            stdout = result.get("stdout", "")
            if "===GENERATED_FILES===" in stdout:
                idx = stdout.index("===GENERATED_FILES===")
                result["stdout"] = stdout[:idx].strip()
                files_section = stdout[idx + len("===GENERATED_FILES==="):].strip()
                result["new_files"] = [f.strip() for f in files_section.split("\n") if f.strip()]
            return result
        finally:
            Path(self.workspace / script_name).unlink(missing_ok=True)

    def run_bash(self, command: str, timeout: int = 30) -> dict[str, object]:
        combined = f"{command}; rc=$?; echo '===GENERATED_FILES==='; find /workspace -type f -not -path '*/.*' | sort; exit $rc"
        result = self._run(["bash", "-lc", combined], timeout=timeout)
        stdout = result.get("stdout", "")
        if "===GENERATED_FILES===" in stdout:
            idx = stdout.index("===GENERATED_FILES===")
            result["stdout"] = stdout[:idx].strip()
            files_section = stdout[idx + len("===GENERATED_FILES==="):].strip()
            result["new_files"] = [f.strip() for f in files_section.split("\n") if f.strip()]
        return result

    def _run(self, command: list[str], timeout: int) -> dict[str, object]:
        container_name = f"research-agent-{uuid4().hex[:8]}"
        docker_command = [
            "docker",
            "run",
            "--name",
            container_name,
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
        try:
            completed = subprocess.run(
                docker_command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", container_name], capture_output=True, text=True, check=False)
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, text=True, check=False)
            return {
                "stdout": "",
                "stderr": "timeout expired",
                "exit_code": -1,
            }
        return {
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "exit_code": completed.returncode,
        }

