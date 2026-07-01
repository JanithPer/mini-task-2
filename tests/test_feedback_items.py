from __future__ import annotations

import json
import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent.loop import ResearchAgent
from agent.message_budget import compact_messages_if_needed, estimate_message_tokens
from agent.models import ToolCallRecord, ToolResult
from agent.state import AgentState
from sandbox.runner import DockerSandbox
from tools.web_search import WebSearchTool


def test_docker_sandbox_kills_named_container_on_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(command)
        if command[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(command, timeout=1)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = DockerSandbox(tmp_path).run_bash("while true; do :; done", timeout=1)

    assert result == {"stdout": "", "stderr": "timeout expired", "exit_code": -1}
    container_name = calls[0][calls[0].index("--name") + 1]
    assert container_name.startswith("research-agent-")
    assert ["docker", "kill", container_name] in calls
    assert ["docker", "rm", "-f", container_name] in calls


def test_web_search_uses_async_httpx_client(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[dict[str, Any]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"results": [{"title": "Result", "url": "https://example.com", "content": "Body", "score": 0.9}]}

    class FakeAsyncClient:
        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, url: str, json: dict[str, Any], timeout: int) -> FakeResponse:
            requests.append({"url": url, "json": json, "timeout": timeout})
            return FakeResponse()

    monkeypatch.setattr("tools.web_search.httpx.AsyncClient", FakeAsyncClient)

    result = asyncio.run(WebSearchTool(api_key="token")({"query": "Tesla revenue", "max_results": 3}))

    assert result.ok
    assert requests[0]["url"] == "https://api.tavily.com/search"
    assert requests[0]["json"]["query"] == "Tesla revenue"
    assert requests[0]["json"]["max_results"] == 3
    assert requests[0]["timeout"] == 30


def test_message_budget_compacts_old_tool_output(tmp_path: Path) -> None:
    state = AgentState(question="Question", workspace=tmp_path, reports_dir=tmp_path, traces_dir=tmp_path)
    state.messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "Question:\nQuestion\n\nPlan:\nPlan"},
        {"role": "assistant", "content": "old thought"},
        {"role": "user", "content": "Tool result: " + ("x" * 2_000)},
        {"role": "assistant", "content": "recent thought"},
        {"role": "user", "content": "recent result"},
    ]
    state.cacheable_prefix_messages = 2
    for index in range(3):
        state.tool_calls.append(
            ToolCallRecord(
                "web_search",
                {"query": str(index)},
                ToolResult(ok=True, output={"content": "finding " + str(index)}),
            )
        )

    assert compact_messages_if_needed(state, token_budget=100, recent_dynamic_messages=2)

    assert state.message_truncations == 1
    assert state.messages[0]["content"] == "stable system"
    assert state.messages[1]["content"].startswith("Question:")
    assert "Previous iterations completed" in state.messages[2]["content"]
    assert state.messages[-1]["content"] == "recent result"
    assert estimate_message_tokens(state.messages) < 100


def test_trace_summary_includes_iteration_costs_and_plot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_plot(per_iteration_costs: list[float], output_path: Path) -> Path:
        output_path.write_text("plot", encoding="utf-8")
        return output_path

    monkeypatch.setattr("agent.loop.write_cumulative_cost_plot", fake_plot)
    agent = ResearchAgent.__new__(ResearchAgent)
    agent.traces_dir = tmp_path

    state = AgentState(question="Question", workspace=tmp_path, reports_dir=tmp_path, traces_dir=tmp_path)
    state.messages = [
        {"role": "system", "content": "stable"},
        {"role": "user", "content": "Question:\nQuestion\n\nPlan:\nPlan"},
    ]
    state.cacheable_prefix_messages = 2
    state.iteration_count = 2
    state.per_iteration_costs = [0.01, 0.02]
    state.estimated_cost = 0.03

    agent._write_trace_summary(state)

    trace_path = next(tmp_path.glob("trace-*.json"))
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["per_iteration_costs"] == [0.01, 0.02]
    assert payload["cumulative_cost_plot"].endswith("-cumulative-cost.png")
    assert "stable cacheable prefix" in payload["cacheable_prefix"]
