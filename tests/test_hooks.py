from __future__ import annotations

from pathlib import Path

from agent.models import ModelDecision, ToolCallRecord, ToolResult
from agent.state import AgentState
from hooks.answer_without_code import AnswerWithoutCodeHook
from hooks.research_completeness import ResearchCompletenessHook


def make_state(tmp_path: Path, question: str = "Analyze revenue trend") -> AgentState:
    return AgentState(question=question, workspace=tmp_path, reports_dir=tmp_path, traces_dir=tmp_path)


def test_answer_without_code_rejects_analysis_final(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    decision = ModelDecision(thought="done", action="final", answer="Final.")
    result = AnswerWithoutCodeHook().before_final(state, decision)
    assert not result.allow


def test_research_completeness_accepts_required_work(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    ok = ToolResult(ok=True, output={})
    for index in range(3):
        state.tool_calls.append(ToolCallRecord("web_search", {"query": str(index)}, ok))
    state.tool_calls.append(ToolCallRecord("execute_python", {"code": "print(1)"}, ok))
    state.generated_files.append(tmp_path / "report.md")

    decision = ModelDecision(thought="done", action="final", answer="Final with https://example.com")
    result = ResearchCompletenessHook().before_final(state, decision)
    assert result.allow

