from __future__ import annotations

from pathlib import Path

from agent.models import ModelDecision
from agent.state import AgentState
from hooks.answer_without_code import ANALYTICAL_TERMS
from hooks.base import Hook, HookResult


class ChartRequiredHook(Hook):
    def before_final(self, state: AgentState, decision: ModelDecision) -> HookResult:
        if not any(term in state.question.lower() for term in ANALYTICAL_TERMS):
            return HookResult(allow=True)
        chart_files = [path for path in state.generated_files if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if chart_files:
            return HookResult(allow=True)
        return HookResult(
            allow=False,
            message="This question requires a generated chart before the final answer.",
        )

