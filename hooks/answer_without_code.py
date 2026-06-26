from __future__ import annotations

from agent.models import ModelDecision
from agent.state import AgentState
from hooks.base import Hook, HookResult

ANALYTICAL_TERMS = ("analyse", "analyze", "trend", "analysis", "forecast", "earnings", "revenue", "predict")


class AnswerWithoutCodeHook(Hook):
    def before_final(self, state: AgentState, decision: ModelDecision) -> HookResult:
        question = state.question.lower()
        needs_code = any(term in question for term in ANALYTICAL_TERMS)
        used_python = any(call.name == "execute_python" and call.result.ok for call in state.tool_calls)
        if needs_code and not used_python:
            return HookResult(
                allow=False,
                message="Analysis tasks require code execution. Generate Python and retry.",
            )
        return HookResult(allow=True)

