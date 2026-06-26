from __future__ import annotations

from agent.models import ModelDecision
from agent.state import AgentState
from hooks.base import Hook, HookResult

MAX_ITERATIONS = 30


class IterationLimitHook(Hook):
    def before_tool(self, state: AgentState, decision: ModelDecision) -> HookResult:
        if state.iteration_count >= MAX_ITERATIONS:
            return HookResult(allow=False, message="Maximum iteration limit reached.", terminate=True)
        return HookResult(allow=True)

    def before_final(self, state: AgentState, decision: ModelDecision) -> HookResult:
        if state.iteration_count >= MAX_ITERATIONS:
            return HookResult(allow=True)
        return HookResult(allow=True)

