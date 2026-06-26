from __future__ import annotations

from dataclasses import dataclass

from agent.models import ModelDecision
from agent.state import AgentState


@dataclass(slots=True)
class HookResult:
    allow: bool
    message: str | None = None
    terminate: bool = False


class Hook:
    def before_final(self, state: AgentState, decision: ModelDecision) -> HookResult:
        return HookResult(allow=True)

    def before_tool(self, state: AgentState, decision: ModelDecision) -> HookResult:
        return HookResult(allow=True)

