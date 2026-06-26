from __future__ import annotations

from agent.models import ModelDecision
from agent.state import AgentState
from hooks.base import Hook, HookResult
from tools.registry import ToolRegistry


class ToolValidationHook(Hook):
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def before_tool(self, state: AgentState, decision: ModelDecision) -> HookResult:
        if decision.tool and self.registry.has(decision.tool):
            return HookResult(allow=True)
        return HookResult(
            allow=False,
            message=f"Tool does not exist. Valid tools: {', '.join(self.registry.names)}.",
        )

