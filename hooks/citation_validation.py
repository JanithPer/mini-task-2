from __future__ import annotations

import re

from agent.models import ModelDecision
from agent.state import AgentState
from hooks.base import Hook, HookResult

URL_RE = re.compile(r"https?://")


class CitationValidationHook(Hook):
    def before_final(self, state: AgentState, decision: ModelDecision) -> HookResult:
        answer = decision.answer or ""
        has_search = any(call.name == "web_search" and call.result.ok for call in state.tool_calls)
        if not has_search or URL_RE.search(answer):
            return HookResult(allow=True)
        return HookResult(
            allow=False,
            message="Final answer must cite source URLs for web-sourced claims.",
        )

