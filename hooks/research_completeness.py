from __future__ import annotations

from pathlib import Path

from agent.models import ModelDecision
from agent.state import AgentState
from hooks.answer_without_code import ANALYTICAL_TERMS
from hooks.base import Hook, HookResult


class ResearchCompletenessHook(Hook):
    def before_final(self, state: AgentState, decision: ModelDecision) -> HookResult:
        question = state.question.lower()
        is_analytical = any(term in question for term in ANALYTICAL_TERMS)

        searches = sum(1 for call in state.tool_calls if call.name == "web_search" and call.result.ok)
        analyses = sum(1 for call in state.tool_calls if call.name == "execute_python" and call.result.ok)
        reports = [path for path in state.generated_files if Path(path).suffix.lower() == ".md"]

        missing: list[str] = []

        if searches < 1:
            missing.append("at least 1 successful search")

        if is_analytical:
            if analyses < 1:
                missing.append("at least 1 successful analysis step")
            if not reports:
                missing.append("at least 1 markdown report file. Use write_file to create a .md report with your findings")

        if missing:
            return HookResult(allow=False, message="Research incomplete: " + "; ".join(missing) + ".")
        return HookResult(allow=True)
