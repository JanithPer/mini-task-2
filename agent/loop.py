from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent.cost_tracker import CostTracker
from agent.models import ModelDecision, ToolCallRecord, ToolResult
from agent.openai_client import OpenAIClient
from agent.planner import Planner
from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState
from hooks.answer_without_code import AnswerWithoutCodeHook
from hooks.base import Hook
from hooks.chart_required import ChartRequiredHook
from hooks.citation_validation import CitationValidationHook
from hooks.iteration_limit import MAX_ITERATIONS, IterationLimitHook
from hooks.research_completeness import ResearchCompletenessHook
from hooks.tool_validation import ToolValidationHook
from tools.registry import ToolRegistry
from tui.events import AnswerEvent, CostEvent, ThoughtEvent, ToolCallEvent, ToolResultEvent, TraceEvent

EventSink = Callable[[TraceEvent], None]


class ResearchAgent:
    def __init__(
        self,
        client: OpenAIClient | None = None,
        event_sink: EventSink | None = None,
        workspace: Path | None = None,
        reports_dir: Path | None = None,
        traces_dir: Path | None = None,
    ) -> None:
        self.client = client or OpenAIClient()
        self.workspace = (workspace or Path(os.getenv("AGENT_WORKSPACE", "workspaces/default"))).resolve()
        self.reports_dir = (reports_dir or Path(os.getenv("AGENT_REPORTS_DIR", "reports"))).resolve()
        self.traces_dir = (traces_dir or Path(os.getenv("AGENT_TRACES_DIR", "traces"))).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ToolRegistry(self.workspace)
        self.cost_tracker = CostTracker.from_pricing(self.client.model_config.pricing)
        self.event_sink = event_sink
        self.hooks: list[Hook] = [
            IterationLimitHook(),
            ToolValidationHook(self.registry),
            AnswerWithoutCodeHook(),
            ResearchCompletenessHook(),
            ChartRequiredHook(),
            CitationValidationHook(),
        ]

    async def run(self, question: str) -> AgentState:
        state = AgentState(
            question=question,
            workspace=self.workspace,
            reports_dir=self.reports_dir,
            traces_dir=self.traces_dir,
        )
        plan = await Planner(self.client).create_plan(question)
        state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question:\n{question}\n\nPlan:\n{plan}"},
        ]
        self._emit(ThoughtEvent(f"Plan created:\n{plan}"))

        while state.iteration_count < MAX_ITERATIONS:
            state.iteration_count += 1
            try:
                decision, usage = await self.client.decide(state.messages)
            except Exception as exc:
                state.messages.append({"role": "assistant", "content": self._error_decision(str(exc))})
                state.messages.append({"role": "user", "content": "Return valid JSON that follows the required schema."})
                continue

            state.estimated_cost = self.cost_tracker.update(state.token_usage, usage)
            self._emit(CostEvent(
                input_tokens=state.token_usage.input_tokens,
                output_tokens=state.token_usage.output_tokens,
                cost=state.estimated_cost,
                iteration=state.iteration_count,
            ))
            self._emit(ThoughtEvent(decision.thought))

            if decision.action == "final":
                if self._handle_final_hooks(state, decision):
                    state.final_answer = decision.answer
                    state.mark_complete()
                    self._emit(AnswerEvent(decision.answer or ""))
                    self._write_trace_summary(state)
                    return state
                continue

            if decision.action == "tool":
                if not self._handle_tool_hooks(state, decision):
                    continue
                await self._execute_tool(state, decision)

        if state.final_answer is None:
            state.final_answer = "Stopped after maximum iteration limit before a complete final answer was produced."
        state.mark_complete()
        self._emit(AnswerEvent(state.final_answer))
        self._write_trace_summary(state)
        return state

    def _handle_tool_hooks(self, state: AgentState, decision: ModelDecision) -> bool:
        for hook in self.hooks:
            result = hook.before_tool(state, decision)
            if result.terminate:
                state.final_answer = result.message
                return False
            if not result.allow:
                self._inject_correction(state, result.message or "Tool call rejected.")
                return False
        return True

    def _handle_final_hooks(self, state: AgentState, decision: ModelDecision) -> bool:
        for hook in self.hooks:
            result = hook.before_final(state, decision)
            if result.allow:
                continue
            self._inject_correction(state, result.message or "Final answer rejected.")
            return False
        return True

    async def _execute_tool(self, state: AgentState, decision: ModelDecision) -> None:
        assert decision.tool is not None
        self._emit(ToolCallEvent(decision.tool, decision.tool_input))
        known_files = self._workspace_files()
        known_root_files = self._files_at(self.workspace.parent)
        result = await self.registry.execute(decision.tool, decision.tool_input)
        self._record_generated_files(state, decision, result, known_files, known_root_files)
        state.tool_calls.append(ToolCallRecord(decision.tool, decision.tool_input, result))
        self._emit(ToolResultEvent(decision.tool, result.ok, result.output if result.ok else result.error))
        state.messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "thought": decision.thought,
                        "action": "tool",
                        "tool": decision.tool,
                        "input": decision.tool_input,
                    }
                ),
            }
        )
        state.messages.append({"role": "user", "content": result.as_message_content()})

    def _record_generated_files(
        self,
        state: AgentState,
        decision: ModelDecision,
        result: ToolResult,
        known_files: set[Path],
        known_root_files: set[Path] | None = None,
    ) -> None:
        if decision.tool == "write_file" and result.ok and isinstance(result.output, dict):
            path = result.output.get("path")
            if path:
                state.add_generated_file(Path(str(path)).resolve())
        for path in self._workspace_files() - known_files:
            state.add_generated_file(path)
        if decision.tool in ("execute_python", "execute_bash") and result.ok and known_root_files is not None:
            for path in self._files_at(self.workspace.parent) - known_root_files:
                state.add_generated_file(path)

    def _workspace_files(self) -> set[Path]:
        return self._files_at(self.workspace)

    @staticmethod
    def _files_at(root: Path) -> set[Path]:
        if not root.exists():
            return set()
        return {path.resolve() for path in root.rglob("*") if path.is_file()}

    def _inject_correction(self, state: AgentState, message: str) -> None:
        state.forced_corrections += 1
        state.messages.append({"role": "user", "content": message})
        self._emit(ThoughtEvent(f"Correction injected: {message}"))

    def _emit(self, event: TraceEvent) -> None:
        if self.event_sink:
            self.event_sink(event)

    def _write_trace_summary(self, state: AgentState) -> None:
        trace_path = self.traces_dir / f"trace-{int(time.time())}.json"
        payload: dict[str, Any] = {
            "question": state.question,
            "iterations": state.iteration_count,
            "tool_calls": [
                {
                    "name": call.name,
                    "input": call.input,
                    "ok": call.result.ok,
                    "output": call.result.output,
                    "error": call.result.error,
                }
                for call in state.tool_calls
            ],
            "generated_files": [str(path) for path in state.generated_files],
            "cost": state.estimated_cost,
            "input_tokens": state.token_usage.input_tokens,
            "output_tokens": state.token_usage.output_tokens,
            "final_answer": state.final_answer,
        }
        trace_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _error_decision(error: str) -> str:
        return json.dumps({"thought": error, "action": "tool", "tool": "invalid", "input": {}})
