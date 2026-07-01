from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.models import TokenUsage, ToolCallRecord


@dataclass
class AgentState:
    question: str
    workspace: Path
    reports_dir: Path
    traces_dir: Path
    messages: list[dict[str, Any]] = field(default_factory=list)
    iteration_count: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    generated_files: list[Path] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    estimated_cost: float = 0.0
    per_iteration_costs: list[float] = field(default_factory=list)
    cacheable_prefix_messages: int = 0
    gemini_cache_name: str | None = None
    message_truncations: int = 0
    final_answer: str | None = None
    started_at: float = field(default_factory=time.monotonic)
    completed_at: float | None = None
    forced_corrections: int = 0

    def runtime_seconds(self) -> float:
        end = self.completed_at or time.monotonic()
        return end - self.started_at

    def mark_complete(self) -> None:
        self.completed_at = time.monotonic()

    def add_generated_file(self, path: Path) -> None:
        if path not in self.generated_files:
            self.generated_files.append(path)

    def cost_report(self) -> str:
        return "\n".join(
            [
                f"Question: {self.question}",
                f"Iterations: {self.iteration_count}",
                f"Input Tokens: {self.token_usage.input_tokens:,}",
                f"Cached Input Tokens: {self.token_usage.cached_input_tokens:,}",
                f"Output Tokens: {self.token_usage.output_tokens:,}",
                f"Total Tokens: {self.token_usage.total_tokens:,}",
                f"Total Cost: ${self.estimated_cost:.4f}",
                f"Tool Calls: {len(self.tool_calls)}",
                f"Runtime: {self.runtime_seconds():.1f} seconds",
            ]
        )

