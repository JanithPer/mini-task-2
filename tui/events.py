from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ThoughtEvent:
    thought: str


@dataclass(slots=True)
class ToolCallEvent:
    name: str
    payload: dict[str, Any]


@dataclass(slots=True)
class ToolResultEvent:
    name: str
    ok: bool
    output: Any


@dataclass(slots=True)
class AnswerEvent:
    answer: str


@dataclass(slots=True)
class CostEvent:
    input_tokens: int
    output_tokens: int
    cost: float
    iteration: int


TraceEvent = ThoughtEvent | ToolCallEvent | ToolResultEvent | AnswerEvent | CostEvent

