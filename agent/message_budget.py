from __future__ import annotations

import json
from typing import Any

from agent.state import AgentState


DEFAULT_INPUT_TOKEN_BUDGET = 32_000
CHARS_PER_TOKEN = 4
RECENT_DYNAMIC_MESSAGES = 4
STABLE_PREFIX_MESSAGES = 2


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(_estimate_tokens(message.get("content", "")) for message in messages)


def compact_messages_if_needed(
    state: AgentState,
    token_budget: int = DEFAULT_INPUT_TOKEN_BUDGET,
    recent_dynamic_messages: int = RECENT_DYNAMIC_MESSAGES,
) -> bool:
    if estimate_message_tokens(state.messages) <= token_budget:
        return False

    prefix_count = state.cacheable_prefix_messages or STABLE_PREFIX_MESSAGES
    prefix = state.messages[:prefix_count]
    dynamic = state.messages[prefix_count:]
    recent = dynamic[-recent_dynamic_messages:] if len(dynamic) > recent_dynamic_messages else dynamic

    summary = {
        "role": "user",
        "content": _build_compaction_summary(state, len(recent)),
    }
    state.messages = [*prefix, summary, *recent]
    state.message_truncations += 1
    return True


def cacheable_prefix_description(state: AgentState) -> str:
    prefix_count = state.cacheable_prefix_messages or STABLE_PREFIX_MESSAGES
    roles = [str(message.get("role", "unknown")) for message in state.messages[:prefix_count]]
    return (
        f"First {prefix_count} messages form the stable cacheable prefix: "
        f"{', '.join(roles)}. The prefix contains the system prompt plus the original question/plan."
    )


def _build_compaction_summary(state: AgentState, preserved_recent_messages: int) -> str:
    completed = []
    findings = []
    compacted_calls = state.tool_calls[:-2] if len(state.tool_calls) > 2 else []
    for index, call in enumerate(compacted_calls, start=1):
        status = "ok" if call.result.ok else "error"
        completed.append(f"{index}. {call.name} -> {status}")
        findings.append(_summarize_tool_result(call.name, call.result.output if call.result.ok else call.result.error))

    if not completed:
        completed.append("No older tool calls were available to summarize.")
    if not findings:
        findings.append("No older tool findings were available to summarize.")

    return "\n".join(
        [
            "Previous iterations completed:",
            *completed,
            "",
            "Key findings so far:",
            *findings,
            "",
            f"Most recent {preserved_recent_messages} dynamic messages are preserved at full fidelity.",
        ]
    )


def _summarize_tool_result(tool_name: str, output: Any) -> str:
    text = output if isinstance(output, str) else json.dumps(output, default=str)
    text = " ".join(text.split())
    if len(text) > 500:
        text = f"{text[:497]}..."
    return f"- {tool_name}: {text}"


def _estimate_tokens(value: Any) -> int:
    if not isinstance(value, str):
        value = json.dumps(value, default=str)
    return max(1, len(value) // CHARS_PER_TOKEN)
