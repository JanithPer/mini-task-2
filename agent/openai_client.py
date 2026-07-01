from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from openai import AsyncOpenAI

from agent.models import ModelConfig, ModelDecision, TokenUsage, get_configured_model, resolve_model


CACHEABLE_PREFIX_SIZE = 2


class OpenAIClient:
    def __init__(self, model: str | ModelConfig | None = None, gemini_cache_name: str | None = None) -> None:
        self.model_config = resolve_model(model) if model else get_configured_model()
        self.model = self.model_config.name
        self.gemini_cache_name = gemini_cache_name
        api_key = os.getenv(self.model_config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing {self.model_config.api_key_env} for model {self.model!r}.")
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.model_config.base_url,
        )

    async def decide(self, messages: list[dict[str, Any]]) -> tuple[ModelDecision, TokenUsage]:
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "response_format": {"type": "json_object"},
        }
        if self.model_config.provider == "openai":
            create_kwargs["messages"] = messages
            create_kwargs["prompt_cache_key"] = _prompt_cache_key(messages)
            if retention := os.getenv("OPENAI_PROMPT_CACHE_RETENTION"):
                create_kwargs["prompt_cache_retention"] = retention
        elif self.model_config.provider == "google" and self.gemini_cache_name:
            create_kwargs["messages"] = messages[CACHEABLE_PREFIX_SIZE:]
            create_kwargs["extra_body"] = {"cached_content": self.gemini_cache_name}
        else:
            create_kwargs["messages"] = messages

        response = await self._client.chat.completions.create(
            **create_kwargs,
        )
        content = response.choices[0].message.content or "{}"
        usage = response.usage
        token_usage = TokenUsage(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cached_input_tokens=_cached_prompt_tokens(usage),
        )
        return parse_decision(content), token_usage

    async def complete_text(self, messages: list[dict[str, Any]]) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content or ""


def _prompt_cache_key(messages: list[dict[str, Any]], prefix_messages: int = 2) -> str:
    stable_prefix = messages[:prefix_messages]
    serialized = json.dumps(stable_prefix, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]
    return f"research-agent-{digest}"


def _cached_prompt_tokens(usage: Any) -> int:
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        if isinstance(details, dict):
            val = details.get("cached_tokens")
        else:
            val = getattr(details, "cached_tokens", None)
        if val:
            return int(val)
    v = getattr(usage, "cached_content_token_count", None)
    if v:
        return int(v)
    return 0


def parse_decision(content: str) -> ModelDecision:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {exc}") from exc

    action = data.get("action")
    if action not in {"tool", "final"}:
        raise ValueError(f"Invalid model action: {action!r}")

    if action == "tool":
        tool = data.get("tool")
        if not isinstance(tool, str) or not tool:
            raise ValueError("Tool decision missing tool name.")
        tool_input = data.get("input", {})
        if not isinstance(tool_input, dict):
            raise ValueError("Tool input must be an object.")
        return ModelDecision(
            thought=str(data.get("thought", "")),
            action="tool",
            tool=tool,
            tool_input=tool_input,
        )

    answer = data.get("answer")
    if not isinstance(answer, str) or not answer:
        raise ValueError("Final decision missing answer.")
    return ModelDecision(
        thought=str(data.get("thought", "")),
        action="final",
        answer=answer,
    )
