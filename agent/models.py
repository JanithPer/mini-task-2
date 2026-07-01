from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal


AgentAction = Literal["tool", "final"]
ModelProvider = Literal["openai", "google"]

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass(frozen=True, slots=True)
class TokenPricing:
    input_rate_per_million: float
    output_rate_per_million: float


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    provider: ModelProvider
    api_key_env: str
    pricing: TokenPricing
    base_url: str | None = None


# Source: https://developers.openai.com/api/docs/pricing | 2026-07-01
OPENAI_DEFAULT_MODEL = ModelConfig(
    name="gpt-5.4-mini",
    provider="openai",
    api_key_env="OPENAI_API_KEY",
    pricing=TokenPricing(input_rate_per_million=0.75, output_rate_per_million=4.50),
)
# Source: https://ai.google.dev/gemini-api/docs/pricing | 2026-07-01
GEMINI_3_5_FLASH = ModelConfig(
    name="gemini-3.5-flash",
    provider="google",
    api_key_env="GEMINI_API_KEY",
    pricing=TokenPricing(input_rate_per_million=1.5, output_rate_per_million=9.0),
    base_url=GEMINI_OPENAI_BASE_URL,
)
# Source: https://ai.google.dev/gemini-api/docs/pricing | 2026-07-01
GEMINI_3_1_FLASH_LITE = ModelConfig(
    name="gemini-3.1-flash-lite",
    provider="google",
    api_key_env="GEMINI_API_KEY",
    pricing=TokenPricing(input_rate_per_million=0.25, output_rate_per_million=1.50),
    base_url=GEMINI_OPENAI_BASE_URL,
)

DEFAULT_MODEL = GEMINI_3_1_FLASH_LITE

MODEL_REGISTRY: dict[str, ModelConfig] = {
    OPENAI_DEFAULT_MODEL.name: OPENAI_DEFAULT_MODEL,
    GEMINI_3_5_FLASH.name: GEMINI_3_5_FLASH,
    GEMINI_3_1_FLASH_LITE.name: GEMINI_3_1_FLASH_LITE,
}


def resolve_model(model: str | ModelConfig | None = None) -> ModelConfig:
    if isinstance(model, ModelConfig):
        return model
    if not model:
        return DEFAULT_MODEL
    if model in MODEL_REGISTRY:
        return MODEL_REGISTRY[model]
    if model.startswith("gemini-"):
        return ModelConfig(
            name=model,
            provider="google",
            api_key_env="GEMINI_API_KEY",
            pricing=GEMINI_3_5_FLASH.pricing,
            base_url=GEMINI_OPENAI_BASE_URL,
        )
    return ModelConfig(
        name=model,
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        pricing=OPENAI_DEFAULT_MODEL.pricing,
    )


def get_configured_model() -> ModelConfig:
    return resolve_model(os.getenv("AGENT_MODEL") or os.getenv("OPENAI_MODEL"))


@dataclass(slots=True)
class ModelDecision:
    thought: str
    action: AgentAction
    tool: str | None = None
    tool_input: dict[str, Any] = field(default_factory=dict)
    answer: str | None = None


@dataclass(slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(slots=True)
class ToolResult:
    ok: bool
    output: Any
    error: str | None = None

    def as_message_content(self) -> str:
        if self.ok:
            return f"Tool result: {self.output}"
        return f"Tool error: {self.error or self.output}"


@dataclass(slots=True)
class ToolCallRecord:
    name: str
    input: dict[str, Any]
    result: ToolResult
