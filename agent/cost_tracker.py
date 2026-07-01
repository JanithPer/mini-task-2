from __future__ import annotations

from dataclasses import dataclass

from agent.models import TokenPricing, TokenUsage


@dataclass(slots=True)
class CostTracker:
    input_rate_per_million: float = 0.25
    output_rate_per_million: float = 2.0

    @classmethod
    def from_pricing(cls, pricing: TokenPricing) -> CostTracker:
        return cls(
            input_rate_per_million=pricing.input_rate_per_million,
            output_rate_per_million=pricing.output_rate_per_million,
        )

    def update(self, usage: TokenUsage, delta: TokenUsage) -> float:
        usage.input_tokens += delta.input_tokens
        usage.output_tokens += delta.output_tokens
        usage.cached_input_tokens += delta.cached_input_tokens
        return self.estimate(usage)

    def estimate(self, usage: TokenUsage) -> float:
        billable_input = max(0, usage.input_tokens - usage.cached_input_tokens)
        input_cost = billable_input / 1_000_000 * self.input_rate_per_million
        output_cost = usage.output_tokens / 1_000_000 * self.output_rate_per_million
        return input_cost + output_cost
