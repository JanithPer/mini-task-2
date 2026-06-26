from __future__ import annotations

from agent.models import GEMINI_OPENAI_BASE_URL, resolve_model
from agent.openai_client import parse_decision


def test_resolve_google_model() -> None:
    model = resolve_model("gemini-3.5-flash")

    assert model.name == "gemini-3.5-flash"
    assert model.provider == "google"
    assert model.api_key_env == "GEMINI_API_KEY"
    assert model.base_url == GEMINI_OPENAI_BASE_URL
    assert model.pricing.input_rate_per_million == 1.5
    assert model.pricing.output_rate_per_million == 9.0


def test_resolve_unknown_openai_model() -> None:
    model = resolve_model("gpt-custom")

    assert model.name == "gpt-custom"
    assert model.provider == "openai"
    assert model.api_key_env == "OPENAI_API_KEY"
    assert model.base_url is None


def test_parse_tool_decision() -> None:
    decision = parse_decision(
        '{"thought":"Need data","action":"tool","tool":"web_search","input":{"query":"Tesla revenue"}}'
    )
    assert decision.action == "tool"
    assert decision.tool == "web_search"
    assert decision.tool_input["query"] == "Tesla revenue"


def test_parse_final_decision() -> None:
    decision = parse_decision('{"thought":"Done","action":"final","answer":"Answer https://example.com"}')
    assert decision.action == "final"
    assert decision.answer == "Answer https://example.com"
