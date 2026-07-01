from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from agent.models import GEMINI_OPENAI_BASE_URL, resolve_model
from agent.openai_client import OpenAIClient, parse_decision


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


def test_openai_decide_sends_prompt_cache_key_and_reads_cached_tokens(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"thought":"cached","action":"final","answer":"Done"}'
                        )
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=2048,
                    completion_tokens=32,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=1024),
                ),
            )

    monkeypatch.setenv("OPENAI_PROMPT_CACHE_RETENTION", "24h")
    client = OpenAIClient.__new__(OpenAIClient)
    client.model_config = resolve_model("gpt-custom")
    client.model = client.model_config.name
    client.gemini_cache_name = None
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    messages = [
        {"role": "system", "content": "stable"},
        {"role": "user", "content": "Question:\nQ\n\nPlan:\nP"},
        {"role": "user", "content": "dynamic"},
    ]

    decision, usage = asyncio.run(client.decide(messages))

    assert decision.action == "final"
    assert calls[0]["prompt_cache_key"].startswith("research-agent-")
    assert calls[0]["prompt_cache_retention"] == "24h"
    assert usage.cached_input_tokens == 1024


def test_google_compatible_decide_omits_openai_prompt_cache_parameters() -> None:
    calls: list[dict[str, Any]] = []

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"thought":"ok","action":"final","answer":"Done"}'
                        )
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

    client = OpenAIClient.__new__(OpenAIClient)
    client.model_config = resolve_model("gemini-3.5-flash")
    client.model = client.model_config.name
    client.gemini_cache_name = None
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    asyncio.run(client.decide([{"role": "user", "content": "hello"}]))

    assert "prompt_cache_key" not in calls[0]
    assert "prompt_cache_retention" not in calls[0]
    assert "extra_body" not in calls[0]


def test_gemini_cached_tokens_from_cached_content_token_count_field() -> None:
    calls: list[dict[str, Any]] = []

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"thought":"cached","action":"final","answer":"Done"}'
                        )
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5,
                    cached_content_token_count=8,
                ),
            )

    client = OpenAIClient.__new__(OpenAIClient)
    client.model_config = resolve_model("gemini-3.5-flash")
    client.model = client.model_config.name
    client.gemini_cache_name = "cachedContents/abc123"
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    _, usage = asyncio.run(client.decide([{"role": "user", "content": "hello"}]))

    assert usage.cached_input_tokens == 8


def test_gemini_with_explicit_cache_trims_messages_and_sends_extra_body() -> None:
    calls: list[dict[str, Any]] = []

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"thought":"cached","action":"final","answer":"Done"}'
                        )
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=8),
                ),
            )

    client = OpenAIClient.__new__(OpenAIClient)
    client.model_config = resolve_model("gemini-3.5-flash")
    client.model = client.model_config.name
    client.gemini_cache_name = "cachedContents/abc123"
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    messages = [
        {"role": "system", "content": "stable system prompt"},
        {"role": "user", "content": "Question:\nQ\n\nPlan:\nP"},
        {"role": "assistant", "content": "dynamic tool call"},
        {"role": "user", "content": "dynamic tool result"},
    ]

    decision, usage = asyncio.run(client.decide(messages))

    assert decision.action == "final"
    assert calls[0]["messages"] == [
        {"role": "assistant", "content": "dynamic tool call"},
        {"role": "user", "content": "dynamic tool result"},
    ]
    assert calls[0]["extra_body"] == {"cached_content": "cachedContents/abc123"}
    assert "prompt_cache_key" not in calls[0]
    assert "prompt_cache_retention" not in calls[0]
    assert usage.cached_input_tokens == 8
