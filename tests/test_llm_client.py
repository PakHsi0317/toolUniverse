"""Tests for llm_client.py.

Real-API tests are gated behind a flag so they don't run by default
(don't burn credits on every test run).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from src.llm_client import (
    LLMResponse,
    MockLLMClient,
    OpenAIClient,
    ToolCall,
    _extract_json,
    get_client,
)


# ---------------------------------------------------------------------------
# JSON extraction (pure function)
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_plain(self):
        assert _extract_json('{"a": 1}') == '{"a": 1}'

    def test_fenced_json(self):
        assert _extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_fenced_no_lang(self):
        assert _extract_json('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_with_whitespace(self):
        assert _extract_json('  {"a": 1}  ') == '{"a": 1}'


# ---------------------------------------------------------------------------
# Mock client (no API needed)
# ---------------------------------------------------------------------------

class TestMockClient:
    def test_default_text_response(self, tmp_path: Path):
        c = MockLLMClient(cache_dir=tmp_path)
        resp = c.complete("hello")
        assert "mock" in resp.text

    def test_default_tool_call(self, tmp_path: Path):
        c = MockLLMClient(cache_dir=tmp_path)
        tools = [{"function": {"name": "search_pubmed"}}]
        resp = c.call_with_tools("find papers", tools=tools)
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search_pubmed"

    def test_handler_match(self, tmp_path: Path):
        c = MockLLMClient(cache_dir=tmp_path)
        c.register("special prompt", lambda p, ctx: LLMResponse(text="custom"))
        assert c.complete("this is a special prompt").text == "custom"
        assert c.complete("nothing special").text != "custom"

    def test_call_log(self, tmp_path: Path):
        c = MockLLMClient(cache_dir=tmp_path)
        c.complete("first")
        c.complete("second")
        # Same prompt twice should hit cache (second call not in log)
        c.complete("second")
        assert len(c.call_log) == 2

    def test_structured_output(self, tmp_path: Path):
        class Out(BaseModel):
            answer: int

        c = MockLLMClient(cache_dir=tmp_path)
        c.register("Return ONLY", lambda p, ctx: LLMResponse(text='{"answer": 42}'))
        result = c.complete_structured("compute 6*7", Out)
        assert result.answer == 42

    def test_structured_retries_on_bad_json(self, tmp_path: Path):
        class Out(BaseModel):
            answer: int

        c = MockLLMClient(cache_dir=tmp_path)
        attempts = []

        def handler(prompt, ctx):
            attempts.append(prompt)
            # First two attempts return bad JSON, third returns good
            if len(attempts) < 3:
                return LLMResponse(text="not json")
            return LLMResponse(text='{"answer": 42}')

        c.register("Return ONLY", handler)
        result = c.complete_structured("question", Out, max_attempts=3)
        assert result.answer == 42
        assert len(attempts) == 3


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------

class TestCache:
    def test_same_prompt_hits_cache(self, tmp_path: Path):
        c = MockLLMClient(cache_dir=tmp_path)
        calls = [0]
        c.register("test", lambda p, ctx: (calls.__setitem__(0, calls[0] + 1), LLMResponse(text=f"call {calls[0]}"))[1])
        r1 = c.complete("test query")
        r2 = c.complete("test query")
        assert r1.text == r2.text
        assert calls[0] == 1  # Handler only called once due to cache

    def test_different_temp_misses_cache(self, tmp_path: Path):
        c = MockLLMClient(cache_dir=tmp_path)
        calls = [0]
        c.register("test", lambda p, ctx: (calls.__setitem__(0, calls[0] + 1), LLMResponse(text=f"call {calls[0]}"))[1])
        c.complete("test query", temperature=0.0)
        c.complete("test query", temperature=0.7)
        assert calls[0] == 2


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_mock(self, tmp_path: Path):
        c = get_client("mock", cache_dir=tmp_path)
        assert isinstance(c, MockLLMClient)

    def test_unknown_provider(self):
        with pytest.raises(ValueError):
            get_client("not-a-provider")


# ---------------------------------------------------------------------------
# Real OpenAI smoke test — only runs if OPENAI_API_KEY is set AND --live flag
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set in environment",
)
def test_openai_live_smoke(tmp_path: Path):
    """Hits the real API. Cheap: one tiny ping to verify connectivity."""
    c = OpenAIClient(model="gpt-4o-mini", cache_dir=tmp_path)
    resp = c.complete(
        "Reply with exactly the word: pong",
        max_tokens=10,
    )
    assert "pong" in resp.text.lower()
