"""LLM client with caching, structured output, and mock mode.

All calls are cached to disk so the same prompt is never paid for twice.
Cache key includes (provider, model, prompt, tools, temperature) — so a
change in any of these forces a fresh call.

Currently supports:
- OpenAIClient — production
- MockLLMClient — testing / CI without API costs

Adding Anthropic later is a 30-line addition (subclass LLMClient).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional, Type, TypeVar

import truststore
from diskcache import Cache
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Use Windows/macOS/Linux system trust store for SSL (fixes
# CERTIFICATE_VERIFY_FAILED on Windows + Anaconda setups).
truststore.inject_into_ssl()

# Load .env at import time so os.environ has the keys.
load_dotenv()

T = TypeVar("T", bound=BaseModel)

_DEFAULT_CACHE_DIR = Path(__file__).parent.parent / ".llm_cache"


# ---------------------------------------------------------------------------
# Unified response shape
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]


class LLMResponse(BaseModel):
    """Normalized response shape across providers."""

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: Optional[str] = None
    raw: dict = Field(default_factory=dict)  # provider-native, for debugging


# ---------------------------------------------------------------------------
# Base client with caching
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Abstract base. Subclasses implement _call() against their provider's SDK.

    The base class handles caching, retries, and structured output parsing —
    so every provider gets these features for free.
    """

    def __init__(self, model: str, cache_dir: Path | str = _DEFAULT_CACHE_DIR):
        self.model = model
        self.cache = Cache(str(cache_dir))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Plain text completion (no tool use). Cached."""
        return self._cached_call(
            prompt=prompt, system=system, tools=None,
            temperature=temperature, max_tokens=max_tokens,
        )

    def call_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Tool-use completion. `tools` must be in OpenAI function format. Cached."""
        return self._cached_call(
            prompt=prompt, system=system, tools=tools,
            temperature=temperature, max_tokens=max_tokens,
        )

    def complete_structured(
        self,
        prompt: str,
        schema: Type[T],
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_attempts: int = 3,
    ) -> T:
        """Returns a parsed Pydantic instance. Retries with error feedback on parse failure."""
        last_err: str = ""
        schema_text = json.dumps(schema.model_json_schema(), indent=2)
        for attempt in range(max_attempts):
            # Attempt counter is included so retries don't collide in the cache
            # when the previous-error message happens to be identical.
            suffix = (
                f"\n\n[Attempt {attempt + 1}/{max_attempts}]\n"
                f"Return ONLY a JSON object matching this schema "
                f"(no prose, no markdown fences):\n{schema_text}"
            )
            if attempt > 0:
                suffix += f"\n\nPrevious attempt failed: {last_err}\nFix the JSON."
            resp = self.complete(
                prompt + suffix, system=system,
                temperature=temperature, max_tokens=max_tokens,
            )
            try:
                return schema.model_validate_json(_extract_json(resp.text))
            except Exception as e:
                last_err = str(e)[:200]
        raise ValueError(
            f"Failed to parse {schema.__name__} after {max_attempts} attempts. "
            f"Last error: {last_err}"
        )

    # ------------------------------------------------------------------
    # Cache + retry layer (concrete; subclasses don't touch this)
    # ------------------------------------------------------------------

    def _cached_call(self, **kwargs) -> LLMResponse:
        key = self._cache_key(kwargs)
        if key in self.cache:
            return LLMResponse.model_validate(self.cache[key])
        resp = self._retry_call(**kwargs)
        self.cache[key] = resp.model_dump()
        return resp

    def _retry_call(self, max_retries: int = 3, **kwargs) -> LLMResponse:
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return self._call(**kwargs)
            except Exception as e:
                last_exc = e
                wait = 2 ** attempt
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_exc}")

    def _cache_key(self, kwargs: dict) -> str:
        canonical = json.dumps({
            "provider": self.__class__.__name__,
            "model": self.model,
            **kwargs,
        }, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Provider-specific (subclass implements)
    # ------------------------------------------------------------------

    @abstractmethod
    def _call(
        self,
        prompt: str,
        system: Optional[str],
        tools: Optional[list[dict]],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------

class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: Optional[str] = None,
        cache_dir: Path | str = _DEFAULT_CACHE_DIR,
    ):
        from openai import OpenAI
        model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        super().__init__(model=model, cache_dir=cache_dir)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Set it in .env or as an environment variable."
            )
        self.client = OpenAI(api_key=api_key)

    def _call(self, prompt, system, tools, temperature, max_tokens) -> LLMResponse:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        raw = self.client.chat.completions.create(**kwargs)
        choice = raw.choices[0]
        text = choice.message.content or ""

        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw_arguments": tc.function.arguments, "_parse_error": True}
                tool_calls.append(ToolCall(name=tc.function.name, arguments=args))

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            raw=raw.model_dump(),
        )


# ---------------------------------------------------------------------------
# Mock client — for end-to-end testing without API costs
# ---------------------------------------------------------------------------

class MockLLMClient(LLMClient):
    """Deterministic mock. Register handlers that match prompt substrings.

    Critical for: CI, local pipeline testing, and bootstrapping Phase 1/2
    before you want to spend on real API calls.
    """

    def __init__(self, cache_dir: Path | str = _DEFAULT_CACHE_DIR):
        super().__init__(model="mock", cache_dir=Path(cache_dir) / "mock")
        self.handlers: list[tuple[str, Callable[[str, dict], LLMResponse]]] = []
        self.call_log: list[dict] = []

    def register(
        self, prompt_substring: str, handler: Callable[[str, dict], LLMResponse]
    ) -> None:
        self.handlers.append((prompt_substring, handler))

    def _call(self, prompt, system, tools, temperature, max_tokens) -> LLMResponse:
        self.call_log.append({"prompt": prompt, "tools": tools})
        for substring, handler in self.handlers:
            if substring in prompt:
                return handler(prompt, {"tools": tools, "system": system})
        # Default: if tools given, call the first one with empty args
        if tools:
            first = tools[0]
            name = first.get("function", {}).get("name") or first.get("name", "unknown")
            return LLMResponse(
                tool_calls=[ToolCall(name=name, arguments={})],
                finish_reason="tool_calls",
            )
        return LLMResponse(text='{"mock": true}', finish_reason="stop")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _extract_json(text: str) -> str:
    """Strip optional markdown code fences. LLMs often wrap JSON in them."""
    text = text.strip()
    m = _FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    return text


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_client(
    provider: str = "openai",
    model: Optional[str] = None,
    cache_dir: Path | str = _DEFAULT_CACHE_DIR,
) -> LLMClient:
    if provider == "openai":
        return OpenAIClient(model=model, cache_dir=cache_dir)
    if provider == "mock":
        return MockLLMClient(cache_dir=cache_dir)
    raise ValueError(f"Unknown provider: {provider!r}. Use 'openai' or 'mock'.")
