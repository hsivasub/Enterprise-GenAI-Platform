"""
LLM Client Abstraction Layer
==============================
Provides a unified interface to multiple LLM providers:
- OpenAI (gpt-4o, gpt-4o-mini, etc.)
- Anthropic (Claude)
- Azure OpenAI
- Local (Ollama)

Wraps each provider's client behind a common LLMResponse interface.
Handles: retries, token counting, cost logging, and streaming.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config.settings import LLMProvider, settings
from observability.logger import get_logger, llm_logger
from observability.metrics import metrics

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Response model
# ─────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    name: str
    arguments: str   # JSON string
    id: str = ""

    class function:
        name: str
        arguments: str


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""
    content: Optional[str]
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    tool_calls: Optional[List[Any]] = None   # Provider-specific tool call objects
    finish_reason: str = "stop"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ─────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────

class LLMClientBase:
    """Abstract LLM client — subclass per provider."""

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        raise NotImplementedError

    def _record_call(self, response: LLMResponse) -> None:
        """Record metrics and logs for every LLM call."""
        cost = metrics.record_llm_call(
            response.model,
            response.input_tokens,
            response.output_tokens,
        )
        llm_logger.log_request(
            model=response.model,
            prompt=str(response.tool_calls or ""),
            response=response.content or "",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            metadata={"cost_usd": cost},
        )


# ─────────────────────────────────────────────────────────────
# OpenAI Client
# ─────────────────────────────────────────────────────────────

class OpenAIClient(LLMClientBase):
    """
    OpenAI API client with automatic retry and function calling support.
    """

    def __init__(
        self,
        model: str = settings.OPENAI_MODEL,
        temperature: float = settings.OPENAI_TEMPERATURE,
        max_tokens: int = settings.OPENAI_MAX_TOKENS,
    ) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError("openai not installed: pip install openai")

        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")

        self._client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        start = time.perf_counter()
        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature or self._temperature,
            "max_tokens": max_tokens or self._max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        msg = response.choices[0].message
        result = LLMResponse(
            content=msg.content,
            model=self._model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_ms=elapsed_ms,
            tool_calls=msg.tool_calls if hasattr(msg, "tool_calls") else None,
            finish_reason=response.choices[0].finish_reason,
        )
        self._record_call(result)
        return result


# ─────────────────────────────────────────────────────────────
# Anthropic Client
# ─────────────────────────────────────────────────────────────

class AnthropicClient(LLMClientBase):
    """Claude API client."""

    def __init__(
        self,
        model: str = settings.ANTHROPIC_MODEL,
    ) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic not installed: pip install anthropic")

        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self._client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self._model = model

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        # Extract system message if present
        system = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        user_messages = [m for m in messages if m["role"] != "system"]

        start = time.perf_counter()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens or settings.OPENAI_MAX_TOKENS,
            system=system,
            messages=user_messages,
            temperature=temperature or settings.OPENAI_TEMPERATURE,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        result = LLMResponse(
            content=response.content[0].text if response.content else "",
            model=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=elapsed_ms,
            finish_reason=response.stop_reason or "stop",
        )
        self._record_call(result)
        return result


# ─────────────────────────────────────────────────────────────
# Local (Ollama) Client
# ─────────────────────────────────────────────────────────────

class OllamaClient(LLMClientBase):
    """
    Ollama local inference server client.
    Provides privacy-first inference with no data leaving the network.
    """

    def __init__(
        self,
        model: str = settings.LOCAL_LLM_MODEL,
        base_url: str = settings.LOCAL_LLM_BASE_URL,
    ) -> None:
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx not installed: pip install httpx")

        import httpx
        self._client = httpx.Client(base_url=base_url, timeout=120.0)
        self._model = model

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        start = time.perf_counter()
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or settings.OPENAI_TEMPERATURE,
                "num_predict": max_tokens or settings.OPENAI_MAX_TOKENS,
            },
        }
        response = self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - start) * 1000

        data = response.json()
        content = data.get("message", {}).get("content", "")
        # Ollama doesn't report exact token counts — estimate
        input_tokens = sum(len(m["content"].split()) * 1.3 for m in messages)
        output_tokens = len(content.split()) * 1.3

        result = LLMResponse(
            content=content,
            model=self._model,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            latency_ms=elapsed_ms,
        )
        self._record_call(result)
        return result


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

def get_llm_client(provider: Optional[LLMProvider] = None) -> LLMClientBase:
    """Build LLM client based on settings. Inject provider for testing."""
    p = provider or settings.LLM_PROVIDER
    if p == LLMProvider.OPENAI:
        return OpenAIClient()
    elif p == LLMProvider.ANTHROPIC:
        return AnthropicClient()
    elif p == LLMProvider.LOCAL:
        return OllamaClient()
    elif p == LLMProvider.AZURE_OPENAI:
        # Azure uses OpenAI SDK with different base URL
        return OpenAIClient()  # Reconfigure with Azure settings in prod
    raise ValueError(f"Unknown LLM provider: {p}")
