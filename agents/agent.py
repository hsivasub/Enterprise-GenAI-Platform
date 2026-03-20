"""
Agent Orchestrator — Agent Layer
==================================
The main entry point for running the financial AI agent.
Wires together: LLM client → graph → tools → response

Also implements:
- Redis-based response caching (skip LLM for repeated queries)  
- Conversation session management
- Structured response formatting
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.graph import AgentState, create_agent_graph
from agents.llm_client import LLMClientBase, get_llm_client
from agents.tools import build_tool_registry
from config.settings import settings
from observability.logger import get_logger
from observability.tracer import create_tracer

logger = get_logger(__name__)
tracer = create_tracer("agent")


@dataclass
class AgentResponse:
    """Structured response from the agent."""
    answer: str
    question: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    iterations: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    error: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "question": self.question,
            "tool_calls": self.tool_calls,
            "iterations": self.iterations,
            "latency_ms": round(self.latency_ms, 2),
            "cached": self.cached,
            "error": self.error,
            "session_id": self.session_id,
        }


class AgentOrchestrator:
    """
    Orchestrates the full agent interaction loop.
    
    Key responsibilities:
    1. Session management (conversation history per user)
    2. Cache lookup (Redis) before expensive LLM calls
    3. Graph execution (LangGraph ReAct loop)
    4. Response formatting and error handling
    """

    def __init__(
        self,
        retriever_engine: Any,
        llm_client: Optional[LLMClientBase] = None,
        redis_client: Optional[Any] = None,
    ) -> None:
        self._llm = llm_client or get_llm_client()
        self._tools = build_tool_registry(retriever_engine)
        self._graph = create_agent_graph(self._llm, self._tools)
        self._redis = redis_client
        self._sessions: Dict[str, List[Dict]] = {}

    def run(
        self,
        question: str,
        session_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> AgentResponse:
        """
        Run the agent for a given question.
        
        Args:
            question: Natural language financial question
            session_id: Optional session ID for multi-turn conversations
            use_cache: Whether to check Redis cache first
        """
        start = time.perf_counter()

        with tracer.start_span("agent_run", attributes={"question_len": len(question)}) as span:
            # 1. Cache lookup
            if use_cache and settings.ENABLE_RESPONSE_CACHE:
                cached = self._get_cached_response(question)
                if cached:
                    logger.info("Cache hit for question", extra={"question": question[:100]})
                    return AgentResponse(
                        answer=cached,
                        question=question,
                        latency_ms=(time.perf_counter() - start) * 1000,
                        cached=True,
                        session_id=session_id,
                    )

            # 2. Build initial state
            history = self._sessions.get(session_id, []) if session_id else []
            initial_state: AgentState = {
                "question": question,
                "messages": history,
                "tool_calls": [],
                "observations": [],
                "final_answer": None,
                "iteration": 0,
                "error": None,
            }

            # 3. Execute agent graph
            try:
                final_state = self._graph.invoke(initial_state)
            except Exception as exc:
                logger.error(f"Agent graph execution failed: {exc}", exc_info=True)
                return AgentResponse(
                    answer="I encountered an error processing your question. Please try again.",
                    question=question,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    error=str(exc),
                    session_id=session_id,
                )

            answer = final_state.get("final_answer") or "Unable to generate a response."
            elapsed_ms = (time.perf_counter() - start) * 1000
            span.set_attribute("iterations", final_state.get("iteration", 0))

            # 4. Cache successful response
            if settings.ENABLE_RESPONSE_CACHE and not final_state.get("error"):
                self._cache_response(question, answer)

            # 5. Update session history
            if session_id:
                self._sessions[session_id] = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]

            return AgentResponse(
                answer=answer,
                question=question,
                tool_calls=final_state.get("tool_calls", []),
                observations=final_state.get("observations", []),
                iterations=final_state.get("iteration", 0),
                latency_ms=elapsed_ms,
                cached=False,
                error=final_state.get("error"),
                session_id=session_id,
            )

    # ── Cache helpers ──────────────────────────────────────────

    def _cache_key(self, question: str) -> str:
        h = hashlib.sha256(question.lower().strip().encode()).hexdigest()[:16]
        return f"genai:response:{h}"

    def _get_cached_response(self, question: str) -> Optional[str]:
        if not self._redis:
            return None
        try:
            key = self._cache_key(question)
            value = self._redis.get(key)
            return value.decode("utf-8") if value else None
        except Exception as exc:
            logger.debug(f"Redis cache get failed: {exc}")
            return None

    def _cache_response(self, question: str, answer: str) -> None:
        if not self._redis:
            return
        try:
            key = self._cache_key(question)
            self._redis.setex(key, settings.REDIS_TTL_SECONDS, answer.encode("utf-8"))
        except Exception as exc:
            logger.debug(f"Redis cache set failed: {exc}")
