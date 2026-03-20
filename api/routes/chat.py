"""
Chat Routes — API Gateway
===========================
Handles conversational queries via the agent orchestrator.
Includes: guardrail pipeline, session management, structured responses.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from agents.agent import AgentOrchestrator, AgentResponse
from api.dependencies import get_agent, get_input_validator, get_output_filter, verify_api_key
from config.settings import settings
from guardrails.input_validator import InputValidator, ValidationStatus
from guardrails.output_filter import OutputFilter
from observability.logger import get_logger, set_correlation_id
from observability.metrics import metrics

import time

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


# ─────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=settings.MAX_INPUT_LENGTH,
        description="The financial question to answer",
        examples=["What was Apple's revenue in Q3 2024?"],
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID for multi-turn conversations. Leave empty for new session.",
    )
    use_cache: bool = Field(True, description="Whether to use Redis cache for repeated queries")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Override default retrieval top-k")


class ToolCallInfo(BaseModel):
    name: str
    arguments: str


class ChatResponse(BaseModel):
    answer: str
    question: str
    session_id: str
    tool_calls: list[ToolCallInfo] = []
    iterations: int
    latency_ms: float
    cached: bool
    hallucination_risk: str
    disclaimer_added: bool
    request_id: str


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=ChatResponse,
    summary="Ask a financial question",
    description=(
        "Send a natural language financial question to the AI agent. "
        "The agent may use document retrieval, SQL queries, and calculations "
        "to formulate a comprehensive, cited response."
    ),
)
async def chat(
    request: ChatRequest,
    agent: AgentOrchestrator = Depends(get_agent),
    validator: InputValidator = Depends(get_input_validator),
    output_filter: OutputFilter = Depends(get_output_filter),
    _api_key: str = Depends(verify_api_key),
):
    request_id = str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())
    set_correlation_id(request_id)
    start = time.perf_counter()

    # 1. Input validation + PII detection
    validation = validator.validate(request.question)
    if not validation.is_allowed:
        logger.warning(
            "Request blocked by input validator",
            extra={"violations": validation.violations, "request_id": request_id},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Input validation failed",
                "violations": validation.violations,
            },
        )

    if validation.pii_detected:
        logger.info(
            f"PII redacted from request: {[m.pii_type for m in validation.pii_found]}"
        )

    # 2. Run agent with (potentially redacted) input
    agent_response: AgentResponse = agent.run(
        question=validation.processed_text,
        session_id=session_id,
        use_cache=request.use_cache,
    )

    if agent_response.error and not agent_response.answer:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": agent_response.error},
        )

    # 3. Output filtering
    filter_result = output_filter.filter(
        response=agent_response.answer,
        context="\n".join(agent_response.observations),
    )

    metrics.record_request(latency_ms=(time.perf_counter() - start) * 1000)

    return ChatResponse(
        answer=filter_result.filtered,
        question=request.question,
        session_id=session_id,
        tool_calls=[
            ToolCallInfo(name=tc.get("name", ""), arguments=str(tc.get("arguments", "")))
            for tc in agent_response.tool_calls
        ],
        iterations=agent_response.iterations,
        latency_ms=agent_response.latency_ms,
        cached=agent_response.cached,
        hallucination_risk=filter_result.hallucination_risk,
        disclaimer_added=filter_result.disclaimer_added,
        request_id=request_id,
    )


@router.get(
    "/health",
    summary="Chat service health check",
    include_in_schema=False,
)
async def health():
    return {"status": "ok", "service": "chat"}
