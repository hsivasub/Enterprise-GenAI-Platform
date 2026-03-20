"""
Prompt Templates — Prompt Management Layer
==========================================
Version-controlled prompt templates for the financial AI assistant.
All templates are Pydantic BaseModel instances for schema validation.

Design principles:
- Templates are versioned (v1, v2, ...) for A/B testing via MLflow
- Jinja2-style variable injection with strict schema
- Prompt registry allows hot-swapping without code changes
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel

from observability.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# System Prompts
# ─────────────────────────────────────────────────────────────

FINANCIAL_ASSISTANT_SYSTEM_V1 = """You are a professional AI Financial Assistant for an enterprise financial services firm. 
You have access to:
1. A knowledge base of financial documents, regulatory filings, and earnings reports
2. A SQL database with real-time financial metrics and historical stock data
3. A mathematical calculator for quantitative analysis

Your responsibilities:
- Provide accurate, well-reasoned financial analysis
- Always cite your sources (document name, section, or data table)
- Clearly distinguish between retrieved facts and your reasoning
- Flag uncertainty when data is incomplete or ambiguous
- Never provide investment advice — provide analysis only
- Maintain regulatory compliance (SEC reporting standards)

Response format:
- Use structured markdown with headers when answering complex questions
- Include relevant numbers, dates, and source citations
- End complex answers with a "Key Takeaways" summary

IMPORTANT: If you cannot find the answer in the provided context, explicitly state that the information is not available in the current knowledge base.
"""

FINANCIAL_ASSISTANT_SYSTEM_V2 = """You are ARIA (Autonomous Research Intelligence Assistant), an enterprise-grade AI Financial Analyst.

Core capabilities you leverage:
- Document analysis: SEC filings, annual reports, earnings calls, research papers
- Quantitative modeling: financial ratios, trend analysis, scenario modeling
- Data synthesis: cross-reference multiple sources for comprehensive insights

Behavioral guidelines:
1. ACCURACY: Every numerical claim must cite a source
2. UNCERTAINTY: Use confidence levels (High/Medium/Low) for derived insights
3. COMPLIANCE: Never speculate about stock price movements or give investment advice
4. COMPLETENESS: If context is insufficient, say so explicitly and suggest what additional data would help
5. STRUCTURE: Use headers, bullet points, and tables for complex responses

Available tools: document_retrieval, sql_query, calculator
"""

# ─────────────────────────────────────────────────────────────
# RAG Prompt Templates
# ─────────────────────────────────────────────────────────────

RAG_TEMPLATE_V1 = """Answer the following question using ONLY the provided context. 
If the answer cannot be found in the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {question}

Instructions:
- Base your answer strictly on the context above
- Cite specific sources when possible
- Be concise but complete

Answer:"""


RAG_TEMPLATE_V2 = """You are analyzing financial documents. Use the retrieved context to answer the question.

Retrieved Context:
{context}

User Question: {question}

Reasoning Process:
1. Identify relevant information in the context
2. Extract key facts, numbers, and dates
3. Synthesize a comprehensive answer

If information is missing, state: "The provided documents do not contain sufficient information about [specific aspect]."

Financial Analysis:"""


# ─────────────────────────────────────────────────────────────
# Agent Prompt Templates
# ─────────────────────────────────────────────────────────────

AGENT_REACT_TEMPLATE = """You are a financial analysis agent with access to the following tools:

{tool_descriptions}

Use the following format:
Thought: Think about what information you need and which tool to use
Action: tool_name
Action Input: {{"key": "value"}}
Observation: [tool result]
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to answer
Final Answer: [comprehensive answer with citations]

Important:
- Use document_retrieval for qualitative information from reports and filings
- Use sql_query for quantitative data and metrics  
- Use calculator for mathematical computations
- Always verify numbers with calculator before including them in your response

Question: {question}
{agent_scratchpad}"""


SQL_GENERATION_TEMPLATE = """Generate a SQL SELECT query for the following request.

Available tables:
- financial_metrics(ticker, period, revenue, net_income, eps, date)
- stock_prices(ticker, date, open, high, low, close, volume)  
- company_info(ticker, name, sector, market_cap, exchange)
- economic_indicators(indicator, date, value, unit)

Request: {request}

Rules:
- Only generate SELECT queries
- Always include a LIMIT clause
- Use parameterized values where appropriate
- Add meaningful column aliases

SQL Query:"""


SUMMARIZATION_TEMPLATE = """Summarize the following financial document excerpt in 3-5 bullet points.
Focus on: key financial metrics, strategic decisions, risk factors, and guidance.

Document: {document_text}

Summary:"""


# ─────────────────────────────────────────────────────────────
# Prompt Manager
# ─────────────────────────────────────────────────────────────

class PromptVersion(BaseModel):
    version: str
    template: str
    description: str
    is_active: bool = True


class PromptRegistry:
    """
    Manages versioned prompts for A/B testing and rollback.
    In production, back this with a database or config service.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, PromptVersion]] = {
            "system": {
                "v1": PromptVersion(
                    version="v1",
                    template=FINANCIAL_ASSISTANT_SYSTEM_V1,
                    description="Standard financial assistant persona",
                    is_active=True,
                ),
                "v2": PromptVersion(
                    version="v2",
                    template=FINANCIAL_ASSISTANT_SYSTEM_V2,
                    description="ARIA persona with confidence levels",
                    is_active=False,
                ),
            },
            "rag": {
                "v1": PromptVersion(
                    version="v1",
                    template=RAG_TEMPLATE_V1,
                    description="Basic RAG template",
                    is_active=True,
                ),
                "v2": PromptVersion(
                    version="v2",
                    template=RAG_TEMPLATE_V2,
                    description="Reasoning-chain RAG template",
                    is_active=False,
                ),
            },
            "agent": {
                "v1": PromptVersion(
                    version="v1",
                    template=AGENT_REACT_TEMPLATE,
                    description="ReAct agent template",
                    is_active=True,
                ),
            },
        }

    def get_active(self, prompt_type: str) -> str:
        """Return the currently active template for a prompt type."""
        versions = self._registry.get(prompt_type, {})
        active = next((v for v in versions.values() if v.is_active), None)
        if not active:
            raise ValueError(f"No active prompt found for type: {prompt_type}")
        return active.template

    def get_version(self, prompt_type: str, version: str) -> str:
        """Return a specific version (for A/B testing)."""
        pvs = self._registry.get(prompt_type, {})
        pv = pvs.get(version)
        if not pv:
            raise ValueError(f"Prompt '{prompt_type}:{version}' not found")
        return pv.template

    def format_rag_prompt(self, context: str, question: str, version: str = "v1") -> str:
        template = self.get_version("rag", version)
        return template.format(context=context, question=question)

    def format_agent_prompt(
        self,
        question: str,
        tool_descriptions: str,
        agent_scratchpad: str = "",
    ) -> str:
        template = self.get_active("agent")
        return template.format(
            question=question,
            tool_descriptions=tool_descriptions,
            agent_scratchpad=agent_scratchpad,
        )

    def list_prompts(self) -> Dict[str, List[dict]]:
        return {
            ptype: [
                {"version": pv.version, "description": pv.description, "active": pv.is_active}
                for pv in pvs.values()
            ]
            for ptype, pvs in self._registry.items()
        }


# Singleton
prompt_registry = PromptRegistry()
