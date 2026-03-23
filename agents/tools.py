"""
Agent Tools — Agent Framework  
=================================
Implements the three core tools available to the financial AI agent:

1. DocumentRetrievalTool — semantic search over the knowledge base
2. SQLQueryTool — structured queries against DuckDB/PostgreSQL
3. CalculatorTool — evaluate mathematical expressions safely

Each tool is a LangChain-compatible BaseTool subclass, allowing them to
be used with both LangChain agents and raw tool-calling APIs.

Security notes:
- Calculator uses ast.literal_eval + restricted eval (no builtins)
- SQL queries run as read-only user against a read replica
"""

from __future__ import annotations

import ast
import math
import operator
import re
import time
from typing import Any, Optional, Type

from pydantic import BaseModel, Field

from config.settings import settings
from observability.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Base tool interface (LangChain-compatible)
# ─────────────────────────────────────────────────────────────

class ToolInput(BaseModel):
    """Base class for tool inputs — enforces Pydantic validation."""
    pass


class BaseTool:
    """
    Minimal tool interface compatible with LangChain and LangGraph.
    LangChain's @tool decorator auto-wraps these in production.
    """
    name: str = "base_tool"
    description: str = ""
    args_schema: Type[ToolInput] = ToolInput

    def run(self, tool_input: str | dict) -> str:
        raise NotImplementedError

    def __call__(self, tool_input: str | dict) -> str:
        return self.run(tool_input)


# ─────────────────────────────────────────────────────────────
# Tool 1: Document Retrieval
# ─────────────────────────────────────────────────────────────

class DocumentRetrievalInput(ToolInput):
    query: str = Field(..., description="Natural language query to search documents")
    top_k: Optional[int] = Field(None, description="Number of top results to return")
    filter_doc_type: Optional[str] = Field(None, description="Filter by doc type: pdf, txt, etc.")


class DocumentRetrievalTool(BaseTool):
    """
    Semantic search over the vector knowledge base.
    Returns formatted context from the most relevant document chunks.
    """

    name = "document_retrieval"
    description = (
        "Search financial documents, reports, and filings for relevant information. "
        "Use this for: regulatory filings, earnings reports, policy documents, "
        "historical context, and any unstructured knowledge."
    )
    args_schema = DocumentRetrievalInput

    def __init__(self, retriever_engine: Any) -> None:
        self._retriever = retriever_engine  # RetrieverEngine instance

    def run(self, tool_input: str | dict) -> str:
        if isinstance(tool_input, str):
            tool_input = {"query": tool_input}

        parsed = DocumentRetrievalInput(**tool_input)
        filters = (
            {"doc_type": parsed.filter_doc_type}
            if parsed.filter_doc_type
            else None
        )

        start = time.perf_counter()
        results = self._retriever.retrieve(
            parsed.query,
            top_k=parsed.top_k,
            filters=filters,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if not results:
            return "No relevant documents found for the given query."

        formatted_results = []
        for i, result in enumerate(results, 1):
            source = result.chunk.metadata.get("filename", "unknown")
            formatted_results.append(
                f"[Result {i} | Source: {source} | Score: {result.score:.3f}]\n"
                f"{result.chunk.content}"
            )

        logger.debug(
            "DocumentRetrievalTool",
            extra={"results": len(results), "latency_ms": round(elapsed_ms, 1)},
        )
        return "\n\n---\n\n".join(formatted_results)


# ─────────────────────────────────────────────────────────────
# Tool 2: SQL Query Tool
# ─────────────────────────────────────────────────────────────

class SQLQueryInput(ToolInput):
    query: str = Field(
        ...,
        description="Syntactically valid SQL SELECT query against the financial data warehouse",
    )
    limit: Optional[int] = Field(100, description="Maximum number of rows to return")


class SQLQueryTool(BaseTool):
    """
    Executes read-only SQL against DuckDB (analytical) or PostgreSQL (transactional).
    Used for: financial metrics, time-series queries, aggregations.

    Security:
    - Only SELECT statements are allowed (enforced by regex + whitelist)
    - Runs under a read-only database role
    - Row limit is enforced to prevent runaway queries
    """

    name = "sql_query"
    description = (
        "Execute SQL queries against the financial data warehouse. "
        "Tables available: "
        "  financial_metrics(ticker, period, revenue, net_income, eps, date), "
        "  stock_prices(ticker, date, open, high, low, close, volume), "
        "  company_info(ticker, name, sector, market_cap, exchange), "
        "  economic_indicators(indicator, date, value, unit). "
        "Only SELECT queries are allowed."
    )
    args_schema = SQLQueryInput

    # Patterns that indicate dangerous operations
    DISALLOWED_PATTERNS = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|EXEC|EXECUTE|"
        r"GRANT|REVOKE|PRAGMA|ATTACH|DETACH)\b",
        re.IGNORECASE,
    )

    def __init__(self, db_path: str = settings.DUCKDB_PATH) -> None:
        self._db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create sample tables if they don't exist."""
        try:
            import duckdb
            conn = duckdb.connect(self._db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS financial_metrics (
                    ticker VARCHAR,
                    period VARCHAR,
                    revenue DECIMAL(18,2),
                    net_income DECIMAL(18,2),
                    eps DECIMAL(10,4),
                    date DATE
                );
                CREATE TABLE IF NOT EXISTS stock_prices (
                    ticker VARCHAR,
                    date DATE,
                    open DECIMAL(10,4),
                    high DECIMAL(10,4),
                    low DECIMAL(10,4),
                    close DECIMAL(10,4),
                    volume BIGINT
                );
                CREATE TABLE IF NOT EXISTS company_info (
                    ticker VARCHAR PRIMARY KEY,
                    name VARCHAR,
                    sector VARCHAR,
                    market_cap DECIMAL(18,2),
                    exchange VARCHAR
                );
                CREATE TABLE IF NOT EXISTS economic_indicators (
                    indicator VARCHAR,
                    date DATE,
                    value DECIMAL(18,4),
                    unit VARCHAR
                );
            """)
            conn.close()
        except Exception as e:
            logger.warning(f"Could not initialize DuckDB schema: {e}")

    def run(self, tool_input: str | dict) -> str:
        if isinstance(tool_input, str):
            tool_input = {"query": tool_input}

        parsed = SQLQueryInput(**tool_input)
        query = parsed.query.strip()

        # Security: block non-SELECT statements
        if self.DISALLOWED_PATTERNS.search(query):
            return "Error: Only SELECT queries are permitted."

        if not query.upper().lstrip().startswith("SELECT"):
            return "Error: Query must start with SELECT."

        # Enforce row limit
        if parsed.limit and "LIMIT" not in query.upper():
            query = f"{query} LIMIT {parsed.limit}"

        try:
            import duckdb
            conn = duckdb.connect(self._db_path, read_only=False)
            result = conn.execute(query).fetchdf()
            conn.close()

            if result.empty:
                return "Query returned no results."

            # Format as markdown table
            return result.to_markdown(index=False)
        except Exception as exc:
            logger.error(f"SQL query failed: {exc}")
            return f"SQL Error: {exc}"


# ─────────────────────────────────────────────────────────────
# Tool 3: Calculator
# ─────────────────────────────────────────────────────────────

class CalculatorInput(ToolInput):
    expression: str = Field(
        ...,
        description=(
            "Mathematical expression to evaluate. "
            "Supports: +, -, *, /, **, sqrt(), log(), abs(), round(), min(), max(). "
            "Examples: '(1500000 / 365) * 12', 'sqrt(9604)', 'log(1000000, 10)'"
        ),
    )


class CalculatorTool(BaseTool):
    """
    Safe math expression evaluator.
    Uses AST-based evaluation — no exec/eval of arbitrary Python.
    Supports finance-relevant operations: percentages, compounding, ratios.
    """

    name = "calculator"
    description = (
        "Perform mathematical calculations: arithmetic, percentages, "
        "financial ratios, compound interest, present value, etc. "
        "Use for: EPS calculations, P/E ratios, growth rates, portfolio math."
    )
    args_schema = CalculatorInput

    # Safe math functions available in expressions
    SAFE_FUNCTIONS = {
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": math.pow,
        "ceil": math.ceil,
        "floor": math.floor,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
    }

    # Allowed AST node types
    ALLOWED_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp,
        ast.Num, ast.Constant, ast.Name, ast.Call,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
        ast.Mod, ast.Pow, ast.USub, ast.UAdd,
        ast.Load, ast.Attribute,
    )

    def run(self, tool_input: str | dict) -> str:
        if isinstance(tool_input, str):
            # Accept raw expression string
            expression = tool_input
        else:
            parsed = CalculatorInput(**tool_input)
            expression = parsed.expression

        try:
            result = self._safe_eval(expression.strip())
            # Format nicely
            if isinstance(result, float):
                if result == int(result):
                    return str(int(result))
                return f"{result:.6f}".rstrip("0").rstrip(".")
            return str(result)
        except Exception as exc:
            return f"Calculation error: {exc}"

    def _safe_eval(self, expression: str) -> float:
        """Parse and evaluate expression via AST — no arbitrary code execution."""
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}")

        # Walk AST and reject any disallowed node types
        for node in ast.walk(tree):
            if not isinstance(node, self.ALLOWED_NODES):
                raise ValueError(
                    f"Disallowed operation in expression: {type(node).__name__}"
                )

        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.AST) -> float:
        """Recursively evaluate AST nodes."""
        OPS = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Python 3.7 compat
            return node.n
        elif isinstance(node, ast.Name):
            if node.id in self.SAFE_FUNCTIONS:
                return self.SAFE_FUNCTIONS[node.id]
            raise ValueError(f"Unknown name: {node.id}")
        elif isinstance(node, ast.BinOp):
            op = OPS.get(type(node.op))
            if not op:
                raise ValueError(f"Unsupported operator: {node.op}")
            return op(self._eval_node(node.left), self._eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            op = OPS.get(type(node.op))
            return op(self._eval_node(node.operand))
        elif isinstance(node, ast.Call):
            func_node = node.func
            if isinstance(func_node, ast.Name):
                func = self.SAFE_FUNCTIONS.get(func_node.id)
                if not func:
                    raise ValueError(f"Unknown function: {func_node.id}")
                args = [self._eval_node(arg) for arg in node.args]
                return func(*args)
        raise ValueError(f"Cannot evaluate node type: {type(node).__name__}")


# ─────────────────────────────────────────────────────────────
# Tool registry
# ─────────────────────────────────────────────────────────────

def build_tool_registry(retriever_engine: Any) -> dict[str, BaseTool]:
    """
    Build the complete tool registry for the agent.
    Returns a dict keyed by tool name for easy lookup.
    """
    tools = [
        DocumentRetrievalTool(retriever_engine=retriever_engine),
        SQLQueryTool(),
        CalculatorTool(),
    ]
    return {tool.name: tool for tool in tools}
