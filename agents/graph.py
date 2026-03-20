"""
LangGraph Agent Workflow
========================
Implements a multi-step ReAct (Reasoning + Acting) agent using LangGraph.
Graph structure:

    START → plan → execute_tool → observe → [continue | finish] → END

This is the heart of the multi-step reasoning system. The agent:
1. Receives a financial question
2. Plans which tool to use
3. Executes the tool
4. Observes the result
5. Decides: need more info (loop) or ready to answer (finish)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, TypedDict

from config.settings import settings
from observability.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """The state object passed between graph nodes."""
    question: str                    # Original user question
    messages: List[Dict[str, Any]]   # Full conversation history (OpenAI format)
    tool_calls: List[Dict[str, Any]] # History of tool invocations
    observations: List[str]          # Tool outputs
    final_answer: Optional[str]      # Set when agent is done
    iteration: int                   # Loop counter for max_iterations guard
    error: Optional[str]             # Error state


# ─────────────────────────────────────────────────────────────
# Node functions
# ─────────────────────────────────────────────────────────────

def create_agent_graph(llm_client: Any, tools: Dict[str, Any]):
    """
    Factory that builds and returns the compiled LangGraph StateGraph.
    Takes an LLM client and tool registry.
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        raise ImportError("langgraph not installed: pip install langgraph")

    def agent_node(state: AgentState) -> AgentState:
        """
        Core reasoning node: sends the current state to the LLM and
        asks it to decide the next action (tool call or final answer).
        """
        from prompts.templates import prompt_registry

        if state["iteration"] >= settings.AGENT_MAX_ITERATIONS:
            logger.warning(
                f"Agent hit max iterations ({settings.AGENT_MAX_ITERATIONS})"
            )
            return {
                **state,
                "final_answer": (
                    "I've reached the maximum reasoning steps. "
                    "Based on information gathered so far: "
                    + "\n".join(state["observations"][-3:])
                ),
            }

        # Build tool descriptions for the system prompt
        tool_desc = "\n".join(
            f"- {name}: {tool.description}"
            for name, tool in tools.items()
        )
        system_prompt = prompt_registry.get_active("system")

        # Build conversation: system + history + current observations
        messages = [{"role": "system", "content": system_prompt}]

        # Add question
        user_content = f"Question: {state['question']}"
        if state["observations"]:
            obs_text = "\n".join(
                f"Tool result {i+1}: {obs}"
                for i, obs in enumerate(state["observations"])
            )
            user_content += f"\n\nPrevious tool results:\n{obs_text}"
            user_content += "\n\nBased on these results, either use another tool or provide your final answer."

        messages.append({"role": "user", "content": user_content})

        # Build tool schemas for function calling
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": _build_tool_schema(tool),
                        "required": _get_required_fields(tool),
                    },
                },
            }
            for tool in tools.values()
        ]

        try:
            response = llm_client.chat(
                messages=messages,
                tools=tool_schemas,
            )

            if response.tool_calls:
                # Agent wants to use a tool
                tool_call = response.tool_calls[0]
                new_tool_calls = state["tool_calls"] + [{
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }]
                return {
                    **state,
                    "messages": messages + [{"role": "assistant", "content": str(response.content)}],
                    "tool_calls": new_tool_calls,
                    "iteration": state["iteration"] + 1,
                }
            else:
                # Agent has a final answer
                return {
                    **state,
                    "final_answer": response.content,
                    "iteration": state["iteration"] + 1,
                }

        except Exception as exc:
            logger.error(f"Agent LLM call failed: {exc}", exc_info=True)
            return {**state, "error": str(exc)}

    def tool_executor_node(state: AgentState) -> AgentState:
        """Execute the last requested tool call and record the observation."""
        if not state["tool_calls"]:
            return state

        last_call = state["tool_calls"][-1]
        tool_name = last_call["name"]
        tool_args = last_call["arguments"]

        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                tool_args = {"query": tool_args}

        tool = tools.get(tool_name)
        if not tool:
            observation = f"Error: Unknown tool '{tool_name}'"
        else:
            try:
                logger.info(
                    f"Executing tool: {tool_name}",
                    extra={"args": str(tool_args)[:200]},
                )
                observation = tool.run(tool_args)
            except Exception as exc:
                observation = f"Tool error: {exc}"
                logger.error(f"Tool '{tool_name}' failed: {exc}")

        new_observations = state["observations"] + [
            f"[{tool_name}]: {observation}"
        ]
        return {**state, "observations": new_observations}

    def should_continue(state: AgentState) -> Literal["continue", "end"]:
        """Router: decide whether to loop or terminate."""
        if state.get("final_answer"):
            return "end"
        if state.get("error"):
            return "end"
        if state["iteration"] >= settings.AGENT_MAX_ITERATIONS:
            return "end"
        if state["tool_calls"]:
            return "continue"
        return "end"

    # ── Build the graph ─────────────────────────────────────────
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tool_executor", tool_executor_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tool_executor",
            "end": END,
        },
    )
    workflow.add_edge("tool_executor", "agent")

    return workflow.compile()


# ─────────────────────────────────────────────────────────────
# Schema helpers for OpenAI function calling
# ─────────────────────────────────────────────────────────────

def _build_tool_schema(tool: Any) -> Dict[str, Any]:
    """Extract parameter schemas from tool's args_schema."""
    try:
        schema = tool.args_schema.model_json_schema()
        return schema.get("properties", {})
    except Exception:
        return {"input": {"type": "string", "description": "Tool input"}}


def _get_required_fields(tool: Any) -> List[str]:
    try:
        schema = tool.args_schema.model_json_schema()
        return schema.get("required", [])
    except Exception:
        return []
