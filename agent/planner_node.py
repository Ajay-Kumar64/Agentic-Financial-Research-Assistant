import json
import re
from agent.state import AgentState
from agent.llm_provider import call_llm_with_telemetry

PLANNER_SYSTEM_PROMPT = """You are the master planning node of an Agentic Financial Research Assistant.
Your single job is to analyze the user's financial query, check what data has been collected so far, and decide on the exact next step.

Available Tools:
1. `rag_search`: Queries internal banking docs, policy mandates, and RBI reports. Use this for specific banking regulations, repo rates, or policy data.
2. `web_search`: Searches the live web for real-time macroeconomic context, external data, or market updates.
3. `financial_calculator`: Computes complex equations (e.g., CAGR, inflation adjusted assets).
4. `final_answer`: Closes the execution look and compiles the final analytical response when all information is gathered.

You MUST respond in strict JSON format matching this schema:
{
  "current_plan": "A concise explanation of the steps required to fulfill the user request.",
  "next_step": "Must be one of: 'rag_search', 'web_search', 'financial_calculator', or 'final_answer'",
  "tool_input": "The exact specific parameter/query string to pass to the tool."
}
Ensure no markdown code fence wrappers are added inside the raw response. Return only valid JSON."""


def planner_node(state: AgentState) -> dict:
    """
    LangGraph execution node responsible for checking remaining budget metrics,
    evaluating collected facts, and directing next orchestration branches.
    """
    # Guardrail Check: Loop Detection & Token Depletion
    if state["tool_call_depth"] >= 5 or state["tokens_consumed"] >= 45000:
        return {
            "next_step": "final_answer",
            "is_budget_exhausted": True,
            "errors_encountered": state["errors_encountered"] + ["Execution capped by guardrail limitations."]
        }

    # Construct context summary from execution history
    context_history = f"User Query: {state['current_query']}\n"
    context_history += f"Current Tool Depth: {state['tool_call_depth']}\n"
    context_history += f"Context Gathered So Far:\n"

    for ctx in state.get("retrieved_contexts", []):
        context_history += f"- [Retrieved Context]: {ctx}\n"
    for calc in state.get("calculation_results", []):
        context_history += f"- [Calculation Result]: {calc}\n"

    # Call out to the core planner model
    try:
        response_text, tokens = call_llm_with_telemetry(
            prompt=context_history,
            system_instruction=PLANNER_SYSTEM_PROMPT
        )

        # Sanitize JSON string against accidental markdown additions
        clean_json_str = re.sub(r"```json\s*|\s*```", "", response_text).strip()
        parsed_plan = json.loads(clean_json_str)

        return {
            "current_plan": parsed_plan.get("current_plan", ""),
            "next_step": parsed_plan.get("next_step", "final_answer"),
            "current_query": parsed_plan.get("tool_input", state["current_query"]),  # update sub-task target
            "tokens_consumed": state["tokens_consumed"] + tokens,
            "tool_call_depth": state["tool_call_depth"] + 1
        }

    except Exception as e:
        # Fallback branch to gracefully route out if JSON or API crashes
        return {
            "next_step": "final_answer",
            "errors_encountered": state["errors_encountered"] + [f"Planner failure: {str(e)}"]
        }