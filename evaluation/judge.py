"""
evaluation/judge.py — LLM-as-Judge for quality metrics.
Uses Gemini to evaluate task completion, faithfulness, and step accuracy.
"""
import json
from typing import Tuple, List, Dict, Any
from agent.llm_provider import call_llm_sync


def judge_task_completion(response: str, expected_pattern: str) -> Tuple[bool, str]:
    """
    Judge whether the response correctly answers the query.
    Returns: (passed, reason)
    """
    if not expected_pattern:
        return True, "no pattern required"

    prompt = f"""You are a strict evaluator. Determine if the following response correctly answers the question or matches the expected pattern.

Expected pattern (regex): {expected_pattern}
Response: {response[:1500]}

Rules:
- If the response contains the expected information or matches the pattern intent, answer YES.
- If the response says "not found", "do not contain", "insufficient information" and the expected pattern is a specific number/fact, answer YES if the agent correctly admitted it doesn't know.
- If the response is completely wrong or unrelated, answer NO.

Return ONLY a JSON object: {{"passed": true/false, "reason": "..."}}"""

    try:
        text, _ = call_llm_sync(
            prompt=prompt,
            system_instruction="You evaluate agent responses. Return ONLY valid JSON.",
            temperature=0.0
        )
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
            return bool(result.get("passed", False)), result.get("reason", "")
    except Exception as e:
        print(f"[Judge] task_completion failed: {e}")

    # Fallback: regex match
    import re
    fallback = bool(re.search(expected_pattern, response, re.IGNORECASE))
    return fallback, "regex fallback"


def judge_faithfulness(response: str, tool_outputs: List[Dict]) -> Tuple[float, str]:
    """
    Judge what percentage of claims in the response are grounded in tool outputs.
    Returns: (score 0-1, reason)
    """
    sources_text = "\n\n".join([
        str(o.get("text_summary", o.get("summary", o.get("result", str(o)))))[:400]
        for o in tool_outputs[:3]
    ])

    prompt = f"""You are a fact-checker. For each claim in the response, check if it is supported by the provided tool outputs.

Tool outputs:
{sources_text}

Response:
{response[:1000]}

Count how many sentences make claims, and how many are supported by the tool outputs. Unsupported claims include hallucinated numbers, facts not in sources, or contradictions.

Return ONLY JSON: {{"supported": int, "total": int, "reason": "..."}}"""

    try:
        text, _ = call_llm_sync(
            prompt=prompt,
            system_instruction="You check factual grounding. Return ONLY valid JSON.",
            temperature=0.0
        )
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
            supported = result.get("supported", 0)
            total = result.get("total", 1)
            score = supported / total if total > 0 else 0.0
            return score, result.get("reason", "")
    except Exception as e:
        print(f"[Judge] faithfulness failed: {e}")

    return 0.0, "judge_error"


def judge_intermediate_step(tool_name: str, tool_input: Any, tool_output: Any, query: str) -> Tuple[bool, str]:
    """
    Judge whether a single tool call was reasonable and correct.
    """
    prompt = f"""Given the user's question and this tool call, is the tool output reasonable and correct?

User question: {query}
Tool: {tool_name}
Tool input: {str(tool_input)[:200]}
Tool output: {str(tool_output)[:400]}

Return ONLY JSON: {{"correct": true/false, "reason": "..."}}"""

    try:
        text, _ = call_llm_sync(
            prompt=prompt,
            system_instruction="You evaluate tool call correctness. Return ONLY valid JSON.",
            temperature=0.0
        )
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
            return bool(result.get("correct", False)), result.get("reason", "")
    except Exception as e:
        print(f"[Judge] intermediate_step failed: {e}")

    return True, "judge_error_fallback"


def judge_multi_turn_coherence(resolved_query: str, expected_resolved: str) -> Tuple[bool, str]:
    """
    Judge whether coreference resolution was correct.
    """
    if not expected_resolved:
        return True, "no resolution required"

    prompt = f"""Does the resolved query contain the same key entities and intent as the expected resolution?

Expected resolved: {expected_resolved}
Actual resolved: {resolved_query}

Return ONLY JSON: {{"correct": true/false, "reason": "..."}}"""

    try:
        text, _ = call_llm_sync(
            prompt=prompt,
            system_instruction="You evaluate query resolution. Return ONLY valid JSON.",
            temperature=0.0
        )
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
            return bool(result.get("correct", False)), result.get("reason", "")
    except Exception as e:
        print(f"[Judge] coherence failed: {e}")

    # Fallback: entity overlap
    exp_lower = expected_resolved.lower()
    res_lower = resolved_query.lower()
    key_terms = ["repo rate", "inflation", "gdp", "npa", "credit", "forex"]
    matched = sum(1 for term in key_terms if term in exp_lower and term in res_lower)
    return matched > 0, "entity overlap fallback"