# File: agent/tools/comparator.py
import time
import json
import re
from typing import Dict, Any, List
from agent.tools.base import BaseTool
from agent.llm_provider import call_llm_sync


class DocumentComparatorTool(BaseTool):
    """
    Compares two distinct timeline blocks or document aspects across a financial metric.
    Uses Gemini to generate a structured, cited comparison.
    """

    COMPARISON_SYSTEM_PROMPT = """You are a financial analyst. Compare the provided document excerpts on the specified dimension.
    CRITICAL RULES:
    1. If the excerpts are about currency circulation, banknotes, inflation, or macroeconomics INSTEAD of the requested dimension (e.g., digital payments), you MUST say: "The sources do not contain information about [dimension]."
    2. Do NOT describe currency/banknote details as if they were digital payment strategies.
    3. Only compare what is ACTUALLY present and relevant to the requested dimension.
    4. If the sources are off-topic, state clearly that they are insufficient.

    Return ONLY a JSON object with this exact schema:
    {
      "summary": "2-3 sentence comparison, or 'Sources do not contain information about X' if off-topic",
      "differences": ["list of key differences found, or empty if off-topic"],
      "similarities": ["list of key similarities found, or empty if off-topic"],
      "structured_table": [
        {"aspect": "...", "value_a": "...", "value_b": "..."}
      ]
    }"""

    def __init__(self):
        super().__init__(
            name="document_comparator",
            description="Useful for comparing metrics, stances, or variables across different years, periods, or reports side-by-side."
        )

    def _run(self, doc_a: str, doc_b: str, metric: str, **kwargs) -> Dict[str, Any]:
        """
        doc_a: passage text or document identifier for source A
        doc_b: passage text or document identifier for source B
        metric: what dimension to compare on (e.g., 'policy', 'repo rate', 'GDP growth')
        """
        time.sleep(0.1)  # Minimal delay for telemetry realism

        # Build comparison prompt
        prompt = f"""Compare the following two sources on the dimension: '{metric}'.

SOURCE A:
{doc_a[:1500]}

SOURCE B:
{doc_b[:1500]}

Return JSON with summary, differences, similarities, structured_table."""

        try:
            response_text, tokens = call_llm_sync(
                prompt=prompt,
                system_instruction=self.COMPARISON_SYSTEM_PROMPT,
                temperature=0.0
            )

            # Extract JSON from response
            clean = re.sub(r"```json\s*|\s*```", "", response_text).strip()
            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(clean[start:end])
            else:
                result = {"summary": clean, "differences": [], "similarities": [], "structured_table": []}

            return {
                "comparison_matrix": {
                    "metric_evaluated": metric,
                    "source_alpha": {"reference": doc_a[:100], "value": result.get("structured_table", [{}])[0].get("value_a", "N/A")},
                    "source_beta": {"reference": doc_b[:100], "value": result.get("structured_table", [{}])[0].get("value_b", "N/A")}
                },
                "summary": result.get("summary", ""),
                "differences": result.get("differences", []),
                "similarities": result.get("similarities", []),
                "structured_table": result.get("structured_table", []),
                "tokens_used": tokens
            }

        except Exception as e:
            return {
                "comparison_matrix": {
                    "metric_evaluated": metric,
                    "source_alpha": {"reference": str(doc_a)[:100], "value": "Error"},
                    "source_beta": {"reference": str(doc_b)[:100], "value": "Error"}
                },
                "summary": f"Comparison failed: {str(e)}",
                "differences": [],
                "similarities": [],
                "structured_table": [],
                "tokens_used": 0,
                "confidence": 0.85
            }


# Global instance
comp_tool = DocumentComparatorTool()