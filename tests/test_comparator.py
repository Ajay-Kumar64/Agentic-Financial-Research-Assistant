# tests/test_comparator.py
import pytest
from agent.tools.comparator import comp_tool, DocumentComparatorTool


def test_comparator_returns_dict():
    result = comp_tool.run(
        doc_a="RBI maintained repo rate at 6.5% in FY2023.",
        doc_b="RBI raised repo rate from 4.0% to 6.5% in FY2022.",
        metric="repo rate policy"
    )
    # ToolResult wrapper
    assert hasattr(result, "result_data") or isinstance(result, dict)
    data = result.result_data if hasattr(result, "result_data") else result
    assert "summary" in data
    assert len(data["summary"]) > 0


def test_comparator_structured_fields():
    result = comp_tool.run(
        doc_a="GDP growth projected at 7.2% for FY2023.",
        doc_b="GDP growth projected at 6.8% for FY2022.",
        metric="GDP growth"
    )
    data = result.result_data if hasattr(result, "result_data") else result
    assert "differences" in data or "structured_table" in data or "summary" in data