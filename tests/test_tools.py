# tests/test_tools.py
import pytest
from agent.tools.calculator import calc_tool


def test_calculator_growth_rate():
    result = calc_tool.run("growth_rate(100, 150)")
    assert result["success"] is True
    assert result["result"] == 50.0


def test_calculator_cagr():
    result = calc_tool.run("cagr(1000, 1500, 3)")
    assert result["success"] is True
    assert round(result["result"], 2) == 14.47


def test_calculator_ratio():
    result = calc_tool.run("ratio(75, 25)")
    assert result["success"] is True
    assert result["result"] == 3.0


def test_calculator_percentage():
    result = calc_tool.run("percentage(25, 100)")
    assert result["success"] is True
    assert result["result"] == 25.0


def test_calculator_arithmetic():
    result = calc_tool.run("((6.5 - 4.0) / 4.0) * 100")
    assert result["success"] is True
    assert result["result"] == 62.5


def test_calculator_invalid_expression():
    result = calc_tool.run("import os")
    assert result["success"] is False
    assert result["result"] is None
    assert "error" in result


def test_calculator_cagr_natural_language():
    """Test that the extractor will eventually parse this (tested in graph.py)."""
    result = calc_tool.run("cagr(1000, 1500, 3)")
    assert result["success"]
    assert result["result"] > 0