# tests/test_memory.py
from agent.tools.memory import memory_tool


def test_memory_no_history():
    """Should return query unchanged if no history."""
    result = memory_tool.resolve_query("What is repo rate?", [])
    assert result == "What is repo rate?"


def test_memory_pronoun_resolution():
    """Test regex fallback for 'they' → RBI."""
    history = [{"turn": 1, "query": "What did RBI say about inflation?", "response": "RBI raised rates."}]
    result = memory_tool.resolve_query("How did they address it?", history)
    assert "RBI" in result or "address" in result


def test_memory_update_history():
    history = []
    history = memory_tool.update_history(history, "Q1", "A1", ["rag_search"])
    assert len(history) == 1
    assert history[0]["query"] == "Q1"
    history = memory_tool.update_history(history, "Q2", "A2", ["rag_search"])
    assert len(history) == 2


def test_memory_window_trim():
    history = []
    for i in range(10):
        history = memory_tool.update_history(history, f"Q{i}", f"A{i}", ["rag_search"])
    assert len(history) == 5  # Window size = 5


def test_memory_summarization_long_history():
    """History > 3 turns should be summarized, last 2 kept verbatim."""
    history = []
    for i in range(6):
        history = memory_tool.update_history(history, f"Query {i}", f"Response {i}", ["rag_search"])

    summary = memory_tool._summarize_history(history)
    assert "Summary" in summary or "Turn 5" in summary or "Turn 6" in summary
    assert len(history) == 5  # Sliding window cap


def test_memory_sliding_window():
    """Window should never exceed 5 turns."""
    history = []
    for i in range(10):
        history = memory_tool.update_history(history, f"Q{i}", f"A{i}", ["rag_search"])
    assert len(history) == 5
    assert history[0]["query"] == "Q5"  # First 5 evicted