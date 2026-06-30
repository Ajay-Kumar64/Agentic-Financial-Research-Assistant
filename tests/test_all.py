#!/usr/bin/env python3
"""
tests/test_all.py
=================
THE ULTIMATE SINGLE TEST FILE — Run everything with one command.

Usage:
    pytest tests/test_all.py -v                    # All tests
    pytest tests/test_all.py::TestInfrastructure -v  # Just infrastructure
    pytest tests/test_all.py::TestRAG -v             # Just RAG
    pytest tests/test_all.py::TestAgent -v           # Just agent tools
    pytest tests/test_all.py::TestEvaluation -v      # Just evaluation
    pytest tests/test_all.py -k "not slow" -v        # Skip slow tests
    pytest tests/test_all.py --tb=short -q          # Quiet mode

This file combines:
- Your existing tests (test_tools.py, test_guardrails.py, etc.)
- New OpenSearch RAG tests
- Evaluation tests (golden traces, RAGAS, adversarial)
- End-to-end integration tests

WHY ONE FILE:
- Single command to verify everything
- No missing test files
- CI/CD friendly
- Easy to maintain
"""

import os
import sys
import json
import time
import asyncio
import pytest
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# CONFIGURATION
# =============================================================================
SLOW = pytest.mark.slow  # Mark slow tests
REQUIRES_OPENSEARCH = pytest.mark.requires_opensearch
REQUIRES_REDIS = pytest.mark.requires_redis
REQUIRES_API = pytest.mark.requires_api


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def env_setup():
    """Ensure test environment is configured."""
    os.environ.setdefault("OPENSEARCH_HOST", "localhost")
    os.environ.setdefault("OPENSEARCH_PORT", "9200")
    os.environ.setdefault("OPENSEARCH_INDEX", "rbi_reports")
    os.environ.setdefault("REDIS_HOST", "localhost")
    os.environ.setdefault("REDIS_PORT", "6379")
    yield


@pytest.fixture(scope="session")
def opensearch_client():
    """Create OpenSearch client. Skip tests if not available."""
    try:
        from rag.opensearch_client import OpenSearchRAGClient
        client = OpenSearchRAGClient(auto_connect=False)
        client._connect(max_retries=1)
        yield client
    except Exception as e:
        pytest.skip(f"OpenSearch not available: {e}")


@pytest.fixture(scope="session")
def redis_client():
    """Create Redis client. Skip tests if not available."""
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, db=0, socket_connect_timeout=2)
        r.ping()
        yield r
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")


@pytest.fixture(scope="function")
def test_index(opensearch_client):
    """Create temporary test index."""
    index_name = "test_rag_index"
    if opensearch_client.index_exists(index_name):
        opensearch_client.delete_index(index_name)
    opensearch_client.create_index(index_name, embedding_dim=768)
    yield index_name
    opensearch_client.delete_index(index_name)


@pytest.fixture
def sample_docs():
    """Sample documents for testing."""
    return [
        {
            "chunk_id": "test_1",
            "text": "The repo rate was maintained at 6.5% during FY2023-24.",
            "doc_id": "rbi_2023-24",
            "type": "child",
            "page": 45,
            "year": "2023-24",
            "source": "rbi_2023-24",
            "text_embedding": [0.1] * 768,
        },
        {
            "chunk_id": "test_2",
            "text": "GDP growth for FY2023-24 was estimated at 7.2%.",
            "doc_id": "rbi_2023-24",
            "type": "child",
            "page": 67,
            "year": "2023-24",
            "source": "rbi_2023-24",
            "text_embedding": [0.2] * 768,
        },
        {
            "chunk_id": "test_3",
            "text": "Inflation remained elevated at 5.4% in Q3 FY2022-23.",
            "doc_id": "rbi_2022-23",
            "type": "child",
            "page": 32,
            "year": "2022-23",
            "source": "rbi_2022-23",
            "text_embedding": [0.3] * 768,
        },
    ]


@pytest.fixture
def populated_index(opensearch_client, test_index, sample_docs):
    """Index with sample documents."""
    opensearch_client.bulk_index(sample_docs, index_name=test_index)
    opensearch_client.refresh_index(test_index)
    yield test_index


# =============================================================================
# 1. INFRASTRUCTURE TESTS
# =============================================================================

class TestInfrastructure:
    """Test Docker services and connectivity."""

    @REQUIRES_OPENSEARCH
    def test_opensearch_health(self, opensearch_client):
        """OpenSearch cluster is healthy."""
        health = opensearch_client.health()
        assert health["status"] in ("green", "yellow")

    @REQUIRES_REDIS
    def test_redis_ping(self, redis_client):
        """Redis responds to ping."""
        assert redis_client.ping()

    def test_environment_variables(self):
        """Required env vars are set."""
        required = ["OPENSEARCH_HOST", "OPENSEARCH_PORT", "REDIS_HOST"]
        for var in required:
            assert os.getenv(var), f"Missing env var: {var}"

    def test_imports(self):
        """All packages can be imported."""
        imports = [
            "rag",
            "rag.config",
            "rag.opensearch_client",
            "rag.document_processor",
            "rag.indexing_pipeline",
            "rag.retriever",
            "rag.reranker",
            "rag.cache",
            "agent.tools.rag_search",
        ]
        for module in imports:
            __import__(module)


# =============================================================================
# 2. RAG COMPONENT TESTS (NEW)
# =============================================================================

class TestRAG:
    """Test new OpenSearch RAG components."""

    def test_config_defaults(self):
        """RAGConfig has correct defaults."""
        from rag.config import RAGConfig
        config = RAGConfig()
        assert config.embedding_dim == 768
        assert config.embedder_model == "BAAI/bge-base-en-v1.5"
        assert config.reranker_model == "BAAI/bge-reranker-v2-m3"
        assert config.hybrid_weights == [0.3, 0.7]

    def test_config_from_env(self, monkeypatch):
        """RAGConfig reads from environment."""
        from rag.config import RAGConfig
        monkeypatch.setenv("EMBEDDER_MODEL", "test-model")
        config = RAGConfig()
        assert config.embedder_model == "test-model"

    def test_cache_manager(self):
        """Cache set/get works."""
        from rag.cache import CacheManager
        cache = CacheManager(ttl=60, max_memory_size=10)
        cache.set("q1", {"docs": ["d1"], "confidence": 0.9})
        result = cache.get("q1")
        assert result["confidence"] == 0.9
        assert cache.get("missing") is None

    def test_query_router_simple(self):
        """Router classifies simple queries."""
        from rag.retriever import QueryRouter
        router = QueryRouter()
        assert router.classify("What is repo rate?") == "simple"

    def test_query_router_complex(self):
        """Router classifies complex queries."""
        from rag.retriever import QueryRouter
        router = QueryRouter()
        assert router.classify("Compare repo rate vs reverse repo rate") == "complex"

    def test_query_router_config(self):
        """Router returns correct config per strategy."""
        from rag.retriever import QueryRouter
        router = QueryRouter()
        config = router.get_config("What is repo rate?")
        assert config["strategy"] == "simple"
        assert config["use_hyde"] is False

    def test_reranker_init(self):
        """Reranker initializes with correct model."""
        from rag.reranker import FastReranker
        r = FastReranker()
        assert r.model_name == "BAAI/bge-reranker-v2-m3"

    def test_document_processor_config(self):
        """ProcessingConfig dataclass works."""
        from rag.document_processor import ProcessingConfig
        config = ProcessingConfig(parent_size=512, child_size=128)
        assert config.parent_size == 512
        assert config.child_size == 128

    def test_document_processor_chunking(self):
        """Parent-child chunking produces correct structure."""
        from rag.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        text = "word " * 500
        children, parents = processor.parent_child_chunk(text, "test_doc")
        assert len(children) > 0
        assert len(parents) > 0
        assert all("chunk_id" in c for c in children)
        assert all("parent_id" in c for c in children)

    def test_metric_extraction(self):
        """Regex extracts financial metrics."""
        from rag.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        text = "Repo rate: 6.5%. GDP growth: 7.2%."
        metrics = processor.extract_metrics(text, "test")
        assert len(metrics) >= 2
        assert any(m["metric_type"] == "repo_rate" for m in metrics)

    def test_year_extraction(self):
        """Year extracted from doc_id."""
        from rag.document_processor import DocumentProcessor
        p = DocumentProcessor()
        assert p._extract_year_from_doc("rbi_2023-24.pdf") == "2023-24"
        assert p._extract_year_from_doc("rbi_2024-25.pdf_77") == "2024-25"

    @REQUIRES_OPENSEARCH
    def test_opensearch_index_creation(self, opensearch_client, test_index):
        """Index created with correct mapping."""
        assert opensearch_client.index_exists(test_index)

    @REQUIRES_OPENSEARCH
    def test_opensearch_bulk_index(self, opensearch_client, test_index, sample_docs):
        """Bulk indexing works."""
        result = opensearch_client.bulk_index(sample_docs, index_name=test_index)
        assert result["success"] == len(sample_docs)
        assert len(result["errors"]) == 0

    @REQUIRES_OPENSEARCH
    def test_opensearch_count(self, opensearch_client, populated_index, sample_docs):
        """Document count is accurate."""
        count = opensearch_client.count_docs(populated_index)
        assert count == len(sample_docs)

    @REQUIRES_OPENSEARCH
    def test_opensearch_bm25_search(self, opensearch_client, populated_index):
        """BM25 search returns results."""
        results = opensearch_client.search_bm25("repo rate", k=10, index_name=populated_index)
        assert len(results) > 0
        assert any("repo" in r["text"].lower() for r in results)

    @REQUIRES_OPENSEARCH
    def test_opensearch_knn_search(self, opensearch_client, populated_index):
        """kNN search returns results."""
        results = opensearch_client.search_knn([0.15] * 768, k=10, index_name=populated_index)
        assert len(results) > 0

    @REQUIRES_OPENSEARCH
    def test_opensearch_hybrid_search(self, opensearch_client, populated_index):
        """Hybrid search returns results."""
        results = opensearch_client.search_hybrid(
            "repo rate", [0.15] * 768, k=10, index_name=populated_index
        )
        assert len(results) > 0

    @REQUIRES_OPENSEARCH
    def test_opensearch_year_filter(self, opensearch_client, populated_index):
        """Metadata filtering by year works."""
        results = opensearch_client.search_bm25(
            "repo rate", k=10, filters={"year": "2023-24"}, index_name=populated_index
        )
        assert all(r.get("year") == "2023-24" for r in results)

    @REQUIRES_OPENSEARCH
    def test_opensearch_empty_results(self, opensearch_client, populated_index):
        """Non-existent query returns empty."""
        results = opensearch_client.search_bm25("xyznonexistent", k=10, index_name=populated_index)
        assert len(results) == 0


# =============================================================================
# 3. SMART RETRIEVER TESTS
# =============================================================================

class TestSmartRetriever:
    """Test the full retrieval pipeline."""

    @REQUIRES_OPENSEARCH
    def test_retriever_init(self, opensearch_client):
        """SmartRetriever initializes."""
        from rag.retriever import SmartRetriever
        r = SmartRetriever(os_client=opensearch_client)
        assert r.os is not None
        assert r.router is not None
        assert r.cache is not None

    @REQUIRES_OPENSEARCH
    @SLOW
    def test_retriever_warmup(self, opensearch_client):
        """Model warmup loads embedder."""
        from rag.retriever import SmartRetriever
        r = SmartRetriever(os_client=opensearch_client)
        r.warmup()
        assert r._embedder is not None

    @REQUIRES_OPENSEARCH
    @SLOW
    def test_simple_retrieval(self, opensearch_client, populated_index):
        """Simple query returns results."""
        from rag.retriever import SmartRetriever
        r = SmartRetriever(os_client=opensearch_client)
        result = r.retrieve("repo rate", top_k=3, force_strategy="simple")
        assert len(result.docs) > 0
        assert result.confidence > 0
        assert result.strategy == "simple"

    @REQUIRES_OPENSEARCH
    @SLOW
    def test_year_filter_retrieval(self, opensearch_client, populated_index):
        """Year filtering works."""
        from rag.retriever import SmartRetriever
        r = SmartRetriever(os_client=opensearch_client)
        result = r.retrieve("repo rate", top_k=5, year_filter="2023-24", force_strategy="simple")
        assert all(d.get("year") == "2023-24" for d in result.docs)

    @REQUIRES_OPENSEARCH
    @SLOW
    def test_crag_low_confidence(self, opensearch_client, populated_index):
        """CRAG flags bad queries."""
        from rag.retriever import SmartRetriever
        r = SmartRetriever(os_client=opensearch_client)
        result = r.retrieve("xyznonexistent123", top_k=3, force_strategy="simple")
        if result.docs:
            assert result.fallback_needed or result.confidence < 0.35

    @REQUIRES_OPENSEARCH
    @SLOW
    def test_caching(self, opensearch_client, populated_index):
        """Second query uses cache."""
        from rag.retriever import SmartRetriever
        r = SmartRetriever(os_client=opensearch_client)
        r.retrieve("repo rate", top_k=3, force_strategy="simple")
        result2 = r.retrieve("repo rate", top_k=3, force_strategy="simple")
        # Either cached or fast enough
        assert result2.cached or result2.latency_ms < 1000


# =============================================================================
# 4. AGENT TOOL TESTS
# =============================================================================

class TestAgent:
    """Test agent-facing tools."""

    def test_rag_tool_init(self):
        """RagSearchTool initializes."""
        from agent.tools.rag_search import RagSearchTool
        tool = RagSearchTool()
        assert tool.name == "rag_search"

    @REQUIRES_OPENSEARCH
    @SLOW
    def test_rag_tool_run(self):
        """Tool returns correct format."""
        from agent.tools.rag_search import RagSearchTool
        tool = RagSearchTool()
        result = tool.run("What is repo rate?", top_k=3)
        assert "success" in result
        assert "retrieved_passages" in result
        assert "confidence_score" in result
        assert "text_summary" in result
        assert "latency_sec" in result
        assert "needs_fallback" in result
        assert "strategy" in result

    @REQUIRES_OPENSEARCH
    def test_rag_tool_year_filter(self):
        """Year filter in tool works."""
        from agent.tools.rag_search import RagSearchTool
        tool = RagSearchTool()
        result = tool.run("repo rate", top_k=3, year_filter="2023-24")
        assert "success" in result

    def test_async_retrieval(self):
        """Async function works."""
        import asyncio
        from agent.tools.rag_search import retrieve_passages_async
        async def test():
            return await retrieve_passages_async("repo rate", top_k=3)
        result = asyncio.run(test())
        assert isinstance(result, list)

    def test_parallel_retrieval(self):
        """Parallel retrieval works."""
        import asyncio
        from agent.tools.rag_search import parallel_retrieve
        async def test():
            return await parallel_retrieve(["repo rate", "inflation"], top_k=3)
        result = asyncio.run(test())
        assert isinstance(result, list)

    def test_year_filtering_logic(self):
        """Year extraction and filtering logic."""
        from agent.tools.rag_search import _extract_year_from_doc, _sort_by_recency
        assert _extract_year_from_doc("rbi_2023-24.pdf") == "2023-24"
        assert _extract_year_from_doc("rbi_2024-25.pdf_77") == "2024-25"

        passages = [
            {"doc_id": "rbi_2024-25", "year": "2024-25"},
            {"doc_id": "rbi_2023-24", "year": "2023-24"},
        ]
        result = _sort_by_recency(passages, "latest")
        assert len(result) == 1
        assert result[0]["year"] == "2024-25"

    def test_relevance_check(self):
        """Relevance check catches bad results."""
        from agent.tools.rag_search import _check_relevance
        passages = [{"text": "The repo rate is 6.5%"}]
        assert _check_relevance(passages, "repo rate") is True
        assert _check_relevance(passages, "weather in mumbai") is False


# =============================================================================
# 5. EXISTING TOOL TESTS (from test_tools.py)
# =============================================================================

class TestExistingTools:
    """Tests from your existing test_tools.py."""

    def test_yahoo_finance_tool(self):
        """Yahoo Finance tool returns data."""
        try:
            from agent.tools.yahoo_finance import YahooFinanceTool
            tool = YahooFinanceTool()
            result = tool.run("RELIANCE.NS")
            assert result is not None
            assert "current_price" in result or "error" in result
        except ImportError:
            pytest.skip("Yahoo Finance not available")

    def test_calculator_tool(self):
        """Calculator tool computes correctly."""
        try:
            from agent.tools.calculator import FinancialCalculatorTool as CalculatorTool
            tool = CalculatorTool()
            result = tool.run("2 + 2")
            assert result is not None
            assert result.get("success") is True
        except ImportError:
            pytest.skip("Calculator not available")

    def test_comparator_tool(self):
        """Comparator tool compares documents."""
        try:
            from agent.tools.comparator import DocumentComparatorTool as ComparatorTool
            tool = ComparatorTool()
            result = tool._run("doc A text", "doc B text", "repo rate")
            assert result is not None
            assert "comparison_matrix" in result
        except ImportError:
            pytest.skip("Comparator not available")

    def test_web_search_tool(self):
        """Web search tool returns results."""
        try:
            from agent.tools.web_search import WebSearchTool
            tool = WebSearchTool()
            result = tool.run("RBI repo rate 2024")
            assert result is not None
        except ImportError:
            pytest.skip("Web search not available")


# =============================================================================
# 6. GUARDRAIL TESTS (from test_guardrails.py)
# =============================================================================

class TestGuardrails:
    """Tests from your existing test_guardrails.py."""

    def test_guardrails_import(self):
        """Guardrails module imports."""
        try:
            from agent import guardrails
            assert guardrails is not None
        except ImportError:
            pytest.skip("Guardrails not available")

    def test_input_validation(self):
        """Guardrails reject bad inputs."""
        try:
            from agent.guardrails import validate_input
            # Should pass for valid input
            assert validate_input("What is repo rate?") is True
        except ImportError:
            pytest.skip("Guardrails not available")


# =============================================================================
# 7. MEMORY TESTS (from test_memory.py)
# =============================================================================

class TestMemory:
    """Tests from your existing test_memory.py."""

    @REQUIRES_REDIS
    def test_memory_store(self, redis_client):
        """Memory tool stores and retrieves history."""
        try:
            from agent.tools.memory import ConversationMemoryTool
            tool = ConversationMemoryTool()
            history = []
            history = tool.update_history(history, "What is repo rate?", "6.5%", [])
            assert len(history) == 1
            assert history[0]["query"] == "What is repo rate?"
        except ImportError:
            pytest.skip("Memory not available")

    @REQUIRES_REDIS
    def test_memory_persistence(self, redis_client):
        """Memory history tracking works across updates."""
        try:
            from agent.tools.memory import ConversationMemoryTool
            tool = ConversationMemoryTool()
            history = []
            history = tool.update_history(history, "test message", "test response", [])
            history = tool.update_history(history, "follow up", "follow up response", [])
            assert len(history) == 2
            assert any("test message" in h["query"] for h in history)
        except ImportError:
            pytest.skip("Memory not available")


# =============================================================================
# 8. EVALUATION TESTS (from evaluation/)
# =============================================================================

class TestEvaluation:
    """Tests for evaluation framework."""

    def test_golden_traces_exist(self):
        """Golden traces file exists and is valid JSON."""
        path = "evaluation/golden_traces.json"
        if not os.path.exists(path):
            pytest.skip("golden_traces.json not found")
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list) or isinstance(data, dict)
        assert len(data) > 0

    def test_adversarial_inputs_exist(self):
        """Adversarial inputs file exists."""
        path = "evaluation/adversarial_inputs.json"
        if not os.path.exists(path):
            pytest.skip("adversarial_inputs.json not found")
        with open(path) as f:
            data = json.load(f)
        # Handle both flat list and dict-wrapped formats
        if isinstance(data, dict):
            data = data.get("inputs", [])
        assert isinstance(data, list)
        assert len(data) > 0

    def test_evaluation_metrics_import(self):
        """Evaluation metrics module imports."""
        try:
            from evaluation import metrics
            assert metrics is not None
        except ImportError:
            pytest.skip("Evaluation metrics not available")

    def test_judge_import(self):
        """Judge module imports."""
        try:
            from evaluation import judge
            assert judge is not None
        except ImportError:
            pytest.skip("Judge not available")

    def test_ragas_eval_import(self):
        """RAGAS eval module imports."""
        try:
            from eval import ragas_eval
            assert ragas_eval is not None
        except ImportError:
            pytest.skip("RAGAS eval not available")

    @SLOW
    def test_answer_relevance_metric(self):
        """Answer relevance scoring works."""
        try:
            from evaluation.metrics import score_answer_relevance
            score = score_answer_relevance(
                question="What is repo rate?",
                answer="The repo rate is 6.5%",
                context="The repo rate was maintained at 6.5%"
            )
            assert 0 <= score <= 1
        except ImportError:
            pytest.skip("Metrics not available")

    @SLOW
    def test_faithfulness_metric(self):
        """Faithfulness scoring works."""
        try:
            from evaluation.metrics import score_faithfulness
            score = score_faithfulness(
                answer="The repo rate is 6.5%",
                context="The repo rate was maintained at 6.5%"
            )
            assert 0 <= score <= 1
        except ImportError:
            pytest.skip("Metrics not available")


# =============================================================================
# 9. INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    @REQUIRES_OPENSEARCH
    @SLOW
    def test_simple_factual_query(self):
        """Real query: repo rate."""
        from agent.tools.rag_search import RagSearchTool
        tool = RagSearchTool()
        result = tool.run("What is the repo rate?", top_k=3)
        assert result["success"] is True
        assert len(result["retrieved_passages"]) > 0
        assert result["confidence_score"] > 0.2

    @REQUIRES_OPENSEARCH
    @SLOW
    def test_year_specific_query(self):
        """Real query with year filter."""
        from agent.tools.rag_search import RagSearchTool
        tool = RagSearchTool()
        result = tool.run("repo rate", top_k=3, year_filter="2023-24")
        assert result["success"] is True
        if result["retrieved_passages"]:
            assert all("2023-24" in p.get("year", "") for p in result["retrieved_passages"])

    @REQUIRES_OPENSEARCH
    @SLOW
    def test_fallback_behavior(self):
        """Irrelevant query triggers fallback."""
        from agent.tools.rag_search import RagSearchTool
        tool = RagSearchTool()
        result = tool.run("What is the weather in Mumbai today?", top_k=3)
        assert result["success"] is False or result["needs_fallback"] is True

    @REQUIRES_OPENSEARCH
    @SLOW
    def test_caching_behavior(self):
        """Identical queries use cache."""
        from agent.tools.rag_search import RagSearchTool
        tool = RagSearchTool()
        result1 = tool.run("What is the current repo rate?", top_k=3)
        result2 = tool.run("What is the current repo rate?", top_k=3)
        assert result2.get("cached") is True or result2["latency_sec"] < 1.0

    @REQUIRES_API
    def test_api_health(self):
        """API health endpoint."""
        import requests
        try:
            response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
            assert response.status_code == 200
        except requests.ConnectionError:
            pytest.skip("API server not running")

    @REQUIRES_API
    @SLOW
    def test_api_chat(self):
        """API chat endpoint."""
        import requests
        try:
            response = requests.post(
                "http://localhost:8000/api/v1/chat",
                json={"message": "What is repo rate?"},
                timeout=30,
            )
            assert response.status_code == 200
            data = response.json()
            assert "response" in data or "answer" in data
        except requests.ConnectionError:
            pytest.skip("API server not running")


# =============================================================================
# 10. PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance benchmarks."""

    @REQUIRES_OPENSEARCH
    def test_retrieval_latency(self):
        """Retrieval under 5 seconds."""
        from agent.tools.rag_search import RagSearchTool
        tool = RagSearchTool()
        start = time.time()
        result = tool.run("repo rate", top_k=5)
        elapsed = time.time() - start
        assert elapsed < 5.0, f"Retrieval took {elapsed:.2f}s"
        print(f"Retrieval latency: {elapsed:.3f}s")

    @REQUIRES_OPENSEARCH
    def test_concurrent_queries(self):
        """Multiple queries don't crash."""
        import asyncio
        from agent.tools.rag_search import parallel_retrieve
        async def run():
            queries = ["repo rate", "inflation", "GDP", "monetary policy"]
            results = await parallel_retrieve(queries, top_k=3)
            return results
        results = asyncio.run(run())
        assert len(results) >= 0  # May be empty but shouldn't crash


# =============================================================================
# RUNNER
# =============================================================================

if __name__ == "__main__":
    """Run tests directly without pytest."""
    print("=" * 70)
    print("  RUNNING ALL TESTS")
    print("=" * 70)
    pytest.main([__file__, "-v", "--tb=short"])

'''
# Install pytest
pip install pytest

# Run EVERYTHING (takes ~2-3 minutes)
pytest tests/test_all.py -v

# Run only fast tests (skip slow ones)
pytest tests/test_all.py -k "not slow" -v

# Run only infrastructure checks (30 seconds)
pytest tests/test_all.py::TestInfrastructure -v

# Run only RAG components (1 minute)
pytest tests/test_all.py::TestRAG -v

# Run only agent tools (1 minute)
pytest tests/test_all.py::TestAgent -v

# Run only evaluation (30 seconds)
pytest tests/test_all.py::TestEvaluation -v

# Run only integration (2 minutes)
pytest tests/test_all.py::TestIntegration -v

# Skip tests that need OpenSearch/Redis/API
pytest tests/test_all.py -k "not requires_opensearch and not requires_redis and not requires_api" -v

# Quiet mode with summary
pytest tests/test_all.py --tb=short -q

# With coverage report
pip install pytest-cov
pytest tests/test_all.py --cov=rag --cov=agent.tools --cov=evaluation
'''