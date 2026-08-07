"""Integration tests for LLM Zoomcamp capstone project.

Tests the full pipeline: ingestion → chunking → search → RAG → agent loop →
monitoring → evaluation → Docker configuration.

Usage:
    cd project/
    uv run pytest tests/test_integration.py -v

No external services required — uses mocked LLM for agent/LLM tests and
validates all existing data files and results.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
RESULTS_DIR = PROJECT_ROOT / "results"


class StubSearchIndex:
    """Search-index stand-in for the guardrail tests (todo 4).

    Exposes the same ``.search(query, num_results) -> list[dict]`` contract as
    HybridSearch without loading the ONNX embedder or any data file, so the
    guardrail tests run independently of the data/model artifacts (todo 1).
    """

    def __init__(self, documents: list[dict] | None = None):
        self._documents = documents if documents is not None else [
            {
                "id": "25",
                "title": "Pikachu (#25)",
                "content": (
                    "Pikachu is an Electric-type Pokémon. Stats: HP 35, "
                    "Attack 55, Defense 40, Special Attack 50, Special "
                    "Defense 50, Speed 90."
                ),
                "section": "Electric",
                "url": "",
                "score": 1.0,
            },
            {
                "id": "6",
                "title": "Charizard (#6)",
                "content": (
                    "Charizard is a Fire/Flying-type Pokémon. It takes 2x "
                    "damage from Water and Electric, 4x from Rock, and is "
                    "immune to Ground."
                ),
                "section": "Fire/Flying",
                "url": "",
                "score": 1.0,
            },
        ]

    def search(self, query: str, num_results: int = 5) -> list[dict]:
        return [dict(doc) for doc in self._documents[:num_results]]


# ===========================================================================
# PHASE 1: Data Ingestion & Chunking
# ===========================================================================


class TestDataIngestion:
    """Verify that data ingestion produced valid output files."""

    def test_corpus_file_exists(self):
        """data/corpus.jsonl must exist after ingestion."""
        corpus_path = DATA_DIR / "corpus.jsonl"
        assert corpus_path.exists(), f"Missing {corpus_path}"

    def test_qa_file_exists(self):
        """data/qa.jsonl must exist after ingestion."""
        qa_path = DATA_DIR / "qa.jsonl"
        assert qa_path.exists(), f"Missing {qa_path}"

    def test_corpus_records_are_valid(self):
        """Each line in corpus.jsonl must be valid JSON with required fields."""
        corpus_path = DATA_DIR / "corpus.jsonl"
        records = []
        with open(corpus_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                assert "passage" in record, f"Record {i} missing 'passage' field"
                assert "id" in record, f"Record {i} missing 'id' field"
                assert isinstance(record["passage"], str), f"Record {i} 'passage' not a string"
                records.append(record)
        # Floor relaxed from >= 3000 (rag-mini-wikipedia) to >= 50: the default
        # dev subset is the first 50 Pokémon by id (user directive 2026-08-07).
        assert len(records) >= 50, f"Expected >= 50 passages, got {len(records)}"

    def test_qa_records_are_valid(self):
        """Each line in qa.jsonl must be valid JSON with required fields."""
        qa_path = DATA_DIR / "qa.jsonl"
        records = []
        with open(qa_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                assert "question" in record, f"Record {i} missing 'question' field"
                assert "answer" in record, f"Record {i} missing 'answer' field"
                assert "id" in record, f"Record {i} missing 'id' field"
                records.append(record)
        # Floor relaxed from >= 900 (rag-mini-wikipedia) to >= 250: the default
        # dev subset generates 50 records × 5 pairs (user directive 2026-08-07).
        assert len(records) >= 250, f"Expected >= 250 Q&A pairs, got {len(records)}"

    def test_chunker_output_exists(self):
        """data/chunks/documents.jsonl must exist after chunking."""
        docs_path = CHUNKS_DIR / "documents.jsonl"
        assert docs_path.exists(), f"Missing {docs_path}"

    def test_chunked_documents_are_valid(self):
        """Each chunked document must have required fields."""
        docs_path = CHUNKS_DIR / "documents.jsonl"
        count = 0
        with open(docs_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                assert "id" in doc, f"Doc {i} missing 'id'"
                assert "content" in doc, f"Doc {i} missing 'content'"
                assert "title" in doc, f"Doc {i} missing 'title'"
                assert "section" in doc, f"Doc {i} missing 'section'"
                assert isinstance(doc["content"], str), f"Doc {i} content not a string"
                assert len(doc["content"]) > 0, f"Doc {i} has empty content"
                count += 1
        # Floor relaxed from >= 3000 (rag-mini-wikipedia) to >= 50: the default
        # dev subset is 50 Pokémon, one document per Pokémon (user directive
        # 2026-08-07; review M4 keeps exactly one doc per Pokémon so every id
        # stays a pure integer).
        assert count >= 50, f"Expected >= 50 chunked docs, got {count}"


# ===========================================================================
# PHASE 2: Chunking Pipeline Unit Tests
# ===========================================================================


class TestChunkingPipeline:
    """Test the chunking logic directly."""

    def test_estimate_tokens(self):
        """Token estimation should be roughly len(text) / 4."""
        from src.data.chunker import estimate_tokens

        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("abcdefgh") == 2
        assert estimate_tokens("") == 0

    def test_re_chunk_passage_short(self):
        """Short passages should not be re-chunked."""
        from src.data.chunker import re_chunk_passage

        short_text = "This is a short passage."
        chunks = re_chunk_passage(short_text, max_tokens=1000)
        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_re_chunk_passage_long(self):
        """Long passages should be split into multiple chunks."""
        from src.data.chunker import re_chunk_passage

        # Create a passage longer than max_tokens
        long_text = "This is sentence number X. " * 500  # ~3500 tokens
        chunks = re_chunk_passage(long_text, min_tokens=500, max_tokens=1000)
        assert len(chunks) > 1, "Long passage should be split into multiple chunks"

        # Each chunk should be within max_tokens limit (approximately)
        from src.data.chunker import estimate_tokens

        for chunk in chunks:
            tokens = estimate_tokens(chunk)
            # Allow some tolerance due to sentence-boundary splitting
            assert tokens <= 1200, f"Chunk too large: {tokens} tokens"

    def test_generate_metadata(self):
        """Metadata should derive Pokémon fields from the raw pokedex cache."""
        from src.data.chunker import generate_metadata

        meta = generate_metadata(1, 0, 1)
        assert meta["title"] == "Bulbasaur (#1)"
        assert meta["section"] == "grass+poison"
        assert meta["url"] == (
            "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
            "sprites/pokemon/other/official-artwork/1.png"
        )

    def test_process_corpus_yields_valid_docs(self):
        """process_corpus should yield documents with required fields."""
        from src.data.chunker import process_corpus

        corpus_path = DATA_DIR / "corpus.jsonl"
        count = 0
        for doc in process_corpus(corpus_path, CHUNKS_DIR / "documents.jsonl"):
            assert "id" in doc
            assert "content" in doc
            assert "title" in doc
            assert "section" in doc
            assert len(doc["content"]) > 0
            count += 1
            if count >= 10:  # Just verify the first few
                break
        assert count >= 10


# ===========================================================================
# PHASE 3: Search Index & Retrieval
# ===========================================================================


class TestHybridSearch:
    """Test the hybrid search index construction and retrieval."""

    @pytest.fixture(scope="class")
    def search_index(self):
        """Build a HybridSearch index (cached per test class)."""
        from src.search.hybrid import HybridSearch

        docs_path = CHUNKS_DIR / "documents.jsonl"
        return HybridSearch(documents_path=docs_path)

    def test_index_loads_documents(self, search_index):
        """Search index should load documents from JSONL."""
        assert len(search_index.documents) >= 3000

    def test_keyword_search_returns_results(self, search_index):
        """Keyword search should return results for a valid query."""
        results = search_index.keyword_search("machine learning", num_results=5)
        assert len(results) > 0
        assert len(results) <= 5
        for doc in results:
            assert "id" in doc
            assert "content" in doc

    def test_vector_search_returns_results(self, search_index):
        """Vector search should return results for a valid query."""
        results = search_index.vector_search("deep learning neural networks", num_results=5)
        assert len(results) > 0
        assert len(results) <= 5
        for doc in results:
            assert "id" in doc

    def test_hybrid_search_returns_results(self, search_index):
        """Hybrid search should return fused results with scores."""
        results = search_index.search("What is Python?", num_results=5)
        assert len(results) > 0
        assert len(results) <= 5
        for doc in results:
            assert "id" in doc
            assert "score" in doc, "Hybrid results should have a 'score' field"
            assert isinstance(doc["score"], (int, float))

    def test_hybrid_search_scores_are_ranked(self, search_index):
        """Results should be ranked by score (descending)."""
        results = search_index.search("machine learning", num_results=5)
        scores = [doc["score"] for doc in results]
        assert scores == sorted(scores, reverse=True), "Scores should be in descending order"

    def test_rrf_fusion(self):
        """Reciprocal Rank Fusion should combine ranked lists correctly."""
        from src.search.hybrid import reciprocal_rank_fusion

        list1 = [{"id": "a", "content": "1"}, {"id": "b", "content": "2"}, {"id": "c", "content": "3"}]
        list2 = [{"id": "a", "content": "1"}, {"id": "d", "content": "4"}, {"id": "e", "content": "5"}]

        fused = reciprocal_rank_fusion([list1, list2], num_results=3)
        assert len(fused) <= 3
        # Document "a" appears at rank 0 in both lists → highest score
        assert fused[0]["id"] == "a"
        # All fused results should have scores
        for doc in fused:
            assert "score" in doc
        # Scores should be descending
        scores = [d["score"] for d in fused]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_with_weights(self):
        """RRF should respect per-list weights."""
        from src.search.hybrid import reciprocal_rank_fusion

        list1 = [{"id": "a", "content": "1"}, {"id": "b", "content": "2"}]
        list2 = [{"id": "c", "content": "3"}, {"id": "d", "content": "4"}]

        # Weight list1 heavily
        fused = reciprocal_rank_fusion([list1, list2], weights=[10.0, 1.0], num_results=2)
        # "a" should be first since it's rank 0 in the heavily weighted list
        assert fused[0]["id"] == "a"

    def test_search_empty_query(self, search_index):
        """Empty query should still return results (not crash)."""
        results = search_index.search("", num_results=3)
        assert isinstance(results, list)


# ===========================================================================
# PHASE 4: RAG Pipeline
# ===========================================================================


class TestRAGPipeline:
    """Test the RAG pipeline (search → context → prompt → LLM)."""

    @pytest.fixture(scope="class")
    def search_index(self):
        from src.search.hybrid import HybridSearch

        return HybridSearch(documents_path=CHUNKS_DIR / "documents.jsonl")

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock OpenAI client for LLM calls."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = "This is a mock answer about machine learning."
        mock_client.responses.create.return_value = mock_response
        return mock_client

    def test_rag_base_search(self, search_index):
        """RAGBase.search should delegate to the search index."""
        from src.rag.pipeline import RAGBase

        rag = RAGBase(search_index=search_index)
        results = rag.search("What is deep learning?")
        assert len(results) > 0
        assert "content" in results[0]

    def test_rag_build_context(self, search_index):
        """build_context should format results into a text block."""
        from src.rag.pipeline import RAGBase

        rag = RAGBase(search_index=search_index)
        results = rag.search("Python programming", num_results=3)
        context = rag.build_context(results)
        assert isinstance(context, str)
        assert len(context) > 0
        # Context should contain content from the results
        for doc in results:
            assert doc["content"] in context or doc["title"] in context

    def test_rag_build_prompt(self, search_index):
        """build_prompt should include both question and context."""
        from src.rag.pipeline import RAGBase

        rag = RAGBase(search_index=search_index)
        results = rag.search("machine learning", num_results=3)
        prompt = rag.build_prompt("What is machine learning?", results)
        assert "What is machine learning?" in prompt
        assert "CONTEXT" in prompt

    def test_rag_llm_call(self, search_index, mock_llm_client):
        """RAGBase.llm should call the LLM and return the answer text."""
        from src.rag.pipeline import RAGBase

        rag = RAGBase(search_index=search_index, llm_client=mock_llm_client)
        answer = rag.llm("Test prompt")
        assert answer == "This is a mock answer about machine learning."
        mock_llm_client.responses.create.assert_called_once()

    def test_rag_full_pipeline(self, search_index, mock_llm_client):
        """Full RAG pipeline: search → prompt → LLM answer."""
        from src.rag.pipeline import RAGBase

        rag = RAGBase(search_index=search_index, llm_client=mock_llm_client)
        answer = rag.rag("What is Python?")
        assert isinstance(answer, str)
        assert len(answer) > 0
        # Should have called search + LLM
        mock_llm_client.responses.create.assert_called()


# ===========================================================================
# PHASE 5: Agent Loop
# ===========================================================================


class TestAgentLoop:
    """Test the agentic RAG with iterative search and query reformulation."""

    @pytest.fixture(scope="class")
    def search_index(self):
        from src.search.hybrid import HybridSearch

        return HybridSearch(documents_path=CHUNKS_DIR / "documents.jsonl")

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM that returns sufficient analysis and an answer."""
        mock_client = MagicMock()

        # First call: analysis (sufficient=True → no reformulation needed)
        analysis_response = MagicMock()
        analysis_response.output_text = json.dumps({
            "sufficient": True,
            "reason": "Results contain relevant information",
            "reformulated_query": "",
            "off_topic": False,
            "off_topic_reason": "",
        })

        # Second call: final answer
        answer_response = MagicMock()
        answer_response.output_text = "Python is a high-level programming language."

        mock_client.responses.create.side_effect = [analysis_response, answer_response]
        return mock_client

    @pytest.fixture
    def mock_llm_client_reformulate(self):
        """Mock LLM that triggers reformulation (insufficient → reformulate → sufficient)."""
        mock_client = MagicMock()

        # Call 1: analysis (insufficient)
        insufficient_response = MagicMock()
        insufficient_response.output_text = json.dumps({
            "sufficient": False,
            "reason": "Results not specific enough",
            "reformulated_query": "Python programming language features",
            "off_topic": False,
            "off_topic_reason": "",
        })

        # Call 2: analysis after reformulation (sufficient)
        sufficient_response = MagicMock()
        sufficient_response.output_text = json.dumps({
            "sufficient": True,
            "reason": "Found relevant information",
            "reformulated_query": "",
            "off_topic": False,
            "off_topic_reason": "",
        })

        # Call 3: final answer
        answer_response = MagicMock()
        answer_response.output_text = "Python is a high-level programming language."

        mock_client.responses.create.side_effect = [
            insufficient_response, sufficient_response, answer_response
        ]
        return mock_client

    def test_agent_dataclasses(self):
        """SearchRecord and AgentResult dataclasses should work correctly."""
        from src.rag.agent import SearchRecord, AgentResult

        record = SearchRecord(query="test", results=[{"id": "1"}], analysis={"sufficient": True})
        assert record.query == "test"
        assert len(record.results) == 1
        assert record.analysis["sufficient"] is True

        result = AgentResult(answer="test answer", searches=[record], iterations=1)
        assert result.answer == "test answer"
        assert result.iterations == 1

    def test_agent_perform_search(self, search_index):
        """Agent.perform_search should return search results."""
        from src.rag.agent import RAGAgent

        agent = RAGAgent(search_index=search_index, llm_client=MagicMock())
        results = agent.perform_search("What is Python?")
        assert len(results) > 0
        assert "id" in results[0]

    def test_agent_analyze_results(self, search_index, mock_llm_client):
        """Agent.analyze_results should call LLM and return analysis dict."""
        from src.rag.agent import RAGAgent

        agent = RAGAgent(search_index=search_index, llm_client=mock_llm_client)
        results = agent.perform_search("What is Python?")
        analysis = agent.analyze_results("What is Python?", results)

        assert isinstance(analysis, dict)
        assert "sufficient" in analysis
        assert "reason" in analysis

    def test_agent_analyze_handles_markdown_json(self):
        """Agent.analyze_results should handle markdown code-block JSON."""
        from src.rag.agent import RAGAgent

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = (
            '```json\n{"sufficient": true, "reason": "ok", '
            '"reformulated_query": "", "off_topic": false, '
            '"off_topic_reason": ""}\n```'
        )
        mock_client.responses.create.return_value = mock_response

        agent = RAGAgent(search_index=StubSearchIndex(), llm_client=mock_client)
        results = agent.perform_search("test")
        analysis = agent.analyze_results("test", results)

        assert analysis["sufficient"] is True

    def test_agent_analyze_handles_invalid_json(self):
        """Agent.analyze_results deterministic fail-safe: unparseable JSON with
        no Pokémon-domain signals → rejection dict (out-of-scope by default)."""
        from src.rag.agent import RAGAgent

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = "This is not JSON at all"
        mock_client.responses.create.return_value = mock_response

        agent = RAGAgent(search_index=StubSearchIndex(), llm_client=mock_client)
        results = agent.perform_search("test")
        analysis = agent.analyze_results("test", results)

        # No Pokémon-domain signals in "test" → deterministic fail-safe rejection
        assert analysis["rejected"] is True

    def test_agent_reformulate_query_from_analysis(self, search_index):
        """Agent.reformulate_query should use analysis.reformulated_query."""
        from src.rag.agent import RAGAgent

        mock_client = MagicMock()
        agent = RAGAgent(search_index=search_index, llm_client=mock_client)
        analysis = {"reformulated_query": "better query about Python"}
        reformulated = agent.reformulate_query("original query", analysis)
        assert reformulated == "better query about Python"
        # Should not call LLM since reformulated_query was provided
        mock_client.responses.create.assert_not_called()

    def test_agent_reformulate_query_via_llm(self, search_index):
        """Agent.reformulate_query should call LLM if no reformulated_query in analysis."""
        from src.rag.agent import RAGAgent

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = "reformulated query from LLM"
        mock_client.responses.create.return_value = mock_response

        agent = RAGAgent(search_index=search_index, llm_client=mock_client)
        analysis = {"reformulated_query": "", "reason": "insufficient"}
        reformulated = agent.reformulate_query("original", analysis)
        assert reformulated == "reformulated query from LLM"
        mock_client.responses.create.assert_called_once()

    def test_agent_generate_answer_deduplicates(self, search_index, mock_llm_client):
        """Agent.generate_answer should deduplicate results by id."""
        from src.rag.agent import RAGAgent

        agent = RAGAgent(search_index=search_index, llm_client=mock_llm_client)
        all_results = [
            {"id": "1", "content": "a", "title": "A", "section": "s"},
            {"id": "1", "content": "a", "title": "A", "section": "s"},  # duplicate
            {"id": "2", "content": "b", "title": "B", "section": "s"},
        ]
        answer = agent.generate_answer("test", all_results)
        assert isinstance(answer, str)
        # Should have been called with deduplicated results
        call_args = mock_llm_client.responses.create.call_args
        assert call_args is not None

    def test_agent_run_single_iteration(self, search_index, mock_llm_client):
        """Agent.run should complete in 1 iteration when results are sufficient."""
        from src.rag.agent import RAGAgent

        agent = RAGAgent(search_index=search_index, llm_client=mock_llm_client)
        result = agent.run("What is Python?")

        assert "answer" in result
        assert "searches" in result
        assert "iterations" in result
        assert result["iterations"] >= 1
        assert len(result["searches"]) >= 1
        assert result["searches"][0].analysis is not None

    def test_agent_run_with_reformulation(self, search_index, mock_llm_client_reformulate):
        """Agent.run should reformulate when first results are insufficient."""
        from src.rag.agent import RAGAgent

        agent = RAGAgent(search_index=search_index, llm_client=mock_llm_client_reformulate)
        result = agent.run("vague query")

        assert result["iterations"] >= 2, "Should have done at least 2 iterations"
        # First search should have been insufficient
        assert result["searches"][0].analysis["sufficient"] is False
        # Second search should have been sufficient
        assert result["searches"][1].analysis["sufficient"] is True

    def test_agent_max_iterations(self, search_index):
        """Agent should not exceed max_iterations; exhausted in-domain runs
        return the uncertainty note (rejected: false), never a rejection dict
        or an LLM-generated fallback."""
        from src.rag.agent import RAGAgent

        # Always insufficient analysis
        mock_client = MagicMock()
        insufficient = MagicMock()
        insufficient.output_text = json.dumps({
            "sufficient": False, "reason": "bad", "reformulated_query": "retry",
            "off_topic": False, "off_topic_reason": "",
        })
        mock_client.responses.create.side_effect = [insufficient, insufficient]

        agent = RAGAgent(search_index=search_index, llm_client=mock_client, max_iterations=2)
        result = agent.run("test")

        assert result["iterations"] <= 2
        # Low-confidence contract (review B1): exhausted iterations with all
        # analyses insufficient → uncertainty note, rejected: false, and no
        # generate_answer LLM call (only the 2 analysis calls happen).
        assert result["rejected"] is False
        assert "couldn't find a confident answer" in result["answer"]
        assert mock_client.responses.create.call_count == 2

    def test_agent_weakness_answer_grounded_in_multipliers(self):
        """Weakness question with damage_taken in context → grounded answer
        citing 'weak' + a multiplier, driven by the Pokémon persona
        instructions (damage_taken citation contract)."""
        from src.rag.agent import RAGAgent

        documents = [
            {
                "id": "6",
                "title": "Charizard (#6)",
                "content": (
                    "damage_taken: normal 1x, fire 0.5x, water 2x, electric 2x, "
                    "grass 0.25x, ice 1x, fighting 1x, poison 1x, ground 0x, "
                    "flying 1x, psychic 1x, bug 0.5x, rock 4x, ghost 1x, "
                    "dragon 1x, steel 1x, dark 1x, fairy 0.5x."
                ),
                "section": "Fire/Flying",
                "url": "",
                "score": 1.0,
            }
        ]
        mock_client = MagicMock()
        analysis_response = MagicMock()
        analysis_response.output_text = json.dumps({
            "sufficient": True,
            "reason": "damage_taken table present",
            "reformulated_query": "",
            "off_topic": False,
            "off_topic_reason": "",
        })
        answer_response = MagicMock()
        answer_response.output_text = (
            "Charizard is 4x weak to Rock, 2x weak to Water and Electric, "
            "and immune to Ground."
        )
        mock_client.responses.create.side_effect = [analysis_response, answer_response]

        agent = RAGAgent(search_index=StubSearchIndex(documents=documents), llm_client=mock_client)
        result = agent.run("What is Charizard weak to?")

        assert result.get("rejected", False) is False
        assert "weak" in result["answer"]
        assert any(mult in result["answer"] for mult in ("2x", "4x"))
        # The answer call must carry the Pokémon persona instructions that
        # require citing damage_taken multipliers.
        answer_call = mock_client.responses.create.call_args_list[-1]
        developer_content = answer_call.kwargs["input"][0]["content"]
        assert "damage_taken" in developer_content
        assert "battles" in developer_content

    def test_agent_team_build_answer_suggests_type_coverage(self):
        """Team-build question → persona instructions ask for type-coverage
        suggestions and the answer provides them, without battle claims."""
        from src.rag.agent import RAGAgent

        mock_client = MagicMock()
        analysis_response = MagicMock()
        analysis_response.output_text = json.dumps({
            "sufficient": True,
            "reason": "type data present",
            "reformulated_query": "",
            "off_topic": False,
            "off_topic_reason": "",
        })
        answer_response = MagicMock()
        answer_response.output_text = (
            "Water types are weak to Electric and Grass, so add a Jolteon or "
            "Venusaur to cover that weakness."
        )
        mock_client.responses.create.side_effect = [analysis_response, answer_response]

        agent = RAGAgent(search_index=StubSearchIndex(), llm_client=mock_client)
        result = agent.run("Help me build a team that covers a water weakness")

        assert result.get("rejected", False) is False
        assert "weak" in result["answer"]
        assert "cover" in result["answer"]
        answer_call = mock_client.responses.create.call_args_list[-1]
        developer_content = answer_call.kwargs["input"][0]["content"]
        assert "team-building" in developer_content
        assert "will beat" in developer_content

    def test_agent_low_confidence_exhausted_iterations(self):
        """In-domain query exhausting max iterations with all analyses
        insufficient → uncertainty note, rejected: false (never a rejection
        dict), and no generate_answer LLM call."""
        from src.rag.agent import RAGAgent, UNCERTAINTY_NOTE

        mock_client = MagicMock()
        insufficient = MagicMock()
        insufficient.output_text = json.dumps({
            "sufficient": False, "reason": "bad", "reformulated_query": "retry",
            "off_topic": False, "off_topic_reason": "",
        })
        mock_client.responses.create.side_effect = [insufficient, insufficient, insufficient]

        agent = RAGAgent(search_index=StubSearchIndex(), llm_client=mock_client, max_iterations=3)
        result = agent.run("What are Gengar's stats?")

        assert result["answer"] == UNCERTAINTY_NOTE
        assert "couldn't find a confident answer" in result["answer"]
        assert result["rejected"] is False
        assert result["iterations"] == 3
        # Only the 3 analysis calls; the answer mock was never invoked.
        assert mock_client.responses.create.call_count == 3

    def test_agent_empty_results_skips_answer_call(self):
        """Empty search results → uncertainty note, rejected: false, and the
        LLM answer call is never made (analysis calls only)."""
        from src.rag.agent import RAGAgent, UNCERTAINTY_NOTE

        mock_client = MagicMock()
        insufficient = MagicMock()
        insufficient.output_text = json.dumps({
            "sufficient": False, "reason": "no results", "reformulated_query": "retry",
            "off_topic": False, "off_topic_reason": "",
        })
        mock_client.responses.create.side_effect = [insufficient, insufficient, insufficient]

        agent = RAGAgent(
            search_index=StubSearchIndex(documents=[]), llm_client=mock_client,
            max_iterations=3,
        )
        result = agent.run("What is the habitat of a rare Pokémon?")

        assert result["answer"] == UNCERTAINTY_NOTE
        assert "couldn't find a confident answer" in result["answer"]
        assert result["rejected"] is False
        # 3 analysis calls, zero answer calls.
        assert mock_client.responses.create.call_count == 3


# ===========================================================================
# PHASE 5b: Agent Guardrails (rule pre-gate + off-topic flag + fail-safe)
# ===========================================================================


class TestAgentGuardrails:
    """Guardrail rejection paths (todo 4): rule pre-gate, LLM off-topic flag,
    and the deterministic fail-safe, all against a stub search index."""

    @staticmethod
    def _response(text: str):
        response = MagicMock()
        response.output_text = text
        return response

    @staticmethod
    def _in_domain_client(answer_text: str = "Pikachu is an Electric-type Pokémon."):
        """Analysis sufficient + off_topic:false, then a grounded answer."""
        mock_client = MagicMock()
        analysis_response = MagicMock()
        analysis_response.output_text = json.dumps({
            "sufficient": True,
            "reason": "Results contain relevant Pokémon information",
            "reformulated_query": "",
            "off_topic": False,
            "off_topic_reason": "",
        })
        answer_response = MagicMock()
        answer_response.output_text = answer_text
        mock_client.responses.create.side_effect = [analysis_response, answer_response]
        return mock_client

    def _agent(self, llm_client, **kwargs):
        from src.rag.agent import RAGAgent

        return RAGAgent(search_index=StubSearchIndex(), llm_client=llm_client, **kwargs)

    def test_rejects_battle_simulation(self):
        """Out-of-scope battle-simulation query → structured rejection."""
        result = self._agent(MagicMock()).run("Who would win Charizard vs Blastoise?")
        assert result["rejected"] is True
        assert result["searches"] == []
        assert result["iterations"] == 0
        assert "can't" in result["answer"]

    def test_rejects_docker_question(self):
        """Non-Pokémon topic (docker) → structured rejection."""
        result = self._agent(MagicMock()).run("What is Docker?")
        assert result["rejected"] is True
        assert result["searches"] == []
        assert result["iterations"] == 0

    def test_rejects_cooking_question(self):
        """Non-Pokémon topic (cooking) → structured rejection."""
        result = self._agent(MagicMock()).run("How do I cook pasta?")
        assert result["rejected"] is True

    def test_rejects_finance_question(self):
        """Non-Pokémon topic (finance) → structured rejection."""
        result = self._agent(MagicMock()).run("Recommend a cheap stock to invest in")
        assert result["rejected"] is True

    def test_rejects_medical_question(self):
        """Non-Pokémon topic (medical) → structured rejection."""
        result = self._agent(MagicMock()).run("I have a fever, what medicine should I take?")
        assert result["rejected"] is True

    def test_rejects_save_file_request(self):
        """Save-file access request → structured rejection."""
        result = self._agent(MagicMock()).run("Can you load my Pokémon save file?")
        assert result["rejected"] is True

    def test_rejects_cheating_request(self):
        """Cheating/hacking request → structured rejection."""
        result = self._agent(MagicMock()).run("Is there a hack to catch Mewtwo easily?")
        assert result["rejected"] is True

    def test_rejection_answer_is_friendly_redirect(self):
        """The rejection answer is the exact friendly redirect message."""
        from src.rag.agent import REJECTION_MESSAGE

        result = self._agent(MagicMock()).run("Who would win Charizard vs Blastoise?")
        assert result["answer"] == REJECTION_MESSAGE

    def test_team_building_is_not_rejected(self):
        """Team-building query (battle TEAM, allowed by scope) → normal path."""
        agent = self._agent(self._in_domain_client("Water types are weak to Electric and Grass."))
        result = agent.run("Help me build a battle team against water types")
        assert result.get("rejected", False) is False
        assert result["iterations"] >= 1
        assert result["answer"] == "Water types are weak to Electric and Grass."

    def test_bare_battle_question_is_not_rejected(self):
        """Bare 'battle' must not trip the pre-gate (battle TEAM is in-scope)."""
        agent = self._agent(self._in_domain_client("Pikachu is a strong special attacker."))
        result = agent.run("Tell me about battle strategies for Pikachu")
        assert result.get("rejected", False) is False
        assert result["answer"] == "Pikachu is a strong special attacker."

    def test_in_domain_question_with_off_topic_false_is_not_rejected(self):
        """Analysis says off_topic:false → normal answer path."""
        agent = self._agent(self._in_domain_client("Pikachu has 90 base Speed."))
        result = agent.run("Tell me about Pikachu")
        assert result.get("rejected", False) is False
        assert result["iterations"] == 1
        assert result["answer"] == "Pikachu has 90 base Speed."

    def test_off_topic_flag_is_rejected(self):
        """Analysis flags the query as outside the Pokémon domain → rejection."""
        mock_client = MagicMock()
        mock_client.responses.create.return_value = self._response(json.dumps({
            "sufficient": False,
            "reason": "Not about Pokémon",
            "reformulated_query": "",
            "off_topic": True,
            "off_topic_reason": "Asks about a non-Pokémon topic",
        }))
        result = self._agent(mock_client).run("Explain the theory of relativity")
        assert result["rejected"] is True
        assert result["searches"] == []
        assert result["iterations"] == 0

    def test_fail_safe_rejects_query_without_domain_signals(self):
        """Unparseable analysis + no Pokémon-domain signals → rejected."""
        mock_client = MagicMock()
        mock_client.responses.create.return_value = self._response("garbage not json")
        result = self._agent(mock_client).run("How does gravity work?")
        assert result["rejected"] is True
        assert result["searches"] == []
        assert result["iterations"] == 0

    def test_fail_safe_allows_query_with_domain_signals(self):
        """Unparseable analysis + 'pikachu stats' (stat signal) → normal path."""
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = [
            self._response("garbage not json"),
            self._response("Pikachu's base Speed is 90."),
        ]
        result = self._agent(mock_client).run("pikachu stats")
        assert result.get("rejected", False) is False
        assert result["iterations"] == 1
        assert result["answer"] == "Pikachu's base Speed is 90."

    def test_fail_safe_analysis_keeps_sufficient_for_domain_signals(self):
        """analyze_results keeps sufficient:true when signals are present."""
        from src.rag.agent import RAGAgent

        mock_client = MagicMock()
        mock_client.responses.create.return_value = self._response("not json")
        agent = RAGAgent(search_index=StubSearchIndex(), llm_client=mock_client)
        analysis = agent.analyze_results("pikachu stats", [])
        assert analysis["sufficient"] is True


# ===========================================================================
# PHASE 5c: Search-type dispatch (todo 6)
# ===========================================================================


class RecordingSearchIndex(StubSearchIndex):
    """Stub that records which search backend method was invoked.

    ``keyword_search`` / ``vector_search`` / ``search`` all satisfy the
    ``(query, num_results) -> list[dict]`` contract; the recorded call tuple
    proves the agent/pipeline dispatched to the right one.
    """

    def __init__(self, documents: list[dict] | None = None):
        super().__init__(documents=documents)
        self.calls: list[tuple[str, str, int]] = []

    def search(self, query: str, num_results: int = 5) -> list[dict]:
        self.calls.append(("search", query, num_results))
        return super().search(query, num_results=num_results)

    def keyword_search(self, query: str, num_results: int = 5) -> list[dict]:
        self.calls.append(("keyword_search", query, num_results))
        return [dict(doc) for doc in self._documents[:num_results]]

    def vector_search(self, query: str, num_results: int = 5) -> list[dict]:
        self.calls.append(("vector_search", query, num_results))
        return [dict(doc) for doc in self._documents[:num_results]]


class TestSearchTypeDispatch:
    """search_type wiring (todo 6): RAGBase/RAGAgent dispatch to the selected
    search backend — stub index only, never a real HybridSearch."""

    @staticmethod
    def _rag(search_type: str | None = None):
        from src.rag.pipeline import RAGBase

        index = RecordingSearchIndex()
        kwargs = {"llm_client": MagicMock()}
        if search_type is not None:
            kwargs["search_type"] = search_type
        return RAGBase(search_index=index, **kwargs), index

    @staticmethod
    def _agent(search_type: str):
        from src.rag.agent import RAGAgent

        index = RecordingSearchIndex()
        agent = RAGAgent(
            search_index=index, llm_client=MagicMock(), search_type=search_type
        )
        return agent, index

    def test_ragbase_keyword_dispatches_to_keyword_search(self):
        rag, index = self._rag("keyword")
        rag.search("pikachu", num_results=3)
        assert index.calls == [("keyword_search", "pikachu", 3)]

    def test_ragbase_vector_dispatches_to_vector_search(self):
        rag, index = self._rag("vector")
        rag.search("pikachu", num_results=3)
        assert index.calls == [("vector_search", "pikachu", 3)]

    def test_ragbase_default_hybrid_dispatches_to_search(self):
        rag, index = self._rag()
        rag.search("pikachu", num_results=3)
        assert index.calls == [("search", "pikachu", 3)]

    def test_agent_keyword_dispatches_to_keyword_search(self):
        agent, index = self._agent("keyword")
        agent.perform_search("pikachu")
        assert index.calls[0][0] == "keyword_search"

    def test_agent_vector_dispatches_to_vector_search(self):
        agent, index = self._agent("vector")
        agent.perform_search("pikachu")
        assert index.calls[0][0] == "vector_search"

    def test_agent_hybrid_dispatches_to_search(self):
        agent, index = self._agent("hybrid")
        agent.perform_search("pikachu")
        assert index.calls[0][0] == "search"


# ===========================================================================
# PHASE 6: Monitoring & Tracing
# ===========================================================================


class TestMonitoring:
    """Test OpenTelemetry tracing with SQLite storage."""

    def test_tracer_setup_creates_db(self, tmp_path):
        """TracerSetup should create the SQLite database."""
        from src.monitoring.tracer import SQLiteSpanExporter

        db_path = tmp_path / "test_traces.db"
        exporter = SQLiteSpanExporter(db_path=db_path)
        assert db_path.exists()
        exporter.shutdown()

    def test_tracer_schema_has_required_columns(self, tmp_path):
        """SQLite spans table should have all required columns."""
        from src.monitoring.tracer import SQLiteSpanExporter

        db_path = tmp_path / "test_traces.db"
        exporter = SQLiteSpanExporter(db_path=db_path)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("PRAGMA table_info(spans)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        exporter.shutdown()

        expected = {
            "name", "start_time", "end_time",
            "input_tokens", "output_tokens", "cost",
            "feedback", "agent_iterations", "query", "search_queries"
        }
        assert expected.issubset(columns), f"Missing columns: {expected - columns}"

    def test_tracer_records_spans(self, tmp_path):
        """SQLiteSpanExporter should record spans to SQLite."""
        from src.monitoring.tracer import SQLiteSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        db_path = tmp_path / "test_traces.db"
        exporter = SQLiteSpanExporter(db_path=db_path)
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test_tracer_records")

        with tracer.start_as_current_span("test.span") as span:
            span.set_attribute("query", "test query")
            span.set_attribute("input_tokens", 100)
            span.set_attribute("output_tokens", 50)
            span.set_attribute("cost", 0.001)

        exporter.force_flush()
        exporter.shutdown()

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT * FROM spans").fetchall()
        conn.close()

        assert len(rows) >= 1
        row = rows[0]
        assert row[0] == "test.span"

    def test_record_feedback(self, tmp_path):
        """record_feedback should update the feedback column."""
        from src.monitoring.tracer import SQLiteSpanExporter, record_feedback
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        db_path = tmp_path / "test_traces.db"
        exporter = SQLiteSpanExporter(db_path=db_path)
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test_record_feedback")

        with tracer.start_as_current_span("agent.run") as span:
            span.set_attribute("query", "test")

        exporter.force_flush()
        exporter.shutdown()

        result = record_feedback("0000000000000001", "positive", db_path=db_path)
        assert isinstance(result, bool)

    def test_get_trace_stats(self, tmp_path):
        """get_trace_stats should return summary statistics."""
        from src.monitoring.tracer import SQLiteSpanExporter, get_trace_stats
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        db_path = tmp_path / "test_traces.db"
        exporter = SQLiteSpanExporter(db_path=db_path)
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test_get_trace_stats")

        with tracer.start_as_current_span("agent.run") as span:
            span.set_attribute("input_tokens", 100)
            span.set_attribute("output_tokens", 50)
            span.set_attribute("cost", 0.01)

        exporter.force_flush()
        exporter.shutdown()

        stats = get_trace_stats(db_path=db_path)
        assert stats["total_traces"] >= 1
        assert "span_names" in stats
        assert stats["total_input_tokens"] >= 100
        assert stats["total_output_tokens"] >= 50
        assert stats["total_cost"] >= 0.01

    def test_traced_ragent_run(self, tmp_path):
        """TracedRAGAgent should wrap agent.run with tracing."""
        from src.monitoring.tracer import SQLiteSpanExporter, TracedRAGAgent, get_trace_stats
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        db_path = tmp_path / "test_traced_agent.db"
        exporter = SQLiteSpanExporter(db_path=db_path)
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test_traced_agent")

        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "answer": "test answer",
            "searches": [],
            "iterations": 1,
        }

        traced = TracedRAGAgent(agent=mock_agent, tracer=tracer)
        result = traced.run("test query")

        assert result["answer"] == "test answer"
        exporter.force_flush()
        exporter.shutdown()

        stats = get_trace_stats(db_path=db_path)
        assert stats["total_traces"] >= 1


# ===========================================================================
# PHASE 7: Evaluation Results Verification
# ===========================================================================


class TestEvaluationResults:
    """Verify that evaluation results are valid and meet minimum thresholds."""

    def test_retrieval_eval_file_exists(self):
        """retrieval_eval.json must exist."""
        path = RESULTS_DIR / "retrieval_eval.json"
        assert path.exists(), f"Missing {path}"

    def test_retrieval_eval_structure(self):
        """retrieval_eval.json must have keyword/vector/hybrid sections."""
        with open(RESULTS_DIR / "retrieval_eval.json") as f:
            data = json.load(f)

        assert "keyword" in data
        assert "vector" in data
        assert "hybrid" in data

        for method in ["keyword", "vector", "hybrid"]:
            section = data[method]
            assert "precision@5" in section
            assert "recall@5" in section
            assert "mrr" in section
            assert "num_questions" in section
            assert "time_seconds" in section
            assert section["num_questions"] >= 900

    def test_retrieval_eval_metrics_are_valid(self):
        """Retrieval metrics should be within valid ranges (0-1)."""
        with open(RESULTS_DIR / "retrieval_eval.json") as f:
            data = json.load(f)

        for method in ["keyword", "vector", "hybrid"]:
            section = data[method]
            for metric in ["precision@5", "recall@5", "mrr"]:
                assert 0 <= section[metric] <= 1, (
                    f"{method}.{metric} = {section[metric]} out of range [0, 1]"
                )

    def test_retrieval_eval_vector_beats_keyword(self):
        """Vector search should outperform keyword-only (recall)."""
        with open(RESULTS_DIR / "retrieval_eval.json") as f:
            data = json.load(f)

        # Vector should have higher recall than keyword
        assert data["vector"]["recall@5"] >= data["keyword"]["recall@5"], (
            f"Vector recall {data['vector']['recall@5']} < keyword recall {data['keyword']['recall@5']}"
        )

    def test_llm_eval_file_exists(self):
        """llm_eval.json must exist."""
        path = RESULTS_DIR / "llm_eval.json"
        assert path.exists(), f"Missing {path}"

    def test_llm_eval_structure(self):
        """llm_eval.json must have simple/detailed/with_examples sections."""
        with open(RESULTS_DIR / "llm_eval.json") as f:
            data = json.load(f)

        assert "simple" in data
        assert "detailed" in data
        assert "with_examples" in data

        for prompt_name in ["simple", "detailed", "with_examples"]:
            section = data[prompt_name]
            assert "faithfulness" in section
            assert "relevance" in section
            assert "coherence" in section
            assert "num_evaluated" in section
            assert section["num_evaluated"] >= 10

    def test_llm_eval_scores_are_high(self):
        """LLM answer quality scores should be >= 4.0 (the model performs well)."""
        with open(RESULTS_DIR / "llm_eval.json") as f:
            data = json.load(f)

        for prompt_name in ["simple", "detailed", "with_examples"]:
            section = data[prompt_name]
            assert section["faithfulness"] >= 4.0, (
                f"{prompt_name} faithfulness {section['faithfulness']} < 4.0"
            )
            assert section["relevance"] >= 4.0, (
                f"{prompt_name} relevance {section['relevance']} < 4.0"
            )
            assert section["coherence"] >= 4.0, (
                f"{prompt_name} coherence {section['coherence']} < 4.0"
            )

    def test_llm_eval_with_examples_is_best(self):
        """The with_examples prompt should be the highest-scoring judge prompt."""
        with open(RESULTS_DIR / "llm_eval.json") as f:
            data = json.load(f)

        best = max(
            ["simple", "detailed", "with_examples"],
            key=lambda k: (
                data[k]["faithfulness"] + data[k]["relevance"] + data[k]["coherence"]
            ) / 3,
        )
        assert best == "with_examples", f"Expected 'with_examples' to be best, got '{best}'"

    def test_agent_eval_file_exists(self):
        """agent_eval.json must exist."""
        path = RESULTS_DIR / "agent_eval.json"
        assert path.exists(), f"Missing {path}"

    def test_agent_eval_structure(self):
        """agent_eval.json must have simple_rag, agentic_rag, comparison sections."""
        with open(RESULTS_DIR / "agent_eval.json") as f:
            data = json.load(f)

        assert "simple_rag" in data
        assert "agentic_rag" in data
        assert "comparison" in data
        assert "config" in data

    def test_agent_eval_comparison_metrics(self):
        """Comparison section should have retrieval_improvement and latency_overhead."""
        with open(RESULTS_DIR / "agent_eval.json") as f:
            data = json.load(f)

        comp = data["comparison"]
        assert "retrieval_improvement" in comp
        assert "answer_quality_improvement" in comp
        assert "latency_overhead" in comp
        assert "search_overhead" in comp

    def test_agent_eval_agentic_beats_simple(self):
        """Agentic RAG should have higher retrieval hit rate than simple RAG."""
        with open(RESULTS_DIR / "agent_eval.json") as f:
            data = json.load(f)

        simple_rate = data["simple_rag"]["retrieval"]["hit_rate"]
        agent_rate = data["agentic_rag"]["retrieval"]["hit_rate"]
        assert agent_rate >= simple_rate, (
            f"Agentic {agent_rate} < Simple {simple_rate}"
        )

    def test_agent_eval_retrieval_improvement_positive(self):
        """Agentic RAG should show positive retrieval improvement over simple."""
        with open(RESULTS_DIR / "agent_eval.json") as f:
            data = json.load(f)

        assert data["comparison"]["retrieval_improvement"] >= 0, (
            f"Retrieval improvement should be >= 0, got {data['comparison']['retrieval_improvement']}"
        )

    def test_agent_eval_config_is_valid(self):
        """Config should specify total_questions and model."""
        with open(RESULTS_DIR / "agent_eval.json") as f:
            data = json.load(f)

        config = data["config"]
        assert config["total_questions"] >= 900
        assert "model" in config


# ===========================================================================
# PHASE 8: Evaluation Script Execution
# ===========================================================================


class TestEvaluationScripts:
    """Verify that evaluation scripts can be imported and their functions work."""

    def test_retrieval_eval_importable(self):
        """retrieval_eval module should be importable."""
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.evaluation.retrieval_eval import (
            load_ground_truth,
            precision_at_k,
            recall_at_k,
            mrr,
            evaluate_search,
        )

    def test_precision_at_k(self):
        """precision_at_k should compute correctly."""
        from src.evaluation.retrieval_eval import precision_at_k

        # Relevant doc is at position 0 in top-5
        assert precision_at_k(["a", "b", "c", "d", "e"], "a", 5) == 1.0 / 5
        # Relevant doc is not in top-5
        assert precision_at_k(["a", "b", "c", "d", "e"], "f", 5) == 0.0

    def test_recall_at_k(self):
        """recall_at_k should return 1.0 if relevant doc is in top-k."""
        from src.evaluation.retrieval_eval import recall_at_k

        assert recall_at_k(["a", "b", "c"], "a", 5) == 1.0
        assert recall_at_k(["a", "b", "c"], "d", 5) == 0.0

    def test_mrr(self):
        """mrr should return 1/rank of first relevant result."""
        from src.evaluation.retrieval_eval import mrr

        assert mrr(["a", "b", "c"], "a") == 1.0  # rank 1
        assert mrr(["a", "b", "c"], "b") == 0.5  # rank 2
        assert mrr(["a", "b", "c"], "d") == 0.0  # not found

    def test_llm_eval_importable(self):
        """llm_eval module should be importable."""
        from src.evaluation.llm_eval import (
            load_qa_pairs,
            evaluate_single,
            JUDGE_PROMPTS,
            JudgeScores,
        )

    def test_judge_prompts_have_required_fields(self):
        """All judge prompts should have instructions and template."""
        from src.evaluation.llm_eval import JUDGE_PROMPTS

        for name, config in JUDGE_PROMPTS.items():
            assert "instructions" in config, f"{name} missing instructions"
            assert "template" in config, f"{name} missing template"
            # Template should have placeholders
            assert "{question}" in config["template"]
            assert "{context}" in config["template"]
            assert "{answer}" in config["template"]

    def test_judge_scores_model(self):
        """JudgeScores Pydantic model should validate correctly."""
        from src.evaluation.llm_eval import JudgeScores

        scores = JudgeScores(faithfulness=5, relevance=4, coherence=5, explanation="Good answer")
        assert scores.faithfulness == 5
        assert scores.relevance == 4

    def test_agent_eval_importable(self):
        """agent_eval module should be importable."""
        from src.evaluation.agent_eval import (
            load_qa_pairs,
            retrieval_accuracy,
            create_comparison_chart,
        )

    def test_retrieval_accuracy_function(self):
        """retrieval_accuracy should compute hit rate correctly."""
        from src.evaluation.agent_eval import retrieval_accuracy

        # Create a simple search function that returns the correct doc
        def perfect_search(query, num_results=5):
            return [{"id": "42", "content": "answer"}, {"id": "1", "content": "other"}]

        questions = [
            {"question": "q1", "answer": "a1", "id": 42},
            {"question": "q2", "answer": "a2", "id": 42},
        ]
        result = retrieval_accuracy(perfect_search, questions, k=5)
        assert result["hit_rate"] == 1.0
        assert result["hits"] == 2
        assert result["total"] == 2

    def test_load_ground_truth(self):
        """load_ground_truth should load qa.jsonl correctly."""
        from src.evaluation.retrieval_eval import load_ground_truth

        qa_path = DATA_DIR / "qa.jsonl"
        questions = load_ground_truth(str(qa_path))
        # Floor relaxed from >= 900 (rag-mini-wikipedia) to >= 250: default dev
        # subset = 50 records × 5 LLM-generated pairs (user directive 2026-08-07).
        assert len(questions) >= 250
        assert "question" in questions[0]
        assert "id" in questions[0]


# ===========================================================================
# PHASE 9: Docker Configuration
# ===========================================================================


class TestDockerConfiguration:
    """Verify Docker files are valid and deployment-ready."""

    def test_dockerfile_exists(self):
        """Dockerfile must exist at project root."""
        assert (PROJECT_ROOT / "Dockerfile").exists()

    def test_dockerfile_uses_official_python(self):
        """Dockerfile should use official Python image."""
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "FROM python:" in dockerfile

    def test_dockerfile_installs_uv(self):
        """Dockerfile should install uv for package management."""
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "uv" in dockerfile.lower()

    def test_dockerfile_has_healthcheck(self):
        """Dockerfile should define a healthcheck."""
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "HEALTHCHECK" in dockerfile

    def test_dockerfile_exposes_8501(self):
        """Dockerfile should expose Streamlit port 8501."""
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "EXPOSE 8501" in dockerfile

    def test_dockerfile_sets_pythonpath(self):
        """Dockerfile should set PYTHONPATH for module imports."""
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "PYTHONPATH" in dockerfile

    def test_docker_compose_exists(self):
        """docker-compose.yml must exist."""
        assert (PROJECT_ROOT / "docker-compose.yml").exists()

    def test_docker_compose_has_services(self):
        """docker-compose.yml should define app and postgres services."""
        # Parse as YAML-like check (just verify structure)
        compose = (PROJECT_ROOT / "docker-compose.yml").read_text()
        assert "services:" in compose
        assert "postgres:" in compose
        assert "app:" in compose

    def test_docker_compose_exposes_port(self):
        """docker-compose.yml should map port 8501."""
        compose = (PROJECT_ROOT / "docker-compose.yml").read_text()
        assert "8501" in compose

    def test_docker_compose_has_healthcheck(self):
        """docker-compose.yml should have health checks."""
        compose = (PROJECT_ROOT / "docker-compose.yml").read_text()
        assert "healthcheck:" in compose

    def test_entrypoint_script_exists(self):
        """docker/entrypoint.sh must exist."""
        assert (PROJECT_ROOT / "docker" / "entrypoint.sh").exists()

    def test_entrypoint_script_is_executable_bash(self):
        """entrypoint.sh should be a bash script."""
        entrypoint = (PROJECT_ROOT / "docker" / "entrypoint.sh").read_text()
        assert "#!/bin/bash" in entrypoint
        assert "set -" in entrypoint  # strict mode

    def test_entrypoint_has_pipeline_steps(self):
        """entrypoint.sh should orchestrate: download → chunk → index → monitor → streamlit."""
        entrypoint = (PROJECT_ROOT / "docker" / "entrypoint.sh").read_text()
        assert "ingest" in entrypoint or "download" in entrypoint.lower()
        assert "chunker" in entrypoint or "chunk" in entrypoint.lower()
        assert "HybridSearch" in entrypoint or "hybrid" in entrypoint.lower()
        assert "streamlit" in entrypoint.lower()

    def test_pyproject_has_required_deps(self):
        """pyproject.toml should list key dependencies."""
        pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()
        assert "openai" in pyproject
        assert "onnxruntime" in pyproject
        assert "tokenizers" in pyproject
        assert "streamlit" in pyproject
        assert "minsearch" in pyproject
        assert "opentelemetry" in pyproject

    def test_pyproject_has_pytest_in_dev(self):
        """pyproject.toml should include pytest in dev dependencies."""
        pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()
        assert "pytest" in pyproject


# ===========================================================================
# PHASE 10: Full Pipeline Integration (End-to-End)
# ===========================================================================


class TestFullPipeline:
    """End-to-end integration: ingestion data → chunking → search → RAG → agent."""

    @pytest.fixture(scope="class")
    def full_pipeline(self):
        """Set up the full pipeline (search index + RAG + agent)."""
        from src.search.hybrid import HybridSearch
        from src.rag.pipeline import RAGBase
        from src.rag.agent import RAGAgent

        search_index = HybridSearch(documents_path=CHUNKS_DIR / "documents.jsonl")
        return search_index

    def test_data_to_search_pipeline(self, full_pipeline):
        """Verify data flows from ingestion → chunking → search index."""
        # Verify data exists
        assert (DATA_DIR / "corpus.jsonl").exists()
        assert (DATA_DIR / "qa.jsonl").exists()
        assert (CHUNKS_DIR / "documents.jsonl").exists()

        # Verify search index loaded data
        assert len(full_pipeline.documents) >= 3000

        # Verify search returns results
        results = full_pipeline.search("machine learning", num_results=3)
        assert len(results) > 0

    def test_search_to_rag_pipeline(self, full_pipeline):
        """Verify search results flow into RAG context and prompt."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = "Machine learning is a subset of AI."
        mock_client.responses.create.return_value = mock_response

        from src.rag.pipeline import RAGBase

        rag = RAGBase(search_index=full_pipeline, llm_client=mock_client)

        # Search
        results = rag.search("What is machine learning?")
        assert len(results) > 0

        # Build context
        context = rag.build_context(results)
        assert len(context) > 0

        # Build prompt
        prompt = rag.build_prompt("What is machine learning?", results)
        assert "machine learning" in prompt.lower()

        # Generate answer
        answer = rag.rag("What is machine learning?")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_full_agent_loop(self, full_pipeline):
        """Full agent loop: search → analyze → answer (with mocked LLM)."""
        mock_client = MagicMock()

        # Analysis response
        analysis_response = MagicMock()
        analysis_response.output_text = json.dumps({
            "sufficient": True,
            "reason": "Results contain information about machine learning",
            "reformulated_query": "",
            "off_topic": False,
            "off_topic_reason": "",
        })

        # Answer response
        answer_response = MagicMock()
        answer_response.output_text = (
            "Machine learning is a branch of artificial intelligence that enables "
            "systems to learn from data."
        )

        mock_client.responses.create.side_effect = [analysis_response, answer_response]

        from src.rag.agent import RAGAgent

        agent = RAGAgent(search_index=full_pipeline, llm_client=mock_client)
        result = agent.run("What is machine learning?")

        # Verify complete pipeline output
        assert "answer" in result
        assert "searches" in result
        assert "iterations" in result
        assert len(result["answer"]) > 0
        assert result["iterations"] >= 1
        assert result["searches"][0].analysis is not None

    def test_full_agent_with_feedback(self, full_pipeline):
        """Full pipeline with monitoring: agent run → trace → feedback."""
        from src.monitoring.tracer import SQLiteSpanExporter, TracedRAGAgent, get_trace_stats
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_traces.db"

            exporter = SQLiteSpanExporter(db_path=db_path)
            provider = TracerProvider()
            provider.add_span_processor(SimpleSpanProcessor(exporter))
            tracer = provider.get_tracer("test_full_feedback")

            mock_inner_agent = MagicMock()
            mock_inner_agent.run.return_value = {
                "answer": "ML is a subset of AI.",
                "searches": [],
                "iterations": 1,
            }

            traced_agent = TracedRAGAgent(agent=mock_inner_agent, tracer=tracer)
            result = traced_agent.run("What is ML?")

            assert result["answer"] == "ML is a subset of AI."

            exporter.force_flush()
            exporter.shutdown()

        stats = get_trace_stats(db_path=db_path)
        assert stats["total_traces"] >= 1

    def test_postgres_export_opt_in_via_env(self, monkeypatch, tmp_path):
        """Without POSTGRES_HOST, tracer setup must not require Postgres."""
        from src.monitoring.tracer import TracerSetup, _postgres_config

        monkeypatch.delenv("POSTGRES_HOST", raising=False)
        for var in ("POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER",
                    "POSTGRES_PASSWORD"):
            monkeypatch.delenv(var, raising=False)

        assert _postgres_config() is None
        setup = TracerSetup()
        assert setup.postgres_exporter is None
        setup.shutdown()

    def test_postgres_down_does_not_break_tracer(self, monkeypatch):
        """Postgres unreachable -> SQLite tracing still works, no crash."""
        from src.monitoring.tracer import TracerSetup, _postgres_config

        monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
        monkeypatch.setenv("POSTGRES_PORT", "59999")
        monkeypatch.setenv("POSTGRES_DB", "nonexistent")
        monkeypatch.setenv("POSTGRES_USER", "nonexistent")
        monkeypatch.setenv("POSTGRES_PASSWORD", "nonexistent")

        assert _postgres_config() is not None
        setup = TracerSetup()  # must not raise even though Postgres is down
        assert setup.postgres_exporter is None
        setup.shutdown()

    def test_qa_pairs_match_search(self, full_pipeline):
        """A sample of Q&A questions should return results with valid document IDs."""
        qa_path = DATA_DIR / "qa.jsonl"
        questions = []
        with open(qa_path) as f:
            for i, line in enumerate(f):
                if i >= 20:
                    break
                questions.append(json.loads(line))

        valid_doc_ids = {str(d["id"]) for d in full_pipeline.documents}

        for q in questions:
            results = full_pipeline.search(q["question"], num_results=5)
            assert len(results) > 0, f"No results for: {q['question'][:60]}"
            for doc in results:
                assert doc["id"] in valid_doc_ids, f"Doc ID {doc['id']} not in index"
