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
import re
import sqlite3
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.llm_call_record import Usage
from src.rag.scoring import AgentResult
from src.search.search_records import PokemonDoc, WebResult

DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
EVAL_QA = PROJECT_ROOT / "evaluation" / "data" / "qa.jsonl"


class FakeConnection:
    """In-memory sqlite stand-in for psycopg connections (test seam).

    Emulates the psycopg surface the monitoring package uses (cursor(),
    %s placeholders, rowcount, fetchone/fetchall, commit/close) so tests never
    need a real Postgres. sqlite3 is thread-safe enough here because every
    statement runs under a lock.
    """

    def __init__(self):
        # check_same_thread=False: spans are exported from the thread that ends
        # them, which may differ from the thread that built the exporter.
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._rows = []
        self.rowcount = 0
        self.statements = []
        self._lock = threading.Lock()

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        with self._lock:
            sql = sql.replace("%s", "?")
            # Postgres SERIAL is not a rowid alias in sqlite, so RETURNING id
            # would yield NULL — make it a real autoincrement column.
            sql = sql.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
            if "ADD COLUMN IF NOT EXISTS" in sql:
                # sqlite has no ADD COLUMN IF NOT EXISTS — skip when present.
                match = re.match(
                    r"ALTER TABLE (\S+) ADD COLUMN IF NOT EXISTS (\S+)", sql
                )
                table, column = match.group(1), match.group(2)
                columns = {
                    row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")
                }
                if column in columns:
                    self.statements.append(sql)
                    return self
                sql = sql.replace("ADD COLUMN IF NOT EXISTS", "ADD COLUMN")
            self.statements.append(sql)
            cur = self._conn.execute(sql, params or ())
            self.rowcount = cur.rowcount if hasattr(cur, "rowcount") else 0
            self._rows = cur.fetchall()
            return self

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows

    def commit(self):
        return None

    def rollback(self):
        return None

    def close(self):
        return None


def make_fake_db(monkeypatch, fake=None):
    if fake is None:
        fake = FakeConnection()
    monkeypatch.setattr("monitoring.db_init.psycopg.connect", lambda **kwargs: fake)
    return fake


class StubSearchIndex:
    """Search-index stand-in for the guardrail tests (todo 4).

    Exposes the same ``.search(query, num_results) -> list[dict]`` contract as
    HybridSearch without loading the ONNX embedder or any data file, so the
    guardrail tests run independently of the data/model artifacts (todo 1).
    """

    def __init__(self, documents=None):
        self.documents = (
            documents
            if documents is not None
            else [
                {
                    "id": 25,
                    "name": "Pikachu",
                    "types": ["Electric"],
                    "generation": "gen-i",
                    "stats": {
                        "hp": 35,
                        "attack": 55,
                        "defense": 40,
                        "sp_attack": 50,
                        "sp_defense": 50,
                        "speed": 90,
                        "base_stat_total": 320,
                    },
                    "height_m": 0.4,
                    "weight_kg": 6.0,
                    "abilities": ["static"],
                    "hidden_ability": "lightning-rod",
                    "egg_groups": ["field", "fairy"],
                    "color": "yellow",
                    "shape": "quadruped",
                    "habitat": "forest",
                    "growth_rate": "medium",
                    "capture_rate": 190,
                    "base_happiness": 70,
                    "base_experience": 112,
                    "genus": "Mouse Pokémon",
                    "is_legendary": False,
                    "is_mythical": False,
                    "is_baby": False,
                    "evolution_chain_id": 10,
                    "flavor_text": (
                        "When several of these POKéMON gather, their electricity "
                        "could build and cause lightning storms."
                    ),
                    "sprite_url": (
                        "https://raw.githubusercontent.com/PokeAPI/sprites/"
                        "master/sprites/pokemon/25.png"
                    ),
                    "evolves_from": None,
                    "evolves_into": ["Raichu"],
                    "type_effectiveness": {"electric": 0.5},
                    "search_text": (
                        "Pokémon: Pikachu (#25)\n"
                        "Types: Electric\n"
                        "Stats: hp 35, attack 55, defense 40, sp. attack 50, "
                        "sp. defense 50, speed 90, total 320\n"
                        "Type effectiveness: electric 0.5\n"
                        "Flavor text: When several of these POKéMON gather, their "
                        "electricity could build and cause lightning storms."
                    ),
                    "score": 1.0,
                },
                {
                    "id": 6,
                    "name": "Charizard",
                    "types": ["Fire", "Flying"],
                    "generation": "gen-i",
                    "stats": {
                        "hp": 78,
                        "attack": 84,
                        "defense": 78,
                        "sp_attack": 109,
                        "sp_defense": 85,
                        "speed": 100,
                        "base_stat_total": 534,
                    },
                    "height_m": 1.7,
                    "weight_kg": 90.5,
                    "abilities": ["blaze"],
                    "hidden_ability": "solar-power",
                    "egg_groups": ["monster", "dragon"],
                    "color": "red",
                    "shape": "upright",
                    "habitat": "mountain",
                    "growth_rate": "medium-slow",
                    "capture_rate": 45,
                    "base_happiness": 70,
                    "base_experience": 240,
                    "genus": "Flame Pokémon",
                    "is_legendary": False,
                    "is_mythical": False,
                    "is_baby": False,
                    "evolution_chain_id": 2,
                    "flavor_text": (
                        "It spits fire that is hot enough to melt boulders."
                    ),
                    "sprite_url": (
                        "https://raw.githubusercontent.com/PokeAPI/sprites/"
                        "master/sprites/pokemon/6.png"
                    ),
                    "evolves_from": "Charmeleon",
                    "evolves_into": [],
                    "type_effectiveness": {"rock": 4.0, "ground": 0.0},
                    "search_text": (
                        "Pokémon: Charizard (#6)\n"
                        "Types: Fire, Flying\n"
                        "Stats: hp 78, attack 84, defense 78, sp. attack 109, "
                        "sp. defense 85, speed 100, total 534\n"
                        "Type effectiveness: rock 4.0, ground 0.0\n"
                        "Flavor text: It spits fire that is hot enough to melt "
                        "boulders."
                    ),
                    "score": 1.0,
                },
            ]
        )

    def search(self, query, num_results=5):
        return [PokemonDoc.from_dict(doc) for doc in self.documents[:num_results]]


# ===========================================================================
# PHASE 1: Data Ingestion & Chunking
# ===========================================================================


class TestDataIngestion:
    def test_documents_file_exists(self):
        docs_path = CHUNKS_DIR / "documents.jsonl"
        assert docs_path.exists(), f"Missing {docs_path}"

    def test_qa_file_exists(self):
        qa_path = EVAL_QA
        assert qa_path.exists(), f"Missing {qa_path}"

    def test_pokemon_records_are_valid(self):
        docs_path = CHUNKS_DIR / "documents.jsonl"
        records = []
        with open(docs_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                if doc.get("kind") == "type_chart":
                    continue
                assert "id" in doc, f"Record {i} missing 'id' field"
                assert "name" in doc, f"Record {i} missing 'name' field"
                assert "search_text" in doc, f"Record {i} missing 'search_text' field"
                assert isinstance(doc["id"], int), f"Record {i} id not an int"
                assert "chunk_id" in doc, f"Record {i} missing 'chunk_id' field"
                assert "start" in doc, f"Record {i} missing 'start' field"
                records.append(doc)
        # Full dataset now (CSV swap): all 1,350 Pokémon, each chunked into
        # multiple docs — count distinct parent ids, not rows.
        assert len({r["id"] for r in records}) == 1350, (
            f"Expected 1350 distinct Pokémon, got {len({r['id'] for r in records})}"
        )

    def test_qa_records_are_valid(self):
        qa_path = EVAL_QA
        records = []
        with open(qa_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                assert "question" in record, f"Record {i} missing 'question' field"
                assert "document" in record, f"Record {i} missing 'document' field"
                # Ground truth links a question to the document containing its
                # answer; no LLM-written answer field.
                assert "answer" not in record, (
                    f"Record {i} has unexpected 'answer' field"
                )
                records.append(record)
        # Floor relaxed from >= 900 (rag-mini-wikipedia) to >= 250: the default
        # dev subset generates a coverage-sampled 50 records × 5 questions (user directive 2026-08-09).
        assert len(records) >= 250, (
            f"Expected >= 250 ground-truth questions, got {len(records)}"
        )

    def test_chunker_output_exists(self):
        docs_path = CHUNKS_DIR / "documents.jsonl"
        assert docs_path.exists(), f"Missing {docs_path}"

    def test_chunked_documents_are_valid(self):
        docs_path = CHUNKS_DIR / "documents.jsonl"
        count = 0
        chart_count = 0
        pokemon_chunk_count = 0
        with open(docs_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                assert "id" in doc, f"Doc {i} missing 'id'"
                assert "search_text" in doc, f"Doc {i} missing 'search_text'"
                assert isinstance(doc["search_text"], str), (
                    f"Doc {i} search_text not a string"
                )
                assert len(doc["search_text"]) > 0, f"Doc {i} has empty search_text"
                if isinstance(doc["id"], str):
                    assert doc.get("kind") == "type_chart", (
                        f"Chart doc {i} missing kind"
                    )
                    chart_count += 1
                else:
                    assert "name" in doc, f"Doc {i} missing 'name'"
                    assert "chunk_id" in doc, f"Doc {i} missing 'chunk_id'"
                    assert "start" in doc, f"Doc {i} missing 'start'"
                    pokemon_chunk_count += 1
                count += 1
        assert chart_count == 18, f"Expected 18 chart docs, got {chart_count}"
        assert pokemon_chunk_count >= 1350, (
            f"Expected >= 1350 Pokémon chunks, got {pokemon_chunk_count}"
        )


# ===========================================================================
# PHASE 2: Chunking Pipeline Unit Tests
# ===========================================================================


class TestChunkingPipeline:
    @staticmethod
    def pokemon_records():
        records = []
        seen = set()
        with open(CHUNKS_DIR / "documents.jsonl", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)
                if doc.get("kind") != "type_chart" and doc["id"] not in seen:
                    seen.add(doc["id"])
                    records.append(doc)
        return records

    @staticmethod
    def record_by_id(records, id_):
        return next(r for r in records if r["id"] == id_)

    @staticmethod
    def full_records():
        from src.data.csv_parsers import load_raw_rows, parse_row

        return sorted((parse_row(r) for r in load_raw_rows()), key=lambda r: r["id"])

    @staticmethod
    def chart():
        from src.data.type_chart import TypeChart

        return TypeChart.load(PROJECT_ROOT / "data" / "raw" / "pokemon_types.csv")

    def test_type_effectiveness_bulbasaur(self):
        chart = self.chart()
        bulbasaur = self.record_by_id(self.pokemon_records(), 1)
        eff = chart.effectiveness(bulbasaur)
        assert eff["fire"] == 2.0
        assert eff["grass"] == 0.25
        assert eff["water"] == 0.5

    def test_evolution_link_ivysaur(self):
        from src.data.evolution import EvolutionChain

        # Evolution linkage needs evolution_chain_id, which the trimmed corpus
        # chunks drop — build the full records from the raw CSVs instead.
        records = self.full_records()
        chains = EvolutionChain.build_map(records)
        ivysaur = self.record_by_id(records, 2)
        chain = chains[ivysaur["evolution_chain_id"]]
        evolves_from, evolves_into = EvolutionChain.link(ivysaur, chain)
        assert evolves_from == "Bulbasaur"
        assert evolves_into == ["Venusaur"]

    def test_alt_form_has_no_evolution_link(self):
        from src.data.evolution import EvolutionChain

        records = self.pokemon_records()
        alt = self.record_by_id(records, 10001)
        evolves_from, evolves_into = EvolutionChain.link(alt, None)
        assert evolves_from is None
        assert evolves_into == []

    def test_build_pokemon_doc_derives_keys(self):
        from src.data.pokemon_doc_builder import PokemonDocBuilder
        from src.data.evolution import EvolutionChain

        records = self.full_records()
        chart = self.chart()
        chains = EvolutionChain.build_map(records)
        ivysaur = self.record_by_id(records, 2)
        doc = PokemonDocBuilder().build(
            ivysaur, chart, chains[ivysaur["evolution_chain_id"]]
        )
        assert doc["id"] == 2  # int id preserved
        assert "Evolution chain" in doc["search_text"]
        assert "fire 2.0" in doc["search_text"]
        assert "Type effectiveness:" in doc["search_text"]
        assert "Flavor text:" in doc["search_text"]

    def test_type_chart_doc_shape(self):
        chart = self.chart()
        fire = chart.doc("fire")
        assert fire["id"] == "type_fire"
        assert fire["kind"] == "type_chart"
        assert fire["type"] == "Fire"
        assert "Fire moves deal 2x damage" in fire["search_text"]
        assert "take 2x damage from" in fire["search_text"]


# ===========================================================================
# PHASE 3: Search Index & Retrieval
# ===========================================================================


class TestHybridSearch:
    @pytest.fixture(scope="class")
    def search_index(self):
        from src.search.hybrid_search import HybridSearch

        docs_path = CHUNKS_DIR / "documents.jsonl"
        return HybridSearch(documents_path=docs_path)

    def test_index_loads_documents(self, search_index):
        # Floor relaxed from >= 3000 (rag-mini-wikipedia) to >= 50: the default
        # dev subset is 50 Pokémon, one document per Pokémon (user directive
        # 2026-08-07) — 3000 can never hold on the dev subset.
        assert len(search_index.documents) >= 50

    def test_keyword_and_vector_search_return_results(self, search_index):
        kw = search_index.keyword_search("pikachu", num_results=5)
        assert len(kw) > 0 and len(kw) <= 5
        for doc in kw:
            assert hasattr(doc, "id") and hasattr(doc, "search_text")
        vec = search_index.vector_search("electric pokemon stats", num_results=5)
        assert len(vec) > 0 and len(vec) <= 5
        for doc in vec:
            assert hasattr(doc, "id")

    def test_hybrid_search_returns_ranked_results(self, search_index):
        results = search_index.search("What are Pikachu's stats?", num_results=5)
        assert len(results) > 0 and len(results) <= 5
        for doc in results:
            assert hasattr(doc, "id")
            assert hasattr(doc, "score"), "Hybrid results should have a 'score' field"
            assert isinstance(doc.score, (int, float))
        scores = [doc.score for doc in results]
        assert scores == sorted(scores, reverse=True), (
            "Scores should be in descending order"
        )

    def test_relevance_filter_keeps_only_relevant_chunks(self):
        from src.search.hybrid_search import HybridSearch

        index = HybridSearch(
            documents_path=CHUNKS_DIR / "documents.jsonl", relevance_threshold=0.3
        )
        results = index.search("What are Pikachu's stats?", num_results=5)
        query_vector = index.embedder.encode(
            "What are Pikachu's stats?", normalize=True
        )
        cosines = index.embeddings @ query_vector
        chunk_cosine = {
            d.get("chunk_id") or d["id"]: float(c)
            for d, c in zip(index.documents, cosines)
        }
        assert len(results) > 0
        for r in results:
            assert chunk_cosine[r.chunk_id] >= 0.3

    def test_relevance_filter_can_return_empty(self):
        from src.search.hybrid_search import HybridSearch

        # A near-1.0 threshold drops every chunk, exercising the empty-results
        # path the agent must handle gracefully.
        index = HybridSearch(
            documents_path=CHUNKS_DIR / "documents.jsonl", relevance_threshold=0.99
        )
        results = index.search("What are Pikachu's stats?", num_results=5)
        assert len(results) == 0

    def test_rrf_fusion(self):
        from src.search.hybrid_search import rrf

        list1 = [
            {"id": "a", "content": "1"},
            {"id": "b", "content": "2"},
            {"id": "c", "content": "3"},
        ]
        list2 = [
            {"id": "a", "content": "1"},
            {"id": "d", "content": "4"},
            {"id": "e", "content": "5"},
        ]

        fused = rrf([list1, list2], num_results=3)
        assert len(fused) <= 3
        # Document "a" appears at rank 0 in both lists → highest score
        assert fused[0]["id"] == "a"
        # All fused results should have scores
        for doc in fused:
            assert "score" in doc
        # Scores should be descending
        scores = [d["score"] for d in fused]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_query(self, search_index):
        results = search_index.search("", num_results=3)
        assert isinstance(results, list)

    def test_reranker_falls_back_when_model_missing(self, monkeypatch):
        from src.search.reranker import Reranker

        monkeypatch.setenv("RERANKER_MODEL_PATH", "/nonexistent/reranker")
        reranker = Reranker()
        assert reranker.available is False
        docs = [{"id": "a", "search_text": "one"}, {"id": "b", "search_text": "two"}]
        assert reranker.rerank("query", docs) == docs

    def test_reranker_reorders_by_cross_encoder_score(self, monkeypatch):
        import numpy as np

        from src.search.reranker import Reranker

        class FakeSession:
            def run(self, _, feed):
                # Higher logit for the second doc → it should rank first.
                return [np.array([[0.0], [3.0]], dtype=np.float32)]

        class FakeTokenizer:
            def enable_truncation(self, **kwargs):
                pass

            def enable_padding(self, **kwargs):
                pass

            def encode_batch(self, pairs):
                return [
                    type(
                        "E", (), {"ids": [0], "attention_mask": [1], "type_ids": [0]}
                    )()
                    for _ in pairs
                ]

        reranker = Reranker()
        reranker.available = True
        reranker._tokenizer = FakeTokenizer()
        reranker._session = FakeSession()
        docs = [{"id": "a", "search_text": "one"}, {"id": "b", "search_text": "two"}]
        ranked = reranker.rerank("query", docs)
        assert [d["id"] for d in ranked] == ["b", "a"]


# ===========================================================================
# PHASE 4: RAG Pipeline
# ===========================================================================


class TestRAGPipeline:
    @pytest.fixture(scope="class")
    def search_index(self):
        from src.search.hybrid_search import HybridSearch

        return HybridSearch(documents_path=CHUNKS_DIR / "documents.jsonl")

    @pytest.fixture
    def mock_llm_client(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = "This is a mock answer about Pikachu."
        mock_client.client.responses.create.return_value = mock_response
        return mock_client

    def test_rag_base_search(self, search_index):
        from src.rag.rag_base import RAGBase

        rag = RAGBase(search_index=search_index)
        results = rag.search("Which Pokémon are weak to fire?")
        assert len(results) > 0
        assert hasattr(results[0], "search_text")

    def test_rag_build_context(self, search_index):
        from src.rag.rag_base import RAGBase

        rag = RAGBase(search_index=search_index)
        results = rag.search("grass type pokemon", num_results=3)
        context = rag.build_context(results)
        assert isinstance(context, str)
        assert len(context) > 0
        for doc in results:
            assert doc.search_text in context

    def test_rag_build_prompt(self, search_index):
        from src.rag.rag_base import RAGBase

        rag = RAGBase(search_index=search_index)
        results = rag.search("electric pokemon", num_results=3)
        prompt = rag.build_prompt("Which Pokémon are weak to fire?", results)
        assert "Which Pokémon are weak to fire?" in prompt
        assert "CONTEXT" in prompt

    def test_rag_llm_call(self, search_index, mock_llm_client):
        from src.rag.rag_base import RAGBase

        rag = RAGBase(search_index=search_index, llm_client=mock_llm_client)
        answer = rag.llm("Test prompt")
        assert answer == "This is a mock answer about Pikachu."
        mock_llm_client.client.responses.create.assert_called_once()

    def test_rag_full_pipeline(self, search_index, mock_llm_client):
        from src.rag.rag_base import RAGBase

        rag = RAGBase(search_index=search_index, llm_client=mock_llm_client)
        answer = rag.rag("What are Pikachu's stats?")
        assert isinstance(answer, str)
        assert len(answer) > 0
        mock_llm_client.client.responses.create.assert_called()


# ===========================================================================
# PHASE 5: Agent Tool Loop (LLM-driven tool use)
# ===========================================================================


class TestAgentToolLoop:
    LOCAL = "search_local_knowledge_base"
    WEB = "search_bulbapedia"

    @staticmethod
    def function_call(name, arguments, call_id="call_1"):
        item = MagicMock()
        item.type = "function_call"
        item.name = name
        item.arguments = json.dumps(arguments)
        item.call_id = call_id
        return item

    @staticmethod
    def response(text="", *calls):
        response = MagicMock()
        response.output = list(calls)
        response.output_text = text
        return response

    @staticmethod
    def script_client(*responses):
        mock_client = MagicMock()
        mock_client.client.responses.create.side_effect = list(responses)
        return mock_client

    @staticmethod
    def agent(llm_client, **kwargs):
        from src.rag.rag_agent import RAGAgent
        from src.search.embedder import Embedder

        index = StubSearchIndex()
        index.embedder = Embedder()
        return RAGAgent(search_index=index, llm_client=llm_client, **kwargs)

    @staticmethod
    def web_fake():
        calls = []

        def fake(query, num_results=5):
            calls.append((query, num_results))
            return [
                WebResult(
                    title="Ikue Otani",
                    url="u",
                    snippet="Ikue Otani voiced Pikachu in the anime",
                )
            ]

        fake.calls = calls
        return fake

    def test_tool_less_memory_answer_escalates_to_web_search(self, monkeypatch):
        web_fake = self.web_fake()
        monkeypatch.setattr("src.search.web_search.web_search", web_fake)
        agent = self.agent(
            self.script_client(
                self.response(text="Pikachu is Electric."),
                self.response("", self.function_call(self.WEB, {"query": "pikachu"})),
                self.response(text="Ikue Otani voiced Pikachu in the anime."),
            )
        )
        result = agent.run("Who voiced Pikachu in the anime?")

        # The tool-less memory answer was ungrounded (no results to ground
        # against); the forced web retry grounded the final answer.
        assert result.rejected is False
        assert result.answer == "Ikue Otani voiced Pikachu in the anime."
        assert result.source == "web"
        assert web_fake.calls == [("pikachu", 5)]

    def test_local_only_ungrounded_answer_escalates_to_web_search(self, monkeypatch):
        web_fake = self.web_fake()
        monkeypatch.setattr("src.search.web_search.web_search", web_fake)
        agent = self.agent(
            self.script_client(
                self.response(
                    "", self.function_call(self.LOCAL, {"query": "pikachu stats"})
                ),
                self.response(text="The answer is 42"),
                self.response("", self.function_call(self.WEB, {"query": "pikachu"})),
                self.response(
                    text="Pikachu has HP 35, Attack 55, Defense 40, Speed 90."
                ),
            )
        )
        result = agent.run("What are Pikachu's stats?")

        # The fabricated local-only answer failed the grounding gate; the
        # forced web retry ran, and the answer now grounds via the local doc.
        assert result.rejected is False
        assert result.source == "local+web"
        assert web_fake.calls == [("pikachu", 5)]
        # The rejected attempt is retained with its failing answer for
        # monitoring/troubleshooting.
        assert result.escalated is True
        assert [g.rejected for g in agent.gate_history] == [True, False]
        assert agent.gate_history[0].rejected_answer == "The answer is 42"

    def test_local_tool_then_answer(self, monkeypatch):
        web_fake = self.web_fake()
        monkeypatch.setattr("src.search.web_search.web_search", web_fake)
        agent = self.agent(
            self.script_client(
                self.response(
                    "", self.function_call(self.LOCAL, {"query": "pikachu stats"})
                ),
                self.response(
                    text="Pikachu has HP 35, Attack 55, Defense 40, Speed 90."
                ),
            )
        )
        result = agent.run("What are Pikachu's stats?")

        assert result.source == "local"
        assert result.iterations == 1
        assert len(result.searches) == 1
        assert result.searches[0].search_query == "pikachu stats"
        assert result.searches[0].source == "local"
        assert result.confidence > 0.65  # grounding cosine vs the Pikachu doc
        assert result.relevance > 0.6  # query-answer cosine
        assert result.escalated is False
        assert [g.rejected for g in agent.gate_history] == [False]
        assert web_fake.calls == []

    def test_local_search_records_query_and_falls_back_to_question(self):
        agent = self.agent(
            self.script_client(
                self.response(
                    "", self.function_call(self.LOCAL, {"query": "pikachu stats"})
                ),
                self.response("", self.function_call(self.LOCAL, {})),
                self.response(
                    text="Pikachu has HP 35, Attack 55, Defense 40, Speed 90."
                ),
            )
        )
        result = agent.run("What are Pikachu's stats?")

        # The model's keyword is recorded per search; a call without a query
        # argument falls back to the original question.
        assert result.searches[0].search_query == "pikachu stats"
        assert result.searches[1].search_query == "What are Pikachu's stats?"

    def test_local_then_web_then_answer(self, monkeypatch):
        web_fake = self.web_fake()
        monkeypatch.setattr("src.search.web_search.web_search", web_fake)
        agent = self.agent(
            self.script_client(
                self.response("", self.function_call(self.LOCAL, {"query": "pikachu"})),
                self.response(
                    "",
                    self.function_call(
                        self.WEB, {"query": "Pikachu voice actor anime"}
                    ),
                ),
                self.response(text="Ikue Otani voiced Pikachu."),
            )
        )
        result = agent.run("Who voiced Pikachu?")

        assert result.source == "local+web"
        assert result.iterations == 2
        assert [s.source for s in result.searches] == ["local", "web"]
        assert result.searches[1].search_query == "Pikachu voice actor anime"
        assert web_fake.calls[0][0] == "Pikachu voice actor anime"
        assert result.confidence is not None  # grounded via the snippet record
        assert result.relevance is not None

    def test_web_tool_failure_returns_empty_results(self, monkeypatch):
        from src.rag.prompts import REJECTION_MESSAGE

        def raise_error(query, num_results=5):
            raise RuntimeError("Tavily down")

        monkeypatch.setattr("src.search.web_search.web_search", raise_error)
        agent = self.agent(
            self.script_client(
                self.response("", self.function_call(self.WEB, {"query": "pikachu"})),
                self.response(text="No answer found."),
            )
        )
        result = agent.run("Needs web")

        assert result.searches[0].results == []
        # Empty tool results -> grounding 0 -> gated to rejection.
        assert result.rejected is True
        assert result.answer == REJECTION_MESSAGE

    def test_out_of_scope_replies_rejection_without_tools(self, monkeypatch):
        from src.rag.prompts import REJECTION_MESSAGE

        def no_web(query, num_results=5):
            raise AssertionError("tools must never run on an out-of-scope question")

        monkeypatch.setattr("src.search.web_search.web_search", no_web)
        agent = self.agent(self.script_client(self.response(text=REJECTION_MESSAGE)))
        result = agent.run("Who would win Charizard vs Blastoise?")

        assert result.rejected is True
        assert result.answer == REJECTION_MESSAGE
        assert result.searches == []
        assert result.source is None

    @pytest.mark.parametrize(
        "question",
        [
            "who would come out on top if my Pikachu and Charizard fought in a tournament bracket",
            "can you fix my game save",
            "help me with my Docker homework",
        ],
    )
    def test_paraphrased_out_of_scope(self, monkeypatch, question):
        from src.rag.prompts import REJECTION_MESSAGE

        agent = self.agent(self.script_client(self.response(text=REJECTION_MESSAGE)))
        result = agent.run(question)

        assert result.rejected is True
        assert result.answer == REJECTION_MESSAGE

    def test_loop_exhaustion_rejects(self):
        from src.rag.prompts import REJECTION_MESSAGE

        responses = [
            self.response(
                "", self.function_call(self.LOCAL, {"query": f"q{i}"}, call_id=f"c{i}")
            )
            for i in range(4)
        ]
        agent = self.agent(self.script_client(*responses), max_iterations=2)
        result = agent.run("Endless question")

        assert result.rejected is True
        assert result.answer == REJECTION_MESSAGE

    def test_empty_final_text_rejects(self):
        from src.rag.prompts import REJECTION_MESSAGE

        agent = self.agent(self.script_client(self.response(text="   ")))
        result = agent.run("Question")

        assert result.rejected is True
        assert result.answer == REJECTION_MESSAGE

    def test_llm_failure_rejects(self):
        from src.rag.prompts import REJECTION_MESSAGE

        mock_client = MagicMock()
        mock_client.client.responses.create.side_effect = RuntimeError("server down")
        agent = self.agent(mock_client)
        result = agent.run("Question")

        assert result.rejected is True
        assert result.answer == REJECTION_MESSAGE

    def test_fabricated_answer_is_gated(self, monkeypatch):
        from src.rag.prompts import REJECTION_MESSAGE

        web_fake = self.web_fake()
        monkeypatch.setattr("src.search.web_search.web_search", web_fake)
        agent = self.agent(
            self.script_client(
                self.response("", self.function_call(self.LOCAL, {"query": "pikachu"})),
                self.response(text="The answer is 42"),
                self.response("", self.function_call(self.WEB, {"query": "pikachu"})),
                self.response(text="The answer is 42"),
            )
        )
        result = agent.run("What are Pikachu's stats?")

        # Fabricated both times (local and web) -> gated; the single forced
        # web retry does not loop forever.
        assert result.rejected is True
        assert result.answer == REJECTION_MESSAGE
        assert web_fake.calls == [("pikachu", 5)]

    def test_confidence_threshold_param_is_enforced(self, monkeypatch):
        from src.rag.prompts import REJECTION_MESSAGE

        web_fake = self.web_fake()
        monkeypatch.setattr("src.search.web_search.web_search", web_fake)
        agent = self.agent(
            self.script_client(
                self.response("", self.function_call(self.LOCAL, {"query": "pikachu"})),
                self.response(
                    text="Pikachu has HP 35, Attack 55, Defense 40, Speed 90."
                ),
                self.response("", self.function_call(self.WEB, {"query": "pikachu"})),
                self.response(
                    text="Pikachu has HP 35, Attack 55, Defense 40, Speed 90."
                ),
            ),
            confidence_threshold=0.9,
        )
        result = agent.run("What are Pikachu's stats?")

        # Grounding cosine ~0.78 < 0.9 even after the web retry -> rejected.
        assert result.rejected is True
        assert result.answer == REJECTION_MESSAGE
        assert web_fake.calls == [("pikachu", 5)]

    def test_web_tool_unlocks_after_first_llm_api_call(self, monkeypatch):
        from src.rag.tools import LOCAL_SEARCH_TOOL, TOOLS

        web_fake = self.web_fake()
        monkeypatch.setattr("src.search.web_search.web_search", web_fake)
        mock_client = MagicMock()
        mock_client.client.responses.create.side_effect = [
            self.response("", self.function_call(self.LOCAL, {"query": "pikachu"})),
            self.response(
                "", self.function_call(self.WEB, {"query": "Pikachu voice actor"})
            ),
            self.response(text="Ikue Otani voiced Pikachu in the anime"),
        ]
        agent = self.agent(mock_client)
        result = agent.run("Who voiced Pikachu?")

        calls = mock_client.client.responses.create.call_args_list
        assert [c.kwargs["tools"] for c in calls] == [
            [LOCAL_SEARCH_TOOL],  # iteration 0: local only
            TOOLS,  # iteration 1+: both
            TOOLS,
        ]
        assert result.rejected is False

    def test_empty_or_punctuation_input_rejected_without_llm_call(self):
        from src.rag.prompts import REJECTION_MESSAGE

        mock_client = MagicMock()
        mock_client.client.responses.create.side_effect = AssertionError(
            "LLM must not be called"
        )
        agent = self.agent(mock_client)

        for bad in ("", "   ", "???!!!", "....", "---"):
            result = agent.run(bad)
            assert result.rejected is True, repr(bad)
            assert result.answer == REJECTION_MESSAGE
            assert result.searches == []
            assert result.iterations == 0

    def test_single_word_query_reaches_llm(self):
        agent = self.agent(
            self.script_client(
                self.response("", self.function_call(self.LOCAL, {"query": "Pikachu"})),
                self.response(text="Pikachu is an Electric Pokémon."),
            )
        )

        result = agent.run("Pikachu")

        assert agent.llm_client.client.responses.create.called
        assert result.rejected is False

    def test_instructions_cover_capabilities_limitations_and_rejection(self):
        from src.rag.prompts import INSTRUCTIONS, REJECTION_MESSAGE

        assert "knowledge base" in INSTRUCTIONS
        assert "Bulbapedia" in INSTRUCTIONS
        for phrase in ("battle", "winner", "save", "cheat", "emulator", "non-Pokémon"):
            assert phrase in INSTRUCTIONS
        assert "never guess" in INSTRUCTIONS
        assert REJECTION_MESSAGE in INSTRUCTIONS
        assert "search_local_knowledge_base" in INSTRUCTIONS
        assert "search_bulbapedia" in INSTRUCTIONS


# ===========================================================================
# PHASE 6: Monitoring & Tracing
# ===========================================================================


class TestMonitoring:
    def test_tracer_export_writes_spans(self, monkeypatch):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from monitoring.span_exporter import PostgresSpanExporter

        fake = make_fake_db(monkeypatch)
        exporter = PostgresSpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test_export")

        with tracer.start_as_current_span("test.span") as span:
            span.set_attribute("query", "test query")

        exporter.force_flush()
        exporter.shutdown()

        fake.execute("SELECT name FROM spans")
        rows = fake.fetchall()
        assert ("test.span",) in rows

    def test_tracer_schema_has_required_columns(self, monkeypatch):
        from monitoring.span_exporter import PostgresSpanExporter

        fake = make_fake_db(monkeypatch)
        exporter = PostgresSpanExporter()
        exporter.shutdown()

        assert any("CREATE TABLE IF NOT EXISTS spans" in s for s in fake.statements)

    def test_tracer_records_spans(self, monkeypatch):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from monitoring.span_exporter import PostgresSpanExporter

        fake = make_fake_db(monkeypatch)
        exporter = PostgresSpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test_tracer_records")

        with tracer.start_as_current_span("test.span") as span:
            span.set_attribute("query", "test query")
            span.set_attribute("input_tokens", 100)
            span.set_attribute("output_tokens", 50)

        exporter.force_flush()
        exporter.shutdown()

        fake.execute("SELECT name, input_tokens, output_tokens FROM spans")
        rows = fake.fetchall()
        assert len(rows) >= 1
        assert rows[0][0] == "test.span"

    def test_tracer_cross_thread_export(self, monkeypatch):
        # The exporter must survive exports from a different thread than the
        # one that built it (Streamlit reruns the script from different
        # threads); the shared fake serializes statements under a lock.
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from monitoring.span_exporter import PostgresSpanExporter

        fake = make_fake_db(monkeypatch)
        exporter = PostgresSpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test_cross_thread")

        errors = []

        def end_span_in_other_thread():
            try:
                with tracer.start_as_current_span("cross.thread.span") as span:
                    span.set_attribute("query", "cross thread query")
            except Exception as exc:  # noqa: BLE001 — catching any cross-thread failure is the point of this test
                errors.append(exc)

        thread = threading.Thread(target=end_span_in_other_thread)
        thread.start()
        thread.join()

        exporter.force_flush()
        exporter.shutdown()

        assert errors == [], f"cross-thread export raised: {errors}"

        fake.execute("SELECT name FROM spans")
        rows = fake.fetchall()
        assert ("cross.thread.span",) in rows

    def test_get_tracer_single_setup_under_concurrency(self, monkeypatch):
        # Streamlit runs multiple sessions concurrently, so the lazy
        # TracerSetup singleton must initialize exactly once under a race.
        import monitoring.tracer as tracer_module

        created = []

        class CountingTracerSetup:
            def __init__(self):
                time.sleep(
                    0.005
                )  # model real TracerSetup overhead (exporter I/O) so the race window exists
                created.append(self)
                self.tracer = object()

        monkeypatch.setattr(tracer_module, "TracerSetup", CountingTracerSetup)
        monkeypatch.setattr(tracer_module, "default_setup", None)

        barrier = threading.Barrier(8)
        tracers = []

        def call_get_tracer():
            barrier.wait()
            tracers.append(tracer_module.get_tracer())

        threads = [threading.Thread(target=call_get_tracer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(created) == 1
        assert len({id(t) for t in tracers}) == 1

    def test_record_feedback(self, monkeypatch):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from monitoring.span_exporter import PostgresSpanExporter
        from monitoring.span_store import record_feedback

        fake = make_fake_db(monkeypatch)
        exporter = PostgresSpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test_record_feedback")

        with tracer.start_as_current_span("agent.run") as span:
            span.set_attribute("query", "test")
            sid = format(span.get_span_context().span_id, "016x")

        exporter.force_flush()
        exporter.shutdown()

        assert record_feedback(sid, "positive") is True

        fake.execute("SELECT span_id, query, feedback FROM spans")
        row = fake.fetchone()
        assert row == (sid, "test", "positive")

    def test_record_feedback_exact_span_attachment(self, monkeypatch):
        """Feedback must attach to the exact span in multi-message sessions.

        Two runs via run_with_feedback, then feedback on the FIRST span id
        only — the second span's feedback must remain NULL (review M3).
        """
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from monitoring.span_exporter import PostgresSpanExporter
        from monitoring.span_store import record_feedback
        from monitoring.tracer import TracedRAGAgent

        fake = make_fake_db(monkeypatch)
        exporter = PostgresSpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test_exact_span")

        mock_agent = MagicMock()
        mock_agent.run.side_effect = [
            AgentResult(
                answer="a1",
                searches=[],
                iterations=1,
                rejected=False,
                source=None,
                confidence=None,
                relevance=None,
            ),
            AgentResult(
                answer="a2",
                searches=[],
                iterations=1,
                rejected=False,
                source=None,
                confidence=None,
                relevance=None,
            ),
        ]
        traced = TracedRAGAgent(agent=mock_agent, tracer=tracer)
        _, first_span_id = traced.run_with_feedback("first query")
        _, second_span_id = traced.run_with_feedback("second query")
        assert first_span_id != second_span_id

        exporter.force_flush()
        exporter.shutdown()

        assert record_feedback(first_span_id, "positive") is True

        fake.execute("SELECT query, feedback FROM spans ORDER BY rowid")
        rows = fake.fetchall()
        assert rows == [("first query", "positive"), ("second query", None)]

    def test_record_feedback_requires_span_id(self, monkeypatch):
        from monitoring.span_store import record_feedback

        make_fake_db(monkeypatch)
        assert record_feedback("", "positive") is False

    def test_tracing_enabled_gate(self, monkeypatch):
        from monitoring.tracer import tracing_enabled

        monkeypatch.delenv("TRACING_ENABLED", raising=False)
        assert tracing_enabled() is True
        for disable in ("0", "false", "no", "off", ""):
            monkeypatch.setenv("TRACING_ENABLED", disable)
            assert tracing_enabled() is False
        monkeypatch.setenv("TRACING_ENABLED", "1")
        assert tracing_enabled() is True

    def test_get_trace_stats(self, monkeypatch):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from monitoring.span_exporter import PostgresSpanExporter
        from monitoring.span_store import get_trace_stats

        fake = make_fake_db(monkeypatch)
        exporter = PostgresSpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test_get_trace_stats")

        with tracer.start_as_current_span("agent.run") as span:
            span.set_attribute("input_tokens", 100)
            span.set_attribute("output_tokens", 50)

        exporter.force_flush()
        exporter.shutdown()

        stats = get_trace_stats()
        assert stats["total_traces"] >= 1
        assert "span_names" in stats
        assert stats["total_input_tokens"] >= 100
        assert stats["total_output_tokens"] >= 50

    def test_traced_ragent_run(self, monkeypatch):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from monitoring.span_exporter import PostgresSpanExporter
        from monitoring.span_store import get_trace_stats
        from monitoring.tracer import TracedRAGAgent

        fake = make_fake_db(monkeypatch)
        exporter = PostgresSpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test_traced_agent")

        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            answer="test answer",
            searches=[],
            iterations=1,
            rejected=False,
            source=None,
            confidence=None,
            relevance=None,
        )

        traced = TracedRAGAgent(agent=mock_agent, tracer=tracer)
        result = traced.run("test query")

        assert result.answer == "test answer"
        exporter.force_flush()
        exporter.shutdown()

        stats = get_trace_stats()
        assert stats["total_traces"] >= 1

    def test_traced_ragent_run_with_feedback_returns_span_id(self, monkeypatch):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from monitoring.span_exporter import PostgresSpanExporter
        from monitoring.tracer import TracedRAGAgent

        fake = make_fake_db(monkeypatch)
        exporter = PostgresSpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test_traced_agent_feedback")

        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            answer="test answer",
            searches=[],
            iterations=1,
            rejected=False,
            source=None,
            confidence=None,
            relevance=None,
        )

        traced = TracedRAGAgent(agent=mock_agent, tracer=tracer)
        result, sid = traced.run_with_feedback("test query")

        assert result.answer == "test answer"
        assert len(sid) == 16
        assert all(c in "0123456789abcdef" for c in sid)

        exporter.force_flush()
        exporter.shutdown()

        fake.execute("SELECT query, span_id FROM spans")
        row = fake.fetchone()
        assert row == ("test query", sid)


# ===========================================================================
# PHASE 8: Evaluation Script Execution
# ===========================================================================


class TestEvaluationScripts:
    def test_judge_prompts_have_required_fields(self):
        from evaluation.notebooks.share.judge_prompts import JUDGE_PROMPTS

        for name, config in JUDGE_PROMPTS.items():
            assert "instructions" in config, f"{name} missing instructions"
            assert "template" in config, f"{name} missing template"
            # Template should have placeholders
            assert "{question}" in config["template"]
            assert "{context}" in config["template"]
            assert "{answer}" in config["template"]


# ===========================================================================
# PHASE 9: Docker Configuration
# ===========================================================================


class TestDockerConfiguration:
    def test_dockerfile_exists(self):
        assert (PROJECT_ROOT / "deployment" / "Dockerfile").exists()

    def test_dockerfile_uses_official_python(self):
        dockerfile = (PROJECT_ROOT / "deployment" / "Dockerfile").read_text()
        assert "FROM python:" in dockerfile

    def test_dockerfile_installs_uv(self):
        dockerfile = (PROJECT_ROOT / "deployment" / "Dockerfile").read_text()
        assert "uv" in dockerfile.lower()

    def test_dockerfile_has_healthcheck(self):
        dockerfile = (PROJECT_ROOT / "deployment" / "Dockerfile").read_text()
        assert "HEALTHCHECK" in dockerfile

    def test_dockerfile_exposes_8501(self):
        dockerfile = (PROJECT_ROOT / "deployment" / "Dockerfile").read_text()
        assert "EXPOSE 8501" in dockerfile

    def test_dockerfile_sets_pythonpath(self):
        dockerfile = (PROJECT_ROOT / "deployment" / "Dockerfile").read_text()
        assert "PYTHONPATH" in dockerfile

    def test_docker_compose_exists(self):
        assert (PROJECT_ROOT / "docker-compose.yml").exists()

    def test_docker_compose_has_services(self):
        # Parse as YAML-like check (just verify structure)
        compose = (PROJECT_ROOT / "docker-compose.yml").read_text()
        assert "services:" in compose
        assert "postgres:" in compose
        assert "app:" in compose

    def test_docker_compose_exposes_port(self):
        compose = (PROJECT_ROOT / "docker-compose.yml").read_text()
        assert "8501" in compose

    def test_docker_compose_has_healthcheck(self):
        compose = (PROJECT_ROOT / "docker-compose.yml").read_text()
        assert "healthcheck:" in compose

    def test_entrypoint_script_exists(self):
        assert (PROJECT_ROOT / "deployment" / "entrypoint.sh").exists()

    def test_entrypoint_script_is_executable_bash(self):
        entrypoint = (PROJECT_ROOT / "deployment" / "entrypoint.sh").read_text()
        assert "#!/bin/bash" in entrypoint
        assert "set -" in entrypoint  # strict mode

    def test_entrypoint_has_pipeline_steps(self):
        entrypoint = (PROJECT_ROOT / "deployment" / "entrypoint.sh").read_text()
        assert "ingest" in entrypoint or "download" in entrypoint.lower()
        assert "chunker" in entrypoint or "chunk" in entrypoint.lower()
        assert "HybridSearch" in entrypoint or "hybrid" in entrypoint.lower()
        assert "streamlit" in entrypoint.lower()

    def test_pyproject_has_required_deps(self):
        pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()
        assert "openai" in pyproject
        assert "onnxruntime" in pyproject
        assert "tokenizers" in pyproject
        assert "streamlit" in pyproject
        assert "minsearch" in pyproject
        assert "opentelemetry" in pyproject

    def test_pyproject_has_pytest_in_dev(self):
        pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()
        assert "pytest" in pyproject


# ===========================================================================
# PHASE 10: Full Pipeline Integration (End-to-End)
# ===========================================================================


class TestFullPipeline:
    @pytest.fixture(scope="class")
    def full_pipeline(self):
        from src.search.hybrid_search import HybridSearch

        search_index = HybridSearch(documents_path=CHUNKS_DIR / "documents.jsonl")
        return search_index

    def test_data_to_search_pipeline(self, full_pipeline):
        # Verify data exists
        assert EVAL_QA.exists()
        assert (CHUNKS_DIR / "documents.jsonl").exists()

        # Verify search index loaded data
        # Floor relaxed from >= 3000 (rag-mini-wikipedia) to >= 50: the default
        # dev subset is 50 Pokémon (user directive 2026-08-09) — 3000 can never
        # hold on the dev subset.
        assert len(full_pipeline.documents) >= 50

        # Verify search returns results
        results = full_pipeline.search("water type pokemon", num_results=3)
        assert len(results) > 0

    def test_search_to_rag_pipeline(self, full_pipeline):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = "Pikachu is an electric type Pokémon."
        mock_client.client.responses.create.return_value = mock_response

        from src.rag.rag_base import RAGBase

        rag = RAGBase(search_index=full_pipeline, llm_client=mock_client)

        # Search
        results = rag.search("Which Pokémon are weak to fire?")
        assert len(results) > 0

        # Build context
        context = rag.build_context(results)
        assert len(context) > 0

        # Build prompt
        prompt = rag.build_prompt("water type pokemon", results)
        assert "water type pokemon" in prompt.lower()

        # Generate answer
        answer = rag.rag("water type pokemon")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_full_agent_loop(self, full_pipeline, monkeypatch):
        def fake_web(query, num_results=5):
            return [{"title": "t", "url": "u", "snippet": "s"}]

        monkeypatch.setattr("src.search.web_search.web_search", fake_web)

        mock_client = MagicMock()
        local_call = MagicMock()
        local_call.type = "function_call"
        local_call.name = "search_local_knowledge_base"
        local_call.arguments = json.dumps({"query": "pikachu stats"})
        local_call.call_id = "c1"
        final = MagicMock()
        final.output = []
        final.output_text = "Pikachu is an Electric Pokémon with high speed."
        mock_client.client.responses.create.side_effect = [
            MagicMock(output=[local_call], output_text=""),
            final,
        ]

        from src.rag.rag_agent import RAGAgent

        agent = RAGAgent(search_index=full_pipeline, llm_client=mock_client)
        result = agent.run("What are Pikachu's stats?")

        # Verify complete pipeline output.
        assert hasattr(result, "answer")
        assert hasattr(result, "searches")
        assert hasattr(result, "iterations")
        assert len(result.answer) > 0
        assert result.iterations >= 1
        assert result.searches[0].source == "local"

    def test_full_agent_with_feedback(self, full_pipeline, monkeypatch):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from monitoring.span_exporter import PostgresSpanExporter
        from monitoring.span_store import get_trace_stats
        from monitoring.tracer import TracedRAGAgent

        fake = make_fake_db(monkeypatch)
        exporter = PostgresSpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test_full_feedback")

        mock_inner_agent = MagicMock()
        mock_inner_agent.run.return_value = AgentResult(
            answer="ML is a subset of AI.",
            searches=[],
            iterations=1,
            rejected=False,
            source=None,
            confidence=None,
            relevance=None,
        )

        traced_agent = TracedRAGAgent(agent=mock_inner_agent, tracer=tracer)
        result = traced_agent.run("What is ML?")

        assert result.answer == "ML is a subset of AI."

        exporter.force_flush()
        exporter.shutdown()

        stats = get_trace_stats()
        assert stats["total_traces"] >= 1

    def test_db_defaults(self, monkeypatch):
        from monitoring.db_init import get_db_connection

        monkeypatch.delenv("POSTGRES_HOST", raising=False)
        for var in (
            "POSTGRES_PORT",
            "POSTGRES_DB",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
        ):
            monkeypatch.delenv(var, raising=False)

        captured = {}

        def fake_connect(**kwargs):
            captured.update(kwargs)
            return FakeConnection()

        monkeypatch.setattr("monitoring.db_init.psycopg.connect", fake_connect)
        get_db_connection()
        assert captured == {
            "host": "localhost",
            "port": "5432",
            "dbname": "capstone",
            "user": "capstone",
            "password": "capstone_secret",
        }

    def test_postgres_down_does_not_break_tracer(self, monkeypatch):
        from monitoring.tracer import TracerSetup

        def raise_connect(**kwargs):
            raise RuntimeError("Postgres down")

        monkeypatch.setattr("monitoring.db_init.psycopg.connect", raise_connect)
        setup = TracerSetup()  # must not raise even though Postgres is down
        assert setup.exporter is None
        setup.shutdown()

    def test_qa_pairs_match_search(self, full_pipeline):
        qa_path = EVAL_QA
        questions = []
        with open(qa_path) as f:
            for i, line in enumerate(f):
                if i >= 20:
                    break
                questions.append(json.loads(line))

        valid_doc_ids = {d["id"] for d in full_pipeline.documents}

        for q in questions:
            results = full_pipeline.search(q["question"], num_results=5)
            assert len(results) > 0, f"No results for: {q['question'][:60]}"
            for doc in results:
                assert doc.id in valid_doc_ids, f"Doc ID {doc.id} not in index"


# ===========================================================================
# Web search backend unit tests
# ===========================================================================


class TestWebSearch:
    @staticmethod
    def fake_client(response):
        fake = MagicMock()
        fake.search.return_value = response
        return fake

    def test_user_namespace_results_filtered(self, monkeypatch):
        from src.search import web_search

        fake = self.fake_client(
            {
                "results": [
                    {
                        "title": "Pikachu (Pokémon)",
                        "url": "https://bulbapedia.bulbagarden.net/wiki/Pikachu_(Pok%C3%A9mon)",
                        "content": "s1",
                        "score": 0.91,
                    },
                    {
                        "title": "User:Landfish7/Overview/Pikachu",
                        "url": "https://bulbapedia.bulbagarden.net/wiki/User:Landfish7/Overview/Pikachu",
                        "content": "s2",
                        "score": 0.9,
                    },
                    {
                        "title": "Volt Tackle (move)",
                        "url": "https://bulbapedia.bulbagarden.net/wiki/Volt_Tackle_(move)",
                        "content": "s3",
                        "score": 0.85,
                    },
                    {
                        "title": "Talk page",
                        "url": "https://bulbapedia.bulbagarden.net/wiki/User_talk:Someone",
                        "content": "s4",
                        "score": 0.8,
                    },
                ]
            }
        )
        monkeypatch.setattr(web_search, "TavilyClient", lambda api_key=None: fake)
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")

        results = web_search.web_search("pikachu", num_results=5)

        assert [r.url for r in results] == [
            "https://bulbapedia.bulbagarden.net/wiki/Pikachu_(Pok%C3%A9mon)",
            "https://bulbapedia.bulbagarden.net/wiki/Volt_Tackle_(move)",
        ]
        assert [r.score for r in results] == [0.91, 0.85]
        fake.search.assert_called_once()

    def test_missing_api_key_raises(self, monkeypatch):
        from src.search import web_search

        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            web_search.web_search("pikachu")


# ===========================================================================
# Query reformulation (web-search step)
# ===========================================================================


# ===========================================================================
# Persistent conversation store (backend)
# ===========================================================================


class TestConversationStore:
    @staticmethod
    def record(
        prompt_tokens=10,
        completion_tokens=5,
        source="local",
        rejected=False,
        span_id="span0",
        model="qwen/qwen3.5-9b",
        error=None,
    ):
        from src.rag.llm_call_record import LLMCallRecord

        return LLMCallRecord(
            model=model,
            prompt="",
            instructions="",
            answer="answer text",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            response_time=0.5,
            source=source,
            rejected=rejected,
            span_id=span_id,
            error=error,
        )

    def test_save_and_get_recent_roundtrip(self, monkeypatch):
        from monitoring.db_init import init_db
        from monitoring.db_query import get_conversations
        from monitoring.db_save import save_conversation

        fake = make_fake_db(monkeypatch)
        init_db()
        ids = []
        for i in range(3):
            # Distinct timestamps so ORDER BY timestamp DESC is deterministic.
            time.sleep(0.002)
            rid = save_conversation(
                self.record(source="local", rejected=(i == 1), span_id=f"span{i}"),
                f"q{i}",
                "llm-zoomcamp",
            )
            ids.append(rid)

        records = get_conversations(limit=2)
        assert len(records) == 2
        assert [r.id for r in records] == [ids[2], ids[1]]  # newest first
        first = records[0]
        assert first.source == "local"
        assert first.rejected is False
        assert first.span_id == "span2"
        assert first.model == "qwen/qwen3.5-9b"
        assert first.prompt_tokens == 10
        assert first.completion_tokens == 5
        assert first.total_tokens == 15
        assert first.response_time == 0.5
        assert records[1].rejected is True

    def test_save_without_usage(self, monkeypatch):
        from monitoring.db_init import init_db
        from monitoring.db_query import get_conversations
        from monitoring.db_save import save_conversation

        fake = make_fake_db(monkeypatch)
        init_db()
        rid = save_conversation(
            self.record(prompt_tokens=0, completion_tokens=0),
            "q",
            "llm-zoomcamp",
        )
        assert rid is not None

        records = get_conversations(limit=10)
        assert len(records) == 1
        assert records[0].prompt_tokens == 0
        assert records[0].completion_tokens == 0
        assert records[0].total_tokens == 0

    def test_stats_aggregates(self, monkeypatch):
        from monitoring.db_init import init_db
        from monitoring.db_stats import get_stats
        from monitoring.db_save import save_conversation

        fake = make_fake_db(monkeypatch)
        init_db()
        for i in range(3):
            save_conversation(
                self.record(prompt_tokens=10, completion_tokens=5),
                f"q{i}",
                "llm-zoomcamp",
            )

        stats = get_stats()
        assert stats.total == 3
        assert stats.avg_tokens == 15.0
        assert stats.avg_response_time == 0.5

    def test_save_never_raises(self, monkeypatch):
        from monitoring.db_save import save_conversation

        def raise_connect(**kwargs):
            raise RuntimeError("Postgres down")

        monkeypatch.setattr("monitoring.db_init.psycopg.connect", raise_connect)
        result = save_conversation(self.record(), "q", "llm-zoomcamp")
        assert result is None

    def test_init_db_creates_tables(self, monkeypatch):
        from monitoring.db_init import init_db, init_feedback

        fake = make_fake_db(monkeypatch)
        init_db()
        init_feedback()
        init_db()
        init_feedback()
        # CREATE TABLE IF NOT EXISTS makes re-initialization idempotent
        # (entrypoint runs init on every container boot).
        assert (
            sum(
                "CREATE TABLE IF NOT EXISTS conversations" in s for s in fake.statements
            )
            >= 1
        )
        assert (
            sum("CREATE TABLE IF NOT EXISTS feedback" in s for s in fake.statements)
            >= 1
        )
        assert (
            sum("CREATE TABLE IF NOT EXISTS searches" in s for s in fake.statements)
            >= 1
        )
        assert (
            sum("CREATE TABLE IF NOT EXISTS llm_calls" in s for s in fake.statements)
            >= 1
        )

    def test_save_conversation_with_session_and_error(self, monkeypatch):
        from monitoring.db_init import init_db
        from monitoring.db_save import save_conversation

        fake = make_fake_db(monkeypatch)
        init_db()
        rid = save_conversation(
            self.record(error="boom"),
            "q",
            "llm-zoomcamp",
            session_id="s1",
        )
        assert rid is not None

        fake.execute("SELECT session_id, error FROM conversations ORDER BY id")
        row = fake.fetchone()
        assert row == ("s1", "boom")

    def test_get_conversations_filters_by_session(self, monkeypatch):
        from monitoring.db_init import init_db
        from monitoring.db_query import get_conversations
        from monitoring.db_save import save_conversation

        fake = make_fake_db(monkeypatch)
        init_db()
        save_conversation(self.record(), "q1", "llm-zoomcamp", session_id="s1")
        save_conversation(self.record(), "q2", "llm-zoomcamp", session_id="s2")

        only_s1 = get_conversations(limit=10, session_id="s1")
        assert len(only_s1) == 1
        assert only_s1[0].question == "q1"

        both = get_conversations(limit=10)
        assert len(both) == 2

    def test_row_error_field_mapped(self, monkeypatch):
        from monitoring.db_init import init_db
        from monitoring.db_query import get_conversations
        from monitoring.db_save import save_conversation

        fake = make_fake_db(monkeypatch)
        init_db()
        save_conversation(self.record(error="boom"), "q", "llm-zoomcamp")

        records = get_conversations(limit=10)
        assert len(records) == 1
        assert records[0].error == "boom"


class TestSearchStore:
    def test_save_search_local_and_web(self, monkeypatch):
        from monitoring.db_init import init_db
        from monitoring.db_save import save_search

        fake = make_fake_db(monkeypatch)
        init_db()
        save_search(
            1,
            "span1",
            "What are Pikachu's stats?",
            "pikachu stats",
            "local",
            [{"id": 25, "name": "Pikachu", "score": 0.9}],
        )
        save_search(
            1,
            "span1",
            "Who voiced Pikachu?",
            "Pikachu voice actor",
            "web",
            [{"title": "Ikue Otani", "url": "u", "snippet": "voiced", "score": 0.8}],
        )

        fake.execute(
            "SELECT source, query, search_query, results FROM searches ORDER BY id"
        )
        rows = fake.fetchall()
        assert len(rows) == 2
        assert rows[0][0] == "local"
        assert rows[0][1] == "What are Pikachu's stats?"
        assert rows[0][2] == "pikachu stats"
        local_results = json.loads(rows[0][3])
        assert local_results[0]["id"] == 25
        assert local_results[0]["name"] == "Pikachu"
        assert rows[1][0] == "web"
        web_results = json.loads(rows[1][3])
        assert "title" in web_results[0]
        assert "snippet" in web_results[0]


class TestLLMCallStore:
    def test_save_llm_call_success(self, monkeypatch):
        from monitoring.db_init import init_db
        from monitoring.db_save import save_llm_call

        fake = make_fake_db(monkeypatch)
        init_db()
        save_llm_call(1, "span1", "qwen/qwen3.5-9b", 100, 50, 150, 0.3, None)

        fake.execute(
            "SELECT model, prompt_tokens, completion_tokens, total_tokens, latency, error "
            "FROM llm_calls ORDER BY id"
        )
        row = fake.fetchone()
        assert row == ("qwen/qwen3.5-9b", 100, 50, 150, 0.3, None)

    def test_save_llm_call_failure(self, monkeypatch):
        from monitoring.db_init import init_db
        from monitoring.db_save import save_llm_call

        fake = make_fake_db(monkeypatch)
        init_db()
        save_llm_call(
            1, "span1", "qwen/qwen3.5-9b", None, None, None, 0.1, "LLM call failed"
        )

        fake.execute(
            "SELECT prompt_tokens, completion_tokens, total_tokens, error "
            "FROM llm_calls ORDER BY id"
        )
        row = fake.fetchone()
        assert row == (None, None, None, "LLM call failed")


class TestAgentLoopSaver:
    def test_save_agent_loop_persists_conversation_search_and_llm_calls(
        self, monkeypatch
    ):
        # Regression: the saver must resolve save_search/save_llm_call from
        # db_save (they are not in db_save); an ImportError was
        # previously swallowed and conversations silently never saved.
        from monitoring.db_init import init_db
        from src.interface.agent_loop_saver import AgentLoopSaver
        from src.rag.tools import SearchRecord

        fake = make_fake_db(monkeypatch)
        init_db()

        agent = SimpleNamespace(
            agent_loop_record=TestConversationStore.record(span_id="span9"),
            calls=[
                SimpleNamespace(
                    model="qwen/qwen3.5-9b",
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    response_time=0.3,
                    error=None,
                ),
                SimpleNamespace(
                    model="qwen/qwen3.5-9b",
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    response_time=0.2,
                    error="LLM call failed",
                ),
            ],
        )
        result = SimpleNamespace(
            searches=[
                SearchRecord(
                    query="What are Pikachu's stats?",
                    results=[],
                    source="local",
                    search_query="pikachu stats",
                )
            ]
        )

        saver = AgentLoopSaver()
        conversation_id = saver.save_agent_loop(
            agent, result, "span9", "What are Pikachu's stats?", "sess1"
        )
        assert conversation_id is not None

        fake.execute("SELECT question, session_id, span_id, source FROM conversations")
        assert fake.fetchone() == (
            "What are Pikachu's stats?",
            "sess1",
            "span9",
            "local",
        )

        fake.execute("SELECT conversation_id, source, search_query FROM searches")
        assert fake.fetchone() == (conversation_id, "local", "pikachu stats")

        fake.execute(
            "SELECT conversation_id, prompt_tokens, error FROM llm_calls ORDER BY id"
        )
        assert fake.fetchall() == [
            (conversation_id, 100, None),
            (conversation_id, 10, "LLM call failed"),
        ]

    def test_save_agent_loop_with_traced_wrapper(self, monkeypatch):
        # Regression: handle_prompt passes the TracedRAGAgent wrapper to the
        # saver; it must delegate the record attributes to the inner agent.
        from monitoring.db_init import init_db
        from monitoring.tracer import TracedRAGAgent
        from src.interface.agent_loop_saver import AgentLoopSaver
        from src.rag.tools import SearchRecord

        fake = make_fake_db(monkeypatch)
        init_db()

        inner = SimpleNamespace(
            agent_loop_record=TestConversationStore.record(span_id="span9"),
            calls=[],
        )
        wrapper = TracedRAGAgent(inner)
        result = SimpleNamespace(searches=[])

        saver = AgentLoopSaver()
        conversation_id = saver.save_agent_loop(wrapper, result, "span9", "q", "sess1")
        assert conversation_id is not None


class TestAgentUsage:
    @staticmethod
    def usage_response(text="", *calls, input_tokens=0, output_tokens=0):
        response = TestAgentToolLoop.response(text, *calls)
        response.usage = SimpleNamespace(
            input_tokens=input_tokens, output_tokens=output_tokens
        )
        return response

    def test_usage_accumulated_across_agent_loops(self, monkeypatch):
        web_fake = TestAgentToolLoop.web_fake()
        monkeypatch.setattr("src.search.web_search.web_search", web_fake)
        agent = TestAgentToolLoop.agent(
            TestAgentToolLoop.script_client(
                self.usage_response(
                    "",
                    TestAgentToolLoop.function_call(
                        TestAgentToolLoop.LOCAL, {"query": "pikachu stats"}
                    ),
                    input_tokens=100,
                    output_tokens=50,
                ),
                self.usage_response(
                    text="Pikachu has HP 35, Attack 55, Defense 40, Speed 90.",
                    input_tokens=100,
                    output_tokens=50,
                ),
            )
        )
        result = agent.run("What are Pikachu's stats?")

        assert result.usage == Usage(input_tokens=200, output_tokens=100)

    def test_usage_zero_on_early_reject(self):
        mock_client = MagicMock()
        mock_client.client.responses.create.side_effect = AssertionError(
            "LLM must not be called"
        )
        agent = TestAgentToolLoop.agent(mock_client)
        result = agent.run("???")

        assert result.usage == Usage(input_tokens=0, output_tokens=0)

    def test_usage_safe_with_magicmock_usage(self):
        response = TestAgentToolLoop.response(text="Pikachu is Electric.")
        response.usage = MagicMock()  # auto-created attrs are not ints
        agent = TestAgentToolLoop.agent(TestAgentToolLoop.script_client(response))
        result = agent.run("What type is Pikachu?")

        assert result.usage == Usage(input_tokens=0, output_tokens=0)

    def test_rejection_result_has_usage(self, monkeypatch):
        from src.rag.prompts import REJECTION_MESSAGE

        def no_web(query, num_results=5):
            raise AssertionError("tools must never run on an out-of-scope question")

        monkeypatch.setattr("src.search.web_search.web_search", no_web)
        response = self.usage_response(
            text=REJECTION_MESSAGE, input_tokens=30, output_tokens=10
        )
        agent = TestAgentToolLoop.agent(TestAgentToolLoop.script_client(response))
        result = agent.run("Who would win Charizard vs Blastoise?")

        assert hasattr(result, "usage")
        assert result.rejected is True
        assert result.usage == Usage(input_tokens=30, output_tokens=10)

    def test_llm_calls_recorded_per_agent_loop(self, monkeypatch):
        web_fake = TestAgentToolLoop.web_fake()
        monkeypatch.setattr("src.search.web_search.web_search", web_fake)
        agent = TestAgentToolLoop.agent(
            TestAgentToolLoop.script_client(
                self.usage_response(
                    "",
                    TestAgentToolLoop.function_call(
                        TestAgentToolLoop.LOCAL, {"query": "pikachu stats"}
                    ),
                    input_tokens=100,
                    output_tokens=50,
                ),
                self.usage_response(
                    text="Pikachu has HP 35, Attack 55, Defense 40, Speed 90.",
                    input_tokens=200,
                    output_tokens=80,
                ),
            )
        )
        result = agent.run("What are Pikachu's stats?")

        assert len(result.llm_calls) == 2
        first, second = result.llm_calls
        assert first.prompt_tokens == 100
        assert first.completion_tokens == 50
        assert first.total_tokens == 150
        assert first.latency >= 0
        assert first.error is None
        assert second.prompt_tokens == 200
        assert second.completion_tokens == 80
        assert second.total_tokens == 280

    def test_llm_calls_error_entry(self):
        from src.rag.prompts import REJECTION_MESSAGE

        mock_client = MagicMock()
        mock_client.client.responses.create.side_effect = RuntimeError("server down")
        agent = TestAgentToolLoop.agent(mock_client)
        result = agent.run("Question")

        assert result.rejected is True
        assert result.answer == REJECTION_MESSAGE
        assert len(result.llm_calls) == 1
        call = result.llm_calls[0]
        assert call.error == "LLM call failed"
        assert call.prompt_tokens is None
        assert call.completion_tokens is None
        assert call.total_tokens is None

    def test_llm_calls_empty_on_early_reject(self):
        mock_client = MagicMock()
        mock_client.client.responses.create.side_effect = AssertionError(
            "LLM must not be called"
        )
        agent = TestAgentToolLoop.agent(mock_client)
        result = agent.run("???")

        assert result.llm_calls == []

    def test_llm_calls_safe_with_magicmock_usage(self, monkeypatch):
        web_fake = TestAgentToolLoop.web_fake()
        monkeypatch.setattr("src.search.web_search.web_search", web_fake)
        response = TestAgentToolLoop.response(text="Pikachu is Electric.")
        response.usage = MagicMock()  # auto-created attrs are not ints
        agent = TestAgentToolLoop.agent(
            TestAgentToolLoop.script_client(
                response,
                TestAgentToolLoop.response(
                    "",
                    TestAgentToolLoop.function_call(
                        TestAgentToolLoop.WEB, {"query": "pikachu"}
                    ),
                ),
                TestAgentToolLoop.response(
                    text="Ikue Otani voiced Pikachu in the anime."
                ),
            )
        )
        result = agent.run("Who voiced Pikachu in the anime?")

        # The tool-less memory answer escalated to web; the MagicMock usage
        # on the first call must be handled without int() crashes.
        assert len(result.llm_calls) == 3
        first = result.llm_calls[0]
        assert first.prompt_tokens == 0
        assert first.completion_tokens == 0
        assert first.total_tokens == 0
        assert first.error is None


class TestRAGOwnsRecords:
    @staticmethod
    def usage_response(text="", *calls, input_tokens=0, output_tokens=0):
        response = TestAgentToolLoop.response(text, *calls)
        response.usage = SimpleNamespace(
            input_tokens=input_tokens, output_tokens=output_tokens
        )
        return response

    def test_calls_are_llmcalls_with_latency(self, monkeypatch):
        from src.rag.llm_call_record import LLMCallRecord

        web_fake = TestAgentToolLoop.web_fake()
        monkeypatch.setattr("src.search.web_search.web_search", web_fake)
        agent = TestAgentToolLoop.agent(
            TestAgentToolLoop.script_client(
                self.usage_response(
                    "",
                    TestAgentToolLoop.function_call(
                        TestAgentToolLoop.LOCAL, {"query": "pikachu stats"}
                    ),
                    input_tokens=100,
                    output_tokens=50,
                ),
                self.usage_response(
                    text="Pikachu has HP 35, Attack 55, Defense 40, Speed 90.",
                    input_tokens=200,
                    output_tokens=80,
                ),
            )
        )
        agent.run("What are Pikachu's stats?")

        assert len(agent.calls) == 2
        assert all(isinstance(c, LLMCallRecord) for c in agent.calls)
        first, second = agent.calls
        assert first.prompt_tokens == 100
        assert first.completion_tokens == 50
        assert first.total_tokens == 150
        assert first.response_time >= 0
        assert first.error is None
        assert second.total_tokens == 280

    def test_agent_loop_record_built(self, monkeypatch):
        web_fake = TestAgentToolLoop.web_fake()
        monkeypatch.setattr("src.search.web_search.web_search", web_fake)
        agent = TestAgentToolLoop.agent(
            TestAgentToolLoop.script_client(
                self.usage_response(
                    "",
                    TestAgentToolLoop.function_call(
                        TestAgentToolLoop.LOCAL, {"query": "pikachu stats"}
                    ),
                    input_tokens=100,
                    output_tokens=50,
                ),
                self.usage_response(
                    text="Pikachu has HP 35, Attack 55, Defense 40, Speed 90.",
                    input_tokens=100,
                    output_tokens=50,
                ),
            )
        )
        result = agent.run("What are Pikachu's stats?")

        record = agent.agent_loop_record
        assert record is not None
        assert record.answer == result.answer
        assert record.total_tokens == 300
        assert record.source == result.source
        assert record.rejected == result.rejected
        assert record.response_time > 0
        assert record.span_id is None  # caller attaches the span id

    def test_agent_loop_record_for_early_reject(self):
        from src.rag.prompts import REJECTION_MESSAGE

        mock_client = MagicMock()
        mock_client.client.responses.create.side_effect = AssertionError(
            "LLM must not be called"
        )
        agent = TestAgentToolLoop.agent(mock_client)
        agent.run("???")

        record = agent.agent_loop_record
        assert record is not None
        assert record.rejected is True
        assert record.answer == REJECTION_MESSAGE
        assert record.total_tokens == 0

    def test_error_call_recorded(self):
        from src.rag.prompts import REJECTION_MESSAGE

        mock_client = MagicMock()
        mock_client.client.responses.create.side_effect = RuntimeError("server down")
        agent = TestAgentToolLoop.agent(mock_client)
        result = agent.run("Question")

        assert result.rejected is True
        assert result.answer == REJECTION_MESSAGE
        assert len(agent.calls) == 1
        assert agent.calls[0].error == "LLM call failed"
        assert agent.calls[0].prompt_tokens is None
        assert agent.agent_loop_record is not None
        assert agent.agent_loop_record.total_tokens == 0

    def test_search_payload_shapes(self):
        from src.rag.tools import SearchRecord

        local = SearchRecord(
            query="q",
            source="local",
            results=[PokemonDoc(id=25, name="Pikachu", score=0.9)],
        )
        assert local.payload == [{"id": 25, "name": "Pikachu", "score": 0.9}]

        long_snippet = "x" * 500
        web = SearchRecord(
            query="q",
            source="web",
            search_query="pikachu",
            results=[WebResult(title="t", url="u", snippet=long_snippet, score=0.8)],
        )
        item = web.payload[0]
        assert set(item) == {"title", "url", "snippet", "score"}
        assert len(item["snippet"]) == 300


class TestTracerConversationIntegration:
    def test_traced_agent_sets_token_attributes(self, monkeypatch):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from monitoring.span_exporter import PostgresSpanExporter
        from monitoring.span_store import get_trace_stats
        from monitoring.tracer import TracedRAGAgent

        fake = make_fake_db(monkeypatch)
        exporter = PostgresSpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test_conv_traced")

        mock_agent = MagicMock()
        mock_agent.model = "qwen/qwen3.5-9b"
        mock_agent.run.return_value = AgentResult(
            answer="a",
            searches=[],
            iterations=1,
            rejected=False,
            source=None,
            confidence=None,
            relevance=None,
            usage=Usage(input_tokens=500, output_tokens=200),
        )

        traced = TracedRAGAgent(agent=mock_agent, tracer=tracer)
        traced.run("test query")
        exporter.force_flush()
        exporter.shutdown()

        stats = get_trace_stats()
        assert stats["total_input_tokens"] >= 500
        assert stats["total_output_tokens"] >= 200

    def test_feedback_table_roundtrip(self, monkeypatch):
        from monitoring.db_feedback import save_feedback
        from monitoring.db_init import init_feedback
        from monitoring.db_query import get_feedback_for_conversations

        fake = make_fake_db(monkeypatch)
        init_feedback()
        save_feedback(7, "user", score=1)
        assert get_feedback_for_conversations([7]) == {7: 1}

    def test_user_feedback_stats(self, monkeypatch):
        from monitoring.db_feedback import save_feedback
        from monitoring.db_init import init_feedback
        from monitoring.db_stats import get_user_feedback_stats

        fake = make_fake_db(monkeypatch)
        init_feedback()
        save_feedback(1, "user", score=1)
        save_feedback(2, "user", score=-1)
        save_feedback(3, "judge", score=1)  # excluded — not a user vote
        assert get_user_feedback_stats() == (1, 1)

    def test_feedback_fk_no_constraint_issue(self, monkeypatch):
        from monitoring.db_feedback import save_feedback
        from monitoring.db_init import init_feedback

        fake = make_fake_db(monkeypatch)
        init_feedback()
        # A conversation_id with no matching conversations row must not raise
        # (FK enforcement is off, mirroring the app's tolerance).
        save_feedback(999, "user", score=1)
        fake.execute("SELECT COUNT(*) FROM feedback")
        assert fake.fetchone()[0] == 1
