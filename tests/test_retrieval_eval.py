"""Tests for retrieval and reranking evaluation (deterministic, no Triton)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eval.retrieval_pipeline import (
    Document,
    EvalQuery,
    cosine_top_k,
    evaluate_queries,
    format_report,
    load_eval_queries,
    load_knowledge_base,
    rank_of,
    recall_at_k,
    reciprocal_rank,
    rerank,
    retrieve,
)

ROOT = Path(__file__).resolve().parent.parent


class _FakeEmbedder:
    def __init__(self, vectors_by_text: dict[str, list[float]]):
        self._vectors = vectors_by_text

    def encode(self, texts: list[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            return np.asarray(self._vectors[texts], dtype=np.float64)
        return np.asarray([self._vectors[text] for text in texts], dtype=np.float64)


class _FakeReranker:
    def __init__(self, scores_by_pair: dict[tuple[str, str], float]):
        self._scores = scores_by_pair

    def predict(self, pairs: list[list[str]]) -> np.ndarray:
        return np.asarray([self._scores[(query, doc)] for query, doc in pairs], dtype=np.float32)


@pytest.fixture
def tiny_documents() -> list[Document]:
    return [
        Document(1, "Router", "Red status LED blinking continuously", "solution-a"),
        Document(2, "Server", "Server does not power on", "solution-b"),
        Document(3, "Switch", "PoE devices not receiving power", "solution-c"),
    ]


def test_eval_query_set_schema_and_expected_ids_exist():
    eval_path = ROOT / "data/retrieval_eval_queries.json"
    kb_path = ROOT / "data/knowledge_base.json"

    with eval_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    kb_ids = {doc["id"] for doc in json.loads(kb_path.read_text(encoding="utf-8"))}
    assert isinstance(raw, list)
    assert 4 <= len(raw) <= 12
    for item in raw:
        assert isinstance(item["query"], str) and item["query"].strip()
        assert isinstance(item["expected_id"], int)
        assert item["expected_id"] in kb_ids


def test_metric_helpers():
    ranked = [3, 1, 2]
    assert rank_of(1, ranked) == 2
    assert rank_of(9, ranked) is None
    assert recall_at_k(1, ranked, 2) == 1.0
    assert recall_at_k(1, ranked, 1) == 0.0
    assert reciprocal_rank(1, ranked) == 0.5


def test_cosine_top_k_is_deterministic():
    doc_vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    query_vector = np.array([1.0, 0.0])
    first = cosine_top_k(query_vector, doc_vectors, [10, 20, 30], limit=2)
    second = cosine_top_k(query_vector, doc_vectors, [10, 20, 30], limit=2)
    assert first == second
    assert first[0] == (10, pytest.approx(1.0))
    assert first[1][0] == 20


def test_retrieve_and_rerank_with_fake_models(tiny_documents: list[Document]):
    embedder = _FakeEmbedder(
        {
            "Router: Red status LED blinking continuously": [1.0, 0.0],
            "Server: Server does not power on": [0.7, 0.7],
            "Switch: PoE devices not receiving power": [0.1, 0.2],
            "router led blinking": [0.9, 0.1],
        },
    )
    reranker = _FakeReranker(
        {
            ("router led blinking", "solution-a"): 0.9,
            ("router led blinking", "solution-b"): 0.2,
            ("router led blinking", "solution-c"): 0.1,
        },
    )

    retrieved = retrieve("router led blinking", tiny_documents, embedder, limit=2)
    assert [doc.id for doc, _ in retrieved] == [1, 2]

    reranked = rerank("router led blinking", [doc for doc, _ in retrieved], reranker)
    assert [doc.id for doc, _ in reranked] == [1, 2]


def test_evaluate_queries_reports_pass_fail(tiny_documents: list[Document]):
    embedder = _FakeEmbedder(
        {
            "Router: Red status LED blinking continuously": [1.0, 0.0],
            "Server: Server does not power on": [0.0, 1.0],
            "Switch: PoE devices not receiving power": [0.5, 0.5],
            "router led issue": [0.98, 0.02],
            "server power issue": [0.02, 0.98],
        },
    )
    reranker = _FakeReranker(
        {
            ("router led issue", "solution-a"): 0.95,
            ("router led issue", "solution-b"): 0.1,
            ("router led issue", "solution-c"): 0.05,
            ("server power issue", "solution-b"): 0.9,
            ("server power issue", "solution-a"): 0.2,
            ("server power issue", "solution-c"): 0.05,
        },
    )
    eval_queries = [
        EvalQuery("router led issue", 1),
        EvalQuery("server power issue", 2),
    ]

    results, summary = evaluate_queries(eval_queries, tiny_documents, embedder, reranker, retrieval_k=2)

    assert results[0].overall_pass is True
    assert results[1].overall_pass is True
    assert summary.recall_at_k == 1.0
    assert summary.rerank_top1 == 1.0
    assert summary.overall_pass_rate == 1.0


def test_format_report_contains_summary_lines(tiny_documents: list[Document]):
    embedder = _FakeEmbedder(
        {
            "Router: Red status LED blinking continuously": [1.0, 0.0],
            "Server: Server does not power on": [0.0, 1.0],
            "Switch: PoE devices not receiving power": [0.5, 0.5],
            "router led issue": [0.98, 0.02],
        },
    )
    reranker = _FakeReranker({("router led issue", "solution-a"): 0.9})
    results, summary = evaluate_queries(
        [EvalQuery("router led issue", 1)],
        tiny_documents,
        embedder,
        reranker,
        retrieval_k=1,
    )
    report = format_report(
        results,
        summary,
        knowledge_base_path="data/knowledge_base.json",
        eval_queries_path="data/retrieval_eval_queries.json",
        embedding_model="fake-embedder",
        reranker_model="fake-reranker",
        backend="test",
    )
    assert "Recall@1 (retrieval)" in report
    assert "Top-1 (rerank)" in report
    assert "does not score LLM answers" in report


def test_loaders_round_trip(tmp_path: Path):
    kb_path = tmp_path / "kb.json"
    eval_path = tmp_path / "eval.json"
    kb_path.write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "category": "Router",
                    "issue_description": "LED blinking",
                    "solution_text": "reboot",
                },
            ],
        ),
        encoding="utf-8",
    )
    eval_path.write_text(json.dumps([{"query": "led", "expected_id": 1}]), encoding="utf-8")

    docs = load_knowledge_base(kb_path)
    queries = load_eval_queries(eval_path)
    assert docs[0].solution_text == "reboot"
    assert queries[0].expected_id == 1
