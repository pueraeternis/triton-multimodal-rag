"""CPU-friendly retrieval and reranking evaluation aligned with the BLS pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

RETRIEVAL_LIMIT = 5


@dataclass(frozen=True)
class Document:
    id: int
    category: str
    issue_description: str
    solution_text: str


@dataclass(frozen=True)
class EvalQuery:
    query: str
    expected_id: int


@dataclass(frozen=True)
class QueryResult:
    query: str
    expected_id: int
    expected_issue: str
    retrieval_rank: int | None
    rerank_rank: int | None
    retrieved_id: int | None
    retrieval_pass: bool
    rerank_pass: bool
    overall_pass: bool


@dataclass(frozen=True)
class EvaluationSummary:
    total: int
    recall_at_k: float
    retrieval_top1: float
    retrieval_mrr: float
    rerank_top1: float
    rerank_mrr: float
    overall_pass_rate: float
    retrieval_k: int


class Embedder(Protocol):
    def encode(self, texts: list[str] | str) -> np.ndarray: ...


class Reranker(Protocol):
    def predict(self, pairs: list[list[str]]) -> np.ndarray: ...


def document_index_text(doc: Document) -> str:
    return f"{doc.category}: {doc.issue_description}"


def load_knowledge_base(path: Path) -> list[Document]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return [
        Document(
            id=doc["id"],
            category=doc["category"],
            issue_description=doc["issue_description"],
            solution_text=doc["solution_text"],
        )
        for doc in raw
    ]


def load_eval_queries(path: Path) -> list[EvalQuery]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return [EvalQuery(query=item["query"], expected_id=item["expected_id"]) for item in raw]


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def cosine_top_k(
    query_vector: np.ndarray,
    doc_vectors: np.ndarray,
    doc_ids: list[int],
    *,
    limit: int,
) -> list[tuple[int, float]]:
    query = query_vector / (np.linalg.norm(query_vector) or 1.0)
    docs = _normalize_rows(doc_vectors)
    scores = docs @ query
    order = np.argsort(scores)[::-1][:limit]
    return [(doc_ids[i], float(scores[i])) for i in order]


def retrieve(
    query: str,
    documents: list[Document],
    embedder: Embedder,
    *,
    limit: int = RETRIEVAL_LIMIT,
) -> list[tuple[Document, float]]:
    index_texts = [document_index_text(doc) for doc in documents]
    doc_vectors = np.asarray(embedder.encode(index_texts), dtype=np.float64)
    query_vector = np.asarray(embedder.encode(query), dtype=np.float64)
    ranked = cosine_top_k(
        query_vector,
        doc_vectors,
        [doc.id for doc in documents],
        limit=limit,
    )
    by_id = {doc.id: doc for doc in documents}
    return [(by_id[doc_id], score) for doc_id, score in ranked]


def rerank(
    query: str,
    candidates: list[Document],
    reranker: Reranker,
) -> list[tuple[Document, float]]:
    if not candidates:
        return []
    pairs = [[query, doc.solution_text] for doc in candidates]
    scores = np.asarray(reranker.predict(pairs), dtype=np.float64)
    order = np.argsort(scores)[::-1]
    return [(candidates[i], float(scores[i])) for i in order]


def rank_of(expected_id: int, ranked_ids: list[int]) -> int | None:
    try:
        return ranked_ids.index(expected_id) + 1
    except ValueError:
        return None


def recall_at_k(expected_id: int, ranked_ids: list[int], k: int) -> float:
    return 1.0 if expected_id in ranked_ids[:k] else 0.0


def reciprocal_rank(expected_id: int, ranked_ids: list[int]) -> float:
    found = rank_of(expected_id, ranked_ids)
    return 0.0 if found is None else 1.0 / found


def evaluate_queries(
    eval_queries: list[EvalQuery],
    documents: list[Document],
    embedder: Embedder,
    reranker: Reranker,
    *,
    retrieval_k: int = RETRIEVAL_LIMIT,
) -> tuple[list[QueryResult], EvaluationSummary]:
    by_id = {doc.id: doc for doc in documents}
    results: list[QueryResult] = []

    retrieval_recalls: list[float] = []
    retrieval_top1s: list[float] = []
    retrieval_rrs: list[float] = []
    rerank_top1s: list[float] = []
    rerank_rrs: list[float] = []
    overall_passes: list[float] = []

    for item in eval_queries:
        expected = by_id[item.expected_id]
        retrieved = retrieve(item.query, documents, embedder, limit=retrieval_k)
        retrieval_ids = [doc.id for doc, _ in retrieved]
        reranked = rerank(item.query, [doc for doc, _ in retrieved], reranker)
        rerank_ids = [doc.id for doc, _ in reranked]

        retrieval_rank = rank_of(item.expected_id, retrieval_ids)
        rerank_rank = rank_of(item.expected_id, rerank_ids)
        retrieval_hit = recall_at_k(item.expected_id, retrieval_ids, retrieval_k)
        rerank_hit = 1.0 if rerank_ids and rerank_ids[0] == item.expected_id else 0.0

        retrieval_recalls.append(retrieval_hit)
        retrieval_top1s.append(1.0 if retrieval_rank == 1 else 0.0)
        retrieval_rrs.append(reciprocal_rank(item.expected_id, retrieval_ids))
        rerank_top1s.append(rerank_hit)
        rerank_rrs.append(reciprocal_rank(item.expected_id, rerank_ids))
        overall = retrieval_hit == 1.0 and rerank_hit == 1.0
        overall_passes.append(1.0 if overall else 0.0)

        top_retrieved = rerank_ids[0] if rerank_ids else (retrieval_ids[0] if retrieval_ids else None)
        results.append(
            QueryResult(
                query=item.query,
                expected_id=item.expected_id,
                expected_issue=expected.issue_description,
                retrieval_rank=retrieval_rank,
                rerank_rank=rerank_rank,
                retrieved_id=top_retrieved,
                retrieval_pass=retrieval_hit == 1.0,
                rerank_pass=rerank_hit == 1.0,
                overall_pass=overall,
            ),
        )

    total = len(eval_queries)
    summary = EvaluationSummary(
        total=total,
        recall_at_k=sum(retrieval_recalls) / total if total else 0.0,
        retrieval_top1=sum(retrieval_top1s) / total if total else 0.0,
        retrieval_mrr=sum(retrieval_rrs) / total if total else 0.0,
        rerank_top1=sum(rerank_top1s) / total if total else 0.0,
        rerank_mrr=sum(rerank_rrs) / total if total else 0.0,
        overall_pass_rate=sum(overall_passes) / total if total else 0.0,
        retrieval_k=retrieval_k,
    )
    return results, summary


def format_report(
    results: list[QueryResult],
    summary: EvaluationSummary,
    *,
    knowledge_base_path: str,
    eval_queries_path: str,
    embedding_model: str,
    reranker_model: str,
    backend: str,
) -> str:
    lines = [
        "Retrieval & Reranking Evaluation Report",
        "======================================",
        f"Knowledge base: {knowledge_base_path}",
        f"Eval queries: {summary.total}",
        f"Retrieval limit: {summary.retrieval_k}",
        f"Embedding model: {embedding_model}",
        f"Reranker model: {reranker_model}",
        f"Backend: {backend}",
        "",
        f"{'Query':<52} {'Expected':<8} {'Retrieved':<9} {'Retr Rank':<10} {'Rerank Rank':<12} Pass",
        "-" * 105,
    ]

    for row in results:
        query_preview = row.query if len(row.query) <= 50 else row.query[:47] + "..."
        retr_rank = str(row.retrieval_rank) if row.retrieval_rank is not None else "-"
        rerank_rank = str(row.rerank_rank) if row.rerank_rank is not None else "-"
        retrieved = str(row.retrieved_id) if row.retrieved_id is not None else "-"
        status = "PASS" if row.overall_pass else "FAIL"
        lines.append(
            f"{query_preview:<52} {row.expected_id:<8} {retrieved:<9} {retr_rank:<10} {rerank_rank:<12} {status}",
        )

    lines.extend(
        [
            "",
            "Summary",
            "-------",
            f"Recall@{summary.retrieval_k} (retrieval): {summary.recall_at_k:.3f}",
            f"Top-1 (retrieval):              {summary.retrieval_top1:.3f}",
            f"MRR (retrieval):                {summary.retrieval_mrr:.3f}",
            f"Top-1 (rerank):                 {summary.rerank_top1:.3f}",
            f"MRR (rerank):                   {summary.rerank_mrr:.3f}",
            f"Overall pass (retrieval+rerank): {summary.overall_pass_rate:.3f}",
            "",
            "Note: This evaluates retrieval and reranking only. It does not score LLM answers.",
        ],
    )
    return "\n".join(lines)
