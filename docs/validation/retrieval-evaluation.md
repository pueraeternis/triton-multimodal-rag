# Retrieval and Reranking Evaluation

Lightweight, CPU-first **deterministic regression checks** for the **retrieval** and **reranking** stages of the multimodal RAG pipeline.

This check validates that vector search and cross-encoder reranking still return the expected knowledge-base documents after code changes. It does **not** score generated LLM answers, compare models, or measure hallucinations.

## Purpose

- Detect regressions when embedding logic, Qdrant indexing assumptions, or reranker wiring change
- Provide a small, human-inspectable query set derived from `data/knowledge_base.json`
- Run without Triton, vLLM, or GPU hardware

## What is evaluated

| Stage | Implementation in eval | Production path |
|-------|------------------------|-----------------|
| Retrieval | In-memory cosine similarity over SentenceTransformer embeddings | Qdrant `query_points` in `bls_orchestrator` |
| Reranking | Direct `CrossEncoder.predict` on CPU | Triton `reranker_py` backend |

The eval mirrors production semantics:

- Documents are indexed as `"{category}: {issue_description}"` (same as `scripts/init_qdrant.py`)
- Queries are embedded as plain user text (same as BLS retrieval)
- Top-5 retrieval candidates are reranked on `solution_text` (same as BLS reranking)

Generation, YOLO vision, and vLLM are **out of scope**.

## Metrics

Simple retrieval-oriented metrics for regression detection:

| Metric | Meaning |
|--------|---------|
| Recall@k | Expected document appears in top-k retrieval results (k = 5, matches BLS) |
| Top-1 (retrieval) | Expected document is the first retrieval hit |
| MRR (retrieval) | Mean reciprocal rank of the expected document in retrieval |
| Top-1 (rerank) | Expected document is selected after reranking |
| MRR (rerank) | Mean reciprocal rank after reranking |
| Overall pass | Expected document is in Recall@k **and** rerank Top-1 |

## Dataset

- Knowledge base: `data/knowledge_base.json`
- Eval queries: `data/retrieval_eval_queries.json` (8 deterministic queries with `expected_id`)

The query set is intentionally small so failures are easy to inspect. Each query maps to a known document in the knowledge base. Not every knowledge-base entry is a stable reranking target when solution text is used as the passage (matching the BLS pipeline), so the fixture covers representative cases rather than the full corpus.

## How to run

### Full evaluation (downloads models on first run)

```bash
uv run scripts/eval_retrieval.py
```

Optional flags:

```bash
uv run scripts/eval_retrieval.py \
  --knowledge-base data/knowledge_base.json \
  --eval-queries data/retrieval_eval_queries.json \
  --min-overall-pass-rate 1.0
```

Or via Make:

```bash
make eval-retrieval
```

### Unit tests (CI-safe, no model downloads)

```bash
uv run pytest tests/test_retrieval_eval.py -q
```

Tests use deterministic fake embedders/rerankers. They validate metrics, pipeline wiring, and eval-set schema without HuggingFace downloads.

## Example output

```
Retrieval & Reranking Evaluation Report
======================================
...
Query                                              Expected Retrieved Retr Rank  Rerank Rank  Pass
---------------------------------------------------------------------------------------------------------
Red status LED is blinking continuously on my...   1        1         1          1            PASS
...

Summary
-------
Recall@5 (retrieval): 1.000
Top-1 (retrieval):              1.000
MRR (retrieval):                1.000
Top-1 (rerank):                 1.000
MRR (rerank):                   1.000
Overall pass (retrieval+rerank): 1.000
```

## Limitations

- **In-memory retrieval** approximates Qdrant cosine search; it does not exercise the Qdrant service itself
- **CPU-only** by design; slower than GPU but suitable for maintainer laptops and CI unit tests
- **Small query set** — regression detection, not academic benchmarking
- **Not an LLM benchmark** — answer quality is explicitly excluded
- First full eval run downloads embedding and reranker weights from HuggingFace

## Files

| File | Role |
|------|------|
| `eval/retrieval_pipeline.py` | Retrieval/reranking logic, metrics, report formatting |
| `data/retrieval_eval_queries.json` | Fixed eval query set |
| `scripts/eval_retrieval.py` | CLI entry point |
| `tests/test_retrieval_eval.py` | Deterministic unit tests |
