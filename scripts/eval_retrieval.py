#!/usr/bin/env python3
"""Evaluate retrieval and reranking without Triton, vLLM, or GPU."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402

from eval.retrieval_pipeline import (  # noqa: E402
    evaluate_queries,
    format_report,
    load_eval_queries,
    load_knowledge_base,
)

load_dotenv()

DEFAULT_KB = ROOT / "data/knowledge_base.json"
DEFAULT_EVAL = ROOT / "data/retrieval_eval_queries.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-friendly retrieval and reranking evaluation (no Triton / vLLM)",
    )
    parser.add_argument("--knowledge-base", type=Path, default=DEFAULT_KB)
    parser.add_argument("--eval-queries", type=Path, default=DEFAULT_EVAL)
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"),
    )
    parser.add_argument(
        "--reranker-model",
        default=os.getenv("RERANKER_MODEL_ID", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    )
    parser.add_argument(
        "--min-overall-pass-rate",
        type=float,
        default=1.0,
        help="Exit non-zero if overall pass rate is below this threshold (default: 1.0)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.knowledge_base.exists():
        print(f"Knowledge base not found: {args.knowledge_base}", file=sys.stderr)
        return 1
    if not args.eval_queries.exists():
        print(f"Eval query set not found: {args.eval_queries}", file=sys.stderr)
        return 1

    documents = load_knowledge_base(args.knowledge_base)
    eval_queries = load_eval_queries(args.eval_queries)

    embedder = SentenceTransformer(args.embedding_model, device="cpu")
    reranker = CrossEncoder(args.reranker_model, device="cpu")

    results, summary = evaluate_queries(eval_queries, documents, embedder, reranker)
    report = format_report(
        results,
        summary,
        knowledge_base_path=str(args.knowledge_base.relative_to(ROOT)),
        eval_queries_path=str(args.eval_queries.relative_to(ROOT)),
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        backend="in-memory cosine similarity (mirrors Qdrant retrieval path)",
    )
    print(report)

    if summary.overall_pass_rate < args.min_overall_pass_rate:
        print(
            f"\nFAILED: overall pass rate {summary.overall_pass_rate:.3f} < required {args.min_overall_pass_rate:.3f}",
            file=sys.stderr,
        )
        return 1

    print("\nPASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
