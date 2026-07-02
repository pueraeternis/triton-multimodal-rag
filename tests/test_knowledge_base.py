"""Validate knowledge_base.json schema."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_KEYS = {"id", "category", "issue_description", "solution_text"}


def test_knowledge_base_is_non_empty_list():
    with open(ROOT / "data/knowledge_base.json", encoding="utf-8") as f:
        docs = json.load(f)
    assert isinstance(docs, list)
    assert len(docs) > 0


def test_knowledge_base_document_schema():
    with open(ROOT / "data/knowledge_base.json", encoding="utf-8") as f:
        docs = json.load(f)
    for i, doc in enumerate(docs):
        missing = REQUIRED_KEYS - set(doc)
        assert not missing, f"document {i} missing keys: {missing}"


def test_knowledge_base_unique_ids():
    with open(ROOT / "data/knowledge_base.json", encoding="utf-8") as f:
        docs = json.load(f)
    ids = [doc["id"] for doc in docs]
    assert len(ids) == len(set(ids)), "duplicate document IDs found"
