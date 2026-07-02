"""Verify .env.example and docs/CONFIGURATION.md stay in sync."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _parse_env_example() -> dict[str, str]:
    env_path = ROOT / ".env.example"
    vars_map: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r'^([A-Z][A-Z0-9_]+)="?([^"]*)"?$', line)
        if match:
            vars_map[match.group(1)] = match.group(2)
    return vars_map


def _config_doc_vars() -> set[str]:
    content = (ROOT / "docs/CONFIGURATION.md").read_text(encoding="utf-8")
    return set(re.findall(r"\| `([A-Z][A-Z0-9_]+)` \|", content))


def test_env_example_vars_documented_in_configuration():
    env_vars = _parse_env_example()
    doc_vars = _config_doc_vars()
    missing = set(env_vars) - doc_vars
    assert not missing, f"Variables in .env.example missing from CONFIGURATION.md: {sorted(missing)}"


def test_configuration_table_has_required_fields():
    content = (ROOT / "docs/CONFIGURATION.md").read_text(encoding="utf-8")
    assert "| Variable | Default | Used by | Description |" in content
    for var in ("QDRANT_URL", "TRITON_URL", "LLM_MODEL_ID", "EMBEDDING_MODEL_ID"):
        assert f"`{var}`" in content


def test_env_example_has_canonical_names():
    env_vars = _parse_env_example()
    assert "EMBEDDING_MODEL_ID" in env_vars
    assert "EMBEDDING_MODEL" not in env_vars
    assert "TRITON_URL" in env_vars
    assert "TRITON_MODEL_NAME" in env_vars
