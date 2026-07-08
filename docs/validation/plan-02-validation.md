# Plan 02 Validation Evidence

Engineering record for maintainer validation of the Plan 02 reproducibility workflow. This document summarizes durable outcomes; it is not a raw log dump.

## Validated platform (tested)

| Field | Value |
|-------|-------|
| Validation date (initial E2E) | 2026-07-02 |
| Validation date (follow-up re-check) | 2026-07-08 |
| Platform | Linux x86_64 (Ubuntu 6.8.0-90-generic) |
| GPU | NVIDIA A100-SXM4-80GB |
| NVIDIA driver | 575.51.03 |
| CUDA | 12.9 (toolkit 12.0 build on host) |
| Python | 3.12.12 |
| uv | 0.9.7 |
| Triton image | `25.05-py3` |
| vLLM | 0.10.2 |
| vLLM backend SHA | `b41f716d15100dc7bcbea27ebea20906452dadf5` |
| Qdrant image | `v1.16.3` |
| LLM model ID | `Qwen/Qwen3-4B-Instruct-2507` |
| Embedding model ID | `sentence-transformers/all-MiniLM-L6-v2` |

**Note:** This GPU is the maintainer validation platform. Consumer GPUs with ~16–24 GB VRAM are *expected-compatible* for the default configuration but were not individually tested. See [QUICKSTART — Validated Environment](../QUICKSTART.md#validated-environment).

## Executed validation steps

### Initial maintainer run (2026-07-02)

| Step | Command | Result |
|------|---------|--------|
| Install dependencies | `uv sync --locked` | Pass |
| Configure environment | `cp .env.example .env` | Pass |
| Export YOLO | `make export-models` | Pass |
| Initialize Qdrant | `make init-qdrant` | Pass |
| Start services | `make up` | Pass |
| Service readiness | `make smoke-test MODE=online` | Pass |
| End-to-end inference | `make smoke-test MODE=full` | Pass |
| Client execution | `make client` | Pass |
| Documentation review | README + QUICKSTART commands verified | Pass |

### Follow-up engineering re-check (2026-07-08)

After Plan 02 follow-up changes (`.env` wiring, locked sync, documentation):

| Step | Command | Result |
|------|---------|--------|
| Locked dependency install | `uv sync --locked --group dev` | Pass |
| Lint | `uv run ruff check .` | Pass |
| Format | `uv run ruff format --check .` | Pass |
| CPU tests | `uv run pytest -q` (14 tests) | Pass |
| Offline smoke | `uv run scripts/smoke_test.py` | Pass |
| Config contract | `make check-config` | Pass |
| Compose config | `docker compose config --quiet` | Pass |
| Makefile targets | `make help` | Pass |
| Online smoke | `QDRANT_URL=http://localhost:6333 TRITON_URL=localhost:8010 make smoke-test MODE=online` | Pass |
| Full inference smoke | `… make smoke-test MODE=full` | Pass (2626-char answer, 4 trace steps) |
| Client execution | `… make client` | Pass (YOLO, Qdrant, reranker, vLLM trace stages) |
| `.env` → Triton container | `docker compose config` shows `env_file: .env` with BLS/reranker vars | Pass |
| Documentation review | README + QUICKSTART validated-platform wording | Pass |

Port `8010` was used for the 2026-07-08 re-check because another unrelated Triton instance occupied the default port `8000` on the maintainer host.

## Outcomes

- **Smoke validation (offline):** all file, schema, and config-doc checks passed.
- **Smoke validation (online + full):** Qdrant, Triton, all required models, and one end-to-end inference passed on the maintainer A100.
- **Client execution:** pipeline trace included YOLO, Qdrant retrieval, cross-encoder reranking, and vLLM generation.
- **README and QUICKSTART:** commands and validated-platform vs expected-compatible wording verified.

## Remaining known limitations

- Validated only on NVIDIA A100-SXM4-80GB; 16–24 GB consumer GPUs are expected-compatible, not tested here.
- CI remains CPU-only; no Triton image build in GitHub Actions.
- `embedding_onnx` is exported but intentionally not on the active BLS path (in-process `SentenceTransformer` is used by design).
- Host-side scripts require `QDRANT_URL=http://localhost:6333` in `.env`; Triton container overrides to `http://qdrant:6333` automatically.
