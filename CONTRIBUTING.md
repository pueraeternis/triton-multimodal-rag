# Contributing

Thank you for your interest in this project. This is a **reference implementation** for learning Triton BLS orchestration patterns — contributions that improve clarity, reproducibility, and educational value are welcome.

---

## Development Setup

### Prerequisites

- NVIDIA GPU with ≥16 GB VRAM (for full pipeline testing)
- Docker with NVIDIA Container Toolkit
- [uv](https://docs.astral.sh/uv/) for Python dependencies

### Local Setup

```bash
git clone <repository-url>
cd triton-multimodal-rag
uv sync --locked --group dev
cp .env.example .env
```

### Makefile Workflow

The documented happy path is available via `make`:

```bash
make help          # list all targets
make export-models # export YOLO ONNX
make init-qdrant   # start Qdrant + upload knowledge base
make up            # build/start Triton
make client        # run inference
make down          # stop containers
```

### Manual Steps

```bash
# Required: YOLO ONNX export
uv run scripts/export_yolo.py

# Optional: embedding ONNX export (not active in BLS path)
uv run scripts/export_embedding.py

# Qdrant
docker compose up -d qdrant
uv run scripts/init_qdrant.py

# Triton
docker compose up -d --build triton
docker logs -f triton-server  # wait for all models READY

# Client
uv run client.py \
  --image data/test_image.jpg \
  --query "Red status LED is blinking continuously on my Router. What to do?"
```

Full setup details: [docs/QUICKSTART.md](docs/QUICKSTART.md)

---

## Automated Validation

### Local CPU tests

```bash
make test            # pytest — no GPU or running services required
make check-config    # .env.example vs docs/CONFIGURATION.md
make smoke-test      # offline file/config checks (CI-safe)
make eval-retrieval  # retrieval/reranking regression check (CPU-only)
```

### Smoke test (online / full)

After Triton is running:

```bash
make smoke-test MODE=online   # Qdrant + Triton readiness
make smoke-test MODE=full     # one end-to-end inference (GPU required)
```

### CI

GitHub Actions runs on every PR and push to `main`:

- Ruff lint and format check
- `pytest` (CPU-only contract tests)
- Offline smoke validation
- `docker compose config` validation

No GPU jobs run in CI. GPU smoke testing is maintainer/local responsibility after merge.

---

## Maintainer End-to-End Validation

Before claiming a release or updating **Validated Platform** in README/QUICKSTART, execute the full [QUICKSTART](docs/QUICKSTART.md) workflow from a **clean clone** on the maintainer validation platform (or document any platform change in `docs/validation/`):

| Step | Action | Pass criterion |
|------|--------|----------------|
| 1 | Fresh clone | Clean working tree |
| 2 | `uv sync --locked` | Completes without error |
| 3 | `make export-models` | `model_repository/yolo_onnx/1/model.onnx` exists |
| 4 | `make init-qdrant` | Collection created; documents uploaded |
| 5 | `make up` | Triton container builds and starts |
| 6 | Wait for READY | All required models READY; `curl localhost:8000/v2/health/ready` succeeds |
| 7 | `make client` | Inference completes without error |
| 8 | Verify response | Client prints AI answer |
| 9 | Verify debug trace | YOLO, Qdrant, reranker, vLLM stages present |
| 10 | Verify documentation | Every README/QUICKSTART command works; update docs if behavior differs |

### Evidence collection

During the run, record from `nvidia-smi`, `uv --version`, `uv run python --version`, image tags, and `.env` model IDs. Transfer exact values into README **Validated Platform** and QUICKSTART **Validated Environment** only after all steps pass. Commit a concise record under `docs/validation/`.

Optional pre-check: `make smoke-test MODE=full` before step 7.

Report validation failures using the [validation failure template](.github/ISSUE_TEMPLATE/validation_failure.md).

---

## Dependency Sources

| Environment | Source |
|-------------|--------|
| Local client & scripts | `pyproject.toml` + `uv.lock` |
| Triton container (BLS Python) | `infra/config/requirements.txt` (pinned from lockfile) |
| Triton container (vLLM, numpy 1.26.4) | `Dockerfile` only — constrained by Triton base image |

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the full environment variable reference.

---

## What to Contribute

| Area | Examples |
|------|----------|
| Documentation | Fix inaccuracies, improve diagrams, add troubleshooting entries |
| Configuration | Sensible defaults, clearer `.env.example` comments |
| Scripts | Export script improvements, better error messages |
| Tests | Contract tests for config/model repository consistency, BLS error handling, retrieval eval |
| Code quality | BLS orchestration, client error display, retrieval pipeline |

---

## Pull Request Guidelines

1. **Keep scope focused** — one logical change per PR
2. **Match existing style** — follow conventions in surrounding code
3. **Update documentation** — if your change affects architecture, config defaults, or setup steps, update the relevant doc
4. **Run local checks** — `make test` and `make smoke-test` must pass; run `make smoke-test MODE=full` if you touch runtime code
5. **No false claims** — do not add compatibility or benchmark claims without maintainer validation evidence

---

## Reporting Issues

- **Bugs:** use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- **Validation failures:** use the [validation failure template](.github/ISSUE_TEMPLATE/validation_failure.md)
- **Security concerns:** see [SECURITY.md](SECURITY.md) — do not file public issues for vulnerabilities

---

## Code of Conduct

Be respectful and constructive. This is an educational reference project — help others learn.
