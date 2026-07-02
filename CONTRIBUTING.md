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
uv sync
cp .env.example .env
```

### Export Models

```bash
# Required: YOLO ONNX export
uv run scripts/export_yolo.py

# Optional: embedding ONNX export (not active in BLS path)
uv run scripts/export_embedding.py
```

### Initialize Qdrant

```bash
docker compose up -d qdrant
uv run scripts/init_qdrant.py
```

### Start Triton

```bash
docker compose up -d --build triton
docker logs -f triton-server  # wait for all models READY
```

### Run the Client

```bash
uv run client.py \
  --image data/test_image.jpg \
  --query "Red status LED is blinking continuously on my Router. What to do?"
```

Or via the project entrypoint:

```bash
uv run main.py --image data/test_image.jpg --query "Your query here"
```

Full setup details: [docs/QUICKSTART.md](docs/QUICKSTART.md)

---

## What to Contribute

| Area | Examples |
|------|----------|
| Documentation | Fix inaccuracies, improve diagrams, add troubleshooting entries |
| Configuration | Sensible defaults, clearer `.env.example` comments |
| Scripts | Export script improvements, better error messages |
| Code quality | BLS error handling, embedding path resolution (see Plan 03) |

---

## What Is Planned (Not Yet Available)

The following are tracked in the implementation plans and are **not** expected in drive-by contributions without prior discussion:

| Capability | Plan |
|------------|------|
| Automated tests (smoke, CPU-only) | [Plan 02](docs/plans/plan-02-reproducibility-validation.md) |
| GitHub Actions CI | [Plan 02](docs/plans/plan-02-reproducibility-validation.md) |
| Dependency pinning and lockfile consolidation | [Plan 02](docs/plans/plan-02-reproducibility-validation.md) |
| Embedding path resolution | [Plan 03](docs/plans/plan-03-engineering-hardening.md) |
| Structured BLS error handling | [Plan 03](docs/plans/plan-03-engineering-hardening.md) |
| Retrieval evaluation metrics | [Plan 03](docs/plans/plan-03-engineering-hardening.md) |

---

## Pull Request Guidelines

1. **Keep scope focused** — one logical change per PR
2. **Match existing style** — follow conventions in surrounding code
3. **Update documentation** — if your change affects architecture, config defaults, or setup steps, update the relevant doc
4. **No false claims** — do not add "production ready", CI badges, or benchmark claims unless the artifacts exist
5. **Test locally** — verify Triton starts and the client runs if your change touches runtime code

---

## Reporting Issues

- **Bugs:** use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- **Validation failures:** use the [validation failure template](.github/ISSUE_TEMPLATE/validation_failure.md)
- **Security concerns:** see [SECURITY.md](SECURITY.md) — do not file public issues for vulnerabilities

---

## Code of Conduct

Be respectful and constructive. This is an educational reference project — help others learn.
