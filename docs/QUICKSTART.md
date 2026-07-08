# Quickstart

Step-by-step guide to run the multimodal RAG pipeline on a single GPU with the default **Qwen3-4B-Instruct-2507** LLM.

---

## Who This Guide Is For

This guide assumes **Linux**, **Docker**, an **NVIDIA GPU**, and basic familiarity with **NVIDIA Triton Inference Server**. If you are new to the repository, start with [README.md](../README.md) for an overview before following the steps below.

---

## Prerequisites

### Hardware

| Requirement | Details |
|-------------|---------|
| GPU | NVIDIA GPU with **16–24 GB VRAM**. Cards such as RTX 3090, RTX 4090, and A10 are typical fits. |
| Disk | Approximately 15 GB free for HuggingFace model weights and Docker images |
| RAM | 16 GB or more system memory recommended |

### Software

| Tool | Purpose |
|------|---------|
| [Docker](https://docs.docker.com/get-docker/) | Run Qdrant and Triton containers |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | GPU access in Docker |
| [uv](https://docs.astral.sh/uv/) | Python dependency management |

### VRAM Budget (Default Configuration)

| Component | Approximate VRAM |
|-----------|-----------------|
| Qwen3-4B-Instruct-2507 (vLLM) | ~8 GB |
| YOLOv8n | ~0.5 GB |
| Cross-encoder reranker | ~0.5 GB |
| SentenceTransformer | ~0.5 GB |
| CUDA overhead | ~1–2 GB |
| **Total** | **~10–14 GB** |

### Download Sizes

Sizes below are approximate and may vary with model revisions, caching, and dependency updates.

| Asset | Approximate Size |
|-------|-----------------|
| Qwen3-4B-Instruct-2507 weights | ~8 GB |
| all-MiniLM-L6-v2 | ~100 MB |
| ms-marco-MiniLM-L-6-v2 | ~100 MB |
| YOLOv8n | ~10 MB |
| Triton Docker image | ~10 GB (first build) |

### Cold-Start Duration

The first startup may take **several minutes** because of:

1. **Docker image build** — longer on first run; cached on subsequent builds
2. **HuggingFace model downloads** — depends on network speed and cache state
3. **Triton initialization** — loading all model backends into memory
4. **CUDA warmup** — kernel compilation on first inference

Subsequent starts are typically faster once images and weights are cached locally.

---

## Setup

### 1. Clone and Install Dependencies

```bash
git clone <repository-url>
cd triton-multimodal-rag
uv sync --locked
```

Or use `make help` after install to see all workflow targets.

### 2. Configure Environment

```bash
cp .env.example .env
```

Review `.env` — defaults target Qwen3-4B-Instruct-2507 on a local GPU. Host scripts and the Triton container read this file automatically; only `QDRANT_URL` is overridden inside the Triton container (`http://qdrant:6333`). Key variables:

| Variable | Default | Notes |
|----------|---------|-------|
| `QDRANT_URL` | `http://localhost:6333` | Use `http://qdrant:6333` only inside Docker network |
| `LLM_MODEL_ID` | `Qwen/Qwen3-4B-Instruct-2507` | Must match `model.json` for consistent tokenization |

### 3. Export YOLO Model

```bash
make export-models
# or: uv run scripts/export_yolo.py
```

Expected output: `model_repository/yolo_onnx/1/model.onnx` created.

### 4. Start Qdrant and Initialize Vector DB

```bash
make init-qdrant
# or manually:
# docker compose up -d qdrant
# uv run scripts/init_qdrant.py
```

Expected log snippet:

```text
Collection 'technical_support' created
Uploaded N documents to Qdrant
```

Verify Qdrant is reachable:

```bash
curl -s http://localhost:6333/collections/technical_support | head
```

### 5. Build and Start Triton

```bash
make up
# or: docker compose up -d --build triton
```

Monitor startup:

```bash
docker logs -f triton-server
```

Wait until all models report `READY`:

```text
| yolo_onnx        | 1       | READY  |
| reranker_py      | 1       | READY  |
| llm_vllm         | 1       | READY  |
| bls_orchestrator | 1       | READY  |
```

> `embedding_onnx` may show `UNAVAILABLE` if the ONNX file has not been exported. This is expected — the active path uses in-process SentenceTransformer.

Optional smoke check before inference:

```bash
make smoke-test MODE=online   # service readiness
make smoke-test MODE=full     # one end-to-end inference (requires GPU)
```

Readiness check:

```bash
curl -s localhost:8000/v2/health/ready
```

### 6. Run Inference

```bash
make client
# or:
uv run client.py \
  --image data/test_image.jpg \
  --query "Red status LED is blinking continuously on my Router. What to do?"
```

Alternative entrypoint:

```bash
uv run main.py \
  --image data/test_image.jpg \
  --query "Red status LED is blinking continuously on my Router. What to do?"
```

---

## Expected Output

The client prints a per-stage trace and the generated answer. The example below illustrates the response structure; latencies are approximate and will vary by hardware:

```text
============================================================
🕵️  PIPELINE EXECUTION REPORT
============================================================
Query: Red status LED is blinking continuously on my Router. What to do?
------------------------------------------------------------
🔹 [YOLOv8 (Vision)] -> ~600ms
------------------------------------------------------------
🔹 [Qdrant (Retrieval)] -> ~350ms
   Found: 5 docs
   Top-1: [Router] Red status LED blinking continuously...
------------------------------------------------------------
🔹 [Cross-Encoder (Reranker)] -> ~240ms
   Best Score: (varies)
   Context Used: "Check the router logs to identify the specific error code..."
------------------------------------------------------------
🔹 [vLLM (Generation)] -> ~3.8s
------------------------------------------------------------
⏱  Total Latency: ~5s
============================================================
🤖 AI RESPONSE:
Red status LED blinking continuously on your router typically indicates a critical error...
============================================================
```

Latencies above are from maintainer validation on NVIDIA A100-SXM4-80GB (2026-07-02); your hardware will differ.

Raw JSON response schema: [OBSERVABILITY.md](OBSERVABILITY.md)

---

## Troubleshooting

### Qdrant Storage Version Mismatch

**Symptoms:** Qdrant container exits on startup after changing the pinned image version; logs mention deserialize or shard holder errors.

**Fixes:**
- Remove incompatible persisted data: `docker compose down` then clear `infra/qdrant_storage` (files are owned by the container — use `docker run --rm -v $(pwd)/infra/qdrant_storage:/data alpine sh -c "rm -rf /data/*"`)
- Re-run `make init-qdrant`

### CUDA Out of Memory

**Symptoms:** Triton logs show `CUDA out of memory` during `llm_vllm` loading.

**Fixes:**
- Reduce `gpu_memory_utilization` in `model_repository/llm_vllm/1/model.json` (e.g., `0.70`)
- Reduce `max_model_len` (e.g., `4096`)
- Ensure no other GPU processes are running (`nvidia-smi`)
- Use a smaller LLM or a GPU with more VRAM

### Qdrant Unreachable

**Symptoms:** BLS trace shows retrieval errors; `curl localhost:6333` fails.

**Fixes:**
- Verify Qdrant is running: `docker compose ps qdrant`
- Check `QDRANT_URL` in `.env` matches your setup (`localhost:6333` for host-side scripts)
- Inside the Triton container, Qdrant is at `http://qdrant:6333` (set via `docker-compose.yml`)

### Triton Models Not Ready

**Symptoms:** `curl localhost:8000/v2/health/ready` returns not-ready; client gets connection errors.

**Fixes:**
- Check logs: `docker logs triton-server 2>&1 | tail -50`
- Ensure `model_repository/yolo_onnx/1/model.onnx` exists (run export script)
- Verify NVIDIA Container Toolkit is installed and GPU is visible: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`
- First LLM load can take several minutes — wait for `READY` in logs
- If port 8000 is already in use, set `TRITON_HTTP_PORT` (and `TRITON_URL` for the client) — see [CONFIGURATION.md](CONFIGURATION.md)

### HuggingFace Authentication

**Symptoms:** `401 Unauthorized` or `Repository Not Found` during model download.

**Fixes:**
- Some models require accepting the license on HuggingFace before download
- Set the `HF_TOKEN` environment variable: `export HF_TOKEN=hf_...`
- Mount the token in Docker: add `HF_TOKEN` to the `docker-compose.yml` environment section
- Verify model ID spelling in `.env` and `model.json`

### Empty or Nonsensical Answers

**Symptoms:** LLM generates irrelevant text.

**Fixes:**
- Verify Qdrant was initialized: `uv run scripts/init_qdrant.py`
- Check that `LLM_MODEL_ID` in `.env` matches `"model"` in `model.json`
- Review reranker scores in the trace — low scores may indicate poor retrieval

---

## Validated Environment

Evidence from maintainer end-to-end validation on **2026-07-02**. Full engineering record: [validation/plan-02-validation.md](validation/plan-02-validation.md).

### Validated platform (tested)

| Field | Value |
|-------|-------|
| GPU | NVIDIA A100-SXM4-80GB |
| NVIDIA driver | 575.51.03 |
| CUDA | 12.9 |
| Python | 3.12.12 |
| uv | 0.9.7 |
| Triton image | `25.05-py3` |
| vLLM | `0.10.2` |
| vLLM backend SHA | `b41f716d15100dc7bcbea27ebea20906452dadf5` |
| Qdrant | `v1.16.3` |
| LLM model ID | `Qwen/Qwen3-4B-Instruct-2507` |
| Embedding model ID | `sentence-transformers/all-MiniLM-L6-v2` |
| Validation date | 2026-07-02 |

### Expected-compatible platforms (not individually tested)

Consumer GPUs with **16–24 GB VRAM** are **expected** to work based on the documented VRAM budget for the default Qwen3-4B-Instruct-2507 configuration. Illustrative fits: RTX 3090, RTX 4090, A10, L40S.

These platforms are **not validated** unless listed in the table above. Expected compatibility is not the same as the maintainer A100 run. Match driver and CUDA requirements for your chosen Triton and vLLM versions.

---

## Next Steps

- [README.md](../README.md) — repository overview
- [ARCHITECTURE.md](ARCHITECTURE.md) — system design and serving boundaries
- [MODEL_REPOSITORY.md](MODEL_REPOSITORY.md) — per-model tensor specifications
- [OBSERVABILITY.md](OBSERVABILITY.md) — trace schema and Prometheus metrics
- [CONTRIBUTING.md](../CONTRIBUTING.md) — development workflow
