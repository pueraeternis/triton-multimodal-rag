# Configuration Reference

Environment variables for the multimodal RAG pipeline. Copy [`.env.example`](../.env.example) to `.env` and adjust as needed.

## How `.env` is loaded

The repository uses a **single configuration file** (`.env`) at the project root. Host-side Python and containerized services read the same file through different mechanisms:

| Consumer | Mechanism |
|----------|-----------|
| Host Python (`client.py`, `scripts/*`) | Each entrypoint calls `load_dotenv()` from `python-dotenv` before reading `os.getenv(...)`. `load_dotenv()` discovers `.env` by walking up from the script path (project root for `client.py`; parent directory for scripts under `scripts/`). |
| Triton container (BLS, reranker) | `env_file: .env` in [`docker-compose.yml`](../docker-compose.yml) injects variables into the container environment at startup. |
| Docker Compose port mapping | Compose reads `.env` from the project root for `${TRITON_HTTP_PORT}` and similar substitutions |

Host scripts and the Triton container therefore share one `.env` file: Python loads it explicitly at process start; Docker Compose loads it for services defined in `docker-compose.yml`.

`QDRANT_URL` is the one variable that differs by runtime: host scripts default to `http://localhost:6333` in `.env.example`, while `docker-compose.yml` overrides it to `http://qdrant:6333` inside the Triton container. All other variables pass through unchanged.

| Variable | Default | Used by | Description |
|----------|---------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | `init_qdrant.py`, BLS | Qdrant HTTP endpoint. Use `http://qdrant:6333` inside the Docker network (set automatically for Triton in `docker-compose.yml`). |
| `QDRANT_COLLECTION` | `technical_support` | `init_qdrant.py`, BLS | Vector collection name for retrieval. |
| `DATA_PATH` | `data/knowledge_base.json` | `init_qdrant.py` | Path to the knowledge base JSON file. |
| `TRITON_URL` | `localhost:8000` | `client.py` | Triton HTTP endpoint (host:port, no scheme). |
| `TRITON_MODEL_NAME` | `bls_orchestrator` | `client.py` | Triton model name for inference requests. |
| `EMBEDDING_MODEL_ID` | `sentence-transformers/all-MiniLM-L6-v2` | `init_qdrant.py`, BLS, `export_embedding.py` | HuggingFace SentenceTransformer model ID for embeddings. |
| `EMBEDDING_DEVICE` | `cuda` | BLS | Device for in-process SentenceTransformer (`cuda` or `cpu`). |
| `EMBEDDING_OUTPUT_DIR` | `model_repository/embedding_onnx/1` | `export_embedding.py` | Output directory for ONNX embedding export (not active BLS path). |
| `YOLO_MODEL_NAME` | `yolov8n` | `export_yolo.py` | YOLO variant to export. |
| `YOLO_EXPORT_PATH` | `model_repository/yolo_onnx/1/model.onnx` | `export_yolo.py` | Output path for YOLO ONNX export. |
| `YOLO_TRITON_MODEL_NAME` | `yolo_onnx` | (documentation) | Triton model name for YOLO; referenced in model repository. |
| `YOLO_IMAGE_SIZE` | `640` | (documentation) | Input image size for YOLO preprocessing. |
| `RERANKER_MODEL_ID` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `reranker_py` backend | Cross-encoder model for reranking. |
| `RERANKER_TRITON_MODEL_NAME` | `reranker_py` | (documentation) | Triton model name for reranker. |
| `LLM_MODEL_ID` | `Qwen/Qwen3-4B-Instruct-2507` | BLS tokenizer | HuggingFace model ID for chat template tokenization. Must match `model.json` `"model"` field. |
| `LLM_TRITON_MODEL_NAME` | `llm_vllm` | (documentation) | Triton model name for vLLM backend. |
| `LLM_TEMPERATURE` | `0.1` | (Plan 03) | Generation temperature — defined here; wired into BLS vLLM requests in Plan 03. |
| `LLM_MAX_TOKENS` | `512` | (Plan 03) | Maximum tokens to generate — defined here; wired in Plan 03. |
| `LLM_TOP_P` | `0.95` | (Plan 03) | Top-p sampling — defined here; wired in Plan 03. |
| `LOG_LEVEL` | `INFO` | export/init scripts | Python logging level. |

### Docker Compose overrides

Optional host port mappings (not in `.env.example`; set as shell env vars when invoking `docker compose`):

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_HTTP_PORT` | `8000` | Host port for Triton HTTP |
| `TRITON_GRPC_PORT` | `8001` | Host port for Triton gRPC |
| `TRITON_METRICS_PORT` | `8002` | Host port for Triton metrics |

## Container image pins

| Component | Pin | Source |
|-----------|-----|--------|
| Triton | `25.05-py3` | `Dockerfile` `TRITON_VERSION` |
| vLLM (Python) | `0.10.2` | `Dockerfile` `VLLM_VERSION` |
| vLLM backend | `b41f716d15100dc7bcbea27ebea20906452dadf5` | `Dockerfile` / `docker-compose.yml` `VLLM_BACKEND_SHA` (branch `r25.05`) |
| Qdrant | `v1.16.3` | `docker-compose.yml` |

Exact versions from the maintainer validation run are recorded in [QUICKSTART.md](QUICKSTART.md#validated-environment).

## Dependency sources

| Environment | Source of truth |
|-------------|-----------------|
| Local client & scripts | `pyproject.toml` + `uv.lock` (`uv sync --locked`) |
| Triton container (BLS Python deps) | `infra/config/requirements.txt` (pinned from lockfile) |
| Triton container (vLLM, numpy) | `Dockerfile` only — constrained by Triton base image |
