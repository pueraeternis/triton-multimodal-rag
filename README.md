# High-Performance Multimodal RAG with NVIDIA Triton & vLLM

![Python](https://img.shields.io/badge/Python-3.12-blue)
![NVIDIA Triton](https://img.shields.io/badge/NVIDIA%20Triton-25.05-green)
![vLLM](https://img.shields.io/badge/vLLM-0.10.2-orange)
![Qdrant](https://img.shields.io/badge/Qdrant-v1.10-red)

An enterprise-grade **Multimodal Retrieval-Augmented Generation (RAG)** pipeline designed for automated technical support. The system processes visual data (equipment photos) and textual queries to retrieve specific repair instructions, utilizing a single GPU (NVIDIA A100) efficiently.

This project demonstrates **System Design** capabilities by orchestrating multiple heterogeneous models (Vision, Embedding, Ranking, LLM) within a single Inference Server using **Business Logic Scripting (BLS)**.

---

## 🏗 Architecture

```mermaid
flowchart LR
    Client([Client Request]) -->|gRPC/HTTP| Triton[Triton Inference Server]
    
    subgraph Triton [Triton Inference Server (BLS Orchestrator)]
        direction TB
        Orchestrator[BLS Python Backend]
        
        subgraph Models
            YOLO[YOLOv8 Vision]
            Embed[SentenceTransformer]
            Rerank[Cross-Encoder]
            LLM[vLLM / Qwen-3]
        end
        
        Orchestrator -->|1. Check Image| YOLO
        Orchestrator -->|2. Vectorize Query| Embed
        Orchestrator -->|4. Re-rank Docs| Rerank
        Orchestrator -->|5. Generate Answer| LLM
    end
    
    Embed <-->|3. ANN Search| Qdrant[(Qdrant DB)]
```

The pipeline is implemented as a **Microservices-in-a-Monolith** pattern within NVIDIA Triton Inference Server. This approach minimizes network overhead by keeping tensor movement within the GPU memory/shared memory.

### Data Flow (BLS Orchestrator)

1.  **Input:** User provides an image (e.g., a router with a red light) and a text query.
2.  **Vision Guardrail:** `YOLOv8` (ONNX) scans the image.
    *   *Purpose:* Fast filtering and context grounding.
3.  **Retrieval:** `SentenceTransformers` converts the query to vectors -> Search in **Qdrant** (Vector DB).
4.  **Reranking:** A `Cross-Encoder` model refines the top-5 candidates to find the single most relevant manual page.
5.  **Reasoning:** `Qwen-3-30B-Instruct` (served via **vLLM**) generates the final answer based on the retrieved context.

---

## 🚀 Key Features

*   **Unified Inference Platform:** All models (Vision, NLP, LLM) run on a single Triton instance.
*   **Business Logic Scripting (BLS):** Complex DAG execution logic (conditionals, loops) is handled inside the server via Python Backend, reducing client-side complexity and latency.
*   **State-of-the-Art LLM Serving:** Uses **vLLM** backend with `decoupled` mode for high-throughput continuous batching.
*   **Precision RAG:** Implements a **Retrieve-then-Rerank** strategy to minimize hallucinations.
*   **Production Ready:**
    *   Fully reproducible environment using `uv` and Docker.
    *   Metric export (Prometheus/Grafana ready).
    *   Comprehensive tracing/logging for observability.

---

## 🛠 Tech Stack

*   **Orchestration:** NVIDIA Triton Inference Server (25.05)
*   **LLM Engine:** vLLM (supporting Qwen 3 MoE)
*   **Vector Database:** Qdrant
*   **Models:**
    *   **LLM:** `Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE)
    *   **Vision:** `YOLOv8n` (ONNX export with dynamic batching)
    *   **Embeddings:** `all-MiniLM-L6-v2`
    *   **Reranker:** `ms-marco-MiniLM-L-6-v2`
*   **Dependency Management:** uv

---

## 📂 Project Structure

```text
.
├── client.py                # Client script with trace reporting
├── data                     # Test images and knowledge base source
├── docker-compose.yml       # Infrastructure (Triton + Qdrant)
├── Dockerfile               # Custom Triton image with vLLM and dependencies
├── model_repository         # Triton Model Store
│   ├── bls_orchestrator     # Python Logic (The "Brain")
│   ├── llm_vllm             # Qwen 3 (vLLM Backend)
│   ├── reranker_py          # Cross-Encoder (Python Backend)
│   └── yolo_onnx            # Vision Model (ONNX Backend)
├── scripts                  # ETL and Model Export scripts
└── pyproject.toml           # Python dependencies
```

---

## ⚡ Quick Start

### 1. Prerequisites
*   NVIDIA GPU (Ampere or newer recommended, e.g., A100/H100)
*   Docker & NVIDIA Container Toolkit
*   `uv` installed

### 2. Setup
Clone the repository and install dependencies:
```bash
uv sync
source .venv/bin/activate
```

### 3. Data Preparation
Prepare the models and the vector database:
```bash
# Export YOLO to ONNX
uv run scripts/export_yolo.py

# Initialize Qdrant and upload knowledge base
docker compose up -d qdrant
uv run scripts/init_qdrant.py
```

### 4. Run Inference Server
Build and start the Triton container:
```bash
docker compose up -d --build triton
```
*Wait until all models show `READY` status in the logs.*

### 5. Run Client
Test the pipeline with a multimodal query:
```bash
uv run client.py \
  --image data/test_image.jpg \
  --query "Red status LED is blinking continuously on my Router. What to do?"
```

## ⚙️ Configuration

The project follows the [12-Factor App](https://12factor.net/config) methodology. Configuration is managed via environment variables.

1. Copy the example configuration:
   ```bash
   cp .env.example .env
   ```

2. Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_URL` | Vector DB endpoint | `http://localhost:6333` (Local) |
| `LLM_MODEL_ID` | Model used by vLLM | `Qwen/Qwen3-30B...` |
| `LLM_TEMPERATURE` | Creativity of the generation | `0.1` |
| `YOLO_MODEL_NAME` | Vision model version | `yolov8n` |

> **Note:** The `docker-compose.yml` is configured to inject these environment variables into the Triton container automatically ensuring consistency between local development and containerized execution.

---

## 📊 Example Output

The system provides a detailed execution trace for observability:

```text
============================================================
🕵️  PIPELINE EXECUTION REPORT
============================================================
Query: Red status LED is blinking continuously on my Router. What to do?
------------------------------------------------------------
🔹 [YOLOv8 (Vision)] -> 47.31ms
------------------------------------------------------------
🔹 [Qdrant (Retrieval)] -> 13.64ms
   Found: 5 docs
   Top-1: [Router] Red status LED blinking continuously...
------------------------------------------------------------
🔹 [Cross-Encoder (Reranker)] -> 10.27ms
   Best Score: -2.6081
   Context Used: "Check the router logs to identify the specific error code..."
------------------------------------------------------------
🔹 [vLLM (Generation)] -> 4496.82ms
------------------------------------------------------------
⏱  Total Latency: 4.57s
============================================================
🤖 AI RESPONSE:
If the red status LED on your router is blinking continuously, follow these steps:

1. **Check Router Logs:** Access your router’s web interface to look for error messages.
2. **Verify Power Stability:** Ensure the router is plugged into a stable outlet.
3. **Reboot:** Power off, wait 30 seconds, and power on.
4. **Update Firmware:** Check the manufacturer’s website for updates.
...
```

---

## 💡 Design Decisions

*   **Why BLS?** Instead of chaining microservices via HTTP (which adds network latency), BLS allows the pipeline to run entirely within the C++ backend of Triton, sharing memory pointers where possible.
*   **Why Qwen 3?** It offers SOTA performance with a Mixture-of-Experts (MoE) architecture, providing the reasoning capabilities of larger models with significantly lower inference costs.
*   **Why Reranking?** Vector search relies on cosine similarity, which captures general semantic meaning. The Cross-Encoder reranker ensures that the specific *nuance* of the query matches the retrieved document, significantly improving RAG accuracy.