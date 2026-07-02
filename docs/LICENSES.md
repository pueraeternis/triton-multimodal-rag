# Licenses

This repository is licensed under the [MIT License](../LICENSE). The application code, documentation, and synthetic knowledge base in this repo are covered by that license.

The pipeline downloads and runs several upstream models and libraries, each governed by its own license. **You are responsible for complying with all upstream licenses when using, modifying, or redistributing this project.**

---

## Upstream Models

| Model | HuggingFace / Source ID | License | Notes |
|-------|------------------------|---------|-------|
| **Qwen3-4B-Instruct-2507** (default LLM) | [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | [Qwen License](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/LICENSE) | Accept license on HuggingFace before download |
| **all-MiniLM-L6-v2** (embeddings) | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | [Apache 2.0](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/README.md) | Used via SentenceTransformer |
| **ms-marco-MiniLM-L-6-v2** (reranker) | [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | [Apache 2.0](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | Used via CrossEncoder |
| **YOLOv8n** (vision) | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) | Exported to ONNX for inference |

### Substituting Larger Models

If you swap the default LLM for a larger model (e.g., Qwen3-30B MoE), that model's license applies instead. Check the model card on HuggingFace before use.

---

## Infrastructure Components

| Component | License | Source |
|-----------|---------|--------|
| NVIDIA Triton Inference Server | [BSD-3-Clause](https://github.com/triton-inference-server/server/blob/main/LICENSE) | NVIDIA NGC |
| vLLM | [Apache 2.0](https://github.com/vllm-project/vllm/blob/main/LICENSE) | vLLM project |
| Qdrant | [Apache 2.0](https://github.com/qdrant/qdrant/blob/master/LICENSE) | Qdrant |

---

## Knowledge Base Data

| Dataset | Location | Provenance | License |
|---------|----------|------------|---------|
| Synthetic technical support KB | `data/knowledge_base.json` | Hand-authored sample data for demonstration | MIT (same as repository) |

The knowledge base contains fictional router troubleshooting entries created for pipeline testing. It is **not** derived from proprietary documentation and should not be used as actual technical support guidance.

---

## Usage Caveats

1. **Model licenses are independent** of the repository MIT license. Downloading and running models requires accepting their respective terms on HuggingFace or the upstream source.
2. **YOLOv8 (AGPL-3.0)** may impose copyleft obligations if you distribute modified versions. Review the Ultralytics license for your deployment scenario.
3. **Qwen models** may have usage restrictions for certain commercial applications. Review the Qwen license before production use.
4. **This is a reference implementation** — license compliance for production deployments is the operator's responsibility.
