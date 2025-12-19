import json
import os
import time

import numpy as np
import triton_python_backend_utils as pb_utils  # pyright: ignore[reportMissingImports]
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant-server:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "technical_support")

EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "Qwen/Qwen3-30B-A3B-Instruct-2507")


class TritonPythonModel:
    def initialize(self, _args: dict[str, str]):
        self.qdrant = QdrantClient(url=QDRANT_URL)
        self.collection_name = QDRANT_COLLECTION
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_ID, device=EMBEDDING_DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)

    def execute(self, requests):  # noqa: ANN001, PLR0915
        responses = []
        for request in requests:
            trace = {"steps": []}
            t0_total = time.time()

            # --- Input ---
            query_tensor = pb_utils.get_input_tensor_by_name(request, "query")
            query_text = query_tensor.as_numpy()[0].decode("utf-8")
            image_tensor = pb_utils.get_input_tensor_by_name(request, "image")

            trace["input_query"] = query_text

            # --- 1. Vision (YOLO) ---
            t0 = time.time()
            yolo_request = pb_utils.InferenceRequest(
                model_name="yolo_onnx",
                requested_output_names=["output0"],
                inputs=[pb_utils.Tensor("images", image_tensor.as_numpy())],
            )
            yolo_response = yolo_request.exec()
            yolo_status = "Success" if not yolo_response.has_error() else f"Error: {yolo_response.error().message()}"
            yolo_details = "Output resided on GPU (Optimization)"

            trace["steps"].append(
                {
                    "component": "YOLOv8 (Vision)",
                    "latency_ms": round((time.time() - t0) * 1000, 2),
                    "status": yolo_status,
                    "details": yolo_details,
                },
            )

            # --- 2. Retrieval (Qdrant) ---
            t0 = time.time()
            query_vector = self.embedder.encode(query_text).tolist()

            search_response = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=5,
                with_payload=True,
            )

            raw_candidates = []
            for hit in search_response.points:
                payload = hit.payload or {}
                raw_candidates.append(
                    {
                        "score": hit.score,
                        "category": payload.get("category", "N/A"),
                        "issue": payload.get("issue_description", "N/A")[:50] + "...",
                        "full_solution": payload.get("solution_text", ""),
                    },
                )

            candidates_text = [c["full_solution"] for c in raw_candidates if c["full_solution"]]

            trace["steps"].append(
                {
                    "component": "Qdrant (Retrieval)",
                    "latency_ms": round((time.time() - t0) * 1000, 2),
                    "candidates_found": len(raw_candidates),
                    "top_candidate_preview": raw_candidates[0] if raw_candidates else None,
                },
            )

            # --- 3. Reranking ---
            t0 = time.time()
            best_context = "No relevant instructions found."
            rerank_score = 0.0

            if candidates_text:
                rerank_inputs = [
                    pb_utils.Tensor("query", np.array([query_text.encode("utf-8")], dtype=np.object_)),
                    pb_utils.Tensor("candidates", np.array([c.encode("utf-8") for c in candidates_text], dtype=np.object_)),
                ]
                rerank_req = pb_utils.InferenceRequest(model_name="reranker_py", requested_output_names=["scores"], inputs=rerank_inputs)
                rerank_resp = rerank_req.exec()

                if not rerank_resp.has_error():
                    scores = pb_utils.get_output_tensor_by_name(rerank_resp, "scores").as_numpy()
                    if len(scores) > 0:
                        best_idx = int(np.argmax(scores))
                        best_context = candidates_text[best_idx]
                        rerank_score = float(scores[best_idx])

            trace["steps"].append(
                {
                    "component": "Cross-Encoder (Reranker)",
                    "latency_ms": round((time.time() - t0) * 1000, 2),
                    "best_score": round(rerank_score, 4),
                    "selected_context_preview": best_context[:100] + "...",
                },
            )

            # --- 4. LLM Generation ---
            t0 = time.time()
            messages = [
                {"role": "system", "content": "You are a helpful technical support assistant."},
                {"role": "user", "content": f"Context: {best_context}\n\nQuestion: {query_text}\n\nSolution:"},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # vLLM params
            sampling_params = json.dumps({"temperature": 0.1, "max_tokens": 512})
            llm_inputs = [
                pb_utils.Tensor("text_input", np.array([prompt.encode("utf-8")], dtype=np.object_)),
                pb_utils.Tensor("sampling_parameters", np.array([sampling_params.encode("utf-8")], dtype=np.object_)),
                pb_utils.Tensor("stream", np.array([False], dtype=bool)),
            ]

            llm_req = pb_utils.InferenceRequest(model_name="llm_vllm", requested_output_names=["text_output"], inputs=llm_inputs)
            llm_responses = llm_req.exec(decoupled=True)

            final_text = ""
            for r in llm_responses:
                if not r.has_error():
                    out = pb_utils.get_output_tensor_by_name(r, "text_output")
                    if out:
                        final_text += out.as_numpy()[0].decode("utf-8")

            trace["steps"].append(
                {
                    "component": "vLLM (Generation)",
                    "latency_ms": round((time.time() - t0) * 1000, 2),
                    "generated_length": len(final_text),
                },
            )

            trace["total_latency_ms"] = round((time.time() - t0_total) * 1000, 2)  # pyright: ignore[reportArgumentType]

            response_payload = json.dumps(
                {
                    "answer": final_text,
                    "debug": trace,
                },
            )

            out_tensor = pb_utils.Tensor("response", np.array([response_payload.encode("utf-8")], dtype=np.object_))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self):
        pass
