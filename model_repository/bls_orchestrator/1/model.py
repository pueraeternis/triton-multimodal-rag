import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils  # pyright: ignore[reportMissingImports]
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant-server:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "technical_support")

EMBEDDING_MODEL_ID = os.getenv(
    "EMBEDDING_MODEL_ID",
    "sentence-transformers/all-MiniLM-L6-v2",
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")

LLM_MODEL_ID = os.getenv(
    "LLM_MODEL_ID",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
)

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))


class TritonPythonModel:
    def initialize(self, _args: dict[str, str]):
        self.qdrant = QdrantClient(url=QDRANT_URL)
        self.collection_name = QDRANT_COLLECTION

        self.embedder = SentenceTransformer(
            EMBEDDING_MODEL_ID,
            device=EMBEDDING_DEVICE,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_ID,
            trust_remote_code=True,
        )

    def execute(self, requests: list[pb_utils.InferenceRequest]):
        responses = []

        for request in requests:
            query_tensor = pb_utils.get_input_tensor_by_name(request, "query")
            query_text = query_tensor.as_numpy()[0].decode("utf-8")

            image_tensor = pb_utils.get_input_tensor_by_name(request, "image")

            # Vision guardrail (YOLO)
            yolo_request = pb_utils.InferenceRequest(
                model_name="yolo_onnx",
                requested_output_names=["output0"],
                inputs=[pb_utils.Tensor("images", image_tensor.as_numpy())],
            )
            _ = yolo_request.exec()

            # Retrieval
            query_vector = self.embedder.encode(query_text).tolist()
            search_result = self.qdrant.search(  # pyright: ignore[reportAttributeAccessIssue]
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=5,
            )

            candidates = [hit.payload.get("solution", "") for hit in search_result]

            # Reranking
            if candidates:
                rerank_inputs = [
                    pb_utils.Tensor(
                        "query",
                        np.array([query_text.encode("utf-8")], dtype=np.object_),
                    ),
                    pb_utils.Tensor(
                        "candidates",
                        np.array(
                            [c.encode("utf-8") for c in candidates],
                            dtype=np.object_,
                        ),
                    ),
                ]

                rerank_request = pb_utils.InferenceRequest(
                    model_name="reranker_py",
                    requested_output_names=["scores"],
                    inputs=rerank_inputs,
                )

                rerank_response = rerank_request.exec()
                best_context = candidates[0]

                if not rerank_response.has_error():
                    scores = pb_utils.get_output_tensor_by_name(
                        rerank_response,
                        "scores",
                    ).as_numpy()
                    best_context = candidates[int(np.argmax(scores))]
            else:
                best_context = "No relevant instructions found in the knowledge base."

            # Prompt construction
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful technical support assistant. Be concise.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Context from manuals:\n{best_context}\n\nUser question:\n{query_text}\n\nProvide a solution based on the context."
                    ),
                },
            ]

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            sampling_params = json.dumps(
                {
                    "temperature": LLM_TEMPERATURE,
                    "max_tokens": LLM_MAX_TOKENS,
                    "top_p": LLM_TOP_P,
                },
            )

            llm_inputs = [
                pb_utils.Tensor(
                    "text_input",
                    np.array([prompt.encode("utf-8")], dtype=np.object_),
                ),
                pb_utils.Tensor(
                    "sampling_parameters",
                    np.array([sampling_params.encode("utf-8")], dtype=np.object_),
                ),
                pb_utils.Tensor(
                    "stream",
                    np.array([False], dtype=bool),
                ),
            ]

            llm_request = pb_utils.InferenceRequest(
                model_name="llm_vllm",
                requested_output_names=["text_output"],
                inputs=llm_inputs,
            )

            llm_response = llm_request.exec()

            if llm_response.has_error():
                final_text = f"LLM error: {llm_response.error().message()}"
            else:
                output_tensor = pb_utils.get_output_tensor_by_name(
                    llm_response,
                    "text_output",
                )
                final_text = output_tensor.as_numpy()[0].decode("utf-8")

            response_tensor = pb_utils.Tensor(
                "response",
                np.array([final_text.encode("utf-8")], dtype=np.object_),
            )

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[response_tensor],
                ),
            )

        return responses

    def finalize(self):
        pass
