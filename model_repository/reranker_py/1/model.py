import os

import numpy as np
import torch
import triton_python_backend_utils as pb_utils  # pyright: ignore[reportMissingImports]
from sentence_transformers import CrossEncoder


class TritonPythonModel:
    def initialize(self, _args: dict[str, str]):
        """Called once when the model is loaded."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = os.getenv(
            "RERANKER_MODEL_ID",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )

        print(f"[Reranker] Loading model: {model_id} on {self.device}")
        self.model = CrossEncoder(model_id, device=self.device)
        print("[Reranker] Model loaded")

    def execute(self, requests):  # noqa: ANN001
        """Executed for each inference request."""
        responses = []

        for request in requests:
            query_tensor = pb_utils.get_input_tensor_by_name(request, "query")
            candidates_tensor = pb_utils.get_input_tensor_by_name(request, "candidates")

            query = query_tensor.as_numpy()[0].decode("utf-8")
            candidates = [c.decode("utf-8") for c in candidates_tensor.as_numpy()]

            if candidates:
                pairs = [[query, doc] for doc in candidates]
                scores = self.model.predict(pairs).astype(np.float32)
            else:
                scores = np.empty(0, dtype=np.float32)

            output_tensor = pb_utils.Tensor("scores", scores)
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[output_tensor],
                ),
            )

        return responses

    def finalize(self):
        """Called once when the model is unloaded."""
        print("[Reranker] Shutdown complete")
