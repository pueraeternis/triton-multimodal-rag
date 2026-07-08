import json
import os
import time

import numpy as np
import triton_python_backend_utils as pb_utils  # pyright: ignore[reportMissingImports]
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "technical_support")

EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")

STAGE_OK = "ok"
STAGE_DEGRADED = "degraded"
STAGE_FAILED = "failed"

EXPECTED_IMAGE_SHAPE = (1, 3, 640, 640)
DEFAULT_CONTEXT = "No relevant instructions found."

DEFAULT_LLM_TEMPERATURE = 0.1
DEFAULT_LLM_MAX_TOKENS = 512
DEFAULT_LLM_TOP_P = 0.95


def _parse_env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, str | None]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default, None
    try:
        value = float(raw.strip())
    except ValueError:
        return default, f"Invalid {name}={raw!r}; using default {default}"
    if minimum is not None and value < minimum:
        return default, f"Invalid {name}={value}; must be >= {minimum}; using default {default}"
    if maximum is not None and value > maximum:
        return default, f"Invalid {name}={value}; must be <= {maximum}; using default {default}"
    return value, None


def _parse_env_int(name: str, default: int, *, minimum: int = 1) -> tuple[int, str | None]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default, None
    try:
        value = int(raw.strip())
    except ValueError:
        return default, f"Invalid {name}={raw!r}; using default {default}"
    if value < minimum:
        return default, f"Invalid {name}={value}; must be >= {minimum}; using default {default}"
    return value, None


def load_generation_config() -> tuple[dict[str, float | int], list[str]]:
    warnings: list[str] = []

    temperature, warning = _parse_env_float("LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE, minimum=0.0)
    if warning:
        warnings.append(warning)

    max_tokens, warning = _parse_env_int("LLM_MAX_TOKENS", DEFAULT_LLM_MAX_TOKENS, minimum=1)
    if warning:
        warnings.append(warning)

    top_p, warning = _parse_env_float("LLM_TOP_P", DEFAULT_LLM_TOP_P, minimum=0.0, maximum=1.0)
    if warning:
        warnings.append(warning)
    elif top_p == 0.0:
        top_p = DEFAULT_LLM_TOP_P
        warnings.append(f"Invalid LLM_TOP_P=0.0; must be > 0; using default {DEFAULT_LLM_TOP_P}")

    params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
    return params, warnings


def validate_inputs(query_text: str | None, image_shape: tuple[int, ...]) -> str | None:
    if not query_text or not query_text.strip():
        return "Query must be a non-empty string"
    if image_shape != EXPECTED_IMAGE_SHAPE:
        return f"Image tensor must have shape {EXPECTED_IMAGE_SHAPE}, got {image_shape}"
    return None


def compute_overall_status(step_statuses: list[str]) -> str:
    if STAGE_FAILED in step_statuses:
        return STAGE_FAILED
    if STAGE_DEGRADED in step_statuses:
        return STAGE_DEGRADED
    return STAGE_OK


def build_response_payload(answer: str, trace: dict, *, error: dict | None = None) -> dict:
    payload: dict = {"answer": answer, "debug": trace}
    if error is not None:
        payload["error"] = error
    return payload


def build_fatal_error(stage: str, message: str) -> dict:
    return {"stage": stage, "stage_status": STAGE_FAILED, "message": message}


class TritonPythonModel:
    def initialize(self, _args: dict[str, str]):
        self.qdrant = QdrantClient(url=QDRANT_URL)
        self.collection_name = QDRANT_COLLECTION
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_ID, device=EMBEDDING_DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
        self.generation_params, self.generation_config_warnings = load_generation_config()

    def execute(self, requests):  # noqa: ANN001
        responses = []
        for request in requests:
            trace = {"steps": []}
            t0_total = time.time()
            step_statuses: list[str] = []

            # --- Input ---
            query_tensor = pb_utils.get_input_tensor_by_name(request, "query")
            image_tensor = pb_utils.get_input_tensor_by_name(request, "image")

            if query_tensor is None or image_tensor is None:
                missing = []
                if query_tensor is None:
                    missing.append("query")
                if image_tensor is None:
                    missing.append("image")
                message = f"Missing required input tensor(s): {', '.join(missing)}"
                trace["input_query"] = ""
                trace["overall_status"] = STAGE_FAILED
                trace["failed_stage"] = "input"
                trace["total_latency_ms"] = round((time.time() - t0_total) * 1000, 2)  # pyright: ignore[reportArgumentType]
                error = build_fatal_error("input", message)
                payload = build_response_payload("", trace, error=error)
                responses.append(self._make_response(payload))
                continue

            query_text = query_tensor.as_numpy()[0].decode("utf-8")
            image_array = image_tensor.as_numpy()
            trace["input_query"] = query_text

            input_error = validate_inputs(query_text, tuple(image_array.shape))
            if input_error is not None:
                trace["overall_status"] = STAGE_FAILED
                trace["failed_stage"] = "input"
                trace["steps"].append(
                    {
                        "component": "Input Validation",
                        "stage_status": STAGE_FAILED,
                        "latency_ms": 0.0,
                        "error": input_error,
                    },
                )
                trace["total_latency_ms"] = round((time.time() - t0_total) * 1000, 2)  # pyright: ignore[reportArgumentType]
                error = build_fatal_error("input", input_error)
                payload = build_response_payload("", trace, error=error)
                responses.append(self._make_response(payload))
                continue

            step_statuses.append(STAGE_OK)
            trace["steps"].append(
                {
                    "component": "Input Validation",
                    "stage_status": STAGE_OK,
                    "latency_ms": 0.0,
                },
            )

            # --- 1. Vision (YOLO) ---
            t0 = time.time()
            yolo_stage_status = STAGE_OK
            yolo_error = None
            yolo_request = pb_utils.InferenceRequest(
                model_name="yolo_onnx",
                requested_output_names=["output0"],
                inputs=[pb_utils.Tensor("images", image_array)],
            )
            yolo_response = yolo_request.exec()
            if yolo_response.has_error():
                yolo_stage_status = STAGE_DEGRADED
                yolo_error = yolo_response.error().message()
            yolo_status = "Success" if yolo_stage_status == STAGE_OK else f"Error: {yolo_error}"
            yolo_details = (
                "Output resided on GPU (Optimization)"
                if yolo_stage_status == STAGE_OK
                else "Continuing without vision output"
            )

            yolo_step: dict = {
                "component": "YOLOv8 (Vision)",
                "stage_status": yolo_stage_status,
                "latency_ms": round((time.time() - t0) * 1000, 2),
                "status": yolo_status,
                "details": yolo_details,
            }
            if yolo_error:
                yolo_step["error"] = yolo_error
                yolo_step["fallback"] = "Pipeline continues; vision detections are not required for retrieval"
            step_statuses.append(yolo_stage_status)
            trace["steps"].append(yolo_step)

            # --- 2. Retrieval (Qdrant) ---
            t0 = time.time()
            retrieval_stage_status = STAGE_OK
            retrieval_error = None
            raw_candidates = []

            try:
                query_vector = self.embedder.encode(query_text).tolist()
                search_response = self.qdrant.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=5,
                    with_payload=True,
                )
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
            except Exception as exc:  # noqa: BLE001
                retrieval_stage_status = STAGE_DEGRADED
                retrieval_error = str(exc)

            if retrieval_stage_status == STAGE_OK and not raw_candidates:
                retrieval_stage_status = STAGE_DEGRADED
                retrieval_error = "No documents matched the query"

            candidates_text = [c["full_solution"] for c in raw_candidates if c["full_solution"]]

            retrieval_step: dict = {
                "component": "Qdrant (Retrieval)",
                "stage_status": retrieval_stage_status,
                "latency_ms": round((time.time() - t0) * 1000, 2),
                "candidates_found": len(raw_candidates),
                "top_candidate_preview": raw_candidates[0] if raw_candidates else None,
            }
            if retrieval_error:
                retrieval_step["error"] = retrieval_error
                retrieval_step["fallback"] = DEFAULT_CONTEXT
            step_statuses.append(retrieval_stage_status)
            trace["steps"].append(retrieval_step)

            # --- 3. Reranking ---
            t0 = time.time()
            rerank_stage_status = STAGE_OK
            rerank_error = None
            rerank_fallback = None
            best_context = DEFAULT_CONTEXT
            rerank_score = 0.0

            if candidates_text:
                rerank_inputs = [
                    pb_utils.Tensor("query", np.array([query_text.encode("utf-8")], dtype=np.object_)),
                    pb_utils.Tensor(
                        "candidates", np.array([c.encode("utf-8") for c in candidates_text], dtype=np.object_)
                    ),
                ]
                rerank_req = pb_utils.InferenceRequest(
                    model_name="reranker_py",
                    requested_output_names=["scores"],
                    inputs=rerank_inputs,
                )
                rerank_resp = rerank_req.exec()

                if rerank_resp.has_error():
                    rerank_stage_status = STAGE_DEGRADED
                    rerank_error = rerank_resp.error().message()
                    best_context = candidates_text[0]
                    rerank_fallback = "Using top retrieval candidate without reranker scores"
                else:
                    scores = pb_utils.get_output_tensor_by_name(rerank_resp, "scores").as_numpy()
                    if len(scores) == 0:
                        rerank_stage_status = STAGE_DEGRADED
                        rerank_error = "Reranker returned no scores"
                        best_context = candidates_text[0]
                        rerank_fallback = "Using top retrieval candidate without reranker scores"
                    else:
                        best_idx = int(np.argmax(scores))
                        best_context = candidates_text[best_idx]
                        rerank_score = float(scores[best_idx])
            elif retrieval_stage_status != STAGE_OK or not raw_candidates:
                rerank_stage_status = STAGE_DEGRADED
                rerank_error = "Skipped reranking because retrieval produced no candidates"
                rerank_fallback = DEFAULT_CONTEXT

            rerank_step: dict = {
                "component": "Cross-Encoder (Reranker)",
                "stage_status": rerank_stage_status,
                "latency_ms": round((time.time() - t0) * 1000, 2),
                "best_score": round(rerank_score, 4),
                "selected_context_preview": best_context[:100] + ("..." if len(best_context) > 100 else ""),
            }
            if rerank_error:
                rerank_step["error"] = rerank_error
            if rerank_fallback:
                rerank_step["fallback"] = rerank_fallback
            step_statuses.append(rerank_stage_status)
            trace["steps"].append(rerank_step)

            # --- 4. LLM Generation ---
            t0 = time.time()
            generation_stage_status = STAGE_OK
            generation_error = None
            final_text = ""

            messages = [
                {"role": "system", "content": "You are a helpful technical support assistant."},
                {"role": "user", "content": f"Context: {best_context}\n\nQuestion: {query_text}\n\nSolution:"},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            sampling_params = json.dumps(self.generation_params)
            llm_inputs = [
                pb_utils.Tensor("text_input", np.array([prompt.encode("utf-8")], dtype=np.object_)),
                pb_utils.Tensor("sampling_parameters", np.array([sampling_params.encode("utf-8")], dtype=np.object_)),
                pb_utils.Tensor("stream", np.array([False], dtype=bool)),
            ]

            llm_req = pb_utils.InferenceRequest(
                model_name="llm_vllm",
                requested_output_names=["text_output"],
                inputs=llm_inputs,
            )
            llm_responses = llm_req.exec(decoupled=True)

            for r in llm_responses:
                if r.has_error():
                    generation_stage_status = STAGE_FAILED
                    generation_error = r.error().message()
                    break
                out = pb_utils.get_output_tensor_by_name(r, "text_output")
                if out:
                    final_text += out.as_numpy()[0].decode("utf-8")

            if generation_stage_status == STAGE_OK and not final_text.strip():
                generation_stage_status = STAGE_FAILED
                generation_error = "vLLM returned empty generation output"

            generation_step: dict = {
                "component": "vLLM (Generation)",
                "stage_status": generation_stage_status,
                "latency_ms": round((time.time() - t0) * 1000, 2),
                "generated_length": len(final_text),
                "sampling_parameters": self.generation_params,
            }
            if self.generation_config_warnings:
                generation_step["config_warnings"] = self.generation_config_warnings
            if generation_error:
                generation_step["error"] = generation_error
            step_statuses.append(generation_stage_status)
            trace["steps"].append(generation_step)

            trace["total_latency_ms"] = round((time.time() - t0_total) * 1000, 2)  # pyright: ignore[reportArgumentType]
            trace["overall_status"] = compute_overall_status(step_statuses)

            if generation_stage_status == STAGE_FAILED:
                trace["failed_stage"] = "generation"
                error = build_fatal_error("generation", generation_error or "Generation failed")
                payload = build_response_payload("", trace, error=error)
            else:
                payload = build_response_payload(final_text, trace)

            responses.append(self._make_response(payload))

        return responses

    def _make_response(self, payload: dict) -> pb_utils.InferenceResponse:
        response_payload = json.dumps(payload)
        out_tensor = pb_utils.Tensor("response", np.array([response_payload.encode("utf-8")], dtype=np.object_))
        return pb_utils.InferenceResponse(output_tensors=[out_tensor])

    def finalize(self):
        pass
