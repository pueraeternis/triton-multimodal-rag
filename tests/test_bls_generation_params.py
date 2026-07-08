"""BLS generation parameter loading and vLLM sampling_parameters wiring."""

import json
from unittest import mock

import numpy as np
import pytest


@pytest.fixture
def generation_env(monkeypatch):
    for name in ("LLM_TEMPERATURE", "LLM_MAX_TOKENS", "LLM_TOP_P"):
        monkeypatch.delenv(name, raising=False)


def test_load_generation_config_defaults(bls_model, generation_env):
    params, warnings = bls_model.load_generation_config()

    assert params == {
        "temperature": 0.1,
        "max_tokens": 512,
        "top_p": 0.95,
    }
    assert warnings == []


def test_load_generation_config_custom_values(bls_model, generation_env, monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("LLM_MAX_TOKENS", "128")
    monkeypatch.setenv("LLM_TOP_P", "0.8")

    params, warnings = bls_model.load_generation_config()

    assert params == {
        "temperature": 0.7,
        "max_tokens": 128,
        "top_p": 0.8,
    }
    assert warnings == []


@pytest.mark.parametrize(
    ("env", "expected_params", "expected_warning_fragment"),
    [
        (
            {"LLM_TEMPERATURE": "hot"},
            {"temperature": 0.1, "max_tokens": 512, "top_p": 0.95},
            "Invalid LLM_TEMPERATURE",
        ),
        (
            {"LLM_MAX_TOKENS": "-1"},
            {"temperature": 0.1, "max_tokens": 512, "top_p": 0.95},
            "Invalid LLM_MAX_TOKENS",
        ),
        (
            {"LLM_TOP_P": "1.5"},
            {"temperature": 0.1, "max_tokens": 512, "top_p": 0.95},
            "Invalid LLM_TOP_P",
        ),
        (
            {"LLM_TOP_P": "0"},
            {"temperature": 0.1, "max_tokens": 512, "top_p": 0.95},
            "Invalid LLM_TOP_P=0.0",
        ),
    ],
)
def test_load_generation_config_invalid_values_fallback(
    bls_model,
    generation_env,
    monkeypatch,
    env,
    expected_params,
    expected_warning_fragment,
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    params, warnings = bls_model.load_generation_config()

    assert params == expected_params
    assert any(expected_warning_fragment in warning for warning in warnings)


def test_initialize_reads_generation_config_once(bls_model, generation_env, monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "0.4")
    monkeypatch.setenv("LLM_MAX_TOKENS", "64")
    monkeypatch.setenv("LLM_TOP_P", "0.9")

    model = bls_model.TritonPythonModel()
    with (
        mock.patch.object(bls_model, "QdrantClient"),
        mock.patch.object(bls_model, "SentenceTransformer"),
        mock.patch.object(bls_model, "AutoTokenizer"),
    ):
        model.initialize({})

    assert model.generation_params == {
        "temperature": 0.4,
        "max_tokens": 64,
        "top_p": 0.9,
    }
    assert model.generation_config_warnings == []


def test_execute_sends_sampling_parameters_to_vllm(bls_model, generation_env, monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("LLM_MAX_TOKENS", "256")
    monkeypatch.setenv("LLM_TOP_P", "0.85")

    captured_sampling_params: list[str] = []

    class FakeTensor:
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def as_numpy(self):
            return self.value

    class FakeInferenceRequest:
        def __init__(self, model_name, requested_output_names, inputs):
            self.model_name = model_name
            self.requested_output_names = requested_output_names
            self.inputs = inputs

        def exec(self, decoupled=False):
            if self.model_name == "llm_vllm":
                for tensor in self.inputs:
                    if tensor.name == "sampling_parameters":
                        captured_sampling_params.append(tensor.value[0].decode("utf-8"))
                response = mock.MagicMock()
                response.has_error.return_value = False
                response.error.return_value = None
                return [response]
            response = mock.MagicMock()
            response.has_error.return_value = False
            if self.model_name == "yolo_onnx":
                return response
            if self.model_name == "reranker_py":
                response.get_output_tensor_by_name = mock.MagicMock(
                    return_value=FakeTensor("scores", np.array([0.9], dtype=np.float32)),
                )
                return response
            return response

    pb_utils = mock.MagicMock()
    pb_utils.Tensor = FakeTensor
    pb_utils.InferenceRequest = FakeInferenceRequest
    pb_utils.get_input_tensor_by_name = lambda request, name: {
        "query": FakeTensor("query", np.array([b"router LED blinking"], dtype=np.object_)),
        "image": FakeTensor("image", np.zeros((1, 3, 640, 640), dtype=np.float32)),
    }[name]

    def fake_get_output_tensor_by_name(_response, name):
        if name == "scores":
            return FakeTensor(name, np.array([0.9], dtype=np.float32))
        return FakeTensor(name, np.array([b"Generated answer"], dtype=np.object_))

    pb_utils.get_output_tensor_by_name = fake_get_output_tensor_by_name
    pb_utils.InferenceResponse = lambda output_tensors: mock.MagicMock(output_tensors=output_tensors)

    model = bls_model.TritonPythonModel()
    model.generation_params = {
        "temperature": 0.2,
        "max_tokens": 256,
        "top_p": 0.85,
    }
    model.generation_config_warnings = []
    model.tokenizer = mock.MagicMock()
    model.tokenizer.apply_chat_template.return_value = "prompt"
    model.embedder = mock.MagicMock()
    model.embedder.encode.return_value = np.zeros(384)
    model.qdrant = mock.MagicMock()
    model.qdrant.query_points.return_value = mock.MagicMock(
        points=[
            mock.MagicMock(
                score=0.9,
                payload={"category": "network", "issue_description": "LED", "solution_text": "Reset router"},
            ),
        ],
    )
    model.collection_name = "technical_support"

    with mock.patch.object(bls_model, "pb_utils", pb_utils):
        responses = model.execute([mock.MagicMock()])

    assert captured_sampling_params == [
        json.dumps({"temperature": 0.2, "max_tokens": 256, "top_p": 0.85}),
    ]
    response_payload = json.loads(responses[0].output_tensors[0].value[0].decode("utf-8"))
    generation_step = next(
        step for step in response_payload["debug"]["steps"] if step["component"] == "vLLM (Generation)"
    )
    assert generation_step["sampling_parameters"] == {
        "temperature": 0.2,
        "max_tokens": 256,
        "top_p": 0.85,
    }
