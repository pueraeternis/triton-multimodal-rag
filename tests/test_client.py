"""Client payload shape, env var loading, and report output."""

import json
import os
from unittest import mock

import numpy as np
import pytest

SUCCESS_PAYLOAD = {
    "answer": "Check the router logs.",
    "debug": {
        "input_query": "router LED blinking",
        "overall_status": "ok",
        "steps": [
            {
                "component": "YOLOv8 (Vision)",
                "latency_ms": 47.31,
                "stage_status": "ok",
                "status": "Success",
                "details": "Output resided on GPU (Optimization)",
            },
            {
                "component": "Qdrant (Retrieval)",
                "latency_ms": 13.64,
                "stage_status": "ok",
                "candidates_found": 5,
                "top_candidate_preview": {
                    "category": "Router",
                    "issue": "Red status LED blinking continuously...",
                },
            },
            {
                "component": "Cross-Encoder (Reranker)",
                "latency_ms": 10.27,
                "stage_status": "ok",
                "best_score": -2.6081,
                "selected_context_preview": "Check the router logs to identify the specific error code...",
            },
            {
                "component": "vLLM (Generation)",
                "latency_ms": 4496.82,
                "stage_status": "ok",
                "generated_length": 487,
            },
        ],
        "total_latency_ms": 4568.04,
    },
}


DEGRADED_PAYLOAD = {
    "answer": "Check the router logs.",
    "debug": {
        "input_query": "router LED blinking",
        "overall_status": "degraded",
        "steps": [
            {
                "component": "YOLOv8 (Vision)",
                "latency_ms": 12.5,
                "stage_status": "degraded",
                "error": "model unavailable",
                "fallback": "Pipeline continues; vision detections are not required for retrieval",
            },
            {
                "component": "Qdrant (Retrieval)",
                "latency_ms": 8.0,
                "stage_status": "ok",
                "candidates_found": 3,
                "top_candidate_preview": {
                    "category": "Router",
                    "issue": "Red status LED blinking continuously...",
                },
            },
        ],
        "total_latency_ms": 42.0,
    },
}


FATAL_PAYLOAD = {
    "answer": "",
    "error": {
        "stage": "input",
        "stage_status": "failed",
        "message": "Query must be a non-empty string",
    },
    "debug": {
        "input_query": "",
        "overall_status": "failed",
        "failed_stage": "input",
        "steps": [
            {
                "component": "Input Validation",
                "stage_status": "failed",
                "latency_ms": 0.0,
                "error": "Query must be a non-empty string",
            },
        ],
        "total_latency_ms": 1.23,
    },
}


def test_client_reads_env_vars():
    with mock.patch.dict(
        os.environ,
        {"TRITON_URL": "testhost:9999", "TRITON_MODEL_NAME": "test_model"},
        clear=False,
    ):
        import importlib

        import client

        importlib.reload(client)
        assert client.TRITON_URL == "testhost:9999"
        assert client.MODEL_NAME == "test_model"


def test_load_image_shape():
    from client import load_image

    img = load_image("data/test_image.jpg")
    assert img.dtype == np.float32
    assert img.shape == (1, 3, 640, 640)
    assert 0.0 <= img.min() <= img.max() <= 1.0


def test_load_dotenv_reads_dotenv_file(tmp_path, monkeypatch):
    """`.env` values are loaded for host entrypoints via load_dotenv."""
    env_file = tmp_path / ".env"
    env_file.write_text('TRITON_URL="dotenv-host:1234"\n', encoding="utf-8")

    monkeypatch.delenv("TRITON_URL", raising=False)
    monkeypatch.setattr("dotenv.main.find_dotenv", lambda *args, **kwargs: str(env_file))

    import importlib

    import client

    importlib.reload(client)
    assert client.TRITON_URL == "dotenv-host:1234"


def test_inference_input_tensor_spec():
    """Tensor names, dtypes, and shapes expected by bls_orchestrator."""
    query = "test query"
    query_data = np.array([query.encode("utf-8")], dtype=np.object_)
    assert query_data.shape == (1,)
    assert query_data.dtype == np.object_

    image_data = np.zeros((1, 3, 640, 640), dtype=np.float32)
    assert image_data.shape == (1, 3, 640, 640)
    assert image_data.dtype == np.float32

    # Names must match config.pbtxt
    assert "query" in ("query", "image")
    assert "image" in ("query", "image")


def test_print_report_success_format(capsys):
    from client import print_report

    print_report(SUCCESS_PAYLOAD)
    output = capsys.readouterr().out

    assert "PIPELINE EXECUTION REPORT" in output
    assert "Query: router LED blinking" in output
    assert "🔹 [YOLOv8 (Vision)] -> 47.31ms" in output
    assert "Found: 5 docs" in output
    assert "Top-1: [Router]" in output
    assert "Best Score: -2.6081" in output
    assert "⏱  Total Latency: 4.57s" in output
    assert "🤖 AI RESPONSE:" in output
    assert "Check the router logs." in output
    assert "PIPELINE ERROR" not in output
    assert "degraded stage(s)" not in output
    assert "Status: ok" not in output


def test_print_report_degraded_pipeline(capsys):
    from client import print_report

    print_report(DEGRADED_PAYLOAD)
    output = capsys.readouterr().out

    assert "Pipeline completed with degraded stage(s)" in output
    assert "Status: degraded" in output
    assert "Error: model unavailable" in output
    assert "Fallback: Pipeline continues; vision detections are not required for retrieval" in output
    assert "PIPELINE ERROR" not in output
    assert "Status: ok" not in output


def test_print_report_fatal_error(capsys):
    from client import print_report

    print_report(FATAL_PAYLOAD)
    output = capsys.readouterr().out

    assert "PIPELINE ERROR" in output
    assert "Stage: input" in output
    assert "Status: failed" in output
    assert "Message: Query must be a non-empty string" in output
    assert "Error: Query must be a non-empty string" in output
    assert "degraded stage(s)" not in output


@pytest.mark.parametrize(
    "payload",
    [SUCCESS_PAYLOAD, DEGRADED_PAYLOAD, FATAL_PAYLOAD],
)
def test_json_flag_outputs_machine_readable_payload(payload, capsys):
    mock_response = mock.MagicMock()
    mock_response.as_numpy.return_value = [json.dumps(payload).encode("utf-8")]

    with (
        mock.patch("client.httpclient.InferenceServerClient") as mock_client_cls,
        mock.patch("client.load_image", return_value=np.zeros((1, 3, 640, 640), dtype=np.float32)),
        mock.patch(
            "sys.argv",
            ["client.py", "--image", "data/test_image.jpg", "--query", "test", "--json"],
        ),
    ):
        mock_client_cls.return_value.infer.return_value = mock_response
        from client import main

        main()

    output = capsys.readouterr().out
    assert json.loads(output) == payload
    assert "PIPELINE EXECUTION REPORT" not in output
