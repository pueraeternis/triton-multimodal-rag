"""Structured BLS error response shape and stage status helpers."""


def test_validate_inputs_rejects_empty_query(bls_model):
    assert bls_model.validate_inputs("", (1, 3, 640, 640)) == "Query must be a non-empty string"
    assert bls_model.validate_inputs("   ", (1, 3, 640, 640)) == "Query must be a non-empty string"


def test_validate_inputs_rejects_bad_image_shape(bls_model):
    message = bls_model.validate_inputs("router LED blinking", (1, 3, 320, 320))
    assert message is not None
    assert "shape" in message


def test_validate_inputs_accepts_valid_payload(bls_model):
    assert bls_model.validate_inputs("router LED blinking", (1, 3, 640, 640)) is None


def test_compute_overall_status_priority(bls_model):
    assert bls_model.compute_overall_status(["ok", "ok"]) == "ok"
    assert bls_model.compute_overall_status(["ok", "degraded"]) == "degraded"
    assert bls_model.compute_overall_status(["degraded", "failed"]) == "failed"


def test_fatal_error_response_shape(bls_model):
    trace = {
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
    }
    error = bls_model.build_fatal_error("input", "Query must be a non-empty string")
    payload = bls_model.build_response_payload("", trace, error=error)

    assert payload["answer"] == ""
    assert payload["error"] == {
        "stage": "input",
        "stage_status": "failed",
        "message": "Query must be a non-empty string",
    }
    assert payload["debug"]["overall_status"] == "failed"
    assert payload["debug"]["steps"][0]["stage_status"] == "failed"


def test_degraded_success_response_shape(bls_model):
    trace = {
        "input_query": "router LED blinking",
        "overall_status": "degraded",
        "steps": [
            {"component": "YOLOv8 (Vision)", "stage_status": "degraded", "error": "model unavailable"},
            {"component": "Qdrant (Retrieval)", "stage_status": "ok", "candidates_found": 3},
        ],
        "total_latency_ms": 42.0,
    }
    payload = bls_model.build_response_payload("Check the router logs.", trace)

    assert payload["answer"] == "Check the router logs."
    assert "error" not in payload
    assert payload["debug"]["overall_status"] == "degraded"
    assert payload["debug"]["steps"][0]["stage_status"] == "degraded"
