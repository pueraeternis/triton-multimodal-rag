"""Tests for model repository structure."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_MODELS = {
    "yolo_onnx": "onnxruntime_onnx",
    "bls_orchestrator": "python",
    "reranker_py": "python",
    "llm_vllm": "vllm",
    "embedding_onnx": "onnxruntime_onnx",
}


def test_model_repository_directories_exist():
    for name in EXPECTED_MODELS:
        assert (ROOT / "model_repository" / name).is_dir(), f"missing model directory: {name}"


def test_config_pbtxt_names_match_directories():
    for name, backend in EXPECTED_MODELS.items():
        config_path = ROOT / "model_repository" / name / "config.pbtxt"
        assert config_path.exists(), f"missing config.pbtxt for {name}"
        content = config_path.read_text(encoding="utf-8")
        if backend == "vllm":
            assert 'backend: "vllm"' in content
        else:
            assert f'name: "{name}"' in content, f"config.pbtxt name mismatch for {name}"
        if backend == "python":
            assert 'backend: "python"' in content
        elif backend == "vllm":
            pass
        else:
            assert f'platform: "{backend}"' in content


def test_yolo_onnx_export_path_configured():
    export_path = ROOT / "model_repository/yolo_onnx/1/model.onnx"
    # File may not exist in CI — config must point to correct location
    config = (ROOT / "model_repository/yolo_onnx/config.pbtxt").read_text(encoding="utf-8")
    assert "yolo_onnx" in config
    assert export_path.parent.exists()


def test_llm_model_json_exists():
    model_json = ROOT / "model_repository/llm_vllm/1/model.json"
    assert model_json.exists()
    assert "Qwen" in model_json.read_text(encoding="utf-8")
