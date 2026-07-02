#!/usr/bin/env python3
"""Smoke validation: offline (CI-safe), online (services), or --full (one inference)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_MODELS = ("yolo_onnx", "reranker_py", "llm_vllm", "bls_orchestrator")
REQUIRED_FILES = (
    "model_repository/yolo_onnx/config.pbtxt",
    "model_repository/bls_orchestrator/config.pbtxt",
    "model_repository/reranker_py/config.pbtxt",
    "model_repository/llm_vllm/config.pbtxt",
    "model_repository/llm_vllm/1/model.json",
    "data/knowledge_base.json",
    "data/test_image.jpg",
    ".env.example",
    "docs/CONFIGURATION.md",
)


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}", file=sys.stderr)


def _http_get(url: str, timeout: float = 5.0) -> tuple[int, str]:
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def check_offline() -> list[str]:
    errors: list[str] = []

    for rel in REQUIRED_FILES:
        path = ROOT / rel
        if path.exists():
            _ok(f"file present: {rel}")
        else:
            errors.append(f"missing file: {rel}")
            _fail(f"missing file: {rel}")

    yolo_onnx = ROOT / "model_repository/yolo_onnx/1/model.onnx"
    if yolo_onnx.exists():
        _ok("YOLO ONNX exported")
    else:
        errors.append("YOLO ONNX not exported — run: make export-models")
        _fail("YOLO ONNX not exported — run: make export-models")

    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    config_doc = (ROOT / "docs/CONFIGURATION.md").read_text(encoding="utf-8")
    env_vars = set(re.findall(r"^([A-Z][A-Z0-9_]+)=", env_example, re.MULTILINE))
    for var in sorted(env_vars):
        if var in config_doc:
            _ok(f"env var documented: {var}")
        else:
            errors.append(f"env var missing from CONFIGURATION.md: {var}")
            _fail(f"env var missing from CONFIGURATION.md: {var}")

    with open(ROOT / "data/knowledge_base.json", encoding="utf-8") as f:
        docs = json.load(f)
    if isinstance(docs, list) and docs:
        required_keys = {"id", "category", "issue_description", "solution_text"}
        bad = [i for i, d in enumerate(docs) if not required_keys.issubset(d)]
        if bad:
            errors.append(f"knowledge_base.json schema errors at indices: {bad[:5]}")
            _fail(f"knowledge_base.json schema errors at indices: {bad[:5]}")
        else:
            _ok(f"knowledge_base.json valid ({len(docs)} documents)")
    else:
        errors.append("knowledge_base.json empty or invalid")
        _fail("knowledge_base.json empty or invalid")

    return errors


def check_online() -> list[str]:
    errors: list[str] = []
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
    triton_url = os.getenv("TRITON_URL", "localhost:8000")

    status, _ = _http_get(f"{qdrant_url}/")
    if status == 200:
        _ok(f"Qdrant reachable at {qdrant_url}")
    else:
        errors.append(f"Qdrant unreachable (HTTP {status})")
        _fail(f"Qdrant unreachable (HTTP {status})")

    collection = os.getenv("QDRANT_COLLECTION", "technical_support")
    status, body = _http_get(f"{qdrant_url}/collections/{collection}")
    if status == 200 and '"status":"green"' in body.replace(" ", ""):
        _ok(f"Qdrant collection '{collection}' ready")
    else:
        errors.append(f"Qdrant collection '{collection}' not ready — run: make init-qdrant")
        _fail(f"Qdrant collection '{collection}' not ready — run: make init-qdrant")

    triton_http = triton_url if triton_url.startswith("http") else f"http://{triton_url}"
    status, _ = _http_get(f"{triton_http}/v2/health/ready")
    if status == 200:
        _ok(f"Triton ready at {triton_http}")
    else:
        errors.append(f"Triton not ready (HTTP {status})")
        _fail(f"Triton not ready (HTTP {status})")

    for model in REQUIRED_MODELS:
        mstatus, mbody = _http_get(f"{triton_http}/v2/models/{model}")
        if mstatus == 200 and f'"name":"{model}"' in mbody.replace(" ", ""):
            _ok(f"Triton model registered: {model}")
        else:
            errors.append(f"Triton model not found: {model}")
            _fail(f"Triton model not found: {model}")

    return errors


def check_full() -> list[str]:
    errors: list[str] = []
    try:
        import numpy as np
        import tritonclient.http as httpclient
        from PIL import Image
    except ImportError as exc:
        errors.append(f"import error: {exc}")
        _fail(f"import error: {exc}")
        return errors

    triton_url = os.getenv("TRITON_URL", "localhost:8000")
    model_name = os.getenv("TRITON_MODEL_NAME", "bls_orchestrator")
    image_path = ROOT / "data/test_image.jpg"
    query = "Red status LED is blinking continuously on my Router. What to do?"

    img = Image.open(image_path).convert("RGB").resize((640, 640))
    img_data = np.transpose(np.array(img).astype(np.float32) / 255.0, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    query_data = np.array([query.encode("utf-8")], dtype=np.object_)

    client = httpclient.InferenceServerClient(url=triton_url)
    inputs = [
        httpclient.InferInput("query", query_data.shape, "BYTES"),
        httpclient.InferInput("image", img_data.shape, "FP32"),
    ]
    inputs[0].set_data_from_numpy(query_data)
    inputs[1].set_data_from_numpy(img_data)
    outputs = [httpclient.InferRequestedOutput("response")]

    try:
        response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        raw = response.as_numpy("response")[0].decode("utf-8")
        result = json.loads(raw)
        answer = result.get("answer", "")
        steps = result.get("debug", {}).get("steps", [])
        components = {s.get("component", "") for s in steps}
        expected = {"YOLOv8 (Vision)", "Qdrant (Retrieval)", "Cross-Encoder (Reranker)", "vLLM (Generation)"}
        if answer and expected.issubset(components):
            _ok(f"full inference succeeded ({len(answer)} char answer, {len(steps)} trace steps)")
        else:
            errors.append("inference response incomplete (missing answer or trace steps)")
            _fail("inference response incomplete (missing answer or trace steps)")
    except Exception as exc:
        errors.append(f"inference failed: {exc}")
        _fail(f"inference failed: {exc}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke validation for triton-multimodal-rag")
    parser.add_argument("--online", action="store_true", help="Check Qdrant and Triton readiness")
    parser.add_argument("--full", action="store_true", help="Run one end-to-end inference (implies --online)")
    args = parser.parse_args()

    print("Smoke test: offline checks")
    all_errors = check_offline()

    if args.online or args.full:
        print("\nSmoke test: online checks")
        all_errors.extend(check_online())

    if args.full:
        print("\nSmoke test: full inference")
        all_errors.extend(check_full())

    if all_errors:
        print(f"\nFAILED ({len(all_errors)} issue(s))", file=sys.stderr)
        return 1

    print("\nPASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
