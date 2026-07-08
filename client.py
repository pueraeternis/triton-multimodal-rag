import argparse
import json
import os
from typing import Any

import numpy as np
import tritonclient.http as httpclient
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "bls_orchestrator")


def load_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((640, 640))
    img_data = np.array(img).astype(np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))
    return np.expand_dims(img_data, axis=0)


def _print_step_issues(step: dict[str, Any]) -> None:
    stage_status = step.get("stage_status")
    if stage_status not in ("degraded", "failed"):
        return
    print(f"   Status: {stage_status}")
    if error := step.get("error"):
        print(f"   Error: {error}")
    if fallback := step.get("fallback"):
        print(f"   Fallback: {fallback}")


def print_report(data: dict[str, Any]) -> None:
    """Pretty-print the BLS pipeline trace and answer."""
    trace = data.get("debug", {})
    error = data.get("error")

    print("\n" + "=" * 60)
    print("🕵️  PIPELINE EXECUTION REPORT")
    print("=" * 60)

    if error:
        print("❌ PIPELINE ERROR")
        print(f"   Stage: {error.get('stage')}")
        print(f"   Status: {error.get('stage_status')}")
        print(f"   Message: {error.get('message')}")
        print("-" * 60)

    overall_status = trace.get("overall_status")
    if overall_status == "degraded" and error is None:
        print("⚠️  Pipeline completed with degraded stage(s)")
        print("-" * 60)

    print(f"Query: {trace.get('input_query')}")
    print("-" * 60)

    for step in trace.get("steps", []):
        name = step.get("component", "")
        latency = step.get("latency_ms")
        print(f"🔹 [{name}] -> {latency}ms")

        _print_step_issues(step)

        if "Qdrant" in name:
            print(f"   Found: {step.get('candidates_found')} docs")
            top = step.get("top_candidate_preview")
            if top:
                print(f"   Top-1: [{top['category']}] {top['issue']}")

        if "Reranker" in name:
            print(f"   Best Score: {step.get('best_score')}")
            print(f'   Context Used: "{step.get("selected_context_preview")}"')

        print("-" * 60)

    total_latency_ms = trace.get("total_latency_ms")
    if total_latency_ms is not None:
        print(f"⏱  Total Latency: {total_latency_ms / 1000:.2f}s")
    print("=" * 60)
    print("🤖 AI RESPONSE:")
    print(data.get("answer"))
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--json", action="store_true", help="Print raw JSON response")
    args = parser.parse_args()

    client = httpclient.InferenceServerClient(url=TRITON_URL)

    image_data = load_image(args.image)
    query_data = np.array([args.query.encode("utf-8")], dtype=np.object_)

    inputs = [
        httpclient.InferInput("query", query_data.shape, "BYTES"),
        httpclient.InferInput("image", image_data.shape, "FP32"),
    ]
    inputs[0].set_data_from_numpy(query_data)
    inputs[1].set_data_from_numpy(image_data)
    outputs = [httpclient.InferRequestedOutput("response")]

    response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

    raw_result = response.as_numpy("response")[0].decode("utf-8")  # pyright: ignore[reportOptionalSubscript]
    try:
        json_result = json.loads(raw_result)
        if args.json:
            print(json.dumps(json_result, indent=2))
        else:
            print_report(json_result)
    except json.JSONDecodeError:
        print("Raw output (not JSON):", raw_result)


if __name__ == "__main__":
    main()
