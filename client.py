import argparse
import json
from typing import Any

import numpy as np
import tritonclient.http as httpclient
from PIL import Image

TRITON_URL = "localhost:8000"
MODEL_NAME = "bls_orchestrator"


def load_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((640, 640))
    img_data = np.array(img).astype(np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))
    return np.expand_dims(img_data, axis=0)


def print_report(data: dict[str, Any]) -> None:
    """Вывод красивого отчета"""
    trace = data.get("debug", {})
    print("\n" + "=" * 60)
    print("🕵️  PIPELINE EXECUTION REPORT")
    print("=" * 60)
    print(f"Query: {trace.get('input_query')}")
    print("-" * 60)

    for step in trace.get("steps", []):
        name = step.get("component")
        latency = step.get("latency_ms")
        print(f"🔹 [{name}] -> {latency}ms")

        if "Qdrant" in name:
            print(f"   Found: {step.get('candidates_found')} docs")
            top = step.get("top_candidate_preview")
            if top:
                print(f"   Top-1: [{top['category']}] {top['issue']}")

        if "Reranker" in name:
            print(f"   Best Score: {step.get('best_score')}")
            print(f'   Context Used: "{step.get("selected_context_preview")}"')

        print("-" * 60)

    print(f"⏱  Total Latency: {trace.get('total_latency_ms') / 1000:.2f}s")
    print("=" * 60)
    print("🤖 AI RESPONSE:")
    print(data.get("answer"))
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
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
        print_report(json_result)
    except json.JSONDecodeError:
        print("Raw output (not JSON):", raw_result)


if __name__ == "__main__":
    main()
