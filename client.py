import argparse
import os
import time

import numpy as np
import tritonclient.http as httpclient
from PIL import Image

TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "bls_orchestrator")

IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "640"))


def load_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess image for YOLO.
    Output shape: (1, 3, H, W), values in range [0, 1].
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image_np = np.asarray(image, dtype=np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))  # HWC -> CHW
    image_np = np.expand_dims(image_np, axis=0)  # Add batch dim

    return image_np


def main():
    parser = argparse.ArgumentParser(description="Triton multimodal client")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User query text",
    )
    args = parser.parse_args()

    client = httpclient.InferenceServerClient(url=TRITON_URL)

    if not client.is_server_live():
        raise RuntimeError("Triton server is not live")

    image_data = load_image(args.image)
    query_data = np.array(
        [args.query.encode("utf-8")],
        dtype=np.object_,
    )

    inputs = [
        httpclient.InferInput(
            "query",
            query_data.shape,
            "BYTES",
        ),
        httpclient.InferInput(
            "image",
            image_data.shape,
            "FP32",
        ),
    ]

    inputs[0].set_data_from_numpy(query_data)
    inputs[1].set_data_from_numpy(image_data)

    outputs = [
        httpclient.InferRequestedOutput("response"),
    ]

    start_time = time.time()
    response = client.infer(
        model_name=MODEL_NAME,
        inputs=inputs,
        outputs=outputs,
    )
    latency = time.time() - start_time

    result = response.as_numpy("response")[0].decode("utf-8")  # pyright: ignore[reportOptionalSubscript]

    print(f"Latency: {latency:.2f}s")
    print(result)


if __name__ == "__main__":
    main()
