import logging
import os
import shutil
from pathlib import Path

from optimum.onnxruntime import ORTModelForFeatureExtraction  # pyright: ignore[reportMissingImports]
from transformers import AutoTokenizer

MODEL_ID = os.getenv(
    "EMBEDDING_MODEL_ID",
    "sentence-transformers/all-MiniLM-L6-v2",
)
OUTPUT_DIR = Path(
    os.getenv(
        "EMBEDDING_OUTPUT_DIR",
        "model_repository/embedding_onnx/1",
    ),
)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Exporting model to ONNX: %s", MODEL_ID)

    model = ORTModelForFeatureExtraction.from_pretrained(
        MODEL_ID,
        export=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    temp_dir = Path("temp_embedding_export")
    model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_file = temp_dir / "model.onnx"
    target_file = OUTPUT_DIR / "model.onnx"

    shutil.move(source_file, target_file)
    logger.info("Model moved to %s", target_file)

    shutil.rmtree(temp_dir)
    logger.info("Temporary files cleaned up")


if __name__ == "__main__":
    main()
