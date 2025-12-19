import logging
import os
import shutil
from pathlib import Path

from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]

MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "yolov8n")
EXPORT_PATH = Path(
    os.getenv(
        "YOLO_EXPORT_PATH",
        "model_repository/yolo_onnx/1/model.onnx",
    ),
)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Loading YOLO model: %s", MODEL_NAME)
    model = YOLO(f"{MODEL_NAME}.pt")

    logger.info("Exporting model to ONNX")
    export_path = Path(model.export(format="onnx", dynamic=True))

    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(export_path, EXPORT_PATH)

    pt_path = Path(f"{MODEL_NAME}.pt")
    if pt_path.exists():
        pt_path.unlink()

    logger.info("Model successfully exported to %s", EXPORT_PATH)


if __name__ == "__main__":
    main()
