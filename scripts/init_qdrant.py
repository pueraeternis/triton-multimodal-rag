import json
import logging
import os
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "technical_support")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DATA_PATH = os.getenv("DATA_PATH", "data/knowledge_base.json")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    client = QdrantClient(url=QDRANT_URL)

    logger.info("Loading model %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    if not Path(DATA_PATH).exists():
        logger.error("File %s not found", DATA_PATH)
        return

    with open(DATA_PATH, encoding="utf-8") as f:
        documents = json.load(f)

    logger.info("Loaded %d documents", len(documents))

    if client.collection_exists(COLLECTION_NAME):
        logger.info("Deleting existing collection '%s'", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
        ),
    )
    logger.info("Collection '%s' created", COLLECTION_NAME)

    logger.info("Generating embeddings")
    texts_to_vectorize = [f"{doc['category']}: {doc['issue_description']}" for doc in documents]
    embeddings = model.encode(texts_to_vectorize)

    points = []
    for i, doc in enumerate(documents):
        points.append(
            models.PointStruct(
                id=doc["id"],
                vector=embeddings[i].tolist(),
                payload=doc,
            ),
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )
    logger.info(
        "Successfully uploaded %d records to Qdrant",
        len(points),
    )


if __name__ == "__main__":
    main()
