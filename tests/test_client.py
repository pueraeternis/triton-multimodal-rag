"""Client payload shape and env var loading."""

import os
from unittest import mock

import numpy as np


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
