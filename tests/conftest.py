"""Shared test fixtures."""

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
BLS_MODEL_PATH = ROOT / "model_repository/bls_orchestrator/1/model.py"
BLS_MODULE_NAME = "bls_orchestrator_model"


@pytest.fixture(scope="session")
def bls_model():
    if BLS_MODULE_NAME in sys.modules:
        return sys.modules[BLS_MODULE_NAME]

    with mock.patch.dict(sys.modules, {"triton_python_backend_utils": mock.MagicMock()}):
        spec = importlib.util.spec_from_file_location(BLS_MODULE_NAME, BLS_MODEL_PATH)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[BLS_MODULE_NAME] = module
        spec.loader.exec_module(module)
        return module
