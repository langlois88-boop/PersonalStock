from __future__ import annotations

"""HTTP push integration for ONNX models."""

from pathlib import Path
from typing import Any

from portfolio.ml_engine.push_model import push_to_portfolio_app


def push_model(model_name: str, onnx_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Push an ONNX model to the Rust backend.

    Args:
        model_name: stable|penny|crypto|intraday.
        onnx_path: Path to ONNX file.
        meta: Metadata payload.

    Returns:
        Response JSON.
    """
    return push_to_portfolio_app(model_name, onnx_path, meta=meta)
