"""
push_model.py
=============
Called by the external training app (Django / scikit-learn side) after a model
is trained and converted to ONNX.  Pushes the file directly to the Rust backend
via the /api/ml/push endpoint — no SCP, no cron lag.

Usage (standalone):
    python push_model.py --model stable --file stable_brain_v1.onnx

Usage (from training code):
    from ml_engine.push_model import push_to_portfolio_app
    push_to_portfolio_app("stable", "/path/to/stable_brain_v1.onnx", meta={...})
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests  # pip install requests

PORTFOLIO_BACKEND_URL = os.getenv(
    "PORTFOLIO_BACKEND_URL", "http://localhost:8081"
)
PUSH_ENDPOINT = f"{PORTFOLIO_BACKEND_URL}/api/ml/push"

VALID_MODELS = {"stable", "penny", "crypto", "intraday"}


def push_to_portfolio_app(
    model_name: str,
    onnx_path: str | Path,
    meta: dict[str, Any] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """
    Push an ONNX model file to the Rust portfolio backend.

    Parameters
    ----------
    model_name : str
        One of: stable, penny, crypto, intraday
    onnx_path : str | Path
        Path to the .onnx file on disk.
    meta : dict, optional
        Metadata dict — will be sent as JSON alongside the file.
        Recommended keys: model_version, trained_at, cv_mean, cv_scores,
        features, universe.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    dict with the backend's JSON response.
    """
    if model_name not in VALID_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Valid: {VALID_MODELS}")

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    files = {
        "model": (None, model_name),
        "file": (onnx_path.name, onnx_path.read_bytes(), "application/octet-stream"),
    }
    if meta:
        files["meta"] = (None, json.dumps(meta), "application/json")

    print(f"[push_model] Pushing '{model_name}' → {PUSH_ENDPOINT}")
    resp = requests.post(PUSH_ENDPOINT, files=files, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    print(f"[push_model] Response: {data}")
    return data


def _build_meta_from_payload(payload: dict) -> dict:
    """Extract metadata from a joblib model payload dict."""
    return {
        "model_version": payload.get("model_version"),
        "trained_at": payload.get("trained_at"),
        "cv_mean": payload.get("cv_mean"),
        "cv_scores": payload.get("cv_scores"),
        "features": payload.get("features"),
        "universe": payload.get("universe") or payload.get("model_type"),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push a trained ONNX model to the portfolio backend.")
    parser.add_argument("--model", required=True, choices=sorted(VALID_MODELS),
                        help="Model key (stable, penny, crypto, intraday)")
    parser.add_argument("--file", required=True, help="Path to the .onnx file")
    parser.add_argument("--meta", default=None,
                        help="Optional JSON string or path to .json file with metadata")
    parser.add_argument("--url", default=None,
                        help="Override backend URL (default: $PORTFOLIO_BACKEND_URL)")
    args = parser.parse_args()

    if args.url:
        PUSH_ENDPOINT = f"{args.url}/api/ml/push"

    meta = None
    if args.meta:
        meta_src = args.meta
        if Path(meta_src).exists():
            meta = json.loads(Path(meta_src).read_text())
        else:
            meta = json.loads(meta_src)

    try:
        result = push_to_portfolio_app(args.model, args.file, meta=meta)
        print("Success:", result)
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
