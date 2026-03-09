from __future__ import annotations

"""ONNX export utilities with sidecar metadata."""

import datetime
import json
from pathlib import Path
from typing import Optional

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from portfolio.ml_engine.training.trainer import TrainResult


class OnnxExporter:
    """Exports a TrainResult to ONNX + JSON sidecar atomically."""

    def export(
        self,
        result: TrainResult,
        output_path: Path,
        universe: str,
        expected_feature_count: int,
        extra_meta: Optional[dict] = None,
    ) -> Path:
        """Export a model to ONNX and write the sidecar metadata.

        Args:
            result: Training result.
            output_path: ONNX output path.
            universe: Model universe name.
            expected_feature_count: Expected feature count.
            extra_meta: Optional extra metadata.

        Returns:
            Path to exported ONNX file.
        """
        if not result.passed_gate:
            raise ValueError(f"Model did not pass quality gate: {result.gate_reason}")
        if len(result.feature_names) != expected_feature_count:
            raise ValueError("Feature count mismatch — update feature registry and Rust app before export")

        n_features = len(result.feature_names)
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        options = {id(result.model): {"zipmap": False}}

        onnx_model = convert_sklearn(result.model, initial_types=initial_type, options=options)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(".onnx.tmp")
        tmp.write_bytes(onnx_model.SerializeToString())
        tmp.rename(output_path)

        meta = {
            "model_version": f"v{datetime.date.today().isoformat()}",
            "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
            "cv_mean": result.cv_mean,
            "cv_scores": result.cv_scores,
            "wf_f1": result.wf_f1,
            "features": result.feature_names,
            "n_features": n_features,
            "universe": universe,
            "n_samples": result.n_samples,
            "label_balance": result.label_balance,
            **(extra_meta or {}),
        }
        sidecar = output_path.with_suffix(".json")
        sidecar.write_text(json.dumps(meta, indent=2))

        print(f"✅ Exported: {output_path} ({output_path.stat().st_size:,} bytes)")
        print(f"   Sidecar: {sidecar}")
        print(f"   CV mean: {result.cv_mean:.3f} | WF F1: {result.wf_f1:.3f} | Samples: {result.n_samples}")
        return output_path
