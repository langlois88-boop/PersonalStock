from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from portfolio.ml_engine.training.trainer import Trainer
from portfolio.ml_engine.export.onnx_exporter import OnnxExporter


def test_export_creates_files(tmp_path: Path) -> None:
    # Seeded and with y actually depending on X, so the model can clear the
    # quality gate deterministically. The previous version used unseeded
    # noise for X and a label independent of it (a fixed 0/1/0/1 pattern),
    # so cv_mean hovered around chance (~0.5) and the test's pass/fail was
    # a coin flip driven by the global numpy RNG state, not by anything the
    # exporter does.
    rng = np.random.RandomState(0)
    a = rng.randn(200)
    b = rng.randn(200)
    X = pd.DataFrame({"a": a, "b": b})
    y = pd.Series(((a + 0.5 * b) > 0).astype(int))

    def pipeline_factory() -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
        ])

    trainer = Trainer(pipeline_factory)
    result = trainer.fit(X, y)
    exporter = OnnxExporter()
    onnx_path = tmp_path / "model.onnx"
    exporter.export(result, onnx_path, "TEST", expected_feature_count=2)
    assert onnx_path.exists()
    assert onnx_path.with_suffix(".json").exists()
