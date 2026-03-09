from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from portfolio.ml_engine.training.trainer import Trainer
from portfolio.ml_engine.export.onnx_exporter import OnnxExporter


def test_export_creates_files(tmp_path: Path) -> None:
    X = pd.DataFrame({"a": np.random.randn(200), "b": np.random.randn(200)})
    y = pd.Series([0, 1] * 100)

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
