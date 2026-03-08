from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

try:
    from portfolio.models import ModelRegistry
except Exception:  # pragma: no cover
    ModelRegistry = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def extract_metric(payload: dict[str, Any] | None, metric_name: str) -> float | None:
    if not payload:
        return None
    metric_key = (metric_name or '').strip().lower()
    raw = payload.get(metric_key)
    if raw is not None:
        try:
            return float(raw)
        except Exception:
            return None

    if metric_key in {'accuracy', 'score', 'cv_mean'}:
        if payload.get('cv_mean') is not None:
            try:
                return float(payload.get('cv_mean'))
            except Exception:
                pass
        scores = payload.get('scores') or payload.get('cv_scores')
        if scores:
            try:
                return _mean([float(s) for s in scores])
            except Exception:
                return None

    if metric_key in {'mae', 'rmse', 'mse'}:
        for key in ('mae', 'rmse'):
            if payload.get(key) is not None:
                try:
                    return float(payload.get(key))
                except Exception:
                    return None
        scores = payload.get('cv_rmse')
        if scores:
            try:
                return _mean([float(s) for s in scores])
            except Exception:
                return None

    return None


def _is_better(new_metric: float | None, old_metric: float | None, direction: str) -> bool:
    if new_metric is None:
        return False
    if old_metric is None:
        return True
    if direction == 'lower':
        return new_metric < old_metric
    return new_metric > old_metric


def _passes_gatekeeper(new_metric: float | None, old_metric: float | None, direction: str) -> bool:
    if new_metric is None:
        return False
    if old_metric is None:
        return True
    tolerance = float(os.getenv('GATEKEEPER_MAX_DEGRADATION_PCT', '0.05'))
    if direction == 'lower':
        return new_metric <= old_metric * (1 + tolerance)
    return new_metric >= old_metric * (1 - tolerance)


def _resolve_export_dir(model_name: str, model_path: Path) -> Path:
    base_dir = Path(os.getenv('ONNX_EXPORT_DIR', str(model_path.parent)))
    use_subdir = os.getenv('ONNX_EXPORT_MODEL_SUBDIR', 'true').lower() in {'1', 'true', 'yes', 'y'}
    if use_subdir:
        export_dir = base_dir / model_name.lower()
    else:
        export_dir = base_dir
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def write_manifest(
    export_dir: Path,
    model_name: str,
    feature_names: list[str],
    extra: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        'schema_version': 1,
        'model_name': model_name,
        'feature_names': feature_names,
        'feature_count': len(feature_names),
        'exported_at': _now_iso(),
    }
    if extra:
        payload.update(extra)
    manifest_path = export_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def export_onnx_with_gatekeeper(
    payload: dict[str, Any],
    model_path: Path,
    model_name: str,
    feature_names: list[str],
    metric_name: str,
    metric_direction: str,
) -> dict[str, Any]:
    model_name = (model_name or 'model').strip()
    feature_names = list(feature_names or [])
    export_dir = _resolve_export_dir(model_name, model_path)
    onnx_path = export_dir / f"{model_path.stem}.onnx"

    new_metric = extract_metric(payload, metric_name)
    old_metric = None
    if model_path.exists():
        try:
            existing = joblib.load(model_path)
            if isinstance(existing, dict):
                old_metric = extract_metric(existing, metric_name)
        except Exception:
            old_metric = None

    if ModelRegistry is not None:
        try:
            active = ModelRegistry.objects.filter(model_name=model_name.upper(), status='ACTIVE').order_by('-trained_at').first()
            if active and isinstance(active.notes, dict):
                stored = active.notes.get(metric_name)
                if stored is not None:
                    old_metric = float(stored)
        except Exception:
            pass

    if not _passes_gatekeeper(new_metric, old_metric, metric_direction):
        _maybe_notify_training(model_name, metric_name, new_metric, False, onnx_path, old_metric)
        return {
            'exported': False,
            'reason': 'metric_not_improved',
            'metric_name': metric_name,
            'metric_value': new_metric,
            'metric_previous': old_metric,
            'onnx_path': str(onnx_path),
        }

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except Exception as exc:
        _maybe_notify_training(model_name, metric_name, new_metric, False, onnx_path, old_metric)
        return {
            'exported': False,
            'reason': f'skl2onnx_missing:{exc}',
            'metric_name': metric_name,
            'metric_value': new_metric,
            'metric_previous': old_metric,
            'onnx_path': str(onnx_path),
        }

    model = payload.get('model') if isinstance(payload, dict) else payload
    initial_type = [("float_input", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    write_manifest(
        export_dir,
        model_name=model_name,
        feature_names=feature_names,
        extra={
            'metric_name': metric_name,
            'metric_value': new_metric,
            'model_path': str(model_path),
            'onnx_path': str(onnx_path),
        },
    )
    _maybe_notify_training(model_name, metric_name, new_metric, True, onnx_path, old_metric)
    return {
        'exported': True,
        'metric_name': metric_name,
        'metric_value': new_metric,
        'metric_previous': old_metric,
        'onnx_path': str(onnx_path),
    }


def _maybe_notify_training(
    model_name: str,
    metric_name: str,
    metric_value: float | None,
    exported: bool,
    onnx_path: Path,
    old_metric: float | None,
) -> None:
    if os.getenv('TELEGRAM_TRAINING_NOTIFY', 'true').lower() not in {'1', 'true', 'yes', 'y'}:
        return
    try:
        from portfolio.tasks import _send_telegram_message
    except Exception:
        return

    status = 'ONNX export OK ✅' if exported else 'ONNX export BLOQUÉ ❌'
    metric_text = f"{metric_name}={metric_value:.4f}" if metric_value is not None else f"{metric_name}=n/a"
    previous_text = f" (prev {old_metric:.4f})" if old_metric is not None else ''
    message = (
        f"🧠 Entraînement terminé: {model_name}\n"
        f"{metric_text}{previous_text}\n"
        f"{status}\n"
        f"ONNX: {onnx_path}"
    )
    try:
        _send_telegram_message(message)
    except Exception:
        return


def save_model_with_version(
    payload: dict[str, Any],
    model_path: Path,
    model_name: str,
    metric_name: str,
    metric_value: float | None,
) -> dict[str, Any]:
    model_name = (model_name or 'model').strip()
    version = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
    metric_suffix = f"{metric_value:.4f}" if metric_value is not None else 'na'
    archive_dir = Path(os.getenv('MODEL_ARCHIVE_DIR', str(model_path.parent / 'models_archive'))) / model_name.lower()
    archive_dir.mkdir(parents=True, exist_ok=True)
    version_path = archive_dir / f"{model_path.stem}_{version}_{metric_suffix}.pkl"
    joblib.dump(payload, version_path)
    joblib.dump(payload, model_path)
    return {
        'archive_path': str(version_path),
        'latest_path': str(model_path),
    }
