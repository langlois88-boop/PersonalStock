from __future__ import annotations

"""Local model registry for versioned archives."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


class LocalModelRegistry:
    """Keep a versioned archive of deployed models."""

    def __init__(self, registry_dir: Path) -> None:
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def register(self, model_name: str, onnx_path: Path, meta: dict) -> str:
        """Register a model into the registry.

        Args:
            model_name: Model key.
            onnx_path: Path to ONNX file.
            meta: Metadata dict.

        Returns:
            Registered version string.
        """
        version = meta.get("model_version", f"v{datetime.utcnow().date().isoformat()}")
        dest_dir = self.registry_dir / model_name / version
        dest_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(onnx_path, dest_dir / "model.onnx")
        (dest_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        latest = self.registry_dir / model_name / "latest"
        if latest.is_symlink():
            latest.unlink()
        latest.symlink_to(dest_dir.resolve())

        print(f"📦 Registered {model_name} {version} → {dest_dir}")
        return version

    def list_versions(self, model_name: str) -> list[dict]:
        """List registered versions for a model."""
        model_dir = self.registry_dir / model_name
        if not model_dir.exists():
            return []
        versions = []
        for v_dir in sorted(model_dir.iterdir()):
            if v_dir.name == "latest":
                continue
            meta_file = v_dir / "meta.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                versions.append({"version": v_dir.name, **meta})
        return versions

    def rollback(self, model_name: str, version: str) -> Path:
        """Rollback to a previous model version.

        Args:
            model_name: Model key.
            version: Version to restore.

        Returns:
            Path to restored ONNX file.
        """
        src = self.registry_dir / model_name / version / "model.onnx"
        if not src.exists():
            raise FileNotFoundError(f"Version {version} not found for {model_name}")
        dest = Path(f"./models/{model_name}.onnx")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"⏪ Rolled back {model_name} to {version}")
        return dest
