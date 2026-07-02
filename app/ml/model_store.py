from __future__ import annotations

import hashlib
import io
from pathlib import Path

import joblib

from app.core.config import get_settings
from app.ml.bundle import ModelBundle


class LocalModelStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bundle(self, bundle: ModelBundle) -> tuple[str, str]:
        target = self.root / f"{bundle.version}.joblib"
        temporary = target.with_suffix(".tmp")
        joblib.dump(bundle, temporary, compress=3)
        temporary.replace(target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        return str(target), digest

    def load_bundle(self, uri: str, expected_sha256: str | None = None) -> ModelBundle:
        path = Path(uri).resolve()
        if self.root not in path.parents:
            raise ValueError("Model file must be inside the configured model directory.")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 and digest != expected_sha256:
            raise ValueError("Model checksum verification failed.")
        bundle = joblib.load(io.BytesIO(payload))
        if not isinstance(bundle, ModelBundle):
            raise TypeError("Stored object is not a ModelBundle.")
        return bundle


def get_model_store() -> LocalModelStore:
    return LocalModelStore(get_settings().model_store_path)
