from __future__ import annotations

import hashlib
import io
from pathlib import Path
from urllib.parse import urlparse

import joblib

from app.core.config import get_settings
from app.ml.bundle import ModelBundle


class ArtifactStore:
    def save_bundle(self, bundle: ModelBundle) -> tuple[str, str]:
        raise NotImplementedError

    def load_bundle(self, uri: str, expected_sha256: str | None = None) -> ModelBundle:
        raise NotImplementedError


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bundle(self, bundle: ModelBundle) -> tuple[str, str]:
        target = self.root / f"{bundle.version}.joblib"
        temporary = target.with_suffix(".tmp")
        joblib.dump(bundle, temporary, compress=3)
        temporary.replace(target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        return str(target), digest

    def load_bundle(self, uri: str, expected_sha256: str | None = None) -> ModelBundle:
        path = Path(uri)
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 and digest != expected_sha256:
            raise ValueError("Model artifact checksum verification failed.")
        bundle = joblib.load(io.BytesIO(payload))
        if not isinstance(bundle, ModelBundle):
            raise TypeError("Artifact is not a ModelBundle.")
        return bundle


class S3ArtifactStore(ArtifactStore):
    def __init__(self, uri: str):
        import boto3

        settings = get_settings()
        parsed = urlparse(uri)
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )

    def _key(self, version: str) -> str:
        name = f"{version}.joblib"
        return f"{self.prefix}/{name}" if self.prefix else name

    def save_bundle(self, bundle: ModelBundle) -> tuple[str, str]:
        buffer = io.BytesIO()
        joblib.dump(bundle, buffer, compress=3)
        payload = buffer.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        key = self._key(bundle.version)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=payload)
        return f"s3://{self.bucket}/{key}", digest

    def load_bundle(self, uri: str, expected_sha256: str | None = None) -> ModelBundle:
        parsed = urlparse(uri)
        response = self.client.get_object(Bucket=parsed.netloc, Key=parsed.path.strip("/"))
        payload = response["Body"].read()
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 and digest != expected_sha256:
            raise ValueError("Model artifact checksum verification failed.")
        bundle = joblib.load(io.BytesIO(payload))
        if not isinstance(bundle, ModelBundle):
            raise TypeError("Artifact is not a ModelBundle.")
        return bundle


def get_artifact_store() -> ArtifactStore:
    settings = get_settings()
    if settings.model_store_uri.startswith("s3://"):
        return S3ArtifactStore(settings.model_store_uri)
    return LocalArtifactStore(settings.model_store_path)
