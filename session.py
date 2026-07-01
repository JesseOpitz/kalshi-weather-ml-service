from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


@dataclass(frozen=True)
class ApiPrincipal:
    key_fingerprint: str
    is_admin: bool = False


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def require_api_key(provided: str | None = Security(api_key_header)) -> ApiPrincipal:
    settings = get_settings()
    if not settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "API_KEY_NOT_CONFIGURED", "message": "Server API key is missing."},
        )
    if not provided or not any(secrets.compare_digest(provided, key) for key in settings.api_keys):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "INVALID_API_KEY", "message": "Invalid API key."},
        )
    return ApiPrincipal(key_fingerprint=_fingerprint(provided))


def require_admin_key(provided: str | None = Security(admin_key_header)) -> ApiPrincipal:
    settings = get_settings()
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "ADMIN_KEY_NOT_CONFIGURED", "message": "Admin key is missing."},
        )
    if not provided or not secrets.compare_digest(provided, settings.admin_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "INVALID_ADMIN_KEY", "message": "Invalid admin key."},
        )
    return ApiPrincipal(key_fingerprint=_fingerprint(provided), is_admin=True)
