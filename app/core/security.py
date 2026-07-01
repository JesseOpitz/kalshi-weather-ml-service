from __future__ import annotations

import hashlib
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
    configured = set(get_settings().api_keys)
    if not configured:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="API key is not configured")
    if provided is None or provided not in configured:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return ApiPrincipal(key_fingerprint=_fingerprint(provided))


def require_admin_key(provided: str | None = Security(admin_key_header)) -> ApiPrincipal:
    configured = get_settings().admin_api_key
    if not configured:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Admin key is not configured")
    if provided is None or provided != configured:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin key")
    return ApiPrincipal(key_fingerprint=_fingerprint(provided), is_admin=True)
