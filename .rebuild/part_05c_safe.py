from pathlib import Path

def write(path: str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding='utf-8')

write('app/core/security.py', '''from __future__ import annotations

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
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "API_KEY_NOT_CONFIGURED", "message": "Server API key is missing."},
        )
    if provided is None or provided not in configured:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "INVALID_API_KEY", "message": "Invalid API key."},
        )
    return ApiPrincipal(key_fingerprint=_fingerprint(provided))

def require_admin_key(provided: str | None = Security(admin_key_header)) -> ApiPrincipal:
    configured = get_settings().admin_api_key
    if not configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "ADMIN_KEY_NOT_CONFIGURED", "message": "Admin key is missing."},
        )
    if provided is None or provided != configured:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "INVALID_ADMIN_KEY", "message": "Invalid admin key."},
        )
    return ApiPrincipal(key_fingerprint=_fingerprint(provided), is_admin=True)
''')
