"""Small compatibility fallback used only when the native orjson package is unavailable."""

from __future__ import annotations

import json
from typing import Any

orjson = None
OPT_NON_STR_KEYS = 1
OPT_SERIALIZE_NUMPY = 2
OPT_UTC_Z = 4


def dumps(value: Any, default=None, option=None) -> bytes:
    return json.dumps(
        value,
        default=default,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def loads(value: bytes | bytearray | memoryview | str) -> Any:
    if not isinstance(value, str):
        value = bytes(value).decode("utf-8")
    return json.loads(value)
