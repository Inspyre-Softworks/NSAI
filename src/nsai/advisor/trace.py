"""API trace formatting helpers for advisor HTTP/model calls."""

from __future__ import annotations

import json
import re
import threading
from typing import Any


MAX_TRACE_CHARS = 12000
REDACTED = '<redacted>'
TRACE_PRINT_LOCK = threading.Lock()


def is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower().replace('_', '-')
    if normalized in {
        'authorization',
        'autologin',
        'api-key',
        'cookie',
        'credential',
        'password',
        'pin',
        'secret',
        'set-cookie',
        'token',
        'x-api-key',
        'x-autologin',
        'x-password',
        'x-pin',
    }:
        return True

    return normalized.endswith('-token') or normalized.endswith('-secret')


def redact_payload(value: Any, *, key: str = '') -> Any:
    if key and is_sensitive_key(key):
        return REDACTED

    if isinstance(value, dict):
        return {
            str(item_key): redact_payload(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }

    if isinstance(value, list | tuple):
        return [redact_payload(item) for item in value]

    if isinstance(value, set):
        return sorted(
            (redact_payload(item) for item in value),
            key=repr,
        )

    if isinstance(value, str):
        return redact_text(value)

    return value


def redact_text(text: str) -> str:
    redacted = text
    redacted = re.sub(
        r'(<TOKEN[^>]*>)(.*?)(</TOKEN>)',
        rf'\1{REDACTED}\3',
        redacted,
        flags=re.IGNORECASE | re.DOTALL,
    )
    redacted = re.sub(
        r'(<SUCCESS[^>]*>)([A-Za-z0-9_-]{6,})(</SUCCESS>)',
        rf'\1{REDACTED}\3',
        redacted,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(
        r'((?:token|password|autologin|pin|api[_-]?key)\s*[=:]\s*)[A-Za-z0-9._~+/=-]+',
        rf'\1{REDACTED}',
        redacted,
        flags=re.IGNORECASE,
    )
    return redacted


def _json_default(value: Any) -> Any:
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json', exclude_none=True)

    if hasattr(value, 'to_dict'):
        return value.to_dict()

    if hasattr(value, '__dict__'):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith('_')
        }

    return str(value)


def format_trace_payload(value: Any) -> str:
    if isinstance(value, str):
        text = redact_text(value)
    else:
        redacted = redact_payload(value)
        text = json.dumps(
            redacted,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )

    if len(text) > MAX_TRACE_CHARS:
        omitted = len(text) - MAX_TRACE_CHARS
        return f'{text[:MAX_TRACE_CHARS]}\n... <truncated {omitted} chars>'

    return text


def print_api_trace(
    *,
    api: str,
    operation: str,
    request: Any,
    response: Any | None = None,
    error: BaseException | str | None = None,
) -> None:
    with TRACE_PRINT_LOCK:
        print()
        print(f'API TRACE [{api}] {operation}')
        print('-' * 88)
        print('REQUEST')
        print(format_trace_payload(request))

        if response is not None:
            print('RESPONSE')
            print(format_trace_payload(response))

        if error is not None:
            print('ERROR')
            print(format_trace_payload(str(error)))
