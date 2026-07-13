"""Compatibility shim re-exporting the standalone nsai-secure-store package.

The Windows Hello / OS credential store implementation now lives in its own
package at packages/nsai-secure-store (import name nsai_secure_store) so its
WinRT/COM interop code has an independent version and test suite. This module
keeps `nsai.secure_store` importable for existing call sites.
"""

from __future__ import annotations

from nsai_secure_store import (
    DEFAULT_WINDOWS_HELLO_TIMEOUT_SECONDS,
    SECRET_BACKEND_KEYRING,
    SECRET_BACKEND_WINDOWS_HELLO,
    SECRET_BACKENDS,
    SECRET_SERVICE,
    WINDOWS_HELLO_TIMEOUT_ENV,
    SecureStoreError,
    default_secret_backend,
    delete_secret,
    get_secret,
    normalize_secret_backend,
    require_windows_hello,
    reset_windows_hello_verification,
    run_windows_hello_async,
    set_secret,
    windows_hello_timeout_seconds,
)


__all__ = [
    'SECRET_SERVICE',
    'SECRET_BACKEND_KEYRING',
    'SECRET_BACKEND_WINDOWS_HELLO',
    'SECRET_BACKENDS',
    'WINDOWS_HELLO_TIMEOUT_ENV',
    'DEFAULT_WINDOWS_HELLO_TIMEOUT_SECONDS',
    'SecureStoreError',
    'default_secret_backend',
    'normalize_secret_backend',
    'windows_hello_timeout_seconds',
    'require_windows_hello',
    'reset_windows_hello_verification',
    'run_windows_hello_async',
    'set_secret',
    'get_secret',
    'delete_secret',
]
