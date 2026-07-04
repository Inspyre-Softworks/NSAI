"""OS-backed secure secret storage for NSAI."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Awaitable, Callable
from typing import Any

import keyring
from keyring.errors import KeyringError


SECRET_SERVICE = 'nsai.nations'
SECRET_BACKEND_KEYRING = 'keyring'
SECRET_BACKEND_WINDOWS_HELLO = 'windows-hello'
SECRET_BACKENDS = (SECRET_BACKEND_KEYRING, SECRET_BACKEND_WINDOWS_HELLO)
WINDOWS_HELLO_TIMEOUT_ENV = 'NSAI_WINDOWS_HELLO_TIMEOUT_SECONDS'
DEFAULT_WINDOWS_HELLO_TIMEOUT_SECONDS = 45.0


class SecureStoreError(RuntimeError):
    """Raised when the OS credential store cannot save or load a secret."""


def default_secret_backend() -> str:
    if os.name == 'nt':
        return SECRET_BACKEND_WINDOWS_HELLO

    return SECRET_BACKEND_KEYRING


def normalize_secret_backend(backend: str | None) -> str:
    normalized = (backend or SECRET_BACKEND_KEYRING).strip().lower()
    if normalized not in SECRET_BACKENDS:
        raise SecureStoreError(f'Unsupported secret backend: {backend!r}')

    return normalized


def windows_hello_timeout_seconds() -> float:
    raw = os.environ.get(WINDOWS_HELLO_TIMEOUT_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_WINDOWS_HELLO_TIMEOUT_SECONDS

    try:
        timeout = float(raw)
    except ValueError as exc:
        raise SecureStoreError(
            f'{WINDOWS_HELLO_TIMEOUT_ENV} must be a positive number of seconds.'
        ) from exc

    if timeout <= 0:
        raise SecureStoreError(
            f'{WINDOWS_HELLO_TIMEOUT_ENV} must be a positive number of seconds.'
        )

    return timeout


async def _check_windows_hello_available() -> None:
    try:
        from winrt.windows.security.credentials.ui import (
            UserConsentVerifier,
            UserConsentVerifierAvailability,
        )
    except ModuleNotFoundError as exc:
        raise SecureStoreError(
            'Windows Hello verification requires the Windows Runtime packages.'
        ) from exc

    availability = await UserConsentVerifier.check_availability_async()
    if availability != UserConsentVerifierAvailability.AVAILABLE:
        raise SecureStoreError(
            'Windows Hello/user verification is not available for this Windows user.'
        )


async def _request_windows_hello_verification(reason: str) -> None:
    try:
        from winrt.windows.security.credentials.ui import (
            UserConsentVerificationResult,
            UserConsentVerifier,
        )
    except ModuleNotFoundError as exc:
        raise SecureStoreError(
            'Windows Hello verification requires the Windows Runtime packages.'
        ) from exc

    result = await UserConsentVerifier.request_verification_async(reason)
    if result != UserConsentVerificationResult.VERIFIED:
        raise SecureStoreError(f'Windows Hello verification failed: {result!s}')


async def _run_with_timeout(
    coro: Awaitable[Any],
    *,
    timeout_seconds: float,
) -> None:
    await asyncio.wait_for(coro, timeout=timeout_seconds)


def format_seconds(seconds: float) -> str:
    return f'{seconds:g}'


def run_windows_hello_async(
    coro_factory: Callable[[], Awaitable[Any]],
    *,
    timeout_seconds: float | None = None,
) -> None:
    timeout = (
        windows_hello_timeout_seconds()
        if timeout_seconds is None
        else timeout_seconds
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            asyncio.run(
                _run_with_timeout(
                    coro_factory(),
                    timeout_seconds=timeout,
                )
            )
        except TimeoutError as exc:
            raise SecureStoreError(
                'Windows Hello verification did not complete after '
                f'{format_seconds(timeout)} seconds. Try again from an '
                'interactive Windows session, provide the secret through the '
                'environment for this run, or re-save the secret with '
                '--secret-backend keyring.'
            ) from exc
        return

    raise SecureStoreError(
        'Windows Hello verification cannot run while another event loop is active.'
    )


def require_windows_hello(reason: str) -> None:
    if os.name != 'nt':
        raise SecureStoreError('Windows Hello secrets are only supported on Windows.')

    timeout = windows_hello_timeout_seconds()
    print(
        f'Windows Hello verification required: {reason}',
        file=sys.stderr,
        flush=True,
    )
    run_windows_hello_async(
        _check_windows_hello_available,
        timeout_seconds=timeout,
    )
    run_windows_hello_async(
        lambda: _request_windows_hello_verification(reason),
        timeout_seconds=timeout,
    )


def set_secret(
    credential_key: str,
    secret: str,
    *,
    backend: str | None = None,
    reason: str | None = None,
) -> None:
    if not secret:
        raise SecureStoreError('Refusing to store an empty secret.')

    backend = normalize_secret_backend(backend)
    if backend == SECRET_BACKEND_WINDOWS_HELLO:
        require_windows_hello(reason or f'Store NSAI secret {credential_key}')

    try:
        keyring.set_password(SECRET_SERVICE, credential_key, secret)
    except KeyringError as exc:
        raise SecureStoreError(
            'Could not store the secret in the OS credential store.'
        ) from exc


def get_secret(
    credential_key: str,
    *,
    backend: str | None = None,
    reason: str | None = None,
) -> str | None:
    backend = normalize_secret_backend(backend)
    if backend == SECRET_BACKEND_WINDOWS_HELLO:
        require_windows_hello(reason or f'Unlock NSAI secret {credential_key}')

    try:
        return keyring.get_password(SECRET_SERVICE, credential_key)
    except KeyringError as exc:
        raise SecureStoreError(
            'Could not read the secret from the OS credential store.'
        ) from exc


def delete_secret(
    credential_key: str,
    *,
    backend: str | None = None,
    reason: str | None = None,
) -> bool:
    backend = normalize_secret_backend(backend)
    if backend == SECRET_BACKEND_WINDOWS_HELLO:
        require_windows_hello(reason or f'Delete NSAI secret {credential_key}')

    try:
        keyring.delete_password(SECRET_SERVICE, credential_key)
        return True
    except keyring.errors.PasswordDeleteError:
        return False
    except KeyringError as exc:
        raise SecureStoreError(
            'Could not delete the secret from the OS credential store.'
        ) from exc
