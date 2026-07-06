from __future__ import annotations

import pytest

import nsai.secure_store as secure_store


def test_windows_hello_backend_verifies_before_storing(monkeypatch) -> None:
    calls: list[str] = []
    stored: dict[tuple[str, str], str] = {}

    monkeypatch.setattr(
        secure_store,
        'require_windows_hello',
        lambda reason: calls.append(reason),
    )
    monkeypatch.setattr(
        secure_store.keyring,
        'set_password',
        lambda service, key, value: stored.__setitem__((service, key), value),
    )

    secure_store.set_secret(
        'nation:oringrad:password',
        'secret',
        backend='windows-hello',
        reason='Store test secret',
    )

    assert calls == ['Store test secret']
    assert stored[(secure_store.SECRET_SERVICE, 'nation:oringrad:password')] == 'secret'


def test_windows_hello_backend_verifies_before_reading(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        secure_store,
        'require_windows_hello',
        lambda reason: calls.append(reason),
    )
    monkeypatch.setattr(
        secure_store.keyring,
        'get_password',
        lambda service, key: 'secret',
    )

    secret = secure_store.get_secret(
        'nation:oringrad:password',
        backend='windows-hello',
        reason='Unlock test secret',
    )

    assert secret == 'secret'
    assert calls == ['Unlock test secret']


def test_keyring_backend_does_not_prompt_windows_hello(monkeypatch) -> None:
    monkeypatch.setattr(
        secure_store,
        'require_windows_hello',
        lambda reason: (_ for _ in ()).throw(AssertionError('should not prompt')),
    )
    monkeypatch.setattr(
        secure_store.keyring,
        'get_password',
        lambda service, key: 'secret',
    )

    assert secure_store.get_secret('plain', backend='keyring') == 'secret'


def test_windows_hello_async_timeout_becomes_store_error(monkeypatch) -> None:
    monkeypatch.setenv(secure_store.WINDOWS_HELLO_TIMEOUT_ENV, '0.01')

    async def never_finishes() -> None:
        await secure_store.asyncio.sleep(3600)

    with pytest.raises(secure_store.SecureStoreError, match='did not complete'):
        secure_store.run_windows_hello_async(never_finishes)


def test_windows_hello_verification_is_cached_per_process(monkeypatch) -> None:
    """Once verified, subsequent require_windows_hello calls are no-ops."""
    import os
    if os.name != 'nt':
        pytest.skip('Windows-only test')

    call_count = 0

    def _fake_run_async(coro_factory, *, timeout_seconds=None):
        nonlocal call_count
        call_count += 1

    secure_store.reset_windows_hello_verification()
    monkeypatch.setattr(secure_store, 'run_windows_hello_async', _fake_run_async)

    secure_store.require_windows_hello('First reason')
    secure_store.require_windows_hello('Second reason')
    secure_store.require_windows_hello('Third reason')

    # Only the first call should have triggered async work (availability + verification = 2).
    assert call_count == 2, f'Expected 2 async calls, got {call_count}'

    # Cleanup: reset so other tests are not affected.
    secure_store.reset_windows_hello_verification()
