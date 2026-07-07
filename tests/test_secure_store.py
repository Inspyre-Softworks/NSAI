from __future__ import annotations

from enum import IntEnum

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


def test_windows_hello_prompt_explains_wait_and_keyring_fallback(
    monkeypatch,
    capsys,
) -> None:
    """The console should not look frozen while Windows Hello is pending."""
    monkeypatch.setattr(secure_store.os, 'name', 'nt')
    monkeypatch.delenv(secure_store.WINDOWS_HELLO_TIMEOUT_ENV, raising=False)
    secure_store.reset_windows_hello_verification()

    try:
        monkeypatch.setattr(
            secure_store,
            'run_windows_hello_async',
            lambda coro_factory, *, timeout_seconds=None: None,
        )

        secure_store.require_windows_hello('Store test secret')

        output = capsys.readouterr().err
        assert 'Approve the Windows security prompt' in output
        assert 'time out after about 45 seconds' in output
        assert '--secret-backend keyring' in output
    finally:
        secure_store.reset_windows_hello_verification()


def test_windows_hello_verify_uses_desktop_interop_on_windows(monkeypatch) -> None:
    class FakeConsentResult(IntEnum):
        VERIFIED = 0
        CANCELED = 6

    calls: list[str] = []

    async def _fake_available() -> type[FakeConsentResult]:
        return FakeConsentResult

    def _fake_request(reason: str) -> int:
        calls.append(reason)
        return int(FakeConsentResult.VERIFIED)

    monkeypatch.setattr(secure_store.os, 'name', 'nt')
    monkeypatch.setattr(secure_store, '_ensure_windows_hello_available', _fake_available)
    monkeypatch.setattr(secure_store, '_request_windows_hello_for_window', _fake_request)

    secure_store.asyncio.run(secure_store._windows_hello_verify('Store test secret'))

    assert calls == ['Store test secret']


def test_windows_hello_verify_rejects_unverified_desktop_result(monkeypatch) -> None:
    class FakeConsentResult(IntEnum):
        VERIFIED = 0
        CANCELED = 6

    async def _fake_available() -> type[FakeConsentResult]:
        return FakeConsentResult

    monkeypatch.setattr(secure_store.os, 'name', 'nt')
    monkeypatch.setattr(secure_store, '_ensure_windows_hello_available', _fake_available)
    monkeypatch.setattr(
        secure_store,
        '_request_windows_hello_for_window',
        lambda reason: int(FakeConsentResult.CANCELED),
    )

    with pytest.raises(secure_store.SecureStoreError, match='failed'):
        secure_store.asyncio.run(secure_store._windows_hello_verify('Store test secret'))


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

    # Only the first call should have triggered async work (combined into 1).
    assert call_count == 1, f'Expected 1 async call, got {call_count}'

    # Cleanup: reset so other tests are not affected.
    secure_store.reset_windows_hello_verification()
