"""OS-backed secure secret storage for NSAI."""

from __future__ import annotations

import asyncio
import ctypes
import os
import sys
import threading
import time
import uuid
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
WINDOWS_HELLO_THREAD_GRACE_SECONDS = 5.0
WINDOWS_HELLO_WAIT_NOTICE_SECONDS = 5.0
WINDOWS_HELLO_JOIN_SLICE_SECONDS = 0.25
WINDOWS_HELLO_RUNTIME_CLASS = 'Windows.Security.Credentials.UI.UserConsentVerifier'
WINDOWS_HELLO_OWNER_CLASS_PREFIX = 'NSAIWindowsHelloOwner'
IID_IUSER_CONSENT_VERIFIER_INTEROP = '39E050C3-4E74-441A-8DC0-B81104DF949C'
IID_IASYNC_INFO = '00000036-0000-0000-C000-000000000046'
IID_IASYNC_OPERATION_USER_CONSENT_RESULT = 'fd596ffd-2318-558f-9dbe-d21df43764a5'
ASYNC_STATUS_STARTED = 0
ASYNC_STATUS_COMPLETED = 1
ASYNC_STATUS_CANCELED = 2
ASYNC_STATUS_ERROR = 3
ERROR_CLASS_ALREADY_EXISTS = 1410


class SecureStoreError(RuntimeError):
    """Raised when the OS credential store cannot save or load a secret."""


class _GUID(ctypes.Structure):
    _fields_ = (
        ('Data1', ctypes.c_uint32),
        ('Data2', ctypes.c_uint16),
        ('Data3', ctypes.c_uint16),
        ('Data4', ctypes.c_ubyte * 8),
    )

    @classmethod
    def from_string(cls, value: str) -> '_GUID':
        parsed = uuid.UUID(value)
        data4 = (ctypes.c_ubyte * 8).from_buffer_copy(parsed.bytes[8:])
        return cls(parsed.time_low, parsed.time_mid, parsed.time_hi_version, data4)


_IID_IUSER_CONSENT_VERIFIER_INTEROP = _GUID.from_string(
    IID_IUSER_CONSENT_VERIFIER_INTEROP
)
_IID_IASYNC_INFO = _GUID.from_string(IID_IASYNC_INFO)
_IID_IASYNC_OPERATION_USER_CONSENT_RESULT = _GUID.from_string(
    IID_IASYNC_OPERATION_USER_CONSENT_RESULT
)
_HRESULT = ctypes.c_int32


# Process-level Windows Hello verification cache.  Once the user has proven
# identity in the current process we trust them for the remainder of that
# process rather than prompting on every secret access.
_windows_hello_verified: bool = False
_windows_hello_lock = threading.Lock()


def reset_windows_hello_verification() -> None:
    """Clear the process-level verification cache (useful for testing)."""
    global _windows_hello_verified
    with _windows_hello_lock:
        _windows_hello_verified = False


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


async def _ensure_windows_hello_available() -> type[Any]:
    """Check whether Windows Hello/user verification can run for this user."""
    try:
        from winrt.windows.security.credentials.ui import (
            UserConsentVerificationResult,
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

    return UserConsentVerificationResult


def format_hresult(hr: int) -> str:
    unsigned = hr & 0xFFFFFFFF
    try:
        message = ctypes.FormatError(unsigned).strip()
    except Exception:
        message = ''

    if message:
        return f'0x{unsigned:08X} ({message})'

    return f'0x{unsigned:08X}'


def check_hresult(hr: int, action: str) -> None:
    if hr < 0:
        raise SecureStoreError(f'{action} failed with HRESULT {format_hresult(hr)}.')


def _winfunctype(*args: Any) -> Any:
    return getattr(ctypes, 'WINFUNCTYPE', ctypes.CFUNCTYPE)(*args)


def _com_method(
    com_pointer: int,
    index: int,
    restype: Any,
    *argtypes: Any,
) -> Any:
    prototype = _winfunctype(restype, ctypes.c_void_p, *argtypes)
    pointer = ctypes.c_void_p(com_pointer)
    vtable = ctypes.cast(
        pointer,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
    ).contents
    return prototype(vtable[index])


def _release_com_object(com_pointer: int | None) -> None:
    if not com_pointer:
        return

    release = _com_method(com_pointer, 2, ctypes.c_uint32)
    release(com_pointer)


def _query_interface(com_pointer: int, iid: _GUID, action: str) -> int:
    query_interface = _com_method(
        com_pointer,
        0,
        _HRESULT,
        ctypes.POINTER(_GUID),
        ctypes.POINTER(ctypes.c_void_p),
    )
    result = ctypes.c_void_p()
    check_hresult(
        query_interface(com_pointer, ctypes.byref(iid), ctypes.byref(result)),
        action,
    )
    if not result.value:
        raise SecureStoreError(f'{action} returned a null COM interface.')

    return result.value


def _create_hstring(value: str) -> ctypes.c_void_p:
    combase = ctypes.windll.combase
    create_string = combase.WindowsCreateString
    create_string.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    )
    create_string.restype = _HRESULT

    hstring = ctypes.c_void_p()
    check_hresult(
        create_string(value, len(value), ctypes.byref(hstring)),
        'Create Windows Runtime string',
    )
    return hstring


def _delete_hstring(hstring: ctypes.c_void_p | None) -> None:
    if not hstring or not hstring.value:
        return

    delete_string = ctypes.windll.combase.WindowsDeleteString
    delete_string.argtypes = (ctypes.c_void_p,)
    delete_string.restype = _HRESULT
    delete_string(hstring)


def _handle_value(handle: Any) -> int:
    return int(getattr(handle, 'value', handle) or 0)


def _windows_hello_owner_window_handle() -> int:
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32
    kernel32.GetConsoleWindow.restype = wintypes.HWND
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.IsWindowVisible.argtypes = (wintypes.HWND,)
    user32.IsWindowVisible.restype = wintypes.BOOL

    console_hwnd = _handle_value(kernel32.GetConsoleWindow())
    if console_hwnd and user32.IsWindowVisible(wintypes.HWND(console_hwnd)):
        return console_hwnd

    foreground_hwnd = _handle_value(user32.GetForegroundWindow())
    return foreground_hwnd or console_hwnd


def _bring_window_to_foreground(hwnd: int) -> None:
    if not hwnd:
        return

    try:
        from ctypes import wintypes

        ctypes.windll.user32.SetForegroundWindow(wintypes.HWND(hwnd))
    except Exception:
        pass


class _WindowsHelloOwnerWindow:
    """Small Win32 owner window used while the Windows Hello prompt is active."""

    def __init__(self) -> None:
        self._hwnd = 0
        self._class_name = (
            f'{WINDOWS_HELLO_OWNER_CLASS_PREFIX}{os.getpid()}{threading.get_ident()}'
        )
        self._hinstance = None
        self._wndproc = None
        self._registered = False

    def __enter__(self) -> int:
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        lresult = getattr(wintypes, 'LRESULT', ctypes.c_ssize_t)
        wndproc_type = _winfunctype(
            lresult,
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )

        user32.DefWindowProcW.argtypes = (
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )
        user32.DefWindowProcW.restype = lresult

        def _window_proc(hwnd: Any, msg: int, wparam: Any, lparam: Any) -> int:
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        class WNDCLASSW(ctypes.Structure):
            _fields_ = (
                ('style', wintypes.UINT),
                ('lpfnWndProc', wndproc_type),
                ('cbClsExtra', ctypes.c_int),
                ('cbWndExtra', ctypes.c_int),
                ('hInstance', wintypes.HINSTANCE),
                ('hIcon', wintypes.HICON),
                ('hCursor', ctypes.c_void_p),
                ('hbrBackground', wintypes.HBRUSH),
                ('lpszMenuName', wintypes.LPCWSTR),
                ('lpszClassName', wintypes.LPCWSTR),
            )

        kernel32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        kernel32.GetLastError.restype = wintypes.DWORD
        user32.RegisterClassW.argtypes = (ctypes.POINTER(WNDCLASSW),)
        user32.RegisterClassW.restype = wintypes.ATOM

        self._wndproc = wndproc_type(_window_proc)
        self._hinstance = kernel32.GetModuleHandleW(None)
        window_class = WNDCLASSW()
        window_class.lpfnWndProc = self._wndproc
        window_class.hInstance = self._hinstance
        window_class.lpszClassName = self._class_name

        atom = user32.RegisterClassW(ctypes.byref(window_class))
        if atom:
            self._registered = True
        else:
            error = int(kernel32.GetLastError())
            if error != ERROR_CLASS_ALREADY_EXISTS:
                raise SecureStoreError(
                    'Could not create Windows Hello owner window class: '
                    f'{format_hresult(error)}.'
                )

        width = 320
        height = 110
        user32.GetSystemMetrics.argtypes = (ctypes.c_int,)
        user32.GetSystemMetrics.restype = ctypes.c_int
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)

        user32.CreateWindowExW.argtypes = (
            wintypes.DWORD,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.HWND,
            wintypes.HMENU,
            wintypes.HINSTANCE,
            wintypes.LPVOID,
        )
        user32.CreateWindowExW.restype = wintypes.HWND

        ws_ex_topmost = 0x00000008
        ws_ex_toolwindow = 0x00000080
        ws_caption = 0x00C00000
        ws_sysmenu = 0x00080000
        self._hwnd = _handle_value(
            user32.CreateWindowExW(
                ws_ex_topmost | ws_ex_toolwindow,
                self._class_name,
                'NSAI Windows Hello Verification',
                ws_caption | ws_sysmenu,
                x,
                y,
                width,
                height,
                None,
                None,
                self._hinstance,
                None,
            )
        )
        if not self._hwnd:
            error = int(kernel32.GetLastError())
            raise SecureStoreError(
                'Could not create Windows Hello owner window: '
                f'{format_hresult(error)}.'
            )

        user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
        user32.ShowWindow.restype = wintypes.BOOL
        user32.UpdateWindow.argtypes = (wintypes.HWND,)
        user32.UpdateWindow.restype = wintypes.BOOL
        user32.ShowWindow(wintypes.HWND(self._hwnd), 5)
        user32.UpdateWindow(wintypes.HWND(self._hwnd))
        _bring_window_to_foreground(self._hwnd)
        self.pump_messages()
        return self._hwnd

    def pump_messages(self) -> None:
        if not self._hwnd:
            return

        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.PeekMessageW.argtypes = (
            ctypes.POINTER(wintypes.MSG),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
            wintypes.UINT,
        )
        user32.PeekMessageW.restype = wintypes.BOOL
        user32.TranslateMessage.argtypes = (ctypes.POINTER(wintypes.MSG),)
        user32.TranslateMessage.restype = wintypes.BOOL
        user32.DispatchMessageW.argtypes = (ctypes.POINTER(wintypes.MSG),)
        user32.DispatchMessageW.restype = getattr(
            wintypes,
            'LRESULT',
            ctypes.c_ssize_t,
        )

        pm_remove = 0x0001
        msg = wintypes.MSG()
        while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, pm_remove):
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        if self._hwnd:
            user32.DestroyWindow.argtypes = (wintypes.HWND,)
            user32.DestroyWindow.restype = wintypes.BOOL
            user32.DestroyWindow(wintypes.HWND(self._hwnd))
            self.pump_messages()
            self._hwnd = 0

        if self._registered and self._hinstance:
            user32.UnregisterClassW.argtypes = (wintypes.LPCWSTR, wintypes.HINSTANCE)
            user32.UnregisterClassW.restype = wintypes.BOOL
            user32.UnregisterClassW(self._class_name, self._hinstance)
            self._registered = False


def _cancel_async_info(async_info_pointer: int) -> None:
    cancel = _com_method(async_info_pointer, 9, _HRESULT)
    cancel(async_info_pointer)


def _close_async_info(async_info_pointer: int) -> None:
    close = _com_method(async_info_pointer, 10, _HRESULT)
    close(async_info_pointer)


def _await_user_consent_operation(
    async_operation_pointer: int,
    pump_messages: Callable[[], None] | None = None,
) -> int:
    async_info_pointer: int | None = None
    try:
        async_info_pointer = _query_interface(
            async_operation_pointer,
            _IID_IASYNC_INFO,
            'Query Windows Hello async status',
        )
        get_status = _com_method(
            async_info_pointer,
            7,
            _HRESULT,
            ctypes.POINTER(ctypes.c_int32),
        )
        get_error_code = _com_method(
            async_info_pointer,
            8,
            _HRESULT,
            ctypes.POINTER(_HRESULT),
        )
        get_results = _com_method(
            async_operation_pointer,
            8,
            _HRESULT,
            ctypes.POINTER(ctypes.c_int32),
        )
        deadline = time.monotonic() + windows_hello_timeout_seconds()

        while True:
            status = ctypes.c_int32()
            check_hresult(
                get_status(async_info_pointer, ctypes.byref(status)),
                'Read Windows Hello async status',
            )

            if status.value == ASYNC_STATUS_COMPLETED:
                result = ctypes.c_int32()
                check_hresult(
                    get_results(async_operation_pointer, ctypes.byref(result)),
                    'Read Windows Hello verification result',
                )
                return result.value

            if status.value == ASYNC_STATUS_CANCELED:
                raise SecureStoreError('Windows Hello verification was canceled.')

            if status.value == ASYNC_STATUS_ERROR:
                error_code = _HRESULT()
                check_hresult(
                    get_error_code(async_info_pointer, ctypes.byref(error_code)),
                    'Read Windows Hello verification error',
                )
                raise SecureStoreError(
                    'Windows Hello verification failed with HRESULT '
                    f'{format_hresult(error_code.value)}.'
                )

            if status.value != ASYNC_STATUS_STARTED:
                raise SecureStoreError(
                    f'Windows Hello verification failed with async status {status.value}.'
                )

            if time.monotonic() >= deadline:
                _cancel_async_info(async_info_pointer)
                raise TimeoutError()

            if pump_messages is not None:
                pump_messages()
            time.sleep(0.05)
    finally:
        if async_info_pointer:
            try:
                _close_async_info(async_info_pointer)
            except Exception:
                pass
            _release_com_object(async_info_pointer)
        _release_com_object(async_operation_pointer)


def _request_windows_hello_for_window(reason: str) -> int:
    if os.name != 'nt':
        raise SecureStoreError('Windows Hello desktop verification requires Windows.')

    from ctypes import wintypes

    combase = ctypes.windll.combase
    ro_get_activation_factory = combase.RoGetActivationFactory
    ro_get_activation_factory.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_GUID),
        ctypes.POINTER(ctypes.c_void_p),
    )
    ro_get_activation_factory.restype = _HRESULT

    runtime_class = _create_hstring(WINDOWS_HELLO_RUNTIME_CLASS)
    message = _create_hstring(reason)
    factory = ctypes.c_void_p()
    async_operation = ctypes.c_void_p()
    operation_owned_by_waiter = False
    try:
        check_hresult(
            ro_get_activation_factory(
                runtime_class,
                ctypes.byref(_IID_IUSER_CONSENT_VERIFIER_INTEROP),
                ctypes.byref(factory),
            ),
            'Get Windows Hello desktop verifier',
        )
        if not factory.value:
            raise SecureStoreError(
                'Windows Hello desktop verifier returned a null activation factory.'
            )

        owner_window = _WindowsHelloOwnerWindow()
        with owner_window as hwnd:
            request_verification = _com_method(
                factory.value,
                6,
                _HRESULT,
                wintypes.HWND,
                ctypes.c_void_p,
                ctypes.POINTER(_GUID),
                ctypes.POINTER(ctypes.c_void_p),
            )
            check_hresult(
                request_verification(
                    factory.value,
                    wintypes.HWND(hwnd),
                    message,
                    ctypes.byref(_IID_IASYNC_OPERATION_USER_CONSENT_RESULT),
                    ctypes.byref(async_operation),
                ),
                'Start Windows Hello desktop verification',
            )
            if not async_operation.value:
                raise SecureStoreError(
                    'Windows Hello desktop verification returned a null async operation.'
                )

            operation_owned_by_waiter = True
            return _await_user_consent_operation(
                async_operation.value,
                pump_messages=owner_window.pump_messages,
            )
    finally:
        if async_operation.value and not operation_owned_by_waiter:
            _release_com_object(async_operation.value)
        _release_com_object(factory.value)
        _delete_hstring(message)
        _delete_hstring(runtime_class)


async def _windows_hello_verify(reason: str) -> None:
    """Check availability then request verification in one apartment context."""
    UserConsentVerificationResult = await _ensure_windows_hello_available()

    if os.name == 'nt':
        result = UserConsentVerificationResult(
            _request_windows_hello_for_window(reason)
        )
    else:
        from winrt.windows.security.credentials.ui import UserConsentVerifier

        result = await UserConsentVerifier.request_verification_async(reason)

    if result != UserConsentVerificationResult.VERIFIED:
        raise SecureStoreError(f'Windows Hello verification failed: {result!s}')


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
        pass
    else:
        raise SecureStoreError(
            'Windows Hello verification cannot run while another event loop is active.'
        )

    errors: list[BaseException] = []

    def _run_on_thread() -> None:
        # WinRT UI operations require a Single-Threaded Apartment (STA).
        # Without init_apartment(STA) the consent dialog's dispatcher has no
        # apartment context and the dialog silently never appears.
        winrt_initialized = False
        try:
            from winrt._winrt import STA, init_apartment, uninit_apartment
            init_apartment(STA)
            winrt_initialized = True
        except Exception as exc:
            errors.append(
                SecureStoreError(
                    'Could not initialize the Windows Runtime STA apartment '
                    'for Windows Hello verification.'
                )
            )
            errors[-1].__cause__ = exc
            return

        # Bring the console window to the foreground so the system-level
        # Windows Hello dialog has a visible anchor when it appears.
        try:
            _bring_window_to_foreground(_windows_hello_owner_window_handle())
        except Exception:
            pass

        # Create and register an event loop for this thread.  Using
        # new_event_loop() instead of ProactorEventLoop() directly keeps the
        # helper portable (on Windows the default policy still returns a
        # ProactorEventLoop) and avoids the deprecation warning for direct
        # ProactorEventLoop instantiation.
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                asyncio.wait_for(coro_factory(), timeout=timeout)
            )
        except Exception as exc:
            errors.append(exc)
        finally:
            if loop is not None:
                try:
                    loop.close()
                except Exception:
                    pass
            asyncio.set_event_loop(None)
            if winrt_initialized:
                try:
                    uninit_apartment()
                except Exception:
                    pass

    thread = threading.Thread(target=_run_on_thread, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout + WINDOWS_HELLO_THREAD_GRACE_SECONDS
    next_notice = time.monotonic() + WINDOWS_HELLO_WAIT_NOTICE_SECONDS
    while thread.is_alive():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        thread.join(min(WINDOWS_HELLO_JOIN_SLICE_SECONDS, remaining))
        now = time.monotonic()
        if thread.is_alive() and now >= next_notice:
            seconds_left = max(0.0, deadline - now)
            print(
                'Still waiting for Windows Hello verification '
                f'({format_seconds(seconds_left)} seconds before timeout).',
                file=sys.stderr,
                flush=True,
            )
            next_notice = now + WINDOWS_HELLO_WAIT_NOTICE_SECONDS

    if thread.is_alive():
        raise SecureStoreError(
            'Windows Hello verification timed out waiting for the consent dialog. '
            'Try again from an interactive Windows session, provide the secret '
            'through the environment for this run, or re-save the secret with '
            '--secret-backend keyring.'
        )

    if errors:
        exc = errors[0]
        if isinstance(exc, TimeoutError):
            raise SecureStoreError(
                'Windows Hello verification did not complete after '
                f'{format_seconds(timeout)} seconds. Try again from an '
                'interactive Windows session, provide the secret through the '
                'environment for this run, or re-save the secret with '
                '--secret-backend keyring.'
            ) from exc
        raise exc


def require_windows_hello(reason: str) -> None:
    global _windows_hello_verified

    if os.name != 'nt':
        raise SecureStoreError('Windows Hello secrets are only supported on Windows.')

    with _windows_hello_lock:
        if _windows_hello_verified:
            return

        timeout = windows_hello_timeout_seconds()
        # Use a fixed message to avoid echoing the reason (which may contain
        # credential key names) to the terminal; the full reason is shown
        # inside the system Windows Hello consent dialog.
        print(
            'Windows Hello verification required. Approve the Windows security '
            'prompt to continue.',
            file=sys.stderr,
            flush=True,
        )
        print(
            'If no prompt appears, this will time out after about '
            f'{format_seconds(timeout)} seconds. Use --secret-backend keyring '
            'to store this secret without Windows Hello verification.',
            file=sys.stderr,
            flush=True,
        )
        # Run availability check AND consent prompt in one thread/apartment so
        # we never need to uninit and re-init the WinRT apartment between calls.
        run_windows_hello_async(
            lambda: _windows_hello_verify(reason),
            timeout_seconds=timeout,
        )
        _windows_hello_verified = True


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
