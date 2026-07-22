# nsai-secure-store

OS-backed secure secret storage used by [NSAI](../..). Wraps `keyring` for
cross-platform credential storage, plus a `windows-hello` backend that gates
secret access behind native Windows Hello / user-verification consent via
direct WinRT/COM interop.

Extracted from the main `nsai` package so the Windows-specific COM/WinRT
apartment-threading code has its own package boundary, versioning, and test
suite independent of the rest of the CLI.

Depended on by `nsai` via a local path dependency (`packages/nsai-secure-store`);
not currently published to PyPI.
