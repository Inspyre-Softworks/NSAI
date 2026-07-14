# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NSAI is a terminal app that builds NationStates AI governance profiles and uses a local OpenAI-compatible model (e.g. LM Studio at `http://localhost:1234/v1`) to advise on — and optionally enact — live NationStates issues. Python 3.11+, Poetry, src layout (`src/nsai`).

## Commands

```powershell
poetry install                 # install with dev dependencies
poetry run nsai --help         # run the CLI
poetry run pytest              # run all tests
poetry run pytest tests/test_advisor.py::test_name   # run a single test
poetry build                   # build distributables
```

There is no lint/format tool configured. Tests are fully offline: they monkeypatch `requests`, the OpenAI client, and keyring, and point `NSAI_CONFIG_HOME` at a tmp dir — never let a test touch the live NationStates API or a real credential store.

## Architecture

`nsai.cli:main` is the single entrypoint; it builds an argparse tree and delegates to subcommand modules. Root-level `ns_live_advisor.py` / `ns_profile_builder.py` are thin compatibility wrappers.

Four domains:

- **`nsai/advisor/`** — the live issue advisor. `live.py` is the orchestration layer only; domain logic lives in:
  - `client.py` — NationStates API HTTP client, auth headers (password/autologin/PIN), XML helpers
  - `governor.py` — local LLM calls (OpenAI client), prompt construction
  - `recommendations.py` — recommendation validation, enactment guardrails, display
  - `safety.py` — fail-safe validation for auto actions (pure functions, no I/O)
  - `cache.py` — SQLite advice cache (issue choices, all-issue order plans, per-issue advice)
  - `audit.py` — JSONL audit-log read/write and publication tracking
  - `trace.py` — redacted API request/response tracing (`--trace-api`)
  - `cli.py` — advisor argument definitions and `publications backfill`
- **`nsai/profile/`** — governance profile JSON files. `builder.py` orchestrates the Textual interview UI; `models.py` holds the interview schema/constants, `enrichment.py` the AI vision/constitution generation, `storage.py` file I/O and backups.
- **`nsai/nations.py`** — per-nation JSON configs plus `secure_store.py`, which keeps secrets (NS password/autologin/PIN, LM API keys) in the OS credential store; on Windows the default backend is `windows-hello` (user verification gates every read/write/delete).
- **`nsai/world_dataset.py`** — downloads the public world dataset and imports it into DuckDB (`nsai world build/search/sql/inspect`).

Config root resolution (in `nations.py:config_root`): `NSAI_CONFIG_HOME` env override → `%APPDATA%\NSAI` on Windows → `~/.config/nsai`. It holds nation configs, managed profile copies, `advice.sqlite3`, and `config.json`. `NS_*` / `LM_STUDIO_*` environment variables override saved config values for the current process (see `.env.example`; `.env` files are never auto-loaded).

## Safety invariants

These are load-bearing design rules, enforced mainly in `safety.py` and `recommendations.py` — preserve them when changing advisor code:

- Advisor-only is the default mode. Nothing is enacted or dismissed without `--enact` (manual) or `--auto` (profile-gated: requires a permissive `enactment_mode`, confidence above the profile threshold, and no triggered red line).
- Every AI recommendation must reference a real live issue ID and a real option ID (or the dismissal option `-1`) before any action path continues.
- Fallback (non-AI) recommendations are review-only and must never enact or dismiss.
- Publication pages (dispatch/factbook) are only posted after a successful issue action; advisor-only runs print drafts without publishing.
- Secrets never go into JSON configs or the audit log; API traces are redacted via `trace.py`.
