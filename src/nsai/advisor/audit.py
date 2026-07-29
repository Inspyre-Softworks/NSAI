"""Advisor audit-log helpers.

Audit records are stored in a small SQLite database (one JSON blob per row)
rather than an append-only JSONL file, so the audit trail is queryable and
doesn't need bespoke line-number bookkeeping. Every function that used to
take a JSONL path now takes a path to that .sqlite3 file; the row's
autoincrement id stands in for the old line number, so callers that already
carry around a `list[tuple[int, dict]]` need no changes.
"""

from __future__ import annotations

import json
import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nsai.advisor.client import NationStatesError
from nsai.advisor.recommendations import get_profile_mode
from nsai.advisor.safety import publication_mismatch_reasons
from nsai.nations import normalize_nation_key


AUDIT_SCHEMA_VERSION = 1


def resolve_audit_store_path(path: Path) -> Path:
    """Map a legacy JSONL setting to its SQLite sibling, migrating once if needed."""

    expanded = path.expanduser()
    if expanded.suffix.lower() != '.jsonl':
        return expanded

    db_path = expanded.with_suffix('.sqlite3')
    if db_path.exists():
        return db_path

    if expanded.exists():
        migrate_jsonl_audit_log(expanded, db_path)

    return db_path


class AuditStore:
    """SQLite-backed store for advisor audit records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self.connect() as connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                '''
            )
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS audit_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    nation TEXT,
                    record_json TEXT NOT NULL
                )
                '''
            )
            connection.execute(
                '''
                CREATE INDEX IF NOT EXISTS idx_audit_records_nation
                ON audit_records (nation)
                '''
            )
            connection.execute(
                '''
                INSERT OR REPLACE INTO meta (key, value)
                VALUES ('schema_version', ?)
                ''',
                (str(AUDIT_SCHEMA_VERSION),),
            )

    def append(self, record: dict[str, Any]) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                '''
                INSERT INTO audit_records (timestamp, nation, record_json)
                VALUES (?, ?, ?)
                ''',
                (
                    str(record.get('timestamp') or ''),
                    record.get('nation'),
                    json.dumps(record, ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def load_all(self) -> list[tuple[int, dict[str, Any]]]:
        with self.connect() as connection:
            rows = connection.execute(
                'SELECT id, record_json FROM audit_records ORDER BY id'
            ).fetchall()

        records = []
        for row in rows:
            try:
                record = json.loads(str(row['record_json']))
            except json.JSONDecodeError as exc:
                raise NationStatesError(
                    f'Audit log {self.path} has invalid JSON in record {row["id"]}: {exc}'
                ) from exc

            if isinstance(record, dict):
                records.append((int(row['id']), record))

        return records


def write_audit_log(
    path: Path,
    *,
    nation: str,
    profile_path: Path | None,
    profile: dict[str, Any] | None,
    recommendation: dict[str, Any],
    action: str,
    action_reasons: list[str],
    result_xml: str | None = None,
    publication_results: list[dict[str, Any]] | None = None,
    action_applied: bool | None = None,
    blocked: bool = False,
    block_reasons: list[str] | None = None,
    ai_step_statuses: list[dict[str, Any]] | None = None,
    fallback_issue_selection_used: bool = False,
) -> None:
    record = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'nation': nation,
        'profile_path': str(profile_path) if profile_path else None,
        'profile_name': profile.get('profile_name') if profile else None,
        'enactment_mode': get_profile_mode(profile),
        'action': action,
        'action_reasons': action_reasons,
        'recommendation': recommendation,
        'result_xml': result_xml,
        'publication_results': publication_results or [],
        'action_applied': action_applied,
        'blocked': blocked,
        'block_reasons': block_reasons or [],
        'ai_step_statuses': ai_step_statuses or [],
        'fallback_issue_selection_used': fallback_issue_selection_used,
    }

    AuditStore(resolve_audit_store_path(path)).append(record)


def append_publication_backfill_log(
    path: Path,
    *,
    source_entry: dict[str, Any],
    source_line: int,
    publication_results: list[dict[str, Any]],
) -> None:
    recommendation = source_entry.get('recommendation') or {}
    record = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'action': 'publication_backfill',
        'nation': source_entry.get('nation'),
        'source_audit_line': source_line,
        'source_timestamp': source_entry.get('timestamp'),
        'source_action': source_entry.get('action'),
        'issue_id': recommendation.get('issue_id'),
        'option_id': recommendation.get('option_id'),
        'publication_results': publication_results,
    }

    AuditStore(resolve_audit_store_path(path)).append(record)


def load_audit_log_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    path = resolve_audit_store_path(path)
    if not path.exists():
        return []

    return AuditStore(path).load_all()


def migrate_jsonl_audit_log(jsonl_path: Path, db_path: Path) -> int:
    """One-time import of a legacy JSONL audit log into the SQLite store.

    Never modifies or deletes jsonl_path. Returns the number of records
    imported.
    """
    if not jsonl_path.exists():
        raise NationStatesError(f'Audit log {jsonl_path} does not exist.')

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(jsonl_path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise NationStatesError(
                f'Audit log {jsonl_path} has invalid JSON on line {line_number}: {exc}'
            ) from exc

        if isinstance(record, dict):
            records.append(record)

    store = AuditStore(db_path)
    for record in records:
        store.append(record)

    return len(records)


def audit_issue_action_succeeded(record: dict[str, Any]) -> bool:
    if record.get('action') not in {'manual_enact', 'auto_enact'}:
        return False

    result_xml = record.get('result_xml')
    if not isinstance(result_xml, str) or not result_xml.strip():
        return False

    try:
        root = ET.fromstring(result_xml)
    except ET.ParseError:
        return False

    return (root.findtext('.//OK') or '').strip() == '1'


def publication_source_key(record: dict[str, Any], kind: str, title: str) -> tuple[str, str, str, str, str, str]:
    recommendation = record.get('recommendation') or {}
    nation = str(record.get('nation') or '')
    timestamp = str(record.get('source_timestamp') or record.get('timestamp') or '')
    issue_id = str(record.get('issue_id') or recommendation.get('issue_id') or '')
    option_id = str(record.get('option_id') or recommendation.get('option_id') or '')
    return (
        normalize_nation_key(nation),
        timestamp,
        issue_id,
        option_id,
        kind,
        title.strip().lower(),
    )


def posted_publication_keys(
    records: list[tuple[int, dict[str, Any]]],
) -> set[tuple[str, str, str, str, str, str]]:
    keys: set[tuple[str, str, str, str, str, str]] = set()

    for _, record in records:
        for result in record.get('publication_results') or []:
            if not isinstance(result, dict) or result.get('status') != 'posted':
                continue

            kind = str(result.get('kind') or '').strip()
            title = str(result.get('title') or '').strip()
            if kind and title:
                keys.add(publication_source_key(record, kind, title))

    return keys


def pending_publication_entries(
    records: list[tuple[int, dict[str, Any]]],
    *,
    nation: str | None = None,
) -> list[dict[str, Any]]:
    posted_keys = posted_publication_keys(records)
    pending = []
    nation_key = normalize_nation_key(nation) if nation else None

    for line_number, record in records:
        if not audit_issue_action_succeeded(record):
            continue

        record_nation = str(record.get('nation') or '')
        if nation_key and normalize_nation_key(record_nation) != nation_key:
            continue

        recommendation = record.get('recommendation')
        if not isinstance(recommendation, dict):
            continue

        if publication_mismatch_reasons(
            recommendation,
            draft_dispatch=True,
            draft_factbook=True,
        ):
            continue

        dispatch = recommendation.get('dispatch_draft')
        factbook = recommendation.get('factbook_draft')
        publish_dispatch = False
        publish_factbook = False

        if isinstance(dispatch, dict) and dispatch.get('requested'):
            title = str(dispatch.get('title') or '').strip()
            if title and publication_source_key(record, 'dispatch', title) not in posted_keys:
                publish_dispatch = True

        if (
            isinstance(factbook, dict)
            and factbook.get('requested')
            and factbook.get('pertinent')
        ):
            title = str(factbook.get('title') or '').strip()
            if title and publication_source_key(record, 'factbook', title) not in posted_keys:
                publish_factbook = True

        if publish_dispatch or publish_factbook:
            pending.append({
                'line': line_number,
                'record': record,
                'draft_dispatch': publish_dispatch,
                'draft_factbook': publish_factbook,
            })

    return pending


__all__ = ['AuditStore', 'resolve_audit_store_path', 'write_audit_log', 'append_publication_backfill_log', 'load_audit_log_records', 'migrate_jsonl_audit_log', 'audit_issue_action_succeeded', 'publication_source_key', 'posted_publication_keys', 'pending_publication_entries']
