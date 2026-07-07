"""Advisor audit-log helpers."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nsai.advisor.client import NationStatesError
from nsai.advisor.recommendations import get_profile_mode
from nsai.advisor.safety import publication_mismatch_reasons
from nsai.nations import normalize_nation_key


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

    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')


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

    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def load_audit_log_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records = []
    if not path.exists():
        return records

    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise NationStatesError(
                f'Audit log {path} has invalid JSON on line {line_number}: {exc}'
            ) from exc

        if isinstance(record, dict):
            records.append((line_number, record))

    return records


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


__all__ = ['write_audit_log', 'append_publication_backfill_log', 'load_audit_log_records', 'audit_issue_action_succeeded', 'publication_source_key', 'posted_publication_keys', 'pending_publication_entries']
