"""Persistent advisory cache for issue choices and per-issue recommendations."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nsai.nations import advice_cache_path


SCHEMA_VERSION = 2


@dataclass
class CachedIssueChoice:
    nation: str
    issue_signature: str
    issue_ids: list[str]
    selected_issue_id: str
    why: str
    source: str
    token_usage: dict[str, Any]
    updated_at: str


@dataclass
class CachedIssuePlan:
    nation: str
    issue_signature: str
    issue_ids: list[str]
    ordered_issue_ids: list[str]
    reasons: dict[str, str]
    source: str
    token_usage: dict[str, Any]
    updated_at: str


@dataclass
class CachedAdvice:
    nation: str
    issue_id: str
    live_issue: dict[str, Any]
    recommendation: dict[str, Any]
    source: str
    model: str | None
    token_usage: dict[str, Any]
    enacted_count: int
    last_effects: list[dict[str, Any]]
    last_headlines: list[str]
    updated_at: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def issue_ids_for(live_issues: list[dict[str, Any]]) -> list[str]:
    return sorted(str(issue['issue_id']) for issue in live_issues)


def issue_set_signature(live_issues: list[dict[str, Any]]) -> str:
    payload = json.dumps(issue_ids_for(live_issues), separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def live_issue_by_id(
    live_issues: list[dict[str, Any]],
    issue_id: str,
) -> dict[str, Any] | None:
    for issue in live_issues:
        if str(issue.get('issue_id')) == str(issue_id):
            return issue

    return None


def clean_text(value: str | None) -> str:
    return ' '.join((value or '').split())


def local_tag(tag: str) -> str:
    return tag.rsplit('}', 1)[-1].upper()


def extract_enactment_outcome(result_xml: str) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        root = ET.fromstring(result_xml)
    except ET.ParseError:
        return [], []

    headlines: list[str] = []
    effects: list[dict[str, Any]] = []
    effect_tags = {
        'CENSUS',
        'CHANGE',
        'EFFECT',
        'EFFECTS',
        'RANK',
        'RANKING',
        'RANKINGS',
        'SCALE',
        'STAT',
        'STATS',
    }

    for element in root.iter():
        tag = local_tag(element.tag)
        text = clean_text(''.join(element.itertext()))

        if tag == 'HEADLINE' and text:
            headlines.append(text)

        if tag in effect_tags:
            effects.append({
                'tag': tag.lower(),
                'attributes': dict(element.attrib),
                'text': text,
            })

    return effects, headlines


class AdviceCache:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or advice_cache_path()
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
                CREATE TABLE IF NOT EXISTS issue_choices (
                    nation TEXT NOT NULL,
                    issue_signature TEXT NOT NULL,
                    issue_ids_json TEXT NOT NULL,
                    selected_issue_id TEXT NOT NULL,
                    why TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT '',
                    token_usage_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (nation, issue_signature)
                )
                '''
            )
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS issue_plans (
                    nation TEXT NOT NULL,
                    issue_signature TEXT NOT NULL,
                    issue_ids_json TEXT NOT NULL,
                    ordered_issue_ids_json TEXT NOT NULL,
                    reasons_json TEXT NOT NULL DEFAULT '{}',
                    source TEXT NOT NULL DEFAULT '',
                    token_usage_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (nation, issue_signature)
                )
                '''
            )
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS issue_advice (
                    nation TEXT NOT NULL,
                    issue_id TEXT NOT NULL,
                    live_issue_json TEXT NOT NULL,
                    recommendation_json TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    model TEXT,
                    token_usage_json TEXT NOT NULL DEFAULT '{}',
                    enacted_count INTEGER NOT NULL DEFAULT 0,
                    last_enacted_at TEXT,
                    last_effects_json TEXT NOT NULL DEFAULT '[]',
                    last_headlines_json TEXT NOT NULL DEFAULT '[]',
                    last_result_xml TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (nation, issue_id)
                )
                '''
            )
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS enactments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nation TEXT NOT NULL,
                    issue_id TEXT NOT NULL,
                    option_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    effects_json TEXT NOT NULL,
                    headlines_json TEXT NOT NULL,
                    result_xml TEXT NOT NULL,
                    enacted_at TEXT NOT NULL
                )
                '''
            )
            connection.execute(
                '''
                INSERT OR REPLACE INTO meta (key, value)
                VALUES ('schema_version', ?)
                ''',
                (str(SCHEMA_VERSION),),
            )
            self._ensure_column(
                connection,
                'issue_choices',
                'token_usage_json',
                "TEXT NOT NULL DEFAULT '{}'",
            )
            self._ensure_column(
                connection,
                'issue_advice',
                'token_usage_json',
                "TEXT NOT NULL DEFAULT '{}'",
            )
            self._ensure_column(
                connection,
                'issue_plans',
                'token_usage_json',
                "TEXT NOT NULL DEFAULT '{}'",
            )

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {
            str(row['name'])
            for row in connection.execute(f'PRAGMA table_info({table})')
        }
        if column not in columns:
            connection.execute(f'ALTER TABLE {table} ADD COLUMN {column} {definition}')

    def get_issue_plan(
        self,
        nation: str,
        live_issues: list[dict[str, Any]],
    ) -> CachedIssuePlan | None:
        signature = issue_set_signature(live_issues)

        with self.connect() as connection:
            row = connection.execute(
                '''
                SELECT *
                FROM issue_plans
                WHERE nation = ? AND issue_signature = ?
                ''',
                (nation, signature),
            ).fetchone()

        if row is None:
            return None

        return self._cached_issue_plan_from_row(row)

    def get_latest_issue_plan(self, nation: str) -> CachedIssuePlan | None:
        """Return the newest cached active-issue snapshot for a nation."""
        with self.connect() as connection:
            row = connection.execute(
                '''
                SELECT *
                FROM issue_plans
                WHERE nation = ?
                ORDER BY updated_at DESC
                LIMIT 1
                ''',
                (nation,),
            ).fetchone()

        if row is None:
            return None

        return self._cached_issue_plan_from_row(row)

    def get_covering_issue_plan(
        self,
        nation: str,
        live_issues: list[dict[str, Any]],
    ) -> CachedIssuePlan | None:
        issue_ids = set(issue_ids_for(live_issues))
        if not issue_ids:
            return None

        with self.connect() as connection:
            rows = connection.execute(
                '''
                SELECT *
                FROM issue_plans
                WHERE nation = ?
                ORDER BY updated_at DESC
                ''',
                (nation,),
            ).fetchall()

        for row in rows:
            cached_issue_ids = set(json.loads(str(row['issue_ids_json'])))
            if issue_ids.issubset(cached_issue_ids):
                return self._cached_issue_plan_from_row(row)

        return None

    def _cached_issue_plan_from_row(self, row: sqlite3.Row) -> CachedIssuePlan:
        return CachedIssuePlan(
            nation=str(row['nation']),
            issue_signature=str(row['issue_signature']),
            issue_ids=json.loads(str(row['issue_ids_json'])),
            ordered_issue_ids=json.loads(str(row['ordered_issue_ids_json'])),
            reasons=json.loads(str(row['reasons_json'] or '{}')),
            source=str(row['source'] or ''),
            token_usage=json.loads(str(row['token_usage_json'] or '{}')),
            updated_at=str(row['updated_at']),
        )

    def save_issue_plan(
        self,
        *,
        nation: str,
        live_issues: list[dict[str, Any]],
        ordered_issue_ids: list[str],
        reasons: dict[str, str] | None = None,
        source: str = '',
        token_usage: dict[str, Any] | None = None,
    ) -> None:
        now = utc_now()
        signature = issue_set_signature(live_issues)
        issue_ids_json = json.dumps(issue_ids_for(live_issues), ensure_ascii=False)

        with self.connect() as connection:
            connection.execute(
                '''
                INSERT INTO issue_plans (
                    nation,
                    issue_signature,
                    issue_ids_json,
                    ordered_issue_ids_json,
                    reasons_json,
                    source,
                    token_usage_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(nation, issue_signature) DO UPDATE SET
                    issue_ids_json = excluded.issue_ids_json,
                    ordered_issue_ids_json = excluded.ordered_issue_ids_json,
                    reasons_json = excluded.reasons_json,
                    source = excluded.source,
                    token_usage_json = excluded.token_usage_json,
                    updated_at = excluded.updated_at
                ''',
                (
                    nation,
                    signature,
                    issue_ids_json,
                    json.dumps([str(issue_id) for issue_id in ordered_issue_ids], ensure_ascii=False),
                    json.dumps(reasons or {}, ensure_ascii=False),
                    source,
                    json.dumps(token_usage or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )

    def get_issue_choice(
        self,
        nation: str,
        live_issues: list[dict[str, Any]],
    ) -> CachedIssueChoice | None:
        signature = issue_set_signature(live_issues)

        with self.connect() as connection:
            row = connection.execute(
                '''
                SELECT *
                FROM issue_choices
                WHERE nation = ? AND issue_signature = ?
                ''',
                (nation, signature),
            ).fetchone()

        if row is None:
            return None

        return CachedIssueChoice(
            nation=str(row['nation']),
            issue_signature=str(row['issue_signature']),
            issue_ids=json.loads(str(row['issue_ids_json'])),
            selected_issue_id=str(row['selected_issue_id']),
            why=str(row['why'] or ''),
            source=str(row['source'] or ''),
            token_usage=json.loads(str(row['token_usage_json'] or '{}')),
            updated_at=str(row['updated_at']),
        )

    def save_issue_choice(
        self,
        *,
        nation: str,
        live_issues: list[dict[str, Any]],
        selected_issue_id: str,
        why: str = '',
        source: str = '',
        token_usage: dict[str, Any] | None = None,
    ) -> None:
        now = utc_now()
        signature = issue_set_signature(live_issues)
        issue_ids_json = json.dumps(issue_ids_for(live_issues), ensure_ascii=False)

        with self.connect() as connection:
            connection.execute(
                '''
                INSERT INTO issue_choices (
                    nation,
                    issue_signature,
                    issue_ids_json,
                    selected_issue_id,
                    why,
                    source,
                    token_usage_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(nation, issue_signature) DO UPDATE SET
                    issue_ids_json = excluded.issue_ids_json,
                    selected_issue_id = excluded.selected_issue_id,
                    why = excluded.why,
                    source = excluded.source,
                    token_usage_json = excluded.token_usage_json,
                    updated_at = excluded.updated_at
                ''',
                (
                    nation,
                    signature,
                    issue_ids_json,
                    str(selected_issue_id),
                    why,
                    source,
                    json.dumps(token_usage or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )

    def get_advice(self, nation: str, issue_id: str) -> CachedAdvice | None:
        with self.connect() as connection:
            row = connection.execute(
                '''
                SELECT *
                FROM issue_advice
                WHERE nation = ? AND issue_id = ?
                ''',
                (nation, str(issue_id)),
            ).fetchone()

        if row is None:
            return None

        return CachedAdvice(
            nation=str(row['nation']),
            issue_id=str(row['issue_id']),
            live_issue=json.loads(str(row['live_issue_json'])),
            recommendation=json.loads(str(row['recommendation_json'])),
            source=str(row['source'] or ''),
            model=row['model'],
            token_usage=json.loads(str(row['token_usage_json'] or '{}')),
            enacted_count=int(row['enacted_count'] or 0),
            last_effects=json.loads(str(row['last_effects_json'] or '[]')),
            last_headlines=json.loads(str(row['last_headlines_json'] or '[]')),
            updated_at=str(row['updated_at']),
        )

    def save_advice(
        self,
        *,
        nation: str,
        live_issue: dict[str, Any],
        recommendation: dict[str, Any],
        source: str,
    ) -> None:
        now = utc_now()
        issue_id = str(recommendation.get('issue_id') or live_issue.get('issue_id'))
        model = recommendation.get('model')

        with self.connect() as connection:
            connection.execute(
                '''
                INSERT INTO issue_advice (
                    nation,
                    issue_id,
                    live_issue_json,
                    recommendation_json,
                    source,
                    model,
                    token_usage_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(nation, issue_id) DO UPDATE SET
                    live_issue_json = excluded.live_issue_json,
                    recommendation_json = excluded.recommendation_json,
                    source = excluded.source,
                    model = excluded.model,
                    token_usage_json = excluded.token_usage_json,
                    updated_at = excluded.updated_at
                ''',
                (
                    nation,
                    issue_id,
                    json.dumps(live_issue, ensure_ascii=False),
                    json.dumps(recommendation, ensure_ascii=False),
                    source,
                    str(model) if model is not None else None,
                    json.dumps(recommendation.get('token_usage') or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )

    def record_enactment(
        self,
        *,
        nation: str,
        issue_id: str,
        option_id: str,
        action: str,
        result_xml: str,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        effects, headlines = extract_enactment_outcome(result_xml)
        effects_json = json.dumps(effects, ensure_ascii=False)
        headlines_json = json.dumps(headlines, ensure_ascii=False)
        now = utc_now()

        with self.connect() as connection:
            connection.execute(
                '''
                INSERT INTO enactments (
                    nation,
                    issue_id,
                    option_id,
                    action,
                    effects_json,
                    headlines_json,
                    result_xml,
                    enacted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    nation,
                    str(issue_id),
                    str(option_id),
                    action,
                    effects_json,
                    headlines_json,
                    result_xml,
                    now,
                ),
            )
            connection.execute(
                '''
                UPDATE issue_advice
                SET enacted_count = enacted_count + 1,
                    last_enacted_at = ?,
                    last_effects_json = ?,
                    last_headlines_json = ?,
                    last_result_xml = ?,
                    updated_at = ?
                WHERE nation = ? AND issue_id = ?
                ''',
                (
                    now,
                    effects_json,
                    headlines_json,
                    result_xml,
                    now,
                    nation,
                    str(issue_id),
                ),
            )

        return effects, headlines
