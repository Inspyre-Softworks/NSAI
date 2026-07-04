"""
NationStates world dataset downloader and searchable DuckDB importer.

Author: Taylor B. | Inspyre-Softworks.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from nsai.advisor.live import NationStatesClient, NationStatesError
from nsai.nations import NationConfig, config_root, maybe_load_nation_config, saved_default_nation_config


WORLD_NATIONS_URL = 'https://www.nationstates.net/pages/nations.xml.gz'
DATASET_SCHEMA_VERSION = 2
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024

TEXT_COLUMNS = [
    'name',
    'type',
    'fullname',
    'motto',
    'category',
    'unstatus',
    'civilrights',
    'economy',
    'politicalfreedom',
    'region',
    'animal',
    'currency',
    'flag',
    'majorindustry',
    'govtpriority',
    'founded',
    'lastactivity',
    'influence',
]

INTEGER_COLUMNS = [
    'population',
    'tax',
    'administration',
    'welfare',
    'healthcare',
    'education',
    'spirituality',
    'defence',
    'lawandorder',
    'commerce',
    'publictransport',
    'environment',
    'socialequality',
    'firstlogin',
    'lastlogin',
]

NATION_COLUMNS = [
    'name',
    'type',
    'fullname',
    'motto',
    'category',
    'unstatus',
    'civilrights',
    'economy',
    'politicalfreedom',
    'region',
    'population',
    'tax',
    'animal',
    'currency',
    'flag',
    'majorindustry',
    'govtpriority',
    'administration',
    'welfare',
    'healthcare',
    'education',
    'spirituality',
    'defence',
    'lawandorder',
    'commerce',
    'publictransport',
    'environment',
    'socialequality',
    'founded',
    'firstlogin',
    'lastlogin',
    'lastactivity',
    'influence',
    'text_blob',
    'data_json',
]

XML_TAG_FOR_COLUMN = {
    'name': 'NAME',
    'type': 'TYPE',
    'fullname': 'FULLNAME',
    'motto': 'MOTTO',
    'category': 'CATEGORY',
    'unstatus': 'UNSTATUS',
    'civilrights': 'CIVILRIGHTS',
    'economy': 'ECONOMY',
    'politicalfreedom': 'POLITICALFREEDOM',
    'region': 'REGION',
    'population': 'POPULATION',
    'tax': 'TAX',
    'animal': 'ANIMAL',
    'currency': 'CURRENCY',
    'flag': 'FLAG',
    'majorindustry': 'MAJORINDUSTRY',
    'govtpriority': 'GOVTPRIORITY',
    'administration': 'ADMINISTRATION',
    'welfare': 'WELFARE',
    'healthcare': 'HEALTHCARE',
    'education': 'EDUCATION',
    'spirituality': 'SPIRITUALITY',
    'defence': 'DEFENCE',
    'lawandorder': 'LAWANDORDER',
    'commerce': 'COMMERCE',
    'publictransport': 'PUBLICTRANSPORT',
    'environment': 'ENVIRONMENT',
    'socialequality': 'SOCIALEQUALITY',
    'founded': 'FOUNDED',
    'firstlogin': 'FIRSTLOGIN',
    'lastlogin': 'LASTLOGIN',
    'lastactivity': 'LASTACTIVITY',
    'influence': 'INFLUENCE',
}

FAST_TAG_PATTERNS = {
    column: re.compile(
        rb'<' + tag.encode('ascii') + rb'>(.*?)</' + tag.encode('ascii') + rb'>',
        re.DOTALL,
    )
    for column, tag in XML_TAG_FOR_COLUMN.items()
}

TAG_STRIPPER = re.compile(rb'<[^>]+>')
WHITESPACE = re.compile(r'\s+')


@dataclass
class DatasetManifest:
    """Local download/import metadata for the world nations dataset."""

    schema_version: int = DATASET_SCHEMA_VERSION
    url: str = WORLD_NATIONS_URL
    last_download_date: str | None = None
    last_download_at: str | None = None
    last_import_at: str | None = None
    gz_path: str | None = None
    db_path: str | None = None
    bytes_downloaded: int | None = None
    nation_count: int | None = None
    import_mode: str | None = None
    import_seconds: float | None = None
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec='seconds'))

    @classmethod
    def load(cls, path: Path) -> 'DatasetManifest':
        if not path.exists():
            return cls()

        data = json.loads(path.read_text(encoding='utf-8'))
        return cls(
            schema_version=int(data.get('schema_version', DATASET_SCHEMA_VERSION)),
            url=str(data.get('url') or WORLD_NATIONS_URL),
            last_download_date=data.get('last_download_date'),
            last_download_at=data.get('last_download_at'),
            last_import_at=data.get('last_import_at'),
            gz_path=data.get('gz_path'),
            db_path=data.get('db_path'),
            bytes_downloaded=data.get('bytes_downloaded'),
            nation_count=data.get('nation_count'),
            import_mode=data.get('import_mode'),
            import_seconds=data.get('import_seconds'),
            updated_at=str(data.get('updated_at') or datetime.now().isoformat(timespec='seconds')),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now().isoformat(timespec='seconds')
        path.write_text(
            json.dumps(self.__dict__, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )


def world_data_dir() -> Path:
    return config_root() / 'world'


def default_gz_path() -> Path:
    return world_data_dir() / 'nations.xml.gz'


def default_db_path() -> Path:
    return world_data_dir() / 'nations.duckdb'


def default_manifest_path() -> Path:
    return world_data_dir() / 'manifest.json'


def console_from_args(args: argparse.Namespace) -> Console:
    return getattr(args, 'console', None) or Console()


def display_value(value: Any, default: str = '-') -> str:
    if value is None:
        return default

    text = str(value).strip()
    return text or default


def display_markup(value: Any, default: str = '-') -> str:
    return escape(display_value(value, default))


def display_count(value: Any, default: str = '-') -> str:
    if value is None:
        return default

    try:
        return f'{int(value):,}'
    except (TypeError, ValueError):
        return display_value(value, default)


def display_seconds(value: Any, default: str = 'unknown') -> str:
    if value is None:
        return default

    try:
        return f'{float(value):.1f}s'
    except (TypeError, ValueError):
        return display_value(value, default)


def add_summary_row(table: Table, label: str, value: Any, default: str = '-') -> None:
    table.add_row(label, display_value(value, default))


def today_key() -> str:
    """Return the local calendar-day key used for the once-per-day download gate."""

    return datetime.now().date().isoformat()


def should_download_dataset(
    *,
    manifest: DatasetManifest,
    gz_path: Path,
    force: bool = False,
    today: str | None = None,
) -> tuple[bool, str]:
    """Decide whether the remote dataset may be downloaded."""

    if force:
        return True, 'force requested'

    if not gz_path.exists():
        return True, 'local gzipped dataset is missing'

    current_day = today or today_key()
    if manifest.last_download_date == current_day:
        return False, f'already downloaded today ({current_day})'

    return True, f'last download was {manifest.last_download_date or "never"}'


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None

    value = value.strip()
    return value if value else None


def maybe_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def normalize_key(key: str) -> str:
    return key.strip().lower()


def element_to_dict(element: ET.Element) -> dict[str, Any]:
    """Parse one ElementTree <NATION> element. Kept for compatibility/tests."""

    result: dict[str, Any] = {}

    for child in list(element):
        key = child.tag.upper()

        if len(list(child)) == 0:
            value: Any = clean_text(child.text)
        else:
            value = element_to_dict(child)

        if key in result:
            existing = result[key]
            if not isinstance(existing, list):
                result[key] = [existing]
            result[key].append(value)
        else:
            result[key] = value

    return result


def flatten_values(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, int | float | bool):
        return [str(value)]

    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            output.extend(flatten_values(item))
        return output

    if isinstance(value, dict):
        output: list[str] = []
        for item in value.values():
            output.extend(flatten_values(item))
        return output

    return [str(value)]


def make_text_blob(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for value in data.values():
        parts.extend(flatten_values(value))
    return ' '.join(part for part in parts if part)


def build_nation_row(data: dict[str, Any]) -> tuple[Any, ...]:
    """Build a nations row from a parsed dict. Kept for compatibility/tests."""

    def text(field: str) -> str | None:
        value = data.get(field)
        if isinstance(value, list | dict):
            return json.dumps(value, ensure_ascii=False)
        return clean_text(None if value is None else str(value))

    def integer(field: str) -> int | None:
        value = data.get(field)
        if isinstance(value, list | dict):
            return None
        return maybe_int(value)

    return (
        text('NAME'),
        text('TYPE'),
        text('FULLNAME'),
        text('MOTTO'),
        text('CATEGORY'),
        text('UNSTATUS'),
        text('CIVILRIGHTS'),
        text('ECONOMY'),
        text('POLITICALFREEDOM'),
        text('REGION'),
        integer('POPULATION'),
        integer('TAX'),
        text('ANIMAL'),
        text('CURRENCY'),
        text('FLAG'),
        text('MAJORINDUSTRY'),
        text('GOVTPRIORITY'),
        integer('ADMINISTRATION'),
        integer('WELFARE'),
        integer('HEALTHCARE'),
        integer('EDUCATION'),
        integer('SPIRITUALITY'),
        integer('DEFENCE'),
        integer('LAWANDORDER'),
        integer('COMMERCE'),
        integer('PUBLICTRANSPORT'),
        integer('ENVIRONMENT'),
        integer('SOCIALEQUALITY'),
        text('FOUNDED'),
        integer('FIRSTLOGIN'),
        integer('LASTLOGIN'),
        text('LASTACTIVITY'),
        text('INFLUENCE'),
        make_text_blob(data),
        json.dumps(data, ensure_ascii=False),
    )


def build_attr_rows(data: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Build slow Python attr rows from parsed dict. Kept for compatibility."""

    nation_name = clean_text(str(data.get('NAME') or ''))
    if not nation_name:
        return []

    rows: list[tuple[str, str, str]] = []
    for key, value in data.items():
        for flat_value in flatten_values(value):
            if flat_value:
                rows.append((nation_name, normalize_key(key), flat_value))
    return rows


def import_duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:
        raise NationStatesError(
            'DuckDB is required for world dataset storage. Install with: pip install duckdb'
        ) from exc

    return duckdb


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def set_best_effort_pragmas(connection: Any, *, threads: int | None = None) -> None:
    pragmas = [
        'PRAGMA preserve_insertion_order=false',
    ]

    if threads and threads > 0:
        pragmas.append(f'PRAGMA threads={threads}')

    for statement in pragmas:
        try:
            connection.execute(statement)
        except Exception:
            pass


def create_schema(connection: Any) -> None:
    connection.execute('DROP TABLE IF EXISTS nation_attrs')
    connection.execute('DROP TABLE IF EXISTS nations')

    connection.execute(
        """
        CREATE TABLE nations (
            name TEXT PRIMARY KEY,
            type TEXT,
            fullname TEXT,
            motto TEXT,
            category TEXT,
            unstatus TEXT,
            civilrights TEXT,
            economy TEXT,
            politicalfreedom TEXT,
            region TEXT,
            population BIGINT,
            tax INTEGER,
            animal TEXT,
            currency TEXT,
            flag TEXT,
            majorindustry TEXT,
            govtpriority TEXT,
            administration INTEGER,
            welfare INTEGER,
            healthcare INTEGER,
            education INTEGER,
            spirituality INTEGER,
            defence INTEGER,
            lawandorder INTEGER,
            commerce INTEGER,
            publictransport INTEGER,
            environment INTEGER,
            socialequality INTEGER,
            founded TEXT,
            firstlogin BIGINT,
            lastlogin BIGINT,
            lastactivity TEXT,
            influence TEXT,
            text_blob TEXT,
            data_json JSON
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE nation_attrs (
            nation_name TEXT,
            key TEXT,
            value TEXT
        )
        """
    )


def decode_xml_text(value: bytes | None) -> str | None:
    if not value:
        return None

    text = value.decode('utf-8', errors='replace').strip()
    if not text:
        return None

    return html.unescape(text)


def extract_fast_value(record: bytes, column: str) -> str | None:
    match = FAST_TAG_PATTERNS[column].search(record)
    if not match:
        return None

    return decode_xml_text(match.group(1))


def make_fast_text_blob(values: dict[str, str | int | None]) -> str | None:
    text_parts = [
        str(value)
        for key, value in values.items()
        if value is not None and key not in {'data_json', 'text_blob'}
    ]

    if not text_parts:
        return None

    return WHITESPACE.sub(' ', ' '.join(text_parts)).strip()


def record_to_fast_row(record: bytes, *, include_json: bool = False) -> tuple[Any, ...]:
    values: dict[str, str | int | None] = {}

    for column in NATION_COLUMNS:
        if column in {'text_blob', 'data_json'}:
            continue

        raw_value = extract_fast_value(record, column)
        if column in INTEGER_COLUMNS:
            values[column] = maybe_int(raw_value)
        else:
            values[column] = raw_value

    text_blob = make_fast_text_blob(values)

    if include_json:
        json_value: str | None = json.dumps(values, ensure_ascii=False)
    else:
        json_value = None

    return tuple(
        text_blob if column == 'text_blob'
        else json_value if column == 'data_json'
        else values.get(column)
        for column in NATION_COLUMNS
    )


def iter_nation_records_from_gzip(gz_path: Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Iterator[bytes]:
    """Yield raw <NATION>...</NATION> records without ElementTree overhead."""

    start_token = b'<NATION>'
    end_token = b'</NATION>'
    end_len = len(end_token)
    buffer = b''

    with gzip.open(gz_path, 'rb') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break

            buffer += chunk

            while True:
                start_index = buffer.find(start_token)
                if start_index < 0:
                    keep = len(start_token) - 1
                    buffer = buffer[-keep:] if len(buffer) > keep else buffer
                    break

                end_index = buffer.find(end_token, start_index)
                if end_index < 0:
                    if start_index > 0:
                        buffer = buffer[start_index:]
                    break

                record_end = end_index + end_len
                yield buffer[start_index:record_end]
                buffer = buffer[record_end:]


def write_staging_csv(
    gz_path: Path,
    csv_path: Path,
    *,
    include_json: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    console: Console | None = None,
    progress_every: int = 10_000,
) -> int:
    """Stream the gzipped XML into one CSV file for DuckDB bulk loading."""

    console = console or Console()
    imported = 0

    with csv_path.open('w', encoding='utf-8', newline='') as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(NATION_COLUMNS)

        for record in iter_nation_records_from_gzip(gz_path, chunk_size=chunk_size):
            row = record_to_fast_row(record, include_json=include_json)

            if not row[0]:
                continue

            writer.writerow(row)
            imported += 1

            if progress_every > 0 and imported % progress_every == 0:
                console.print(f'[cyan]Parsed {imported:,} nations...[/]', end='\r')

    if imported:
        console.print(f'[green]Parsed {imported:,} nations.[/]             ')

    return imported


def copy_staging_csv_into_duckdb(
    connection: Any,
    csv_path: Path,
) -> None:
    csv_literal = sql_literal(csv_path.as_posix())
    connection.execute(
        f"""
        COPY nations
        FROM {csv_literal}
        (
            HEADER true,
            DELIMITER ',',
            QUOTE '"',
            ESCAPE '"',
            NULL '',
            SAMPLE_SIZE -1
        )
        """
    )


def create_indexes(connection: Any) -> None:
    connection.execute('CREATE INDEX idx_nations_name ON nations(name)')
    connection.execute('CREATE INDEX idx_nations_region ON nations(region)')
    connection.execute('CREATE INDEX idx_nations_category ON nations(category)')
    connection.execute('CREATE INDEX idx_nation_attrs_key ON nation_attrs(key)')
    connection.execute('CREATE INDEX idx_nation_attrs_value ON nation_attrs(value)')


def populate_attr_table_from_columns(connection: Any) -> None:
    """Populate nation_attrs inside DuckDB, not through Python row inserts."""

    attr_columns = [
        column
        for column in NATION_COLUMNS
        if column not in {'name', 'text_blob', 'data_json'}
    ]
    selects = []

    for column in attr_columns:
        ident = quote_ident(column)
        selects.append(
            f"SELECT name AS nation_name, '{column}' AS key, CAST({ident} AS TEXT) AS value "
            f'FROM nations WHERE {ident} IS NOT NULL'
        )

    if selects:
        connection.execute('INSERT INTO nation_attrs ' + ' UNION ALL '.join(selects))


def import_xml_gz(
    gz_path: Path,
    db_path: Path,
    *,
    include_attrs: bool = False,
    include_json: bool = False,
    keep_staging: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    threads: int | None = None,
    console: Console | None = None,
    batch_size: int | None = None,
) -> int:
    """
    Fast import path.

    Instead of Python executemany() calls, this writes one staging CSV and lets DuckDB
    bulk COPY it. The optional attr table is derived with SQL after the main load.
    """

    del batch_size  # Backwards-compatible parameter; the fast path does not batch inserts.

    duckdb = import_duckdb()
    console = console or Console()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    with tempfile.TemporaryDirectory(prefix='nsai-world-', dir=str(db_path.parent)) as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        csv_path = temp_dir / 'nations_staging.csv'

        parse_started = time.monotonic()
        row_count = write_staging_csv(
            gz_path,
            csv_path,
            include_json=include_json,
            chunk_size=chunk_size,
            console=console,
        )
        parse_elapsed = time.monotonic() - parse_started
        csv_size = csv_path.stat().st_size if csv_path.exists() else 0
        console.print(
            Panel(
                (
                    f'[bold]Rows:[/] {row_count:,}\n'
                    f'[bold]Staging CSV:[/] {csv_size / 1024 / 1024:.1f} MiB\n'
                    f'[bold]Parse time:[/] {parse_elapsed:.1f}s'
                ),
                title='World Import Staging',
                border_style='cyan',
            )
        )

        load_started = time.monotonic()
        connection = duckdb.connect(str(db_path))
        set_best_effort_pragmas(connection, threads=threads)
        create_schema(connection)
        copy_staging_csv_into_duckdb(connection, csv_path)

        if include_attrs:
            console.print('[cyan]Building nation_attrs from loaded columns...[/]')
            populate_attr_table_from_columns(connection)

        console.print('[cyan]Creating indexes...[/]')
        create_indexes(connection)
        connection.execute('CHECKPOINT')
        connection.close()
        load_elapsed = time.monotonic() - load_started
        console.print(f'[green]DuckDB bulk load complete in {load_elapsed:.1f}s.[/]')

        if keep_staging:
            kept_path = db_path.with_suffix('.staging.csv')
            if kept_path.exists():
                kept_path.unlink()
            csv_path.replace(kept_path)
            console.print(f'[yellow]Kept staging CSV:[/] {display_markup(kept_path)}')

    return row_count


def download_world_dataset(
    *,
    client: NationStatesClient,
    url: str,
    gz_path: Path,
    manifest_path: Path,
    force: bool = False,
    console: Console | None = None,
) -> tuple[bool, DatasetManifest]:
    console = console or Console()
    manifest = DatasetManifest.load(manifest_path)
    allowed, reason = should_download_dataset(
        manifest=manifest,
        gz_path=gz_path,
        force=force,
    )

    if not allowed:
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style='bold yellow', no_wrap=True)
        summary.add_column()
        add_summary_row(summary, 'Dataset', gz_path)
        add_summary_row(summary, 'Reason', reason)
        summary.add_row('Override', 'Use --force to download again.')
        console.print(
            Panel(
                summary,
                title='World Dataset Cache',
                border_style='yellow',
            )
        )
        return False, manifest

    gz_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = gz_path.with_suffix(gz_path.suffix + '.part')

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style='bold cyan', no_wrap=True)
    summary.add_column()
    add_summary_row(summary, 'Source', url)
    add_summary_row(summary, 'Destination', gz_path)
    add_summary_row(summary, 'Reason', reason)
    console.print(
        Panel(
            summary,
            title='World Dataset Download',
            border_style='cyan',
        )
    )

    with client.get_stream(
        url,
        accept='application/gzip,application/octet-stream,*/*',
        timeout=180,
    ) as response:
        total_raw = response.headers.get('Content-Length')
        total = int(total_raw) if total_raw and total_raw.isdigit() else None

        with Progress(
            SpinnerColumn(),
            TextColumn('[progress.description]{task.description}'),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task_id = progress.add_task('Downloading', total=total)
            written = 0

            with temp_path.open('wb') as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    output.write(chunk)
                    written += len(chunk)
                    progress.update(task_id, advance=len(chunk))

    temp_path.replace(gz_path)

    manifest.schema_version = DATASET_SCHEMA_VERSION
    manifest.url = url
    manifest.last_download_date = today_key()
    manifest.last_download_at = datetime.now().isoformat(timespec='seconds')
    manifest.gz_path = str(gz_path)
    manifest.bytes_downloaded = gz_path.stat().st_size
    manifest.save(manifest_path)

    console.print(
        Panel(
            (
                f'[bold]Dataset:[/] {display_markup(gz_path)}\n'
                f'[bold]Bytes:[/] {manifest.bytes_downloaded:,}'
            ),
            title='Download Complete',
            border_style='green',
        )
    )
    return True, manifest


def resolve_nation_config(args: argparse.Namespace) -> NationConfig | None:
    if getattr(args, 'no_nation_config', False):
        return None

    nation = getattr(args, 'nation', None)
    if nation:
        return maybe_load_nation_config(nation)

    config, _source = saved_default_nation_config()
    return config


def run_world_build(args: argparse.Namespace) -> None:
    console = console_from_args(args)
    nation_config = resolve_nation_config(args)
    client = NationStatesClient.from_env(nation_config)

    gz_path = Path(args.gz_path).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()

    downloaded, manifest = download_world_dataset(
        client=client,
        url=args.url,
        gz_path=gz_path,
        manifest_path=manifest_path,
        force=args.force,
        console=console,
    )

    if db_path.exists() and not downloaded and not args.rebuild:
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style='bold yellow', no_wrap=True)
        summary.add_column()
        add_summary_row(summary, 'Database', db_path)
        summary.add_row('Status', 'Import skipped.')
        summary.add_row('Override', 'Use --rebuild to rebuild from the cached gzip.')
        console.print(
            Panel(
                summary,
                title='World Dataset Ready',
                border_style='yellow',
            )
        )
        return

    started = time.monotonic()
    plan = Table.grid(padding=(0, 2))
    plan.add_column(style='bold cyan', no_wrap=True)
    plan.add_column()
    add_summary_row(plan, 'Source gzip', gz_path)
    add_summary_row(plan, 'Database', db_path)
    add_summary_row(plan, 'Attributes', 'enabled' if args.with_attrs else 'skipped')
    add_summary_row(plan, 'JSON payloads', 'enabled' if args.include_json else 'skipped')
    add_summary_row(plan, 'Threads', args.threads)
    add_summary_row(plan, 'Chunk size', f'{args.chunk_size:,} bytes')
    console.print(
        Panel(
            plan,
            title='World Dataset Import',
            border_style='cyan',
        )
    )
    count = import_xml_gz(
        gz_path,
        db_path,
        include_attrs=args.with_attrs,
        include_json=args.include_json,
        keep_staging=args.keep_staging,
        chunk_size=args.chunk_size,
        threads=args.threads,
        console=console,
    )
    elapsed = time.monotonic() - started

    manifest.schema_version = DATASET_SCHEMA_VERSION
    manifest.nation_count = count
    manifest.db_path = str(db_path)
    manifest.last_import_at = datetime.now().isoformat(timespec='seconds')
    manifest.import_mode = 'fast-csv-copy'
    manifest.import_seconds = round(elapsed, 3)
    manifest.save(manifest_path)

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style='bold green', no_wrap=True)
    summary.add_column()
    summary.add_row('Imported', f'{count:,} nations')
    summary.add_row('Elapsed', f'{elapsed:.1f}s')
    add_summary_row(summary, 'Database', db_path)
    add_summary_row(summary, 'Manifest', manifest_path)
    console.print(
        Panel(
            summary,
            title='World Dataset Ready',
            border_style='green',
        )
    )


def print_world_search_results(
    rows: list[tuple[Any, ...]],
    *,
    query: str,
    db_path: Path,
    limit: int,
    console: Console,
) -> None:
    if not rows:
        console.print(
            Panel(
                f'No nations matched [bold]{display_markup(query)}[/].',
                title='World Search',
                border_style='yellow',
            )
        )
        return

    table = Table(
        title=f'World Search: {display_markup(query)}',
        caption=(
            f'{len(rows):,} result(s) shown, limit {limit:,}; '
            f'database: {display_markup(db_path)}'
        ),
        box=box.SIMPLE_HEAVY,
        header_style='bold cyan',
        show_lines=False,
    )
    table.add_column('Nation', style='bold', no_wrap=True)
    table.add_column('Region')
    table.add_column('Category')
    table.add_column('Population', justify='right', no_wrap=True)
    table.add_column('Influence', no_wrap=True)
    table.add_column('Motto', overflow='fold', ratio=2)

    for name, region, category, population, influence, motto in rows:
        table.add_row(
            display_value(name),
            display_value(region),
            display_value(category),
            display_count(population),
            display_value(influence),
            display_value(motto),
        )

    console.print(table)


def run_world_search(args: argparse.Namespace) -> None:
    duckdb = import_duckdb()
    console = console_from_args(args)
    db_path = Path(args.db_path).expanduser().resolve()
    like = f'%{args.query.lower()}%'

    connection = duckdb.connect(str(db_path), read_only=True)
    rows = connection.execute(
        """
        SELECT name, region, category, population, influence, motto
        FROM nations
        WHERE lower(text_blob) LIKE ?
        ORDER BY name
        LIMIT ?
        """,
        [like, args.limit],
    ).fetchall()
    connection.close()

    print_world_search_results(
        rows,
        query=args.query,
        db_path=db_path,
        limit=args.limit,
        console=console,
    )


def run_world_sql(args: argparse.Namespace) -> None:
    duckdb = import_duckdb()
    db_path = Path(args.db_path).expanduser().resolve()
    connection = duckdb.connect(str(db_path), read_only=True)
    result = connection.execute(args.sql)
    columns = [description[0] for description in result.description]
    rows = result.fetchall()
    connection.close()

    print('\t'.join(columns))
    for row in rows:
        print('\t'.join('' if value is None else str(value) for value in row))


def print_world_inspection(
    *,
    db_path: Path,
    manifest: DatasetManifest,
    nation_count: int,
    region_count: int,
    attr_count: int,
    region_rows: list[tuple[Any, ...]],
    console: Console,
) -> None:
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style='bold cyan', no_wrap=True)
    summary.add_column()
    add_summary_row(summary, 'Database', db_path)
    summary.add_row('Size', f'{db_path.stat().st_size / 1024 / 1024:.2f} MiB')
    summary.add_row('Nations', f'{nation_count:,}')
    summary.add_row('Regions', f'{region_count:,}')
    summary.add_row('Attribute rows', f'{attr_count:,}')
    add_summary_row(summary, 'Import mode', manifest.import_mode, 'unknown')
    summary.add_row('Last import seconds', display_seconds(manifest.import_seconds))
    add_summary_row(summary, 'Last download date', manifest.last_download_date, 'never')
    add_summary_row(summary, 'Last import', manifest.last_import_at, 'never')

    console.print(
        Panel(
            summary,
            title='World Dataset Inspection',
            border_style='cyan',
        )
    )

    regions = Table(
        title='Top Regions by Nation Count',
        box=box.SIMPLE,
        header_style='bold green',
    )
    regions.add_column('#', justify='right', style='dim', no_wrap=True)
    regions.add_column('Region', style='bold')
    regions.add_column('Nations', justify='right', no_wrap=True)

    for index, (region, count) in enumerate(region_rows, start=1):
        regions.add_row(
            str(index),
            display_value(region, 'Unknown'),
            display_count(count),
        )

    console.print(regions)


def run_world_inspect(args: argparse.Namespace) -> None:
    duckdb = import_duckdb()
    console = console_from_args(args)
    db_path = Path(args.db_path).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()
    manifest = DatasetManifest.load(manifest_path)

    connection = duckdb.connect(str(db_path), read_only=True)
    nation_count = connection.execute('SELECT count(*) FROM nations').fetchone()[0]
    region_count = connection.execute('SELECT count(DISTINCT region) FROM nations').fetchone()[0]
    attr_count = connection.execute('SELECT count(*) FROM nation_attrs').fetchone()[0]
    rows = connection.execute(
        """
        SELECT region, count(*) AS nations
        FROM nations
        GROUP BY region
        ORDER BY nations DESC
        LIMIT 20
        """
    ).fetchall()
    connection.close()

    print_world_inspection(
        db_path=db_path,
        manifest=manifest,
        nation_count=nation_count,
        region_count=region_count,
        attr_count=attr_count,
        region_rows=rows,
        console=console,
    )


def add_world_arguments(subparsers: argparse._SubParsersAction) -> None:
    world_parser = subparsers.add_parser(
        'world',
        help='Download, build, and search the public NationStates world dataset.',
    )
    world_subparsers = world_parser.add_subparsers(dest='world_command')

    build_parser = world_subparsers.add_parser(
        'build',
        help='Download nations.xml.gz at most once per local day and build DuckDB.',
    )
    build_parser.add_argument('--url', default=WORLD_NATIONS_URL)
    build_parser.add_argument('--gz-path', type=Path, default=default_gz_path())
    build_parser.add_argument('--db-path', type=Path, default=default_db_path())
    build_parser.add_argument('--manifest-path', type=Path, default=default_manifest_path())
    build_parser.add_argument('--nation', help='Saved NSAI nation config to use for User-Agent.')
    build_parser.add_argument('--no-nation-config', action='store_true')
    build_parser.add_argument('--force', action='store_true', help='Download even if already downloaded today.')
    build_parser.add_argument('--rebuild', action='store_true', help='Rebuild DuckDB from cached gzip even if DB exists.')
    build_parser.add_argument(
        '--with-attrs',
        action='store_true',
        help='Also populate nation_attrs. Slower and larger; off by default for fast imports.',
    )
    build_parser.add_argument(
        '--include-json',
        action='store_true',
        help='Fill data_json for every row. Slower; off by default because typed columns are searchable.',
    )
    build_parser.add_argument(
        '--keep-staging',
        action='store_true',
        help='Keep the temporary staging CSV next to the database for debugging.',
    )
    build_parser.add_argument(
        '--chunk-size',
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help='Gzip streaming chunk size in bytes. Larger is usually faster.',
    )
    build_parser.add_argument(
        '--threads',
        type=int,
        default=os.cpu_count() or 4,
        help='DuckDB worker threads. Defaults to detected CPU count.',
    )
    build_parser.set_defaults(func=run_world_build)

    search_parser = world_subparsers.add_parser('search', help='Simple text search.')
    search_parser.add_argument('query')
    search_parser.add_argument('--db-path', type=Path, default=default_db_path())
    search_parser.add_argument('--limit', type=int, default=25)
    search_parser.set_defaults(func=run_world_search)

    sql_parser = world_subparsers.add_parser('sql', help='Run read-only SQL against the dataset.')
    sql_parser.add_argument('sql')
    sql_parser.add_argument('--db-path', type=Path, default=default_db_path())
    sql_parser.set_defaults(func=run_world_sql)

    inspect_parser = world_subparsers.add_parser('inspect', help='Show local dataset metadata.')
    inspect_parser.add_argument('--db-path', type=Path, default=default_db_path())
    inspect_parser.add_argument('--manifest-path', type=Path, default=default_manifest_path())
    inspect_parser.set_defaults(func=run_world_inspect)
