from __future__ import annotations

import argparse
import gzip
import xml.etree.ElementTree as ET

from rich.console import Console

import nsai.world_dataset as world_dataset
from nsai.world_dataset import (
    DatasetManifest,
    build_nation_row,
    create_schema,
    element_to_dict,
    should_download_dataset,
)


def build_small_world_db(tmp_path):
    db_path = tmp_path / 'nations.duckdb'
    duckdb = world_dataset.import_duckdb()
    connection = duckdb.connect(str(db_path))
    create_schema(connection)
    connection.executemany(
        """
        INSERT INTO nations
            (name, region, category, population, influence, motto, text_blob)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                'Oringrad',
                'New Columbia',
                'Inoffensive Centrist Democracy',
                1234,
                'Powerbroker',
                'Order through reason',
                'oringrad new columbia order through reason',
            ),
            (
                'Auralia',
                'The Pacific',
                'Civil Rights Lovefest',
                4321,
                'Minnow',
                'Lanterns over the sea',
                'auralia pacific lanterns over the sea',
            ),
        ],
    )
    connection.executemany(
        'INSERT INTO nation_attrs (nation_name, key, value) VALUES (?, ?, ?)',
        [
            ('Oringrad', 'region', 'New Columbia'),
            ('Auralia', 'region', 'The Pacific'),
        ],
    )
    connection.close()
    return db_path


def test_world_dataset_daily_gate_blocks_same_day_when_file_exists(tmp_path) -> None:
    gz_path = tmp_path / 'nations.xml.gz'
    gz_path.write_bytes(b'fake-gz-data')
    manifest = DatasetManifest(last_download_date='2026-07-02')

    should_download, reason = should_download_dataset(
        manifest=manifest,
        gz_path=gz_path,
        today='2026-07-02',
    )

    assert should_download is False
    assert 'already downloaded today' in reason


def test_world_dataset_daily_gate_resets_after_midnight(tmp_path) -> None:
    gz_path = tmp_path / 'nations.xml.gz'
    gz_path.write_bytes(b'fake-gz-data')
    manifest = DatasetManifest(last_download_date='2026-07-02')

    should_download, reason = should_download_dataset(
        manifest=manifest,
        gz_path=gz_path,
        today='2026-07-03',
    )

    assert should_download is True
    assert '2026-07-02' in reason


def test_world_dataset_daily_gate_force_overrides_same_day(tmp_path) -> None:
    gz_path = tmp_path / 'nations.xml.gz'
    gz_path.write_bytes(b'fake-gz-data')
    manifest = DatasetManifest(last_download_date='2026-07-02')

    should_download, reason = should_download_dataset(
        manifest=manifest,
        gz_path=gz_path,
        force=True,
        today='2026-07-02',
    )

    assert should_download is True
    assert 'force' in reason


def test_world_dataset_parses_nation_element() -> None:
    element = ET.fromstring(
        '<NATION>'
        '<NAME>Oringrad</NAME>'
        '<REGION>New Columbia</REGION>'
        '<POPULATION>1234</POPULATION>'
        '<MOTTO>Order through reason</MOTTO>'
        '</NATION>'
    )

    data = element_to_dict(element)
    row = build_nation_row(data)

    assert data['NAME'] == 'Oringrad'
    assert data['REGION'] == 'New Columbia'
    assert row[0] == 'Oringrad'
    assert row[9] == 'New Columbia'
    assert row[10] == 1234


def test_world_dataset_fast_record_parser() -> None:
    from nsai.world_dataset import record_to_fast_row

    row = record_to_fast_row(
        b'<NATION>'
        b'<NAME>Oringrad</NAME>'
        b'<REGION>New Columbia</REGION>'
        b'<POPULATION>1234</POPULATION>'
        b'<MOTTO>Order &amp; reason</MOTTO>'
        b'</NATION>'
    )

    assert row[0] == 'Oringrad'
    assert row[3] == 'Order & reason'
    assert row[9] == 'New Columbia'
    assert row[10] == 1234
    assert 'Order & reason' in row[33]
    assert row[34] is None


def test_world_dataset_record_streamer_crosses_chunks(tmp_path) -> None:
    from nsai.world_dataset import iter_nation_records_from_gzip

    gz_path = tmp_path / 'nations.xml.gz'
    xml = (
        b'<NATIONS>'
        b'<NATION><NAME>One</NAME></NATION>'
        b'<NATION><NAME>Two</NAME></NATION>'
        b'</NATIONS>'
    )
    with gzip.open(gz_path, 'wb') as file:
        file.write(xml)

    records = list(iter_nation_records_from_gzip(gz_path, chunk_size=11))

    assert records == [
        b'<NATION><NAME>One</NAME></NATION>',
        b'<NATION><NAME>Two</NAME></NATION>',
    ]


def test_world_search_uses_rich_table_output(tmp_path) -> None:
    db_path = build_small_world_db(tmp_path)
    console = Console(record=True, width=140, color_system=None)

    world_dataset.run_world_search(
        argparse.Namespace(
            db_path=db_path,
            query='oringrad',
            limit=5,
            console=console,
        )
    )

    output = console.export_text()
    assert 'World Search: oringrad' in output
    assert 'Oringrad' in output
    assert 'New Columbia' in output
    assert 'Order through reason' in output
    assert '1 result(s) shown' in output


def test_world_inspect_uses_rich_summary_and_region_table(tmp_path) -> None:
    db_path = build_small_world_db(tmp_path)
    manifest_path = tmp_path / 'manifest.json'
    DatasetManifest(
        last_download_date='2026-07-03',
        last_import_at='2026-07-03T12:00:00',
        import_mode='fast-csv-copy',
        import_seconds=1.234,
    ).save(manifest_path)
    console = Console(record=True, width=140, color_system=None)

    world_dataset.run_world_inspect(
        argparse.Namespace(
            db_path=db_path,
            manifest_path=manifest_path,
            console=console,
        )
    )

    output = console.export_text()
    assert 'World Dataset Inspection' in output
    assert 'Top Regions by Nation Count' in output
    assert 'Nations' in output
    assert '2' in output
    assert 'New Columbia' in output
    assert 'fast-csv-copy' in output


def test_world_import_uses_rich_status_output(tmp_path) -> None:
    gz_path = tmp_path / 'nations.xml.gz'
    db_path = tmp_path / 'nations.duckdb'
    xml = (
        b'<NATIONS>'
        b'<NATION><NAME>One</NAME><REGION>Alpha</REGION></NATION>'
        b'<NATION><NAME>Two</NAME><REGION>Beta</REGION></NATION>'
        b'</NATIONS>'
    )
    with gzip.open(gz_path, 'wb') as file:
        file.write(xml)
    console = Console(record=True, width=140, color_system=None)

    count = world_dataset.import_xml_gz(
        gz_path,
        db_path,
        console=console,
        chunk_size=11,
    )

    output = console.export_text()
    assert count == 2
    assert 'Parsed 2 nations' in output
    assert 'World Import Staging' in output
    assert 'DuckDB bulk load complete' in output
