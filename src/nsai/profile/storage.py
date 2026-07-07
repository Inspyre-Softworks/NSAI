"""Profile JSON file storage helpers."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from nsai.profile.enrichment import append_ai_generated_governance


def load_profile_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'Profile does not exist: {path}')

    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'Profile is not valid JSON: {path}') from exc

    if not isinstance(data, dict):
        raise ValueError('Profile JSON root must be an object.')

    return data


def write_profile_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )


def default_enriched_path(input_path: Path) -> Path:
    return input_path.with_name(f'{input_path.stem}_enriched{input_path.suffix}')


def make_backup(path: Path) -> Path:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = path.with_name(f'{path.stem}.backup_{stamp}{path.suffix}')
    shutil.copy2(path, backup_path)
    return backup_path


def enrich_profile_file(
    input_path: Path,
    *,
    output_path: Path | None = None,
    in_place: bool = False,
    force: bool = False,
    no_backup: bool = False,
    strict: bool = False,
) -> Path:
    profile_data = load_profile_json(input_path)

    if 'ai_generated' in profile_data and not force:
        print('Profile already contains ai_generated. Use --force to regenerate.')
        if in_place:
            return input_path

    enriched = append_ai_generated_governance(
        profile_data,
        force=force,
        use_fallback=not strict,
    )

    if in_place:
        target_path = input_path

        if not no_backup:
            backup_path = make_backup(input_path)
            print(f'Backup written to: {backup_path}')
    else:
        target_path = output_path or default_enriched_path(input_path)

    write_profile_json(target_path, enriched)
    return target_path
