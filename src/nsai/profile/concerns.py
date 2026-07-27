"""Manage user-authored concerns in a nation's saved governance profile."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nsai.nations import load_nation_config
from nsai.profile.storage import load_profile_json, make_backup, write_profile_json


def normalize_concern(value: str) -> str:
    """Return a stable, single-line concern suitable for storage and matching."""

    concern = ' '.join(value.split())
    if not concern:
        raise ValueError('Concern must not be empty.')
    return concern


def profile_path_for_nation(nation: str) -> Path:
    config = load_nation_config(nation)
    if not config.profile_path:
        raise ValueError(
            f'Nation {config.nation_name} has no saved profile. Configure one with '
            f'`nsai nation set "{config.nation_name}" --profile <path>`.'
        )

    path = Path(config.profile_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f'Saved profile for {config.nation_name} does not exist: {path}'
        )
    return path


def profile_concerns(profile: dict[str, Any]) -> list[str]:
    raw = profile.get('concerns', [])
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError('Profile field "concerns" must be a list of strings.')
    return [normalize_concern(item) for item in raw if item.strip()]


def save_concerns(path: Path, profile: dict[str, Any], concerns: list[str]) -> Path:
    backup_path = make_backup(path)
    profile['concerns'] = concerns
    profile['updated_at'] = datetime.now(timezone.utc).isoformat()
    write_profile_json(path, profile)
    return backup_path


def run_concern_add(args: argparse.Namespace) -> None:
    path = profile_path_for_nation(args.nation)
    profile = load_profile_json(path)
    concerns = profile_concerns(profile)
    concern = normalize_concern(args.concern)

    if any(existing.casefold() == concern.casefold() for existing in concerns):
        print(f'Concern already exists for {args.nation}: {concern}')
        return

    concerns.append(concern)
    backup_path = save_concerns(path, profile, concerns)
    print(f'Added concern for {args.nation}: {concern}')
    print(f'Profile: {path}')
    print(f'Backup: {backup_path}')


def run_concern_list(args: argparse.Namespace) -> None:
    path = profile_path_for_nation(args.nation)
    concerns = profile_concerns(load_profile_json(path))
    if not concerns:
        print(f'No concerns configured for {args.nation}.')
        return

    print(f'Concerns for {args.nation}:')
    for index, concern in enumerate(concerns, start=1):
        print(f'  {index}. {concern}')


def run_concern_remove(args: argparse.Namespace) -> None:
    path = profile_path_for_nation(args.nation)
    profile = load_profile_json(path)
    concerns = profile_concerns(profile)
    requested = normalize_concern(args.concern)

    removed: str | None = None
    if requested.isdigit():
        index = int(requested) - 1
        if 0 <= index < len(concerns):
            removed = concerns.pop(index)
    else:
        for index, concern in enumerate(concerns):
            if concern.casefold() == requested.casefold():
                removed = concerns.pop(index)
                break

    if removed is None:
        raise ValueError(
            f'Concern not found for {args.nation}: {requested}. '
            'Use `nsai profile concern list <nation>` to see exact text or indexes.'
        )

    backup_path = save_concerns(path, profile, concerns)
    print(f'Removed concern for {args.nation}: {removed}')
    print(f'Profile: {path}')
    print(f'Backup: {backup_path}')


def add_concern_arguments(profile_subparsers: argparse._SubParsersAction) -> None:
    concern_parser = profile_subparsers.add_parser(
        'concern',
        help='Add, list, or remove explicit concerns in a nation profile.',
    )
    concern_subparsers = concern_parser.add_subparsers(dest='concern_command')

    add_parser = concern_subparsers.add_parser(
        'add',
        help='Add a concern to the nation profile.',
    )
    add_parser.add_argument('nation', help='Nation whose saved profile should be updated.')
    add_parser.add_argument(
        'concern',
        help='Concern or focus instruction, such as "Income equality should be the biggest focus".',
    )
    add_parser.set_defaults(func=run_concern_add)

    list_parser = concern_subparsers.add_parser(
        'list',
        help='List concerns in the nation profile.',
    )
    list_parser.add_argument('nation', help='Nation whose concerns should be listed.')
    list_parser.set_defaults(func=run_concern_list)

    remove_parser = concern_subparsers.add_parser(
        'remove',
        help='Remove a concern by its exact text or list number.',
    )
    remove_parser.add_argument('nation', help='Nation whose saved profile should be updated.')
    remove_parser.add_argument(
        'concern',
        help='Exact concern text or its 1-based number from the list command.',
    )
    remove_parser.set_defaults(func=run_concern_remove)


__all__ = [
    'add_concern_arguments',
    'normalize_concern',
    'profile_concerns',
    'profile_path_for_nation',
    'run_concern_add',
    'run_concern_list',
    'run_concern_remove',
]
