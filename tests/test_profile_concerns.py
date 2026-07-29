from __future__ import annotations

import argparse
import json

import pytest

from nsai.advisor.governor import build_governor_instruction, compact_profile_for_ai
from nsai.cli import build_parser, main
from nsai.nations import NationConfig, save_nation_config
from nsai.profile.concerns import (
    run_concern_add,
    run_concern_list,
    run_concern_remove,
)


def configured_profile(tmp_path, monkeypatch):
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config'))
    profile_path = tmp_path / 'oringrad.json'
    profile_path.write_text(
        json.dumps({
            'nation_name': 'Oringrad',
            'profile_name': 'Test Profile',
            'top_priorities': ['economy'],
            'red_lines': ['Do not collapse the economy.'],
        }),
        encoding='utf-8',
    )
    save_nation_config(NationConfig(
        nation_name='Oringrad',
        profile_path=str(profile_path),
    ))
    return profile_path


def test_concern_add_list_and_remove_by_number(tmp_path, monkeypatch, capsys) -> None:
    profile_path = configured_profile(tmp_path, monkeypatch)
    concern = 'Income equality should be the biggest focus'

    run_concern_add(argparse.Namespace(nation='Oringrad', concern=concern))
    saved = json.loads(profile_path.read_text(encoding='utf-8'))
    assert saved['concerns'] == [concern]
    assert saved['updated_at']
    assert list(tmp_path.glob('oringrad.backup_*.json'))

    run_concern_add(argparse.Namespace(
        nation='Oringrad',
        concern='income equality should be the biggest focus',
    ))
    saved = json.loads(profile_path.read_text(encoding='utf-8'))
    assert saved['concerns'] == [concern]
    assert 'already exists' in capsys.readouterr().out

    run_concern_list(argparse.Namespace(nation='Oringrad'))
    assert f'1. {concern}' in capsys.readouterr().out

    run_concern_remove(argparse.Namespace(nation='Oringrad', concern='1'))
    saved = json.loads(profile_path.read_text(encoding='utf-8'))
    assert saved['concerns'] == []
    assert f'Removed concern for Oringrad: {concern}' in capsys.readouterr().out


def test_concern_remove_accepts_exact_text_and_rejects_missing(
    tmp_path,
    monkeypatch,
) -> None:
    profile_path = configured_profile(tmp_path, monkeypatch)
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    profile['concerns'] = ['Protect rural healthcare']
    profile_path.write_text(json.dumps(profile), encoding='utf-8')

    run_concern_remove(argparse.Namespace(
        nation='Oringrad',
        concern='protect rural healthcare',
    ))
    saved = json.loads(profile_path.read_text(encoding='utf-8'))
    assert saved['concerns'] == []

    with pytest.raises(ValueError, match='Concern not found'):
        run_concern_remove(argparse.Namespace(
            nation='Oringrad',
            concern='missing concern',
        ))


def test_concerns_are_binding_advisor_prompt_context() -> None:
    concern = 'Income equality should be the biggest focus'
    profile = {
        'nation_name': 'Oringrad',
        'profile_name': 'Test Profile',
        'top_priorities': ['economy'],
        'concerns': [concern],
    }

    summary = compact_profile_for_ai(profile)
    instruction = build_governor_instruction(profile, 'keep things stable')

    assert summary is not None
    assert summary['concerns'] == [concern]
    assert concern in instruction
    assert 'explicit user amendments' in instruction
    assert 'outranks' in instruction


def test_profile_concern_cli_shape_and_help(tmp_path, monkeypatch, capsys) -> None:
    args = build_parser().parse_args([
        'profile',
        'concern',
        'add',
        'Oringrad',
        'Income equality should be the biggest focus',
    ])
    assert args.nation == 'Oringrad'
    assert args.concern_command == 'add'

    configured_profile(tmp_path, monkeypatch)
    assert main([
        'profile',
        'concern',
        'add',
        'Oringrad',
        'Income equality should be the biggest focus',
    ]) == 0

    output = capsys.readouterr().out
    assert 'Added concern for Oringrad' in output

    assert main(['profile', 'concern']) == 0
    help_output = capsys.readouterr().out
    assert 'add' in help_output
    assert 'list' in help_output
    assert 'remove' in help_output
