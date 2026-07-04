from __future__ import annotations

import pytest

from nsai import __version__
from nsai.cli import build_parser, main


def test_cli_help_surfaces_commands(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(['--help'])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert '--save-opts' in output
    assert 'profile' in output
    assert 'advise' in output
    assert 'publications' in output
    assert 'nation' in output
    assert 'world' in output


def test_cli_subcommand_help_surfaces_shapes(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(['profile', '--help'])

    assert exc.value.code == 0
    profile_output = capsys.readouterr().out
    assert 'interview' in profile_output
    assert 'enrich' in profile_output
    assert 'preview' in profile_output

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(['profile', 'interview', '--help'])

    assert exc.value.code == 0
    interview_output = capsys.readouterr().out
    assert '--dev' in interview_output
    assert '--no-ai-append' in interview_output

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(['advise', '--help'])

    assert exc.value.code == 0
    advise_output = capsys.readouterr().out
    assert '--save-opts' in advise_output
    assert '--base-url' in advise_output
    assert '--model' in advise_output
    assert '--lm-api-key' in advise_output
    assert '--secret-backend' in advise_output
    assert '--draft-dispatch' in advise_output
    assert '--draft-factbook' in advise_output
    assert '--enact' in advise_output
    assert '--auto' in advise_output
    assert '--audit-log' in advise_output
    assert '--no-nation-config' in advise_output
    assert '--refresh-advice' in advise_output

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(['publications', '--help'])

    assert exc.value.code == 0
    publications_output = capsys.readouterr().out
    assert 'backfill' in publications_output

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(['publications', 'backfill', '--help'])

    assert exc.value.code == 0
    backfill_output = capsys.readouterr().out
    assert '--execute' in backfill_output
    assert '--audit-log' in backfill_output
    assert '--cooldown-seconds' in backfill_output
    assert '--cooldown-retries' in backfill_output

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(['world', '--help'])

    assert exc.value.code == 0
    world_output = capsys.readouterr().out
    assert 'build' in world_output
    assert 'search' in world_output
    assert 'inspect' in world_output

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(['world', 'build', '--help'])

    assert exc.value.code == 0
    world_build_output = capsys.readouterr().out
    assert '--force' in world_build_output
    assert '--rebuild' in world_build_output

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(['nation', '--help'])

    assert exc.value.code == 0
    nation_output = capsys.readouterr().out
    assert 'set' in nation_output
    assert 'show' in nation_output
    assert 'paths' in nation_output

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(['nation', 'set', '--help'])

    assert exc.value.code == 0
    nation_set_output = capsys.readouterr().out
    assert '--base-url' in nation_set_output
    assert '--model' in nation_set_output
    assert '--lm-api-key' in nation_set_output
    assert '--secret-backend' in nation_set_output


def test_cli_version(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(['--version'])

    assert exc.value.code == 0
    assert f'nsai {__version__}' in capsys.readouterr().out


def test_global_save_opts_can_be_placed_before_or_after_advise() -> None:
    before = build_parser().parse_args(['--save-opts', 'advise', '--nation', 'Oringrad'])
    after = build_parser().parse_args(['advise', '--nation', 'Oringrad', '--save-opts'])

    assert before.save_opts is True
    assert after.save_opts is True


def test_profile_interview_accepts_textual_dev_flag() -> None:
    args = build_parser().parse_args(['profile', 'interview', '--dev'])

    assert args.dev is True


def test_cli_without_command_prints_help(capsys) -> None:
    main([])

    output = capsys.readouterr().out
    assert 'NationStates AI profile builder' in output
    assert 'advise' in output
