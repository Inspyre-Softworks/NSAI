from __future__ import annotations

import pytest

import nsai.nations as nations
from nsai import __version__
from nsai.cli import build_parser, main
from nsai.help import NSAIArgumentParser, NSAIHelpFormatter


def _subparser(parser, name: str):
    subparsers_action = next(
        action
        for action in parser._actions
        if name in (getattr(action, 'choices', None) or {})
    )
    return subparsers_action.choices[name]


def test_cli_help_uses_rich_formatter() -> None:
    parser = build_parser()
    profile_parser = _subparser(parser, 'profile')
    advise_parser = _subparser(parser, 'advise')
    profile_interview_parser = _subparser(profile_parser, 'interview')

    assert isinstance(parser, NSAIArgumentParser)
    assert parser.formatter_class is NSAIHelpFormatter
    assert isinstance(profile_parser, NSAIArgumentParser)
    assert profile_parser.formatter_class is NSAIHelpFormatter
    assert isinstance(advise_parser, NSAIArgumentParser)
    assert advise_parser.formatter_class is NSAIHelpFormatter
    assert isinstance(profile_interview_parser, NSAIArgumentParser)
    assert profile_interview_parser.formatter_class is NSAIHelpFormatter


def test_cli_help_surfaces_commands(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(['--help'])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert '--save-opts' in output
    assert 'profile' in output
    assert 'check' in output
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
    assert '--all-issues' in advise_output
    assert '--issue-order' in advise_output
    assert '--no-issue-ordering' in advise_output
    assert '--issue-id-order' in advise_output
    assert '--parallel-requests' in advise_output
    assert '--trace-api' in advise_output
    assert '--publication-cooldown-seconds' in advise_output
    assert '--issue-cooldown-seconds' in advise_output
    assert '--decision-summary' in advise_output

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(['check', '--help'])

    assert exc.value.code == 0
    check_output = capsys.readouterr().out
    assert 'issues' in check_output

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(['check', 'issues', '--help'])

    assert exc.value.code == 0
    check_issues_output = capsys.readouterr().out
    assert '--nation' in check_issues_output
    assert '--profile' in check_issues_output
    assert '--no-nation-config' in check_issues_output
    assert '--trace-api' in check_issues_output

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


def test_check_issues_command_parses() -> None:
    args = build_parser().parse_args(['check', 'issues', '--nation', 'Oringrad'])

    assert args.command == 'check'
    assert args.check_command == 'issues'
    assert args.nation == 'Oringrad'


def test_profile_interview_accepts_textual_dev_flag() -> None:
    args = build_parser().parse_args(['profile', 'interview', '--dev'])

    assert args.dev is True


def test_cli_without_command_prints_help(capsys) -> None:
    assert main([]) == 0

    output = capsys.readouterr().out
    assert 'NationStates AI profile builder' in output
    assert 'advise' in output


def test_main_reports_action_errors_without_traceback(monkeypatch, capsys) -> None:
    def _raise_error(args) -> None:
        raise RuntimeError('verification failed')

    monkeypatch.setattr(nations, 'run_nation_set', _raise_error)

    assert main(['nation', 'set', 'Oringrad']) == 1

    captured = capsys.readouterr()
    assert 'ERROR: verification failed' in captured.err
    assert 'Traceback' not in captured.out
    assert 'Traceback' not in captured.err
