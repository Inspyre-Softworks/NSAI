"""Unified command-line entrypoint for NSAI."""

from __future__ import annotations

import argparse
import sys

from nsai import __version__
from nsai.help import NSAIArgumentParser
from nsai.advisor.live import (
    add_advise_arguments,
    add_check_issues_arguments,
    add_publications_arguments,
    run_advise,
    run_check_issues,
)
from nsai.nations import add_nation_arguments
from nsai.world_dataset import add_world_arguments
from nsai.profile.builder import (
    add_textual_dev_argument,
    run_enrich,
    run_interview,
    run_preview,
)


def build_parser() -> argparse.ArgumentParser:
    parser = NSAIArgumentParser(
        prog='nsai',
        description='NationStates AI profile builder and live advisor.',
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}',
    )
    parser.add_argument(
        '--save-opts',
        action='store_true',
        default=argparse.SUPPRESS,
        help='Save supported command-line options as program defaults.',
    )

    subparsers = parser.add_subparsers(dest='command')

    profile_parser = subparsers.add_parser(
        'profile',
        help='Build, enrich, and inspect governance profile JSON files.',
    )
    profile_subparsers = profile_parser.add_subparsers(dest='profile_command')

    interview_parser = profile_subparsers.add_parser(
        'interview',
        help='Launch the Textual interview UI.',
    )
    add_textual_dev_argument(interview_parser)
    interview_parser.add_argument(
        '--no-ai-append',
        action='store_true',
        help='Save only the interview profile without asking the AI to append vision/constitution.',
    )
    interview_parser.set_defaults(func=run_interview)

    enrich_parser = profile_subparsers.add_parser(
        'enrich',
        help='Append AI-generated vision/constitution to an existing profile JSON.',
    )
    enrich_parser.add_argument(
        'profile',
        help='Path to an existing governance profile JSON.',
    )
    enrich_parser.add_argument(
        '-o',
        '--output',
        help='Output path. Defaults to <name>_enriched.json unless --in-place is used.',
    )
    enrich_parser.add_argument(
        '--in-place',
        action='store_true',
        help='Overwrite the input profile after making a backup.',
    )
    enrich_parser.add_argument(
        '--no-backup',
        action='store_true',
        help='When using --in-place, do not create a backup file.',
    )
    enrich_parser.add_argument(
        '--force',
        action='store_true',
        help='Regenerate ai_generated even if it already exists.',
    )
    enrich_parser.add_argument(
        '--strict',
        action='store_true',
        help='Fail instead of using fallback text if the local AI fails.',
    )
    enrich_parser.set_defaults(func=run_enrich)

    preview_parser = profile_subparsers.add_parser(
        'preview',
        help='Pretty-print a profile JSON.',
    )
    preview_parser.add_argument(
        'profile',
        help='Path to a governance profile JSON.',
    )
    preview_parser.set_defaults(func=run_preview)

    advise_parser = subparsers.add_parser(
        'advise',
        help='Recommend, audit, and optionally enact a live NationStates issue choice.',
    )
    add_advise_arguments(advise_parser)
    advise_parser.set_defaults(func=run_advise)

    check_parser = subparsers.add_parser(
        'check',
        help='Run read-only NationStates checks.',
    )
    check_subparsers = check_parser.add_subparsers(dest='check_command')

    check_issues_parser = check_subparsers.add_parser(
        'issues',
        help='List current live NationStates issues without AI or actions.',
    )
    add_check_issues_arguments(check_issues_parser)
    check_issues_parser.set_defaults(func=run_check_issues)

    add_publications_arguments(subparsers)

    add_nation_arguments(subparsers)
    add_world_arguments(subparsers)

    return parser


def _run(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, 'save_opts'):
        args.save_opts = False

    if args.command is None:
        parser.print_help()
        return

    if args.command == 'profile' and args.profile_command is None:
        profile_parser = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ).choices['profile']
        profile_parser.print_help()
        return

    if args.command == 'nation' and args.nation_command is None:
        nation_parser = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ).choices['nation']
        nation_parser.print_help()
        return

    if args.command == 'world' and args.world_command is None:
        world_parser = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ).choices['world']
        world_parser.print_help()
        return

    if args.command == 'publications' and args.publications_command is None:
        publications_parser = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ).choices['publications']
        publications_parser.print_help()
        return

    if args.command == 'check' and args.check_command is None:
        check_parser = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ).choices['check']
        check_parser.print_help()
        return

    args.func(args)


def main(argv: list[str] | None = None) -> int:
    try:
        _run(argv)
    except KeyboardInterrupt:
        print('\nCancelled.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'\nERROR: {exc}', file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
