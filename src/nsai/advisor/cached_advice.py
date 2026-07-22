"""Read-only Rich reporting for cached advisor recommendations."""

from __future__ import annotations

import argparse
import copy
import os
import re
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Footer, Header, Label, Static

from nsai.advisor.cache import AdviceCache, CachedAdvice
from nsai.nations import saved_default_nation_config


OPTION_REFERENCE_PATTERN = re.compile(r'\bOption\s+(-?\d+)\b', re.IGNORECASE)
WEBSITE_OPTION_LABEL_BASIS = 'website_position'


def relief_table() -> Table:
    """Create a two-column detail table with horizontal field separators."""
    table = Table(
        box=box.HORIZONTALS,
        show_header=False,
        show_lines=True,
        expand=True,
        padding=(0, 1),
    )
    table.add_column(style='bold cyan', no_wrap=True)
    table.add_column(ratio=1)
    return table


def option_number_by_id(live_issue: dict[str, Any]) -> dict[str, int]:
    """Map NationStates' possibly sparse raw IDs to website list positions."""
    return {
        str(option.get('option_id')): position
        for position, option in enumerate(live_issue.get('options') or [], start=1)
        if isinstance(option, dict) and option.get('option_id') is not None
    }


def website_option_label(cached: CachedAdvice) -> tuple[str, str, str]:
    """Return display label, style, and chosen option text for cached advice."""
    recommendation = cached.recommendation
    raw_option_id = str(recommendation.get('option_id', '')).strip()
    action = str(recommendation.get('action', '')).strip().lower()

    if action == 'dismiss' or raw_option_id == '-1':
        return 'dismissed', 'bold magenta', ''

    positions = option_number_by_id(cached.live_issue)
    website_number = positions.get(raw_option_id)
    if website_number is None:
        return f'unknown option ID {raw_option_id!r}', 'bold red', ''

    option_text = next(
        (
            str(option.get('text', '')).strip()
            for option in cached.live_issue.get('options') or []
            if isinstance(option, dict)
            and str(option.get('option_id')) == raw_option_id
        ),
        '',
    )
    return f'Option {website_number}', 'bold green', option_text


def display_reason(cached: CachedAdvice) -> str:
    """Return readable reasoning with legacy raw-ID references corrected."""
    recommendation = cached.recommendation
    reason = str(
        recommendation.get('reasoning')
        or recommendation.get('why_this_issue_first')
        or '(no reason cached)'
    )

    # New recommendations are explicitly prompted to use website positions.
    # Older cache entries referred to raw option IDs and require translation.
    if recommendation.get('option_label_basis') != WEBSITE_OPTION_LABEL_BASIS:
        positions = option_number_by_id(cached.live_issue)

        def replace_option(match: re.Match[str]) -> str:
            raw_option_id = match.group(1)
            if raw_option_id == '-1':
                return 'dismissed'
            website_number = positions.get(raw_option_id)
            return (
                f'Option {website_number}'
                if website_number is not None
                else match.group(0)
            )

        reason = OPTION_REFERENCE_PATTERN.sub(replace_option, reason)

    return re.sub(
        r'\bthe enacted option\b',
        lambda match: (
            'The chosen option'
            if match.group(0)[0].isupper()
            else 'the chosen option'
        ),
        reason,
        flags=re.IGNORECASE,
    )


def resolve_report_nation(explicit_nation: str | None) -> str:
    if explicit_nation and explicit_nation.strip():
        return explicit_nation.strip()

    environment_nation = os.environ.get('NS_NATION', '').strip()
    if environment_nation:
        return environment_nation

    nation_config, _ = saved_default_nation_config()
    if nation_config:
        return nation_config.nation_name

    raise SystemExit(
        'Provide --nation, set NS_NATION, or save a default nation with '
        '`nsai nation set`.'
    )


def _metadata_text(cached: CachedAdvice) -> Text:
    parts: list[tuple[str, str]] = []
    confidence = cached.recommendation.get('confidence')
    if confidence not in (None, ''):
        try:
            confidence_text = f'{float(confidence):.0%}'
        except (TypeError, ValueError):
            confidence_text = str(confidence)
        parts.append((f'confidence {confidence_text}', 'cyan'))

    model = cached.model or cached.recommendation.get('model')
    if model:
        parts.append((f'model {model}', 'blue'))
    if cached.source:
        parts.append((f'source {cached.source}', 'yellow'))
    parts.append((f'cached {cached.updated_at}', 'dim'))

    text = Text()
    for index, (value, style) in enumerate(parts):
        if index:
            text.append('  •  ', style='dim')
        text.append(value, style=style)
    return text


def render_cached_advice_report(
    nation: str,
    *,
    cache: AdviceCache | None = None,
    console: Console | None = None,
) -> None:
    cache = cache or AdviceCache()
    console = console or Console()
    plan = cache.get_latest_issue_plan(nation)

    if plan is None:
        console.print(f'[yellow]No cached active-issue plan found for {nation}.[/yellow]')
        return

    active_ids = {str(issue_id) for issue_id in plan.issue_ids}
    ordered_ids = [str(issue_id) for issue_id in plan.ordered_issue_ids]
    issue_ids = [
        *[issue_id for issue_id in ordered_ids if issue_id in active_ids],
        *sorted(active_ids - set(ordered_ids)),
    ]

    summary = relief_table()
    summary.add_row('Nation', nation)
    summary.add_row('Advisor snapshot', plan.updated_at)
    summary.add_row('Active issues', str(len(issue_ids)))
    summary.add_row('Cache', str(cache.path))
    console.print(Panel(summary, title='[bold blue]Cached Advisor Decisions[/bold blue]'))

    for issue_id in issue_ids:
        cached = cache.get_advice(nation, issue_id)
        if cached is None:
            console.print(
                Panel(
                    '[yellow]No cached recommendation is available.[/yellow]',
                    title=f'Issue {issue_id}',
                    border_style='yellow',
                )
            )
            continue

        title = str(cached.live_issue.get('title') or 'Untitled issue')
        choice, choice_style, option_text = website_option_label(cached)

        details = relief_table()
        details.add_row('Choice', Text(choice, style=choice_style))
        if option_text:
            details.add_row('Option text', option_text)
        details.add_row('Rationale', display_reason(cached))
        details.add_row('Details', _metadata_text(cached))
        if cached.enacted_count:
            details.add_row(
                'History',
                f'Applied by NSAI {cached.enacted_count} time(s)',
            )

        console.print(
            Panel(
                Group(details),
                title=Text.assemble(
                    (title, 'bold white'),
                    (f'  ·  Issue {issue_id}', 'dim'),
                ),
                border_style='green' if choice != 'dismissed' else 'magenta',
            )
        )


class ConfirmAdviceActionScreen(ModalScreen[bool]):
    """Require an explicit second action before returning an enact request."""

    CSS = """
    ConfirmAdviceActionScreen {
        align: center middle;
        background: rgba(3, 7, 18, 0.82);
    }

    #confirm-dialog {
        width: 72;
        height: auto;
        padding: 1 2;
        background: #111a2d;
        border: heavy #4f8cff;
    }

    #confirm-title {
        margin-bottom: 1;
        text-style: bold;
        color: #f4f7fb;
    }

    #confirm-detail {
        margin-bottom: 1;
        color: #cbd5e1;
    }

    #confirm-buttons {
        height: auto;
        align-horizontal: right;
    }

    #confirm-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, *, issue_title: str, choice: str) -> None:
        super().__init__()
        self.issue_title = issue_title
        self.choice = choice

    def compose(self) -> ComposeResult:
        verb = 'dismiss this issue' if self.choice == 'dismissed' else f'enact {self.choice}'
        with Container(id='confirm-dialog'):
            yield Label(f'Confirm {verb}?', id='confirm-title')
            yield Label(
                f'{self.issue_title}\n\nNSAI will re-check the live issue and run the '
                'normal validation, safety, audit, and publication path. A request '
                'that fails guardrails will remain advisor-only.',
                id='confirm-detail',
            )
            with Horizontal(id='confirm-buttons'):
                yield Button('Cancel', id='cancel-action')
                yield Button(
                    'Confirm dismissal' if self.choice == 'dismissed' else 'Confirm enactment',
                    id='confirm-action',
                    variant='warning' if self.choice == 'dismissed' else 'success',
                )

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == 'confirm-action')


class CachedAdviceApp(App[str | None]):
    """Interactive viewer for the latest cached active-issue decisions."""

    TITLE = 'NSAI Cached Advisor Decisions'
    SUB_TITLE = 'Read-only cached snapshot'
    BINDINGS = [
        ('q', 'quit', 'Quit'),
        ('e', 'expand_all', 'Expand all'),
        ('c', 'collapse_all', 'Collapse all'),
        *[
            Binding(
                str(number),
                f'toggle_issue({number - 1})',
                f'Toggle issue {number}',
                show=False,
            )
            for number in range(1, 10)
        ],
    ]
    CSS = """
    Screen {
        background: #090d18;
        color: #e7ecf5;
    }

    #summary {
        height: auto;
        margin: 1 2;
        padding: 1 2;
        background: #111a2d;
        border: round #4f8cff;
    }

    #issues {
        padding: 0 2 1 2;
    }

    Collapsible {
        margin-bottom: 1;
        padding: 0 1;
        background: #111827;
        border: round #31415f;
    }

    Collapsible:focus-within {
        border: heavy #4f8cff;
    }

    Collapsible.dismissed {
        border: round #c05acb;
    }

    Collapsible.missing {
        border: round #d9a441;
    }

    .issue-details {
        height: auto;
        padding: 1 2 2 2;
    }

    .issue-action {
        width: auto;
        min-width: 20;
        margin: 0 2 1 2;
    }
    """

    def __init__(
        self,
        nation: str,
        cache: AdviceCache | None = None,
        *,
        allow_actions: bool = False,
    ) -> None:
        super().__init__()
        self.nation = nation
        self.cache = cache or AdviceCache()
        self.allow_actions = allow_actions
        self.plan = self.cache.get_latest_issue_plan(nation)
        self.issue_ids = self._ordered_issue_ids()

    def _ordered_issue_ids(self) -> list[str]:
        if self.plan is None:
            return []
        active_ids = {str(issue_id) for issue_id in self.plan.issue_ids}
        ordered_ids = [str(issue_id) for issue_id in self.plan.ordered_issue_ids]
        return [
            *[issue_id for issue_id in ordered_ids if issue_id in active_ids],
            *sorted(active_ids - set(ordered_ids)),
        ]

    def compose(self) -> ComposeResult:
        yield Header()

        if self.plan is None:
            yield Static(
                Text(f'No cached active-issue plan found for {self.nation}.', style='yellow'),
                id='summary',
            )
            yield Footer()
            return

        summary = relief_table()
        summary.add_row('Nation', self.nation)
        summary.add_row('Advisor snapshot', self.plan.updated_at)
        summary.add_row('Active issues', str(len(self.issue_ids)))
        summary.add_row('Cache', str(self.cache.path))
        summary.add_row('Shortcuts', '1–9 toggle cards  •  e expand all  •  c collapse all  •  q quit')
        yield Static(summary, id='summary')

        with VerticalScroll(id='issues'):
            for index, issue_id in enumerate(self.issue_ids):
                cached = self.cache.get_advice(self.nation, issue_id)
                if cached is None:
                    with Collapsible(
                        Static(
                            '[yellow]No cached recommendation is available.[/yellow]',
                            classes='issue-details',
                        ),
                        title=f'Issue {issue_id}  ·  no cached recommendation',
                        collapsed=index != 0,
                        classes='missing',
                    ):
                        pass
                    continue

                title = str(cached.live_issue.get('title') or 'Untitled issue')
                choice, choice_style, option_text = website_option_label(cached)
                details = relief_table()
                details.add_row('Choice', Text(choice, style=choice_style))
                if option_text:
                    details.add_row('Option text', option_text)
                details.add_row('Rationale', display_reason(cached))
                details.add_row('Details', _metadata_text(cached))
                if cached.enacted_count:
                    details.add_row(
                        'History',
                        f'Applied by NSAI {cached.enacted_count} time(s)',
                    )

                classes = 'dismissed' if choice == 'dismissed' else ''
                children: list[Static | Button] = [
                    Static(details, classes='issue-details')
                ]
                if self.allow_actions:
                    action_label = (
                        'Dismiss issue'
                        if choice == 'dismissed'
                        else f'Enact {choice}'
                    )
                    children.append(
                        Button(
                            action_label,
                            id=f'issue-action-{index}',
                            classes='issue-action',
                            variant='warning' if choice == 'dismissed' else 'success',
                            disabled=choice.startswith('unknown option ID'),
                        )
                    )

                with Collapsible(
                    *children,
                    title=f'{choice}  ·  {title}  ·  Issue {issue_id}',
                    collapsed=index != 0,
                    classes=classes,
                ):
                    pass

        yield Footer()

    def action_expand_all(self) -> None:
        for collapsible in self.query(Collapsible):
            collapsible.collapsed = False

    def action_collapse_all(self) -> None:
        for collapsible in self.query(Collapsible):
            collapsible.collapsed = True

    def action_toggle_issue(self, index: int) -> None:
        cards = list(self.query(Collapsible))
        if 0 <= index < len(cards):
            cards[index].collapsed = not cards[index].collapsed

    @on(Button.Pressed, '.issue-action')
    def request_issue_action(self, event: Button.Pressed) -> None:
        if event.button.id is None:
            return
        try:
            index = int(event.button.id.rsplit('-', 1)[1])
            issue_id = self.issue_ids[index]
        except (IndexError, ValueError):
            self.notify('Could not identify that issue.', severity='error')
            return

        cached = self.cache.get_advice(self.nation, issue_id)
        if cached is None:
            self.notify('That recommendation is no longer cached.', severity='error')
            return

        choice, _, _ = website_option_label(cached)
        issue_title = str(cached.live_issue.get('title') or f'Issue {issue_id}')

        def finish_request(confirmed: bool | None) -> None:
            if confirmed:
                self.exit(issue_id)

        self.push_screen(
            ConfirmAdviceActionScreen(issue_title=issue_title, choice=choice),
            finish_request,
        )


def run_cached_advice_list(args: argparse.Namespace) -> None:
    nation = resolve_report_nation(args.nation)
    if args.plain:
        render_cached_advice_report(nation)
        return

    from nsai.profile.builder import textual_devtools_enabled

    with textual_devtools_enabled(bool(getattr(args, 'dev', False))):
        CachedAdviceApp(nation).run()


def run_advisor_tui(args: argparse.Namespace) -> None:
    """Generate all live advice, then safely service actions requested by the TUI."""
    if bool(getattr(args, 'enact', False)) or bool(getattr(args, 'auto', False)):
        raise SystemExit(
            'Do not combine --tui with --enact or --auto. Use the per-issue '
            'action buttons inside the TUI.'
        )

    # Imported lazily to avoid a module cycle while live.py re-exports CLI helpers.
    from nsai.advisor.cli import resolve_nation_name
    from nsai.advisor.governor import load_profile
    from nsai.advisor.live import run_advise
    from nsai.profile.builder import textual_devtools_enabled

    profile = (
        load_profile(Path(args.profile).expanduser().resolve())
        if getattr(args, 'profile', None)
        else None
    )
    try:
        nation = resolve_nation_name(cli_nation=args.nation, profile=profile)
    except SystemExit:
        nation = resolve_report_nation(args.nation)

    advising_args = copy.copy(args)
    advising_args.tui = False
    advising_args._tui_child = True
    advising_args.all_issues = True
    advising_args.enact = False
    advising_args.auto = False
    advising_args.decision_summary = False

    run_advise(advising_args)
    advising_args.refresh_advice = False

    while True:
        with textual_devtools_enabled(bool(getattr(args, 'dev', False))):
            issue_id = CachedAdviceApp(nation, allow_actions=True).run()

        if not issue_id:
            return

        action_args = copy.copy(args)
        action_args.tui = False
        action_args._tui_child = True
        action_args.all_issues = False
        action_args.enact = True
        action_args.auto = False
        action_args.refresh_advice = False
        action_args._target_issue_id = issue_id
        action_args._target_issue_reason = 'Selected interactively in the advice TUI.'
        action_args._target_issue_source = 'advice_tui'
        action_args._target_issue_order_fallback = False

        run_advise(action_args)

        # Refresh the live issue set after every request. If guardrails blocked the
        # action, the issue remains; if NationStates accepted it, the card disappears.
        try:
            run_advise(advising_args)
        except SystemExit as exc:
            if 'No live issues found' in str(exc):
                print('No live issues remain for this nation.')
                return
            raise


def add_advice_arguments(subparsers: argparse._SubParsersAction) -> None:
    advice_parser = subparsers.add_parser(
        'advice',
        help='Inspect cached advisor decisions without contacting NationStates.',
    )
    advice_subparsers = advice_parser.add_subparsers(dest='advice_command')
    list_parser = advice_subparsers.add_parser(
        'list',
        help='Show recommendations for the latest cached active-issue snapshot.',
    )
    list_parser.add_argument(
        '--nation',
        default=None,
        help='Nation name. Defaults to NS_NATION or the saved default nation.',
    )
    list_parser.add_argument(
        '--plain',
        action='store_true',
        help='Print a Rich report instead of opening the interactive Textual viewer.',
    )
    list_parser.add_argument(
        '--dev',
        action='store_true',
        help='Enable Textual devtools for the interactive viewer.',
    )
    list_parser.set_defaults(func=run_cached_advice_list)


__all__ = [
    'WEBSITE_OPTION_LABEL_BASIS',
    'CachedAdviceApp',
    'add_advice_arguments',
    'display_reason',
    'option_number_by_id',
    'relief_table',
    'render_cached_advice_report',
    'resolve_report_nation',
    'run_cached_advice_list',
    'run_advisor_tui',
    'website_option_label',
]
