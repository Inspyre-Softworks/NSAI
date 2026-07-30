"""Read-only Rich reporting for cached advisor recommendations."""

from __future__ import annotations

import argparse
import copy
import os
import re
from decimal import Decimal, InvalidOperation
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

from nsai.advisor.cache import AdviceCache, CachedAdvice, extract_enactment_outcome
from nsai.advisor.census import census_scale_label
from nsai.nations import saved_default_nation_config


OPTION_REFERENCE_PATTERN = re.compile(r'\bOption\s+(-?\d+)\b', re.IGNORECASE)
WEBSITE_OPTION_LABEL_BASIS = 'website_position'
ENACT_ALL_ACTION = '__enact_all__'


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


def enactment_outcome_table(cached: CachedAdvice) -> Table:
    """Render the terminal state of an answered issue from its cached API result."""

    choice, choice_style, _ = website_option_label(cached)
    effects = cached.last_effects
    headlines = cached.last_headlines
    if cached.last_result_xml:
        parsed_effects, parsed_headlines = extract_enactment_outcome(
            cached.last_result_xml
        )
        effects = parsed_effects or effects
        headlines = parsed_headlines or headlines

    details = relief_table()
    details.add_row('Status', Text('Answered successfully', style='bold green'))
    details.add_row('Choice', Text(choice, style=choice_style))
    details.add_row('History', f'Applied by NSAI {cached.enacted_count} time(s)')

    rankings = [effect for effect in effects if effect.get('tag') == 'rank']
    if rankings:
        stats = Table(
            box=box.SIMPLE,
            show_header=True,
            expand=True,
            padding=(0, 1),
        )
        stats.add_column('Stat', style='bold cyan')
        stats.add_column('Score', justify='right')
        stats.add_column('Change', justify='right')
        stats.add_column('% change', justify='right')
        for effect in rankings:
            attributes = effect.get('attributes') or {}
            values = effect.get('values') or {}
            stats.add_row(
                census_scale_label(attributes.get('id') or '?'),
                str(values.get('score') or effect.get('text') or '—'),
                str(values.get('change') or '—'),
                str(values.get('pchange') or '—'),
            )
        details.add_row('Stats', stats)
    elif effects:
        outcome_lines = []
        for effect in effects:
            tag = str(effect.get('tag') or 'effect').replace('_', ' ').title()
            attributes = effect.get('attributes') or {}
            identifier = attributes.get('id')
            label = f'{tag} {identifier}' if identifier is not None else tag
            outcome_lines.append(f'{label}: {effect.get("text") or "—"}')
        details.add_row('Stats', '\n'.join(outcome_lines))

    if headlines:
        details.add_row('Headlines', '\n'.join(f'• {headline}' for headline in headlines))

    if not effects and not headlines:
        details.add_row('Outcome data', 'NationStates returned no rankings or headlines.')

    return details


def _decimal_change_text(value: Decimal, *, suffix: str = '') -> str:
    rounded = value.quantize(Decimal('0.000001'))
    rendered = format(rounded, 'f').rstrip('0').rstrip('.') or '0'
    if value > 0:
        rendered = f'+{rendered}'
    return f'{rendered}{suffix}'


def cumulative_enactment_effects_table(
    enactments: list[CachedAdvice],
) -> Table:
    """Combine ranking changes across recently enacted issue outcomes."""

    totals: dict[str, dict[str, Any]] = {}
    for cached in enactments:
        effects = cached.last_effects
        if cached.last_result_xml:
            parsed_effects, _ = extract_enactment_outcome(cached.last_result_xml)
            effects = parsed_effects or effects

        for effect in effects:
            if effect.get('tag') != 'rank':
                continue
            attributes = effect.get('attributes') or {}
            values = effect.get('values') or {}
            stat_id = str(attributes.get('id') or '?')
            total = totals.setdefault(stat_id, {
                'latest_score': '—',
                'change': Decimal('0'),
                'percentage_factor': Decimal('1'),
                'affected_issues': 0,
            })
            if values.get('score') not in (None, ''):
                total['latest_score'] = str(values['score'])
            try:
                total['change'] += Decimal(str(values.get('change') or '0'))
            except InvalidOperation:
                pass
            try:
                percentage = Decimal(str(values.get('pchange') or '0'))
                total['percentage_factor'] *= Decimal('1') + (
                    percentage / Decimal('100')
                )
            except InvalidOperation:
                pass
            total['affected_issues'] += 1

    details = relief_table()
    details.add_row(
        'Scope',
        f'{len(enactments)} successfully enacted issue(s) from this TUI session',
    )
    if not totals:
        details.add_row(
            'Stats',
            'NationStates returned no ranking changes for these enactments.',
        )
        return details

    stats = Table(
        box=box.SIMPLE,
        show_header=True,
        expand=True,
        padding=(0, 1),
    )
    stats.add_column('Stat', style='bold cyan')
    stats.add_column('Latest score', justify='right')
    stats.add_column('Total change', justify='right')
    stats.add_column('Combined % change', justify='right')
    stats.add_column('Affected issues', justify='right')

    def stat_sort_key(stat_id: str) -> tuple[int, int | str]:
        return (0, int(stat_id)) if stat_id.isdigit() else (1, stat_id)

    for stat_id in sorted(totals, key=stat_sort_key):
        total = totals[stat_id]
        combined_percentage = (
            total['percentage_factor'] - Decimal('1')
        ) * Decimal('100')
        stats.add_row(
            census_scale_label(stat_id),
            str(total['latest_score']),
            _decimal_change_text(total['change']),
            _decimal_change_text(combined_percentage, suffix='%'),
            str(total['affected_issues']),
        )
    details.add_row('Stats', stats)
    return details


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

        if cached.enacted_count:
            details = enactment_outcome_table(cached)
        else:
            details = relief_table()
            details.add_row('Choice', Text(choice, style=choice_style))
            if option_text:
                details.add_row('Option text', option_text)
            details.add_row('Rationale', display_reason(cached))
            details.add_row('Details', _metadata_text(cached))

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

    def __init__(
        self,
        *,
        issue_title: str,
        choice: str,
        bulk_count: int | None = None,
    ) -> None:
        super().__init__()
        self.issue_title = issue_title
        self.choice = choice
        self.bulk_count = bulk_count

    def compose(self) -> ComposeResult:
        if self.bulk_count is not None:
            verb = f'enact all {self.bulk_count} remaining recommendations'
            detail = (
                f'{self.issue_title}\n\nNSAI will process each issue separately in '
                'the current plan order. Every recommendation will be re-checked '
                'against the live issue and will run through the normal validation, '
                'safety, cooldown, audit, and publication path. Dismissal '
                'recommendations will dismiss their issue. Guardrail failures will '
                'remain advisor-only.'
            )
            confirm_label = 'Confirm enact all'
            confirm_variant = 'warning'
        else:
            verb = (
                'dismiss this issue'
                if self.choice == 'dismissed'
                else f'enact {self.choice}'
            )
            detail = (
                f'{self.issue_title}\n\nNSAI will re-check the live issue and run the '
                'normal validation, safety, audit, and publication path. A request '
                'that fails guardrails will remain advisor-only.'
            )
            confirm_label = (
                'Confirm dismissal'
                if self.choice == 'dismissed'
                else 'Confirm enactment'
            )
            confirm_variant = 'warning' if self.choice == 'dismissed' else 'success'

        with Container(id='confirm-dialog'):
            yield Label(f'Confirm {verb}?', id='confirm-title')
            yield Label(detail, id='confirm-detail')
            with Horizontal(id='confirm-buttons'):
                yield Button('Cancel', id='cancel-action')
                yield Button(
                    confirm_label,
                    id='confirm-action',
                    variant=confirm_variant,
                )

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == 'confirm-action')


class RecentEnactmentEffectsScreen(ModalScreen[None]):
    """Show the combined outcomes enacted during the current TUI session."""

    BINDINGS = [
        Binding('escape', 'close_effects', 'Back', priority=True),
    ]

    CSS = """
    RecentEnactmentEffectsScreen {
        align: center middle;
        background: rgba(3, 7, 18, 0.88);
    }

    #recent-effects-dialog {
        width: 94%;
        height: 92%;
        padding: 1 2;
        background: #090f1f;
        border: heavy #36c98f;
    }

    #recent-effects-title {
        height: auto;
        margin-bottom: 1;
        text-style: bold;
        color: #f4f7fb;
    }

    #recent-effects-list {
        height: 1fr;
    }

    .recent-effect, .cumulative-effect {
        height: auto;
        margin-bottom: 1;
    }

    #recent-effects-hint {
        height: auto;
        margin-top: 1;
        color: #94a3b8;
    }
    """

    def __init__(self, enactments: list[CachedAdvice]) -> None:
        super().__init__()
        self.enactments = enactments

    def compose(self) -> ComposeResult:
        with Container(id='recent-effects-dialog'):
            yield Label(
                f'Recent Enactment Effects · {len(self.enactments)} issue(s)',
                id='recent-effects-title',
            )
            with VerticalScroll(id='recent-effects-list'):
                yield Static(
                    Panel(
                        cumulative_enactment_effects_table(self.enactments),
                        title='Cumulative Effects',
                        border_style='bold cyan',
                    ),
                    classes='cumulative-effect',
                )
                for cached in self.enactments:
                    title = str(
                        cached.live_issue.get('title')
                        or f'Issue {cached.issue_id}'
                    )
                    yield Static(
                        Panel(
                            enactment_outcome_table(cached),
                            title=f'{title} · Issue {cached.issue_id}',
                            border_style='green',
                        ),
                        classes='recent-effect',
                    )
            yield Label('Press Tab or Escape to return.', id='recent-effects-hint')

    def action_close_effects(self) -> None:
        self.dismiss(None)


class CachedAdviceApp(App[str | None]):
    """Interactive viewer for the latest cached active-issue decisions."""

    TITLE = 'NSAI Cached Advisor Decisions'
    SUB_TITLE = 'Read-only cached snapshot'
    BINDINGS = [
        ('q', 'quit', 'Quit'),
        Binding('tab', 'show_recent_effects', 'Recent effects', priority=True),
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

    #bulk-actions {
        height: auto;
        padding: 0 2 1 2;
        align-horizontal: right;
    }

    #enact-all {
        width: auto;
        min-width: 28;
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

    Collapsible.answered {
        border: heavy #36c98f;
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
        outcome_issue_ids: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.nation = nation
        self.cache = cache or AdviceCache()
        self.allow_actions = allow_actions
        self.outcome_issue_ids = [str(issue_id) for issue_id in outcome_issue_ids or []]
        self.plan = self.cache.get_latest_issue_plan(nation)
        self.issue_ids = self._ordered_issue_ids()

    def _ordered_issue_ids(self) -> list[str]:
        active_ids = (
            {str(issue_id) for issue_id in self.plan.issue_ids}
            if self.plan is not None
            else set()
        )
        ordered_ids = (
            [str(issue_id) for issue_id in self.plan.ordered_issue_ids]
            if self.plan is not None
            else []
        )
        outcome_ids = list(dict.fromkeys(self.outcome_issue_ids))
        outcome_set = set(outcome_ids)
        return [
            *outcome_ids,
            *[
                issue_id
                for issue_id in ordered_ids
                if issue_id in active_ids and issue_id not in outcome_set
            ],
            *sorted(active_ids - set(ordered_ids) - outcome_set),
        ]

    def actionable_issue_ids(self) -> list[str]:
        outcome_set = set(self.outcome_issue_ids)
        actionable: list[str] = []
        for issue_id in self.issue_ids:
            if issue_id in outcome_set:
                continue
            cached = self.cache.get_advice(self.nation, issue_id)
            if cached is None or cached.enacted_count:
                continue
            choice, _, _ = website_option_label(cached)
            if not choice.startswith('unknown option ID'):
                actionable.append(issue_id)
        return actionable

    def _default_expanded_issue_id(self) -> str | None:
        outcome_set = set(self.outcome_issue_ids)
        return next(
            (
                issue_id
                for issue_id in self.issue_ids
                if issue_id not in outcome_set
            ),
            self.issue_ids[0] if self.issue_ids else None,
        )

    def recent_enactments(self) -> list[CachedAdvice]:
        enactments: list[CachedAdvice] = []
        for issue_id in self.outcome_issue_ids:
            cached = self.cache.get_advice(self.nation, issue_id)
            if cached is not None and cached.enacted_count:
                enactments.append(cached)
        return enactments

    def compose(self) -> ComposeResult:
        yield Header()

        if self.plan is None and not self.issue_ids:
            yield Static(
                Text(f'No cached active-issue plan found for {self.nation}.', style='yellow'),
                id='summary',
            )
            yield Footer()
            return

        summary = relief_table()
        summary.add_row('Nation', self.nation)
        summary.add_row(
            'Advisor snapshot',
            self.plan.updated_at if self.plan is not None else 'No live issues remain',
        )
        active_count = len(self.plan.issue_ids) if self.plan is not None else 0
        summary.add_row('Active issues', str(active_count))
        if self.outcome_issue_ids:
            summary.add_row('Answered this session', str(len(self.outcome_issue_ids)))
        summary.add_row('Cache', str(self.cache.path))
        summary.add_row('Shortcuts', '1–9 toggle cards  •  e expand all  •  c collapse all  •  q quit')
        yield Static(summary, id='summary')

        default_expanded_issue_id = self._default_expanded_issue_id()
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
                        collapsed=issue_id != default_expanded_issue_id,
                        classes='missing',
                    ):
                        pass
                    continue

                title = str(cached.live_issue.get('title') or 'Untitled issue')
                choice, choice_style, option_text = website_option_label(cached)
                if cached.enacted_count:
                    details = enactment_outcome_table(cached)
                else:
                    details = relief_table()
                    details.add_row('Choice', Text(choice, style=choice_style))
                    if option_text:
                        details.add_row('Option text', option_text)
                    details.add_row('Rationale', display_reason(cached))
                    details.add_row('Details', _metadata_text(cached))

                classes = (
                    'answered'
                    if cached.enacted_count
                    else ('dismissed' if choice == 'dismissed' else '')
                )
                children: list[Static | Button] = [
                    Static(details, classes='issue-details')
                ]
                if self.allow_actions and not cached.enacted_count:
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
                    title=(
                        f'Answered  ·  {choice}  ·  {title}  ·  Issue {issue_id}'
                        if cached.enacted_count
                        else f'{choice}  ·  {title}  ·  Issue {issue_id}'
                    ),
                    collapsed=issue_id != default_expanded_issue_id,
                    classes=classes,
                ):
                    pass

        actionable_count = len(self.actionable_issue_ids())
        if self.allow_actions and actionable_count:
            with Horizontal(id='bulk-actions'):
                yield Button(
                    f'Enact all recommendations ({actionable_count})',
                    id='enact-all',
                    variant='warning',
                )

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

    def action_show_recent_effects(self) -> None:
        if isinstance(self.screen, RecentEnactmentEffectsScreen):
            self.screen.dismiss(None)
            return

        enactments = self.recent_enactments()
        if not enactments:
            self.notify(
                'No issues have been enacted during this TUI session.',
                severity='information',
            )
            return
        self.push_screen(RecentEnactmentEffectsScreen(enactments))

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

    @on(Button.Pressed, '#enact-all')
    def request_all_actions(self) -> None:
        issue_ids = self.actionable_issue_ids()
        if not issue_ids:
            self.notify('No unanswered recommendations remain.', severity='warning')
            return

        def finish_request(confirmed: bool | None) -> None:
            if confirmed:
                self.exit(ENACT_ALL_ACTION)

        self.push_screen(
            ConfirmAdviceActionScreen(
                issue_title='All currently unanswered issues',
                choice='all',
                bulk_count=len(issue_ids),
            ),
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
    outcome_issue_ids: list[str] = []

    while True:
        with textual_devtools_enabled(bool(getattr(args, 'dev', False))):
            viewer = CachedAdviceApp(
                nation,
                allow_actions=True,
                outcome_issue_ids=outcome_issue_ids,
            )
            selection = viewer.run()

        if not selection:
            return

        issue_ids = (
            viewer.actionable_issue_ids()
            if selection == ENACT_ALL_ACTION
            else [selection]
        )
        shared_issue_state = {'last_action_at': None}
        shared_publication_state = {'cooldown_hit': False, 'last_post_at': None}
        viewer_cache = getattr(viewer, 'cache', None)
        for index, issue_id in enumerate(issue_ids):
            cached_before = (
                viewer_cache.get_advice(nation, issue_id)
                if viewer_cache is not None
                else None
            )
            enacted_before = (
                cached_before.enacted_count
                if cached_before is not None
                else None
            )
            action_args = copy.copy(args)
            action_args.tui = False
            action_args._tui_child = True
            action_args.all_issues = False
            action_args.enact = True
            action_args.auto = False
            action_args.refresh_advice = False
            action_args._target_issue_id = issue_id
            action_args._target_issue_reason = (
                'Selected by Enact all in the advice TUI.'
                if selection == ENACT_ALL_ACTION
                else 'Selected interactively in the advice TUI.'
            )
            action_args._target_issue_source = (
                'advice_tui_enact_all'
                if selection == ENACT_ALL_ACTION
                else 'advice_tui'
            )
            action_args._target_issue_order_fallback = False
            action_args._issue_state = shared_issue_state
            action_args._publication_state = shared_publication_state
            action_args._issue_cooldown_required = index > 0

            run_advise(action_args)
            cached_after = (
                viewer_cache.get_advice(nation, issue_id)
                if viewer_cache is not None
                else None
            )
            action_succeeded = (
                viewer_cache is None
                or (
                    cached_after is not None
                    and enacted_before is not None
                    and cached_after.enacted_count > enacted_before
                )
            )
            if action_succeeded and issue_id not in outcome_issue_ids:
                outcome_issue_ids.append(issue_id)

        # Refresh the live issue set after every request. If guardrails blocked the
        # action, the issue remains; if NationStates accepted it, the card disappears.
        try:
            run_advise(advising_args)
        except SystemExit as exc:
            if 'No live issues found' in str(exc):
                print('No live issues remain for this nation.')
            else:
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
    'ENACT_ALL_ACTION',
    'WEBSITE_OPTION_LABEL_BASIS',
    'CachedAdviceApp',
    'RecentEnactmentEffectsScreen',
    'add_advice_arguments',
    'cumulative_enactment_effects_table',
    'display_reason',
    'enactment_outcome_table',
    'option_number_by_id',
    'relief_table',
    'render_cached_advice_report',
    'resolve_report_nation',
    'run_cached_advice_list',
    'run_advisor_tui',
    'website_option_label',
]
