"""
Textual NationStates AI governance profile builder and enricher.

This module serves as the primary orchestration layer. Domain logic is organized in:
- nsai.profile.models — Data models, interview schema, and constants
- nsai.profile.enrichment — AI addendum generation and validation
- nsai.profile.storage — JSON file I/O and backup helpers

Author: Taylor B. | Inspyre-Softworks.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Footer, Header, Input, Static, TextArea

# Re-export from submodules for backward compatibility
from nsai.profile.models import (  # noqa: F401
    ENACTMENT_MODES,
    ISSUE_SELECTION_STRATEGIES,
    POLICY_AREAS,
    RISK_LEVELS,
    STEPS,
    GovernanceProfile,
    Step,
)
from nsai.profile.enrichment import (  # noqa: F401
    LM_BASE_URL,
    LM_MODEL,
    append_ai_generated_governance,
    fallback_ai_governance_addendum,
    generate_ai_governance_addendum,
    get_completion_text,
    get_local_model_name,
    parse_json_object,
    validate_addendum,
)
from nsai.profile.storage import (  # noqa: F401
    default_enriched_path,
    enrich_profile_file,
    load_profile_json,
    make_backup,
    write_profile_json,
)


TEXTUAL_FEATURES_ENV = 'TEXTUAL'
TEXTUAL_DEVTOOLS_FEATURE = 'devtools'

class GovernanceProfileApp(App):
    CSS = """
    Screen {
        background: $surface;
    }

    #root {
        width: 92%;
        height: 90%;
        margin: 1 2;
        padding: 1 2;
        border: round $accent;
        background: $panel;
    }

    #brand {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    #progress {
        color: $text-muted;
        margin-bottom: 1;
    }

    #question {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    #help {
        color: $text-muted;
        margin-bottom: 1;
        height: auto;
    }

    #answer_input {
        margin-top: 1;
        margin-bottom: 1;
    }

    #answer_textarea {
        height: 12;
        margin-top: 1;
        margin-bottom: 1;
        border: round $accent;
    }

    #preview {
        height: 1fr;
        border: round $accent;
        padding: 1 2;
        margin-top: 1;
        margin-bottom: 1;
        overflow-y: auto;
    }

    #validation {
        color: $warning;
        margin-top: 1;
        min-height: 1;
    }

    #buttons {
        height: auto;
        dock: bottom;
        margin-top: 1;
    }

    Button {
        margin-right: 1;
    }
    """

    BINDINGS = [
        ('ctrl+q', 'quit', 'Quit'),
        ('ctrl+s', 'save', 'Save'),
        ('ctrl+n', 'next_step', 'Next'),
        ('ctrl+b', 'previous_step', 'Back'),
    ]

    def __init__(self, *, enrich_on_save: bool = True) -> None:
        super().__init__()
        self.index = 0
        self.answers: dict[str, Any] = {}
        self.enrich_on_save = enrich_on_save

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id='root'):
            yield Static('⚖ NationStates AI Governor Charter Builder', id='brand')
            yield Static('', id='progress')
            yield Static('', id='question')
            yield Static('', id='help')

            yield Input(id='answer_input')
            yield TextArea(id='answer_textarea')
            yield Static('', id='preview')
            yield Static('', id='validation')

            with Horizontal(id='buttons'):
                yield Button('← Back', id='back', variant='default')
                yield Button('Next →', id='next', variant='primary')
                yield Button('Save Charter', id='save', variant='success')

        yield Footer()

    def on_mount(self) -> None:
        self.load_step()

    def current_step(self) -> Step:
        return STEPS[self.index]

    def load_step(self) -> None:
        step = self.current_step()

        self.query_one('#validation', Static).update('')
        self.query_one('#progress', Static).update(self.make_progress_line())
        self.query_one('#question', Static).update(step.title)
        self.query_one('#help', Static).update(step.help_text)

        answer_input = self.query_one('#answer_input', Input)
        answer_textarea = self.query_one('#answer_textarea', TextArea)
        preview = self.query_one('#preview', Static)

        answer_input.display = step.kind in {'input', 'choice', 'csv_policy', 'number', 'weights'}
        answer_textarea.display = step.kind in {'textarea', 'lines'}
        preview.display = step.kind == 'review'

        value = self.answers.get(step.key, step.default)

        if step.kind == 'choice':
            answer_input.placeholder = self.choice_placeholder(step)
            answer_input.value = self.format_choice_value(value)
            answer_input.type = 'text'
            self.set_focus(answer_input)

        elif step.kind == 'number':
            answer_input.placeholder = str(step.default)
            answer_input.value = str(value)
            answer_input.type = 'number'
            self.set_focus(answer_input)

        elif step.kind in {'csv_policy', 'weights'}:
            answer_input.placeholder = str(step.default)
            answer_input.value = self.format_value_for_input(value)
            answer_input.type = 'text'
            self.set_focus(answer_input)

        elif step.kind == 'input':
            answer_input.placeholder = str(step.default)
            answer_input.value = str(value)
            answer_input.type = 'text'
            self.set_focus(answer_input)

        elif step.kind in {'textarea', 'lines'}:
            self.set_text_area_text(answer_textarea, self.format_value_for_textarea(value))
            self.set_focus(answer_textarea)

        elif step.kind == 'review':
            preview.update(self.build_preview_text())
            self.set_focus(self.query_one('#save', Button))

        self.query_one('#back', Button).disabled = self.index == 0
        self.query_one('#next', Button).disabled = self.index == len(STEPS) - 1
        self.query_one('#save', Button).disabled = step.kind != 'review'

    def set_text_area_text(self, text_area: TextArea, text: str) -> None:
        if hasattr(text_area, 'load_text'):
            text_area.load_text(text)
        else:
            text_area.text = text

    def make_progress_line(self) -> str:
        total = len(STEPS)
        current = self.index + 1
        width = 28
        filled = int(width * current / total)
        bar = '█' * filled + '░' * (width - filled)
        return f'Step {current}/{total}  {bar}'

    def choice_placeholder(self, step: Step) -> str:
        assert step.choices is not None
        options = ', '.join(f'{index + 1}={choice}' for index, choice in enumerate(step.choices))
        return options

    def format_choice_value(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        return str(value)

    def format_value_for_input(self, value: Any) -> str:
        if isinstance(value, list):
            return ', '.join(value)

        if isinstance(value, dict):
            pairs = [
                f'{key}={amount}'
                for key, amount in value.items()
                if amount != 0
            ]
            return ', '.join(pairs)

        return str(value)

    def format_value_for_textarea(self, value: Any) -> str:
        if isinstance(value, list):
            return '\n'.join(str(item) for item in value)

        return str(value)

    def save_current_answer(self) -> bool:
        step = self.current_step()

        if step.kind == 'review':
            return True

        answer_input = self.query_one('#answer_input', Input)
        answer_textarea = self.query_one('#answer_textarea', TextArea)

        raw = answer_textarea.text if step.kind in {'textarea', 'lines'} else answer_input.value
        raw = raw.strip()

        if not raw:
            raw = str(step.default)

        try:
            parsed = self.parse_answer(step, raw)
        except ValueError as exc:
            self.query_one('#validation', Static).update(f'⚠ {exc}')
            return False

        self.answers[step.key] = parsed
        self.query_one('#validation', Static).update('')
        return True

    def parse_answer(self, step: Step, raw: str) -> Any:
        if step.kind in {'input', 'textarea'}:
            return raw

        if step.kind == 'lines':
            return [
                line.strip()
                for line in raw.splitlines()
                if line.strip()
            ]

        if step.kind == 'csv_policy':
            values = [
                item.strip().lower().replace(' ', '_')
                for item in raw.split(',')
                if item.strip()
            ]

            if not values:
                raise ValueError('Pick at least one policy area.')

            invalid = [
                item
                for item in values
                if item not in POLICY_AREAS
            ]

            if invalid:
                raise ValueError(f'Unknown policy area(s): {", ".join(invalid)}')

            return values[:5]

        if step.kind == 'choice':
            assert step.choices is not None

            if raw.isdigit():
                index = int(raw) - 1
                if 0 <= index < len(step.choices):
                    return step.choices[index]

            normalized = raw.strip().lower()
            for choice in step.choices:
                if normalized == choice.lower():
                    return choice

            raise ValueError(f'Choose one of: {", ".join(step.choices)}')

        if step.kind == 'number':
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError('Enter a number from 0.0 to 1.0.') from exc

            if not 0.0 <= value <= 1.0:
                raise ValueError('Confidence must be from 0.0 to 1.0.')

            return value

        if step.kind == 'weights':
            weights = {area: 0 for area in POLICY_AREAS}

            if not raw:
                return weights

            parts = [
                part.strip()
                for part in raw.split(',')
                if part.strip()
            ]

            for part in parts:
                match = re.fullmatch(r'([a-zA-Z_ ]+)\s*=\s*(-?\d+)', part)
                if not match:
                    raise ValueError(
                        f'Bad weight pair: {part!r}. Use area=value, like economy=5.'
                    )

                area = match.group(1).strip().lower().replace(' ', '_')
                value = int(match.group(2))

                if area not in POLICY_AREAS:
                    raise ValueError(f'Unknown policy area: {area}')

                if not -5 <= value <= 5:
                    raise ValueError(f'Weight for {area} must be -5 through 5.')

                weights[area] = value

            return weights

        raise ValueError(f'Unknown step kind: {step.kind}')

    def build_profile(self) -> GovernanceProfile:
        defaults = {
            step.key: self.parse_answer(step, str(step.default))
            for step in STEPS
            if step.kind != 'review'
        }

        data = defaults | self.answers

        return GovernanceProfile(
            profile_name=data['profile_name'],
            nation_name=data['nation_name'],
            created_at=datetime.now(timezone.utc).isoformat(),
            roleplay_premise=data['roleplay_premise'],
            national_vision=data['national_vision'],
            governing_style=data['governing_style'],
            top_priorities=data['top_priorities'],
            secondary_priorities=data['secondary_priorities'],
            red_lines=data['red_lines'],
            preferred_tradeoffs=data['preferred_tradeoffs'],
            unacceptable_tradeoffs=data['unacceptable_tradeoffs'],
            risk_tolerance=data['risk_tolerance'],
            enactment_mode=data['enactment_mode'],
            minimum_confidence_to_enact=float(data['minimum_confidence_to_enact']),
            issue_selection_strategy=data['issue_selection_strategy'],
            tone=data['tone'],
            custom_instruction=data['custom_instruction'],
            scoring_weights=data['scoring_weights'],
        )

    def build_preview_text(self) -> str:
        profile = self.build_profile()
        data = asdict(profile)

        lines = [
            '⚖ GOVERNANCE CHARTER PREVIEW',
            '',
            f'Nation: {profile.nation_name}',
            f'Profile: {profile.profile_name}',
            f'Control Level: {profile.enactment_mode}',
            f'Min Confidence: {profile.minimum_confidence_to_enact}',
            f'Risk: {profile.risk_tolerance}',
            '',
            'Top Priorities:',
            *[f'  • {item}' for item in profile.top_priorities],
            '',
            'Red Lines:',
            *[f'  • {item}' for item in profile.red_lines],
            '',
            'JSON Preview:',
            json.dumps(data, indent=2),
        ]

        return '\n'.join(lines)

    def save_profile(self) -> Path:
        profile = self.build_profile()
        profile_data = asdict(profile)

        if self.enrich_on_save:
            enriched_profile_data = append_ai_generated_governance(
                profile_data,
                force=True,
                use_fallback=True,
            )
        else:
            enriched_profile_data = profile_data

        safe_name = profile.nation_name.lower().replace(' ', '_')
        safe_name = re.sub(r'[^a-z0-9_]+', '', safe_name)
        path = Path(f'{safe_name}_governance_profile.json')

        write_profile_json(path, enriched_profile_data)
        return path

    @on(Button.Pressed, '#next')
    def next_pressed(self, event: Button.Pressed) -> None:
        self.action_next_step()

    @on(Button.Pressed, '#back')
    def back_pressed(self, event: Button.Pressed) -> None:
        self.action_previous_step()

    @on(Button.Pressed, '#save')
    def save_pressed(self, event: Button.Pressed) -> None:
        self.action_save()

    def action_next_step(self) -> None:
        if not self.save_current_answer():
            return

        if self.index < len(STEPS) - 1:
            self.index += 1
            self.load_step()

    def action_previous_step(self) -> None:
        if self.index > 0:
            if self.current_step().kind != 'review':
                self.save_current_answer()

            self.index -= 1
            self.load_step()

    def action_save(self) -> None:
        if self.current_step().kind != 'review':
            if not self.save_current_answer():
                return

        if self.enrich_on_save:
            self.query_one('#validation', Static).update(
                '⏳ Asking local AI to draft vision and constitution...'
            )
        else:
            self.query_one('#validation', Static).update(
                '⏳ Saving governance profile...'
            )

        try:
            path = self.save_profile()
        except Exception as exc:
            self.query_one('#validation', Static).update(f'❌ Save failed: {exc}')
            return

        self.query_one('#validation', Static).update(
            f'✅ Saved governance profile to {path}'
        )


def add_textual_dev_argument(
    parser: argparse.ArgumentParser,
    *,
    default: Any = False,
) -> None:
    parser.add_argument(
        '--dev',
        action='store_true',
        default=default,
        help='Run the Textual UI with Textual devtools enabled.',
    )


def textual_features_with_devtools(features: str | None) -> str:
    enabled_features = [
        feature.strip()
        for feature in (features or '').split(',')
        if feature.strip()
    ]
    if not any(
        feature.lower() == TEXTUAL_DEVTOOLS_FEATURE
        for feature in enabled_features
    ):
        enabled_features.append(TEXTUAL_DEVTOOLS_FEATURE)

    return ','.join(enabled_features)


@contextmanager
def textual_devtools_enabled(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    previous_features = os.environ.get(TEXTUAL_FEATURES_ENV)
    os.environ[TEXTUAL_FEATURES_ENV] = textual_features_with_devtools(previous_features)
    try:
        yield
    finally:
        if previous_features is None:
            os.environ.pop(TEXTUAL_FEATURES_ENV, None)
        else:
            os.environ[TEXTUAL_FEATURES_ENV] = previous_features


def run_interview(args: argparse.Namespace) -> None:
    with textual_devtools_enabled(bool(getattr(args, 'dev', False))):
        app = GovernanceProfileApp(enrich_on_save=not args.no_ai_append)
        app.run()


def run_enrich(args: argparse.Namespace) -> None:
    input_path = Path(args.profile).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else None

    print(f'Reading profile: {input_path}')

    target_path = enrich_profile_file(
        input_path,
        output_path=output_path,
        in_place=args.in_place,
        force=args.force,
        no_backup=args.no_backup,
        strict=args.strict,
    )

    print(f'Enriched profile written to: {target_path}')


def run_preview(args: argparse.Namespace) -> None:
    input_path = Path(args.profile).expanduser().resolve()
    profile_data = load_profile_json(input_path)

    print_profile_preview(profile_data, source_path=input_path)


def format_preview_scalar(value: Any, default: str = '-') -> str:
    if value is None:
        return default

    text = str(value).strip()
    return text or default


def format_preview_list(value: Any) -> str:
    if not value:
        return '-'

    if not isinstance(value, list):
        return format_preview_scalar(value)

    return '\n'.join(f'- {item}' for item in value) if value else '-'


def add_summary_row(table: Table, label: str, value: Any) -> None:
    table.add_row(label, format_preview_scalar(value))


def print_profile_preview(
    profile_data: dict[str, Any],
    *,
    source_path: Path | None = None,
    console: Console | None = None,
) -> None:
    console = console or Console()
    profile_name = format_preview_scalar(profile_data.get('profile_name'), 'Profile')
    nation_name = format_preview_scalar(profile_data.get('nation_name'), 'Unknown nation')

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style='bold cyan', no_wrap=True)
    summary.add_column()
    add_summary_row(summary, 'Nation', nation_name)
    add_summary_row(summary, 'Profile', profile_name)
    add_summary_row(summary, 'Created', profile_data.get('created_at'))
    add_summary_row(summary, 'Updated', profile_data.get('updated_at'))
    add_summary_row(summary, 'Governing Style', profile_data.get('governing_style'))
    add_summary_row(summary, 'Risk Tolerance', profile_data.get('risk_tolerance'))
    add_summary_row(summary, 'Enactment Mode', profile_data.get('enactment_mode'))
    add_summary_row(
        summary,
        'Min Confidence',
        profile_data.get('minimum_confidence_to_enact'),
    )
    add_summary_row(
        summary,
        'Issue Strategy',
        profile_data.get('issue_selection_strategy'),
    )
    add_summary_row(summary, 'Tone', profile_data.get('tone'))
    if source_path:
        add_summary_row(summary, 'Source', source_path)

    console.print(Panel(summary, title='Governance Profile', border_style='cyan'))

    narrative = Table.grid(padding=(0, 1))
    narrative.add_column(style='bold magenta', no_wrap=True)
    narrative.add_column()
    add_summary_row(narrative, 'Premise', profile_data.get('roleplay_premise'))
    add_summary_row(narrative, 'Vision', profile_data.get('national_vision'))
    add_summary_row(narrative, 'Custom Instruction', profile_data.get('custom_instruction'))
    console.print(Panel(narrative, title='Roleplay Direction', border_style='magenta'))

    priorities = Table(
        title='Priorities, Guardrails, and Tradeoffs',
        show_lines=True,
        header_style='bold green',
    )
    priorities.add_column('Area', style='bold')
    priorities.add_column('Values')
    priorities.add_row('Top Priorities', format_preview_list(profile_data.get('top_priorities')))
    priorities.add_row(
        'Secondary Priorities',
        format_preview_list(profile_data.get('secondary_priorities')),
    )
    priorities.add_row('Red Lines', format_preview_list(profile_data.get('red_lines')))
    priorities.add_row(
        'Preferred Tradeoffs',
        format_preview_list(profile_data.get('preferred_tradeoffs')),
    )
    priorities.add_row(
        'Unacceptable Tradeoffs',
        format_preview_list(profile_data.get('unacceptable_tradeoffs')),
    )
    console.print(priorities)

    weights = profile_data.get('scoring_weights')
    if isinstance(weights, dict) and weights:
        weights_table = Table(title='Scoring Weights', header_style='bold yellow')
        weights_table.add_column('Policy Area')
        weights_table.add_column('Weight', justify='right')
        for area, value in sorted(weights.items()):
            weights_table.add_row(str(area), str(value))
        console.print(weights_table)

    ai_generated = profile_data.get('ai_generated')
    if isinstance(ai_generated, dict):
        print_ai_generated_preview(ai_generated, console)


def print_ai_generated_preview(ai_generated: dict[str, Any], console: Console) -> None:
    metadata = Table.grid(padding=(0, 2))
    metadata.add_column(style='bold blue', no_wrap=True)
    metadata.add_column()
    add_summary_row(metadata, 'Source', ai_generated.get('generation_source'))
    add_summary_row(metadata, 'Model', ai_generated.get('model'))
    add_summary_row(metadata, 'Generated', ai_generated.get('generated_at'))
    add_summary_row(metadata, 'Warning', ai_generated.get('generation_warning'))
    console.print(Panel(metadata, title='AI Generation', border_style='blue'))

    vision = ai_generated.get('vision_description')
    if vision:
        console.print(
            Panel(
                format_preview_scalar(vision),
                title='AI Vision',
                border_style='blue',
            )
        )

    constitution = ai_generated.get('short_constitution')
    if not isinstance(constitution, dict):
        return

    articles = constitution.get('articles') or []
    if not articles:
        return

    article_table = Table(
        title=format_preview_scalar(constitution.get('title'), 'Short Constitution'),
        show_lines=True,
        header_style='bold blue',
    )
    article_table.add_column('Article')
    article_table.add_column('Text')

    for article in articles:
        if not isinstance(article, dict):
            continue

        article_table.add_row(
            format_preview_scalar(article.get('name')),
            format_preview_scalar(article.get('text')),
        )

    console.print(article_table)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Build and enrich NationStates AI governor profiles.'
    )
    add_textual_dev_argument(parser, default=argparse.SUPPRESS)

    subparsers = parser.add_subparsers(dest='command')

    interview_parser = subparsers.add_parser(
        'interview',
        help='Launch the Textual interview UI.',
    )
    add_textual_dev_argument(interview_parser, default=argparse.SUPPRESS)
    interview_parser.add_argument(
        '--no-ai-append',
        action='store_true',
        help='Save only the interview profile without asking the AI to append vision/constitution.',
    )
    interview_parser.set_defaults(func=run_interview)

    enrich_parser = subparsers.add_parser(
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

    preview_parser = subparsers.add_parser(
        'preview',
        help='Pretty-print a profile JSON.',
    )
    preview_parser.add_argument(
        'profile',
        help='Path to a governance profile JSON.',
    )
    preview_parser.set_defaults(func=run_preview)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        default_args = ['interview']
        if getattr(args, 'dev', False):
            default_args.append('--dev')
        args = parser.parse_args(default_args)

    args.func(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled.')
        sys.exit(130)
    except Exception as exc:
        print(f'\nERROR: {exc}')
        sys.exit(1)
