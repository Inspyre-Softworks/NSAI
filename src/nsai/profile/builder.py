"""
Textual NationStates AI governance profile builder and enricher.

Author: Taylor B. | Inspyre-Softworks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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


LM_BASE_URL = os.environ.get('LM_STUDIO_BASE_URL', 'http://localhost:1234/v1')
LM_MODEL = os.environ.get('LM_STUDIO_MODEL')
TEXTUAL_FEATURES_ENV = 'TEXTUAL'
TEXTUAL_DEVTOOLS_FEATURE = 'devtools'


POLICY_AREAS = [
    'economy',
    'civil_rights',
    'political_freedom',
    'public_order',
    'national_security',
    'technology',
    'environment',
    'healthcare',
    'education',
    'religion',
    'immigration',
    'corporate_power',
    'worker_rights',
    'military',
    'international_reputation',
    'weirdness',
]


RISK_LEVELS = [
    'very_cautious',
    'cautious',
    'balanced',
    'bold',
    'chaotic',
]


ENACTMENT_MODES = [
    'advise_only',
    'auto_enact_high_confidence',
    'auto_enact_unless_red_line',
    'fully_autonomous',
]


ISSUE_SELECTION_STRATEGIES = [
    'highest_impact',
    'most_aligned_with_priorities',
    'most_dangerous_if_ignored',
    'oldest_first',
    'random_roleplay',
]


@dataclass
class Step:
    key: str
    title: str
    help_text: str
    kind: str = 'input'
    default: Any = ''
    choices: list[str] | None = None


@dataclass
class GovernanceProfile:
    profile_name: str
    nation_name: str
    created_at: str
    roleplay_premise: str
    national_vision: str
    governing_style: str
    top_priorities: list[str]
    secondary_priorities: list[str]
    red_lines: list[str]
    preferred_tradeoffs: list[str]
    unacceptable_tradeoffs: list[str]
    risk_tolerance: str
    enactment_mode: str
    minimum_confidence_to_enact: float
    issue_selection_strategy: str
    tone: str
    custom_instruction: str
    scoring_weights: dict[str, int]


STEPS = [
    Step(
        key='nation_name',
        title='Nation Name',
        help_text='Which NationStates nation should this AI governor manage?',
        default='Oringrad',
    ),
    Step(
        key='profile_name',
        title='Governance Profile Name',
        help_text='Name the governing personality. Examples: Prosperous Technocracy, Benevolent Tyrant, Free Market Utopia.',
        default='Prosperous Technocracy',
    ),
    Step(
        key='roleplay_premise',
        title='Roleplay Premise',
        help_text='Describe what this nation is supposed to be. This tells the AI what kind of country it is roleplaying.',
        kind='textarea',
        default=(
            'A pragmatic, technologically advanced nation that wants prosperity, stability, '
            'and personal freedom without becoming absurdly authoritarian.'
        ),
    ),
    Step(
        key='national_vision',
        title='National Vision',
        help_text='What should this country become over time?',
        kind='textarea',
        default=(
            'A wealthy, stable, high-tech society with strong institutions, high standards '
            'of living, and enough liberty that citizens do not feel crushed by the state.'
        ),
    ),
    Step(
        key='governing_style',
        title='Governing Style',
        help_text='Describe the ruler’s personality: pragmatic, ruthless, compassionate, technocratic, populist, religious, libertarian, chaotic, etc.',
        default='pragmatic technocratic reformer',
    ),
    Step(
        key='top_priorities',
        title='Top Priorities',
        help_text=(
            'Enter up to 5 comma-separated policy areas in priority order.\n\n'
            f'Available: {", ".join(POLICY_AREAS)}'
        ),
        kind='csv_policy',
        default='economy, technology, public_order, civil_rights, education',
    ),
    Step(
        key='secondary_priorities',
        title='Secondary Priorities',
        help_text=(
            'Enter up to 5 secondary policy areas.\n\n'
            f'Available: {", ".join(POLICY_AREAS)}'
        ),
        kind='csv_policy',
        default='healthcare, environment, national_security, worker_rights, international_reputation',
    ),
    Step(
        key='red_lines',
        title='Red Lines',
        help_text='Things the AI should almost never do. One per line.',
        kind='lines',
        default=(
            'Do not intentionally collapse the economy.\n'
            'Do not enact obviously joke-only policies unless the whole profile is a joke nation.\n'
            'Do not make the nation brutally authoritarian unless explicitly required by the profile.\n'
            'Do not sacrifice long-term stability for tiny short-term approval gains.'
        ),
    ),
    Step(
        key='preferred_tradeoffs',
        title='Preferred Tradeoffs',
        help_text='Tradeoffs the AI is allowed to make. One per line.',
        kind='lines',
        default=(
            'Accept modest economic cost for long-term stability.\n'
            'Accept some regulation if it improves public welfare or reduces corruption.\n'
            'Accept limited defense spending if it protects sovereignty.'
        ),
    ),
    Step(
        key='unacceptable_tradeoffs',
        title='Unacceptable Tradeoffs',
        help_text='Tradeoffs the AI should reject. One per line.',
        kind='lines',
        default=(
            'Do not trade basic civil order for chaos.\n'
            'Do not sell out national sovereignty for short-term money.\n'
            'Do not let corporations completely replace the government.'
        ),
    ),
    Step(
        key='risk_tolerance',
        title='Risk Tolerance',
        help_text='Choose a risk level by typing the number or exact value.',
        kind='choice',
        choices=RISK_LEVELS,
        default='balanced',
    ),
    Step(
        key='enactment_mode',
        title='AI Control Level',
        help_text='Choose how much direct control the AI should have.',
        kind='choice',
        choices=ENACTMENT_MODES,
        default='advise_only',
    ),
    Step(
        key='minimum_confidence_to_enact',
        title='Minimum Auto-Enact Confidence',
        help_text='Use 0.0 to 1.0. Example: 0.85 means the AI must be very confident before auto-enacting.',
        kind='number',
        default='0.85',
    ),
    Step(
        key='issue_selection_strategy',
        title='Issue Selection Strategy',
        help_text='When multiple issues are available, which should the AI handle first?',
        kind='choice',
        choices=ISSUE_SELECTION_STRATEGIES,
        default='most_aligned_with_priorities',
    ),
    Step(
        key='tone',
        title='Advisor Tone',
        help_text='How should the AI explain its choices?',
        default='direct, witty, and honest about tradeoffs',
    ),
    Step(
        key='custom_instruction',
        title='Custom Instruction',
        help_text='Any final instruction for the AI governor?',
        kind='textarea',
        default='When in doubt, prefer durable national strength over flashy short-term wins.',
    ),
    Step(
        key='scoring_weights',
        title='Scoring Weights',
        help_text=(
            'Set rough weights from -5 to 5 using area=value pairs.\n\n'
            'Example:\n'
            'economy=5, technology=4, civil_rights=2, weirdness=-3\n\n'
            f'Available: {", ".join(POLICY_AREAS)}'
        ),
        kind='weights',
        default='economy=5, technology=4, public_order=3, civil_rights=2, weirdness=-2',
    ),
    Step(
        key='review',
        title='Review & Save',
        help_text='Review the generated governance charter. Press Save when it looks right.',
        kind='review',
        default='',
    ),
]


def get_local_model_name(client: OpenAI) -> str:
    models = client.models.list()

    if not models.data:
        raise RuntimeError('No local Studio model is loaded.')

    return LM_MODEL or models.data[0].id


def get_completion_text(response: Any) -> str:
    """Extract assistant text from normal content or LM Studio reasoning fields."""
    message = response.choices[0].message

    content = getattr(message, 'content', None)
    if isinstance(content, str) and content.strip():
        return content.strip()

    reasoning_content = getattr(message, 'reasoning_content', None)
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return reasoning_content.strip()

    reasoning = getattr(message, 'reasoning', None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    model_extra = getattr(message, 'model_extra', None)
    if isinstance(model_extra, dict):
        for key in ('content', 'reasoning_content', 'reasoning'):
            value = model_extra.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    if hasattr(message, 'model_dump'):
        dumped = message.model_dump()
        for key in ('content', 'reasoning_content', 'reasoning'):
            value = dumped.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    if hasattr(response, 'model_dump'):
        dumped_response = response.model_dump()
        try:
            dumped_message = dumped_response['choices'][0]['message']
            for key in ('content', 'reasoning_content', 'reasoning'):
                value = dumped_message.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        except (KeyError, IndexError, TypeError):
            pass

    return ''


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def fallback_ai_governance_addendum(profile_data: dict[str, Any]) -> dict[str, Any]:
    """Used if the local AI is unavailable or returns malformed output."""
    nation_name = profile_data.get('nation_name', 'the nation')
    profile_name = profile_data.get('profile_name', 'Governance Profile')
    vision = profile_data.get('national_vision', '')
    priorities = profile_data.get('top_priorities', [])

    priority_text = ', '.join(priorities) if priorities else 'national stability'

    return {
        'vision_description': (
            f'{nation_name} shall be governed under the {profile_name} doctrine: '
            f'a state guided by {priority_text}, with policy decisions measured against '
            f'the long-term national vision. {vision}'
        ),
        'short_constitution': {
            'title': f'Compact Charter of {nation_name}',
            'preamble': (
                f'We establish this charter to guide the AI governor of {nation_name} '
                'toward stable, consistent, and user-approved rule.'
            ),
            'articles': [
                {
                    'number': 1,
                    'title': 'Purpose of Government',
                    'text': 'The government shall pursue the national vision defined by the user-approved governance profile.',
                },
                {
                    'number': 2,
                    'title': 'Policy Priorities',
                    'text': f'The AI governor shall prioritize: {priority_text}.',
                },
                {
                    'number': 3,
                    'title': 'Limits on Authority',
                    'text': 'The AI governor shall respect all red lines and reject policies that violate unacceptable tradeoffs.',
                },
                {
                    'number': 4,
                    'title': 'Risk and Autonomy',
                    'text': 'The AI governor shall act only within the selected risk tolerance and enactment mode.',
                },
                {
                    'number': 5,
                    'title': 'Auditability',
                    'text': 'Every autonomous decision should be explainable, logged, and reviewable by the user.',
                },
            ],
        },
        'generation_source': 'fallback',
    }


def validate_addendum(addendum: dict[str, Any]) -> None:
    if not isinstance(addendum.get('vision_description'), str):
        raise ValueError('AI addendum missing vision_description string.')

    constitution = addendum.get('short_constitution')
    if not isinstance(constitution, dict):
        raise ValueError('AI addendum missing short_constitution object.')

    if not isinstance(constitution.get('title'), str):
        raise ValueError('Constitution missing title.')

    if not isinstance(constitution.get('preamble'), str):
        raise ValueError('Constitution missing preamble.')

    articles = constitution.get('articles')
    if not isinstance(articles, list) or len(articles) != 5:
        raise ValueError('Constitution must contain exactly 5 articles.')

    for index, article in enumerate(articles, start=1):
        if not isinstance(article, dict):
            raise ValueError(f'Article {index} is not an object.')

        if not isinstance(article.get('number'), int):
            raise ValueError(f'Article {index} missing integer number.')

        if not isinstance(article.get('title'), str):
            raise ValueError(f'Article {index} missing title.')

        if not isinstance(article.get('text'), str):
            raise ValueError(f'Article {index} missing text.')


def generate_ai_governance_addendum(profile_data: dict[str, Any]) -> dict[str, Any]:
    """
    Send the completed interview/profile JSON to the local AI and ask it to append
    a vision description and short constitution.
    """
    client = OpenAI(
        base_url=LM_BASE_URL,
        api_key=os.environ.get('LM_STUDIO_API_KEY', 'lm-studio'),
    )

    model = get_local_model_name(client)

    system = """
You are a political worldbuilding assistant for a NationStates AI governor.

You will receive a JSON governance profile created from a user interview.

Create a polished but concise governance addendum.

Return strict JSON only. No markdown.

Required JSON shape:
{
  "vision_description": "string",
  "short_constitution": {
    "title": "string",
    "preamble": "string",
    "articles": [
      {
        "number": 1,
        "title": "string",
        "text": "string"
      }
    ]
  }
}

Rules:
- Keep the vision_description to 1 or 2 paragraphs.
- Create exactly 5 constitution articles.
- The constitution must reflect the user's priorities, red lines, risk tolerance, and enactment mode.
- Do not override the user's profile.
- Do not add new political goals that are not implied by the profile.
- Keep it usable as instructions for an AI governor.
- If the profile is silly, theatrical, authoritarian, libertarian, religious, corporate, or utopian, preserve that chosen flavor.
- The user is sovereign; the AI governor is bounded by this charter.
"""

    user = {
        'task': 'Generate vision_description and short_constitution for this AI governor profile.',
        'profile': profile_data,
    }

    schema = {
        'type': 'object',
        'properties': {
            'vision_description': {
                'type': 'string',
            },
            'short_constitution': {
                'type': 'object',
                'properties': {
                    'title': {'type': 'string'},
                    'preamble': {'type': 'string'},
                    'articles': {
                        'type': 'array',
                        'minItems': 5,
                        'maxItems': 5,
                        'items': {
                            'type': 'object',
                            'properties': {
                                'number': {'type': 'integer'},
                                'title': {'type': 'string'},
                                'text': {'type': 'string'},
                            },
                            'required': ['number', 'title', 'text'],
                            'additionalProperties': False,
                        },
                    },
                },
                'required': ['title', 'preamble', 'articles'],
                'additionalProperties': False,
            },
        },
        'required': ['vision_description', 'short_constitution'],
        'additionalProperties': False,
    }

    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': json.dumps(user, indent=2)},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.45,
            response_format={
                'type': 'json_schema',
                'json_schema': {
                    'name': 'GovernanceAddendum',
                    'schema': schema,
                    'strict': True,
                },
            },
        )
    except Exception:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.45,
            response_format={'type': 'text'},
        )

    text = get_completion_text(response)

    if not text:
        raise ValueError('Local AI returned no content.')

    addendum = parse_json_object(text)
    validate_addendum(addendum)

    addendum['generation_source'] = 'local_ai'
    addendum['generated_at'] = datetime.now(timezone.utc).isoformat()
    addendum['model'] = model

    return addendum


def append_ai_generated_governance(
    profile_data: dict[str, Any],
    *,
    force: bool = False,
    use_fallback: bool = True,
) -> dict[str, Any]:
    """
    Append AI-generated vision/constitution material to the profile JSON.
    """
    enriched = dict(profile_data)

    if 'ai_generated' in enriched and not force:
        return enriched

    try:
        addendum = generate_ai_governance_addendum(profile_data)
    except Exception as exc:
        if not use_fallback:
            raise

        addendum = fallback_ai_governance_addendum(profile_data)
        addendum['generation_warning'] = str(exc)
        addendum['generated_at'] = datetime.now(timezone.utc).isoformat()

    enriched['ai_generated'] = addendum
    enriched['updated_at'] = datetime.now(timezone.utc).isoformat()

    return enriched


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
