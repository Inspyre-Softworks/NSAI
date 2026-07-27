"""Profile data models, interview schema, and constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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


class GovernanceProfile(BaseModel):
    """Schema for a completed governance profile.

    ``extra='allow'`` because profile JSON on disk accumulates fields this
    model doesn't declare (``ai_generated``, ``updated_at``) once
    nsai.profile.enrichment appends AI-generated material after the
    interview completes.
    """

    model_config = ConfigDict(extra='allow')

    profile_name: str
    nation_name: str
    created_at: str
    roleplay_premise: str
    national_vision: str
    governing_style: str
    top_priorities: list[str]
    secondary_priorities: list[str]
    concerns: list[str] = Field(default_factory=list)
    red_lines: list[str]
    preferred_tradeoffs: list[str]
    unacceptable_tradeoffs: list[str]
    risk_tolerance: str
    enactment_mode: str
    minimum_confidence_to_enact: float = Field(ge=0.0, le=1.0)
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
        help_text=(
            'How willing should the AI governor be to accept uncertain, disruptive, or '
            'hard-to-reverse outcomes? This guides its recommendations; it does not bypass '
            'enactment safeguards.\n\n'
            '1. very_cautious — avoid major uncertainty and prefer proven, reversible choices\n'
            '2. cautious — accept limited risk when the likely benefit is clear\n'
            '3. balanced — weigh risk and reward without strongly favoring either\n'
            '4. bold — accept substantial risk for important long-term gains\n'
            '5. chaotic — embrace unpredictable or extreme outcomes for roleplay\n\n'
            'Type the number or exact value.'
        ),
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
