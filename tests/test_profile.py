from __future__ import annotations

import argparse
import json
import os

import pytest
from rich.console import Console

import nsai.profile.builder as profile_builder
from nsai.profile.builder import (
    GovernanceProfileApp,
    Step,
    build_arg_parser,
    print_profile_preview,
)
from nsai.profile.enrichment import (
    fallback_ai_governance_addendum,
    validate_addendum,
)
from nsai.profile.storage import (
    default_enriched_path,
    load_profile_json,
    write_profile_json,
)


def sample_profile() -> dict[str, object]:
    return {
        'profile_name': 'Prosperous Technocracy',
        'nation_name': 'Oringrad',
        'roleplay_premise': 'A pragmatic high-tech nation.',
        'national_vision': 'Stable, free, and wealthy.',
        'governing_style': 'pragmatic technocrat',
        'top_priorities': ['economy', 'technology'],
        'secondary_priorities': ['education'],
        'red_lines': ['Do not collapse the economy.'],
        'preferred_tradeoffs': ['Prefer durable institutions.'],
        'unacceptable_tradeoffs': ['No needless authoritarianism.'],
        'risk_tolerance': 'balanced',
        'enactment_mode': 'advise_only',
        'minimum_confidence_to_enact': 0.9,
        'issue_selection_strategy': 'most_aligned_with_priorities',
        'tone': 'measured',
        'custom_instruction': '',
        'scoring_weights': {'economy': 5},
    }


def test_profile_answer_parsing() -> None:
    app = GovernanceProfileApp(enrich_on_save=False)

    assert app.parse_answer(
        Step('top_priorities', 'Top Priorities', '', kind='csv_policy'),
        'economy, technology, civil rights',
    ) == ['economy', 'technology', 'civil_rights']

    assert app.parse_answer(
        Step(
            'risk_tolerance',
            'Risk Tolerance',
            '',
            kind='choice',
            choices=['cautious', 'balanced'],
        ),
        '2',
    ) == 'balanced'

    assert app.parse_answer(
        Step('minimum_confidence_to_enact', 'Minimum Confidence', '', kind='number'),
        '0.75',
    ) == 0.75

    weights = app.parse_answer(
        Step('scoring_weights', 'Weights', '', kind='weights'),
        'economy=5, civil rights=-2',
    )
    assert weights['economy'] == 5
    assert weights['civil_rights'] == -2

    with pytest.raises(ValueError, match='Unknown policy area'):
        app.parse_answer(
            Step('top_priorities', 'Top Priorities', '', kind='csv_policy'),
            'moonbase',
        )


def test_profile_builder_accepts_textual_dev_flag() -> None:
    explicit = build_arg_parser().parse_args(['interview', '--dev'])
    implicit = build_arg_parser().parse_args(['--dev'])

    assert explicit.dev is True
    assert implicit.dev is True


def test_run_interview_enables_textual_devtools(monkeypatch) -> None:
    recorded: dict[str, object] = {}

    class FakeGovernanceProfileApp:
        def __init__(self, *, enrich_on_save: bool) -> None:
            recorded['enrich_on_save'] = enrich_on_save
            recorded['textual_features'] = os.environ.get('TEXTUAL')

        def run(self) -> None:
            recorded['ran'] = True

    monkeypatch.setenv('TEXTUAL', 'debug')
    monkeypatch.setattr(profile_builder, 'GovernanceProfileApp', FakeGovernanceProfileApp)

    profile_builder.run_interview(
        argparse.Namespace(no_ai_append=True, dev=True)
    )

    assert recorded == {
        'enrich_on_save': False,
        'textual_features': 'debug,devtools',
        'ran': True,
    }
    assert os.environ['TEXTUAL'] == 'debug'


def test_profile_builder_main_uses_dev_for_default_interview(monkeypatch) -> None:
    recorded: dict[str, object] = {}

    class FakeGovernanceProfileApp:
        def __init__(self, *, enrich_on_save: bool) -> None:
            recorded['enrich_on_save'] = enrich_on_save
            recorded['textual_features'] = os.environ.get('TEXTUAL')

        def run(self) -> None:
            recorded['ran'] = True

    monkeypatch.delenv('TEXTUAL', raising=False)
    monkeypatch.setattr(profile_builder, 'GovernanceProfileApp', FakeGovernanceProfileApp)

    profile_builder.main(['--dev'])

    assert recorded == {
        'enrich_on_save': True,
        'textual_features': 'devtools',
        'ran': True,
    }
    assert 'TEXTUAL' not in os.environ


def test_profile_json_round_trip(tmp_path) -> None:
    path = tmp_path / 'profile.json'
    profile = sample_profile()

    write_profile_json(path, profile)

    assert load_profile_json(path) == profile
    assert json.loads(path.read_text(encoding='utf-8')) == profile
    assert default_enriched_path(path) == tmp_path / 'profile_enriched.json'


def test_addendum_validation_and_fallback(monkeypatch) -> None:
    addendum = fallback_ai_governance_addendum(sample_profile())

    validate_addendum(addendum)
    assert addendum['short_constitution']['title'] == 'Compact Charter of Oringrad'

    def fail_generation(profile_data):
        raise RuntimeError('offline')

    monkeypatch.setattr(profile_builder, 'generate_ai_governance_addendum', fail_generation)

    profile = profile_builder.append_ai_generated_governance(sample_profile())
    assert profile['ai_generated']['generation_source'] == 'fallback'
    assert profile['ai_generated']['generation_warning'] == 'offline'


def test_addendum_validation_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match='vision_description'):
        validate_addendum({'short_constitution': {'title': 'x', 'articles': []}})


def test_profile_preview_uses_rich_rendering(tmp_path) -> None:
    profile = sample_profile()
    profile['ai_generated'] = {
        'generation_source': 'fallback',
        'model': 'fallback',
        'generated_at': '2026-07-01T00:00:00+00:00',
        'vision_description': 'Oringrad shall remain prosperous and stable.',
        'short_constitution': {
            'title': 'Compact Charter of Oringrad',
            'articles': [
                {
                    'name': 'Article I',
                    'text': 'Preserve prosperity and civic stability.',
                }
            ],
        },
    }
    console = Console(record=True, width=120, color_system=None)

    print_profile_preview(profile, source_path=tmp_path / 'profile.json', console=console)

    output = console.export_text()
    assert 'Governance Profile' in output
    assert 'Roleplay Direction' in output
    assert 'Priorities, Guardrails, and Tradeoffs' in output
    assert 'Scoring Weights' in output
    assert 'AI Vision' in output
    assert 'Compact Charter of Oringrad' in output
    assert 'Prosperous Technocracy' in output
    assert '"profile_name"' not in output
