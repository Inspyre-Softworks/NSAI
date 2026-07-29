from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import pytest
from rich.console import Console

import nsai.profile.builder as profile_builder
import nsai.profile.enrichment as profile_enrichment
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


def test_risk_tolerance_step_explains_each_choice_and_safety_scope() -> None:
    step = next(step for step in profile_builder.STEPS if step.key == 'risk_tolerance')

    assert all(choice in step.help_text for choice in step.choices or [])
    assert 'uncertain, disruptive, or hard-to-reverse outcomes' in step.help_text
    assert 'does not bypass enactment safeguards' in step.help_text


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


def test_enter_submits_single_line_answer_and_advances() -> None:
    async def exercise() -> None:
        app = GovernanceProfileApp(enrich_on_save=False)

        async with app.run_test() as pilot:
            answer_input = app.query_one('#answer_input')
            answer_input.value = 'Test Nation'

            await pilot.press('enter')

            assert app.answers['nation_name'] == 'Test Nation'
            assert app.current_step().key == 'profile_name'

    asyncio.run(exercise())


def test_successful_save_disables_save_and_enables_finish() -> None:
    async def exercise() -> None:
        app = GovernanceProfileApp(enrich_on_save=True)
        app.index = len(profile_builder.STEPS) - 1
        save_calls = 0

        def fake_save_profile() -> Path:
            nonlocal save_calls
            save_calls += 1
            return Path('test_nation_governance_profile.json')

        app.save_profile = fake_save_profile

        async with app.run_test() as pilot:
            app.action_save()
            await pilot.pause()

            assert save_calls == 1
            assert app.query_one('#save').disabled is True
            assert app.query_one('#next').disabled is False
            assert app.query_one('#next').label.plain == 'Finish →'

            app.action_save()
            assert save_calls == 1

    asyncio.run(exercise())


def test_failed_save_reenables_save_and_keeps_next_disabled() -> None:
    async def exercise() -> None:
        app = GovernanceProfileApp(enrich_on_save=True)
        app.index = len(profile_builder.STEPS) - 1

        def fail_save_profile() -> Path:
            raise RuntimeError('model unavailable')

        app.save_profile = fail_save_profile

        async with app.run_test() as pilot:
            app.action_save()
            await pilot.pause()

            assert app.query_one('#save').disabled is False
            assert app.query_one('#next').disabled is True
            assert app.saved_path is None

    asyncio.run(exercise())


def test_profile_builder_accepts_textual_dev_flag() -> None:
    explicit = build_arg_parser().parse_args(['interview', '--dev'])
    implicit = build_arg_parser().parse_args(['--dev'])

    assert explicit.dev is True
    assert implicit.dev is True


def test_profile_enrich_accepts_ai_connection_options() -> None:
    args = build_arg_parser().parse_args([
        'enrich',
        'profile.json',
        '--base-url',
        'https://api.openai.com/v1',
        '--model',
        'gpt-test',
        '--lm-api-key',
        'secret',
        '--save-opts',
    ])

    assert args.base_url == 'https://api.openai.com/v1'
    assert args.model == 'gpt-test'
    assert args.lm_api_key == 'secret'
    assert args.save_opts is True


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


def test_run_enrich_passes_ai_connection_options(tmp_path, monkeypatch) -> None:
    recorded: dict[str, object] = {}
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))

    def fake_enrich_profile_file(input_path, **kwargs):  # noqa: ANN001
        recorded['input_path'] = input_path
        recorded.update(kwargs)
        return kwargs['output_path']

    monkeypatch.setattr(profile_builder, 'enrich_profile_file', fake_enrich_profile_file)

    input_path = tmp_path / 'profile.json'
    output_path = tmp_path / 'profile.enriched.json'
    profile_builder.run_enrich(
        argparse.Namespace(
            profile=str(input_path),
            output=str(output_path),
            in_place=False,
            force=True,
            no_backup=False,
            strict=True,
            base_url='https://api.openai.com/v1',
            model='gpt-test',
            lm_api_key='secret',
            secret_backend=None,
        )
    )

    assert recorded['input_path'] == input_path.resolve()
    assert recorded['output_path'] == output_path.resolve()
    assert recorded['force'] is True
    assert recorded['strict'] is True
    assert recorded['base_url'] == 'https://api.openai.com/v1'
    assert recorded['model'] == 'gpt-test'
    assert recorded['api_key'] == 'secret'


def test_save_enrich_options_persists_program_config_and_secret_ref(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    recorded: dict[str, object] = {}

    def fake_set_secret(key, value, *, backend, reason):  # noqa: ANN001
        recorded['key'] = key
        recorded['value'] = value
        recorded['backend'] = backend
        recorded['reason'] = reason

    monkeypatch.setattr(profile_builder, 'set_secret', fake_set_secret)

    config_path = profile_builder.save_enrich_options(
        lm_base_url='https://api.openai.com/v1',
        lm_model='gpt-test',
        lm_api_key='secret',
        secret_backend='keyring',
    )

    saved_text = config_path.read_text(encoding='utf-8')
    saved = json.loads(saved_text)
    assert saved['profile_enrich_lm_base_url'] == 'https://api.openai.com/v1'
    assert saved['profile_enrich_lm_model'] == 'gpt-test'
    assert saved['profile_enrich_lm_api_key_credential_key'] == (
        profile_builder.PROFILE_ENRICH_LM_API_KEY_CREDENTIAL_KEY
    )
    assert saved['profile_enrich_lm_api_key_backend'] == 'keyring'
    assert 'secret' not in saved_text
    assert recorded == {
        'key': profile_builder.PROFILE_ENRICH_LM_API_KEY_CREDENTIAL_KEY,
        'value': 'secret',
        'backend': 'keyring',
        'reason': 'Store profile enrichment LM API key',
    }


def test_resolve_enrich_lm_settings_uses_saved_program_config(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    monkeypatch.delenv('LM_STUDIO_BASE_URL', raising=False)
    monkeypatch.delenv('LM_STUDIO_MODEL', raising=False)
    monkeypatch.delenv('LM_STUDIO_API_KEY', raising=False)
    monkeypatch.setattr(
        profile_builder,
        'set_secret',
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        profile_builder,
        'get_secret',
        lambda key, *, backend, reason: 'stored-secret',
    )
    profile_builder.save_enrich_options(
        lm_base_url='https://api.openai.com/v1',
        lm_model='gpt-test',
        lm_api_key='secret',
        secret_backend='keyring',
    )

    base_url, model, api_key = profile_builder.resolve_enrich_lm_settings(
        argparse.Namespace(base_url=None, model=None, lm_api_key=None)
    )

    assert base_url == 'https://api.openai.com/v1'
    assert model == 'gpt-test'
    assert api_key == 'stored-secret'


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


def test_append_ai_generated_governance_passes_model_settings(monkeypatch) -> None:
    recorded: dict[str, object] = {}

    def fake_generate(profile_data, **kwargs):  # noqa: ANN001
        recorded['profile_data'] = profile_data
        recorded['kwargs'] = kwargs
        return fallback_ai_governance_addendum(profile_data)

    monkeypatch.setattr(profile_builder, 'generate_ai_governance_addendum', fake_generate)

    profile_data = sample_profile()
    enriched = profile_builder.append_ai_generated_governance(
        profile_data,
        base_url='https://api.openai.com/v1',
        model='gpt-test',
        api_key='secret',
    )

    assert recorded['profile_data'] == profile_data
    assert recorded['kwargs'] == {
        'base_url': 'https://api.openai.com/v1',
        'model': 'gpt-test',
        'api_key': 'secret',
    }
    assert enriched['ai_generated']['generation_source'] == 'fallback'


def test_generate_ai_governance_addendum_uses_explicit_model_settings(monkeypatch) -> None:
    recorded: dict[str, object] = {}
    addendum = fallback_ai_governance_addendum(sample_profile())
    addendum.pop('generation_source', None)

    class FakeMessage:
        content = json.dumps(addendum)

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):  # noqa: ANN001
            recorded['chat_kwargs'] = kwargs
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):  # noqa: ANN001
            recorded['client_kwargs'] = kwargs
            self.chat = FakeChat()

    monkeypatch.setattr(profile_enrichment, 'OpenAI', FakeOpenAI)

    generated = profile_enrichment.generate_ai_governance_addendum(
        sample_profile(),
        base_url='https://api.openai.com/v1',
        model='gpt-test',
        api_key='secret',
    )

    assert recorded['client_kwargs'] == {
        'base_url': 'https://api.openai.com/v1',
        'api_key': 'secret',
    }
    assert recorded['chat_kwargs']['model'] == 'gpt-test'
    assert generated['generation_source'] == 'local_ai'
    assert generated['model'] == 'gpt-test'


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
