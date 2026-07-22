from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from io import BytesIO
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
from PIL import Image
from rich.console import Console

import nsai.advisor.live as live
from nsai.advisor.audit import AuditStore, load_audit_log_records, migrate_jsonl_audit_log, write_audit_log
from nsai.advisor.live import (
    NationStatesError,
    NationStatesClient,
    extract_token_usage,
    pending_publication_entries,
    publish_backfill_draft_with_retry,
    publish_publication_drafts,
    resolve_publication_category,
    run_advise,
    save_advise_options,
    xml_to_string,
)
from nsai.advisor.governor import LocalGovernor, build_governor_instruction
from nsai.advisor.recommendations import (
    collect_issue_option_ids,
    extract_live_issues,
    fallback_recommendation,
    is_dismiss_recommendation,
    print_recommendation,
    recommendation_action,
    should_auto_enact,
    should_manual_enact,
    validate_auto_action,
    validate_recommendation_consistency,
    validate_recommendation,
    validate_publication_drafts,
)
from nsai.nations import load_app_config, load_nation_config, profiles_dir


def sample_issues() -> list[dict[str, object]]:
    root = ET.fromstring(
        '''
        <NATION>
          <ISSUES>
            <ISSUE id="123">
              <TITLE>Robot Teachers</TITLE>
              <TEXT>Schools want robot teachers.</TEXT>
              <OPTION id="1">Ban them.</OPTION>
              <OPTION id="2">Regulate them.</OPTION>
              <OPTION id="3">Mandate them everywhere.</OPTION>
            </ISSUE>
          </ISSUES>
        </NATION>
        '''
    )
    return extract_live_issues(root)


def sample_recommendation() -> dict[str, object]:
    return {
        'issue_id': '123',
        'option_id': '2',
        'action': 'enact',
        'issue_summary': 'Schools want robot teachers.',
        'option_summaries': [
            {
                'option_id': '1',
                'summary': 'Ban robot teachers.',
                'expected_effect': 'Technology in education would slow down.',
            },
            {
                'option_id': '2',
                'summary': 'Regulate robot teachers.',
                'expected_effect': 'Schools can use technology with oversight.',
            },
            {
                'option_id': '3',
                'summary': 'Mandate robot teachers.',
                'expected_effect': 'Automation in education would accelerate.',
            },
        ],
        'headline': 'Regulate Robot Teachers',
        'summary': 'A balanced education technology policy.',
        'dispatch_draft': {
            'requested': False,
            'title': '',
            'text': '',
            'category_hint': '',
            'subcategory_hint': '',
        },
        'factbook_draft': {
            'requested': False,
            'pertinent': False,
            'title': '',
            'text': '',
            'category_hint': '',
            'subcategory_hint': '',
            'reason': '',
        },
        'confidence': 0.91,
        'red_line_triggered': False,
        'do_not_enact_if': [],
    }


def test_recommendation_output_uses_structured_sections(capsys) -> None:
    recommendation = {
        **sample_recommendation(),
        'why_this_issue_first': 'Education policy has immediate effects on schools.',
        'reasoning': 'Regulation balances innovation with public oversight.',
        'expected_tradeoffs': ['Schools gain oversight but implementation slows.'],
        'red_line_notes': ['No red lines are triggered.'],
        'do_not_enact_if': ['Do not enact if the education ministry objects.'],
    }

    print_recommendation(recommendation)
    output = capsys.readouterr().out

    assert 'AI Governor Recommendation' in output
    assert '  Decision' in output
    assert '  Issue' in output
    assert '  Reasoning' in output
    assert '  Options' in output
    assert '  Guardrails' in output
    assert '    Option 2' in output
    assert '      Effect:' in output
    assert '      - Schools gain oversight but implementation slows.' in output


def write_auto_profile(tmp_path, *, minimum_confidence: float = 0.8):
    profile_path = tmp_path / 'profile.json'
    profile_path.write_text(
        json.dumps({
            'nation_name': 'Oringrad',
            'profile_name': 'Auto Test Profile',
            'enactment_mode': 'auto_enact_high_confidence',
            'minimum_confidence_to_enact': minimum_confidence,
            'red_lines': [],
        }),
        encoding='utf-8',
    )
    return profile_path


def advise_args(tmp_path, *, profile_path, draft_dispatch=False, draft_factbook=False):
    return SimpleNamespace(
        save_opts=False,
        profile=str(profile_path),
        nation='Oringrad',
        no_nation_config=True,
        strategy=None,
        show_issues=False,
        show_instruction=False,
        no_ai=False,
        audit_log=str(tmp_path / 'audit.sqlite3'),
        base_url='http://localhost:1234/v1',
        model='test-model',
        lm_api_key=None,
        secret_backend=None,
        draft_dispatch=draft_dispatch,
        draft_factbook=draft_factbook,
        refresh_advice=True,
        enact=False,
        auto=True,
        allow_fallback_auto=False,
        override_red_line=False,
        trace_api=False,
        issue_cooldown_seconds=0.0,
        publication_cooldown_seconds=0.0,
    )


def test_extract_and_validate_live_issue_options() -> None:
    issues = sample_issues()

    assert issues == [
        {
            'issue_id': '123',
            'title': 'Robot Teachers',
            'text': 'Schools want robot teachers.',
            'options': [
                {'option_id': '1', 'text': 'Ban them.'},
                {'option_id': '2', 'text': 'Regulate them.'},
                {'option_id': '3', 'text': 'Mandate them everywhere.'},
            ],
        }
    ]

    valid_options = collect_issue_option_ids(issues)
    assert validate_recommendation(sample_recommendation(), valid_options) == ('123', '2')
    assert validate_recommendation(
        {
            **sample_recommendation(),
            'action': 'dismiss',
            'option_id': '-1',
        },
        valid_options,
    ) == ('123', '-1')

    with pytest.raises(NationStatesError, match='invalid option_id'):
        validate_recommendation(
            {'issue_id': '123', 'option_id': '999'},
            valid_options,
        )

    with pytest.raises(NationStatesError, match='Dismissal must use option_id'):
        validate_recommendation(
            {
                **sample_recommendation(),
                'action': 'dismiss',
                'option_id': '2',
            },
            valid_options,
        )


def test_fallback_recommendation_is_review_only() -> None:
    recommendation = fallback_recommendation(
        sample_issues(),
        'keep things stable',
        draft_dispatch=True,
        draft_factbook=True,
    )

    assert recommendation['issue_id'] == '123'
    assert recommendation['option_id'] == '-1'
    assert recommendation['action'] == 'dismiss'
    assert recommendation['headline'] == 'Fallback Recommendation Selected'
    assert is_dismiss_recommendation(recommendation) is True
    assert recommendation['dispatch_draft']['requested'] is True
    assert recommendation['dispatch_draft']['title']
    assert recommendation['factbook_draft']['requested'] is True
    assert recommendation['factbook_draft']['pertinent'] is False
    assert recommendation['factbook_draft']['reason']
    assert recommendation['reasoning_matches_action'] is True

    allowed, reasons = should_manual_enact(
        recommendation=recommendation,
        enact_requested=True,
        override_red_line=False,
    )
    assert allowed is False
    assert reasons == ['Fallback recommendations may not be enacted.']


def test_governor_instruction_allows_dismissal() -> None:
    simple = build_governor_instruction(None, 'keep things stable')
    profiled = build_governor_instruction(
        {
            'nation_name': 'Oringrad',
            'profile_name': 'Prosperous Technocracy',
            'red_lines': ['Do not collapse the economy.'],
        },
        'keep things stable',
    )

    assert 'dismiss the issue' in simple
    assert 'dismiss the issue' in profiled


def test_ai_prompt_context_is_compact_and_clear() -> None:
    profile = {
        'nation_name': 'Oringrad',
        'profile_name': 'Verbose Profile',
        'national_vision': 'x' * 5000,
        'minimum_confidence_to_enact': 0.8,
        'red_lines': [f'red line {index}' for index in range(20)],
        'ai_generated': {
            'vision_description': 'y' * 5000,
            'short_constitution': {'article': 'z' * 5000},
        },
    }

    summary = live.compact_profile_for_ai(profile)
    instruction = build_governor_instruction(profile, 'keep things stable')
    nation_context = live.compact_nation_context(
        '<NATION><FULLNAME>Oringrad</FULLNAME>'
        f'<LEGISLATION>{"law " * 2000}</LEGISLATION>'
        '<UNUSED>ignore me</UNUSED></NATION>'
    )
    issues = live.compact_live_issues_for_ai([
        {
            'issue_id': '123',
            'title': 'Robot Teachers',
            'text': 'a' * 5000,
            'options': [{'option_id': '2', 'text': 'b' * 5000}],
        }
    ])

    assert summary is not None
    assert summary['minimum_confidence_to_enact'] == 0.8
    assert len(summary['national_vision']) <= live.PROMPT_TEXT_LIMIT
    assert len(summary['red_lines']) == live.PROMPT_LIST_LIMIT
    assert len(summary['vision_description']) <= live.PROMPT_LONG_TEXT_LIMIT
    assert 'Never describe dismissing an issue "with Option N"' in instruction
    assert 'charter_alignment_score is required' in instruction
    assert nation_context['fullname'] == 'Oringrad'
    assert 'unused' not in nation_context
    assert len(nation_context['legislation']) <= live.PROMPT_LONG_TEXT_LIMIT
    assert issues[0]['issue_id'] == '123'
    assert issues[0]['options'][0]['option_id'] == '2'
    assert issues[0]['options'][0]['option_label'] == 'Option 1'
    assert len(issues[0]['text']) <= live.PROMPT_LONG_TEXT_LIMIT


def test_render_ascii_flag_from_png(monkeypatch) -> None:
    image = Image.new('RGB', (2, 2))
    image.putdata([
        (0, 0, 0),
        (255, 255, 255),
        (255, 255, 255),
        (0, 0, 0),
    ])
    buffer = BytesIO()
    image.save(buffer, format='PNG')

    class FakeResponse:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(
        live.requests,
        'get',
        lambda url, timeout=15: FakeResponse(buffer.getvalue()),
    )

    ascii_flag = live.render_ascii_flag('https://example.com/flag.png', width=8)
    lines = ascii_flag.splitlines()

    assert len(lines) == 4
    assert all(len(line) == 8 for line in lines)


def test_render_banner_includes_nation_name() -> None:
    banner = live.render_banner('Oringrad')

    assert 'Oringrad' in banner
    assert banner.count('=') >= 40


def test_check_issues_lists_live_issues_without_advisor_side_effects(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))

    class FakeNationStatesClient:
        def __init__(self) -> None:
            self.issue_calls = []

        def issues(self, nation):
            self.issue_calls.append(nation)
            return ET.fromstring(
                '''
                <NATION>
                  <ISSUES>
                    <ISSUE id="123">
                      <TITLE>Robot Teachers</TITLE>
                      <TEXT>Schools want robot teachers.</TEXT>
                      <OPTION id="1">Ban them.</OPTION>
                      <OPTION id="2">Regulate them.</OPTION>
                    </ISSUE>
                  </ISSUES>
                </NATION>
                '''
            )

    fake_ns = FakeNationStatesClient()
    monkeypatch.setattr(
        live.NationStatesClient,
        'from_env',
        classmethod(lambda cls, nation_config=None: fake_ns),
    )

    args = SimpleNamespace(
        profile=None,
        nation='Oringrad',
        no_nation_config=True,
        trace_api=False,
    )

    live.run_check_issues(args)

    output = capsys.readouterr().out
    assert fake_ns.issue_calls == ['Oringrad']
    assert 'Live NationStates Issues' in output
    assert 'Issue 123: Robot Teachers' in output
    assert 'Found 1 live issue(s) for Oringrad.' in output
    assert not (tmp_path / 'audit.jsonl').exists()


def test_advise_skips_ai_issue_selection_when_only_one_issue(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    shards_seen: list[list[str]] = []

    class FakeNationStatesClient:
        user_agent = 'NSAI-Test/0.1 contact:test@example.com nation:Oringrad'
        api_version = None

        def public_nation(self, nation, shards):
            shards_seen.append(list(shards))
            return ET.fromstring('<NATION id="oringrad"><FULLNAME>Oringrad</FULLNAME></NATION>')

        def issues(self, nation):
            return ET.fromstring(
                '''
                <NATION>
                  <ISSUES>
                    <ISSUE id="123">
                      <TITLE>Robot Teachers</TITLE>
                      <TEXT>Schools want robot teachers.</TEXT>
                      <OPTION id="1">Ban them.</OPTION>
                      <OPTION id="2">Regulate them.</OPTION>
                    </ISSUE>
                  </ISSUES>
                </NATION>
                '''
            )

    class FakeGovernor:
        advise_called = False

        def __init__(self, *, base_url=None, model=None, api_key=None):
            self.base_url = base_url or 'http://localhost:1234/v1'
            self.model = model or 'test-model'

        def select_issue(self, **kwargs):
            raise AssertionError('select_issue should not be called for one live issue')

        def advise(self, *, live_issues, **kwargs):
            assert len(live_issues) == 1
            FakeGovernor.advise_called = True
            return {
                **sample_recommendation(),
                'issue_id': '123',
                'option_id': '2',
                'model': self.model,
            }

    monkeypatch.setattr(
        live.NationStatesClient,
        'from_env',
        classmethod(lambda cls, nation_config=None: FakeNationStatesClient()),
    )
    monkeypatch.setattr(live, 'LocalGovernor', FakeGovernor)

    args = SimpleNamespace(
        save_opts=False,
        profile=None,
        nation='Oringrad',
        no_nation_config=True,
        strategy=None,
        show_issues=False,
        show_instruction=False,
        no_ai=False,
        audit_log=str(tmp_path / 'audit.sqlite3'),
        base_url='http://localhost:1234/v1',
        model='test-model',
        lm_api_key=None,
        secret_backend=None,
        draft_dispatch=False,
        draft_factbook=False,
        refresh_advice=True,
        enact=False,
        auto=False,
        override_red_line=False,
    )

    run_advise(args)

    output = capsys.readouterr().out
    assert 'AI step skipped: only one live issue is present.' in output
    assert FakeGovernor.advise_called is True
    assert 'flag' in shards_seen[0]


def test_advise_skips_flag_shard_when_flag_display_none(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    shards_seen: list[list[str]] = []

    class FakeNationStatesClient:
        user_agent = 'NSAI-Test/0.1 contact:test@example.com nation:Oringrad'
        api_version = None

        def public_nation(self, nation, shards):
            shards_seen.append(list(shards))
            return ET.fromstring('<NATION id="oringrad"><FULLNAME>Oringrad</FULLNAME></NATION>')

        def issues(self, nation):
            return ET.fromstring(
                '''
                <NATION>
                  <ISSUES>
                    <ISSUE id="123">
                      <TITLE>Robot Teachers</TITLE>
                      <TEXT>Schools want robot teachers.</TEXT>
                      <OPTION id="1">Ban them.</OPTION>
                      <OPTION id="2">Regulate them.</OPTION>
                    </ISSUE>
                  </ISSUES>
                </NATION>
                '''
            )

    class FakeGovernor:
        def __init__(self, *, base_url=None, model=None, api_key=None):
            self.base_url = base_url or 'http://localhost:1234/v1'
            self.model = model or 'test-model'

        def select_issue(self, **kwargs):
            raise AssertionError('select_issue should not be called for one live issue')

        def advise(self, *, live_issues, **kwargs):
            return {
                **sample_recommendation(),
                'issue_id': '123',
                'option_id': '2',
                'model': self.model,
            }

    monkeypatch.setattr(
        live.NationStatesClient,
        'from_env',
        classmethod(lambda cls, nation_config=None: FakeNationStatesClient()),
    )
    monkeypatch.setattr(live, 'LocalGovernor', FakeGovernor)

    args = SimpleNamespace(
        save_opts=False,
        profile=None,
        nation='Oringrad',
        no_nation_config=True,
        strategy=None,
        show_issues=False,
        show_instruction=False,
        no_ai=False,
        audit_log=str(tmp_path / 'audit.sqlite3'),
        base_url='http://localhost:1234/v1',
        model='test-model',
        lm_api_key=None,
        secret_backend=None,
        draft_dispatch=False,
        draft_factbook=False,
        refresh_advice=True,
        enact=False,
        auto=False,
        override_red_line=False,
        flag_display='none',
    )

    run_advise(args)

    assert shards_seen, 'public_nation should have been called'
    assert 'flag' not in shards_seen[0], (
        "flag shard should not be requested when flag_display='none'"
    )


def test_model_reload_issue_selection_retry_blocks_auto_action(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile_path = write_auto_profile(tmp_path)

    class FakeReloadingCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise RuntimeError("Error code: 400 - {'error': 'Model reloaded.'}")

    class FakeOpenAIClient:
        def __init__(self) -> None:
            self.completions = FakeReloadingCompletions()
            self.chat = SimpleNamespace(completions=self.completions)

    class FakeGovernor(live.LocalGovernor):
        completions: FakeReloadingCompletions | None = None

        def __init__(self, *, base_url=None, model=None, api_key=None):
            self.base_url = base_url or 'http://localhost:1234/v1'
            self.model = model or 'test-model'
            self.client = FakeOpenAIClient()
            FakeGovernor.completions = self.client.completions

        def advise(self, *, live_issues, **kwargs):
            assert live_issues[0]['issue_id'] == '123'
            return {
                **sample_recommendation(),
                'issue_id': '123',
                'option_id': '2',
                'confidence': 0.95,
                'charter_alignment_score': 95,
                'model': self.model,
                'dispatch_draft': {
                    'requested': True,
                    'title': 'Robot Teacher Reform Approved',
                    'text': 'New oversight has been adopted.',
                    'category_hint': 'Bulletin',
                    'subcategory_hint': 'News',
                },
                'factbook_draft': {
                    'requested': True,
                    'pertinent': True,
                    'title': 'Education Automation Framework',
                    'text': 'Oringrad regulates classroom automation.',
                    'category_hint': 'Factbook',
                    'subcategory_hint': 'Legislation',
                    'reason': 'The change creates durable education law.',
                },
            }

    class FakeNationStatesClient:
        user_agent = 'NSAI-Test/0.1 contact:test@example.com nation:Oringrad'
        api_version = None

        def __init__(self) -> None:
            self.answer_calls = []
            self.publish_calls = []

        def public_nation(self, nation, shards):
            return ET.fromstring('<NATION id="oringrad"><FULLNAME>Oringrad</FULLNAME></NATION>')

        def issues(self, nation):
            return ET.fromstring(
                '''
                <NATION>
                  <ISSUES>
                    <ISSUE id="123">
                      <TITLE>Robot Teachers</TITLE>
                      <TEXT>Schools want robot teachers.</TEXT>
                      <OPTION id="1">Ban them.</OPTION>
                      <OPTION id="2">Regulate them.</OPTION>
                    </ISSUE>
                    <ISSUE id="456">
                      <TITLE>Drone Parks</TITLE>
                      <TEXT>Parks want delivery drones.</TEXT>
                      <OPTION id="7">Ban drones.</OPTION>
                      <OPTION id="8">Permit quiet drones.</OPTION>
                    </ISSUE>
                  </ISSUES>
                </NATION>
                '''
            )

        def answer_issue(self, nation, issue_id, option_id):
            self.answer_calls.append((nation, issue_id, option_id))
            return ET.fromstring('<NATION><ISSUE><OK>1</OK></ISSUE></NATION>')

        def create_dispatch(self, nation, *, title, text, category, subcategory):
            self.publish_calls.append((nation, title, text, category, subcategory))
            return ET.fromstring('<NATION><SUCCESS>Created.</SUCCESS></NATION>')

    fake_ns = FakeNationStatesClient()
    monkeypatch.setattr(
        live.NationStatesClient,
        'from_env',
        classmethod(lambda cls, nation_config=None: fake_ns),
    )
    monkeypatch.setattr(live, 'LocalGovernor', FakeGovernor)

    run_advise(advise_args(
        tmp_path,
        profile_path=profile_path,
        draft_dispatch=True,
        draft_factbook=True,
    ))

    output = capsys.readouterr().out
    assert FakeGovernor.completions is not None
    assert FakeGovernor.completions.calls == 2
    assert 'Local model reloaded during issue selection; retrying AI step (2/2)' in output
    assert 'AUTO ACTION BLOCKED' in output
    assert 'fallback_issue_selection_used: true' in output
    assert fake_ns.answer_calls == []
    assert fake_ns.publish_calls == []

    _, audit_entry = load_audit_log_records(tmp_path / 'audit.sqlite3')[0]
    assert audit_entry['action'] == 'requires_review'
    assert audit_entry['action_applied'] is False
    assert audit_entry['blocked'] is True
    assert audit_entry['fallback_issue_selection_used'] is True
    assert any('AI step failed during issue_selection' in reason for reason in audit_entry['block_reasons'])


def test_recommendation_can_disable_text_retry_for_prefetch() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise RuntimeError('local model request failed')

    class FakeOpenAIClient:
        def __init__(self) -> None:
            self.completions = FakeCompletions()
            self.chat = SimpleNamespace(completions=self.completions)

    governor = LocalGovernor.__new__(LocalGovernor)
    governor.base_url = 'http://localhost:1234/v1'
    governor.model = 'test-model'
    governor.api_trace = False
    governor.client = FakeOpenAIClient()

    with pytest.raises(RuntimeError, match='local model request failed'):
        governor.advise(
            nation_snapshot_xml='<NATION />',
            live_issues=sample_issues(),
            strategy='keep things stable',
            profile=None,
            draft_dispatch=False,
            draft_factbook=False,
            pulse_label=None,
            retry_text_mode=False,
            fallback_on_failure=False,
        )

    assert governor.client.completions.calls == 1


def test_contradictory_dismiss_recommendation_fails_validation() -> None:
    recommendation = {
        **sample_recommendation(),
        'action': 'dismiss',
        'option_id': '-1',
        'confidence': 0.9,
        'charter_alignment_score': 80,
        'reasoning': 'Dismissal would ignore a real crisis. Option 4 best aligns with the nation.',
    }

    result = validate_recommendation_consistency(
        recommendation,
        live_issues=sample_issues(),
        selected_issue=sample_issues()[0],
    )

    assert result.passed is False
    assert any('reasoning recommends' in reason for reason in result.reasons)


def test_dismiss_with_specific_option_fails_validation() -> None:
    recommendation = {
        **sample_recommendation(),
        'action': 'dismiss',
        'option_id': '-1',
        'confidence': 0.9,
        'charter_alignment_score': 80,
        'reasoning': 'Dismissing it with Option 2 resolves the immediate threat.',
    }

    result = validate_recommendation_consistency(
        recommendation,
        live_issues=sample_issues(),
        selected_issue=sample_issues()[0],
    )

    assert result.passed is False
    assert any('reasoning recommends' in reason for reason in result.reasons)


def test_suspicious_confidence_alignment_fails_validation() -> None:
    recommendation = {
        **sample_recommendation(),
        'action': 'dismiss',
        'option_id': '-1',
        'confidence': 0.95,
        'alignment_score': 0,
        'charter_alignment_score': 0,
        'reasoning': 'All options should be deferred until a safer answer exists.',
    }

    result = validate_recommendation_consistency(
        recommendation,
        live_issues=sample_issues(),
        selected_issue=sample_issues()[0],
    )

    assert result.passed is False
    assert any('suspiciously high' in reason for reason in result.reasons)


def test_neutral_alignment_at_auto_threshold_passes_validation() -> None:
    recommendation = {
        **sample_recommendation(),
        'action': 'enact',
        'option_id': '2',
        'confidence': 0.90,
        'alignment_score': 0,
        'charter_alignment_score': 0,
    }

    result = validate_recommendation_consistency(
        recommendation,
        live_issues=sample_issues(),
        selected_issue=sample_issues()[0],
    )

    assert result.passed is True
    assert not any('suspiciously high' in reason for reason in result.reasons)


def test_auto_validation_passes_for_valid_enact() -> None:
    result = validate_auto_action(
        live_issues=sample_issues(),
        selected_issue=sample_issues()[0],
        recommendation={
            **sample_recommendation(),
            'confidence': 0.92,
            'charter_alignment_score': 92,
            'red_line_triggered': False,
        },
        ai_step_statuses=[
            {'step': 'issue_selection', 'status': 'ok'},
            {'step': 'recommendation_generation', 'status': 'ok'},
        ],
        draft_dispatch=False,
        draft_factbook=False,
        minimum_confidence=0.8,
    )

    assert result.passed is True
    assert result.reasons == []


def test_auto_validation_blocks_zero_alignment_enact() -> None:
    result = validate_auto_action(
        live_issues=sample_issues(),
        selected_issue=sample_issues()[0],
        recommendation={
            **sample_recommendation(),
            'action': 'enact',
            'option_id': '2',
            'confidence': 0.90,
            'alignment_score': 0,
            'charter_alignment_score': 0,
            'red_line_triggered': False,
        },
        ai_step_statuses=[
            {'step': 'issue_selection', 'status': 'ok'},
            {'step': 'recommendation_generation', 'status': 'ok'},
        ],
        draft_dispatch=False,
        draft_factbook=False,
        minimum_confidence=0.8,
    )

    assert result.passed is False
    assert any('positive alignment score' in reason for reason in result.reasons)


def test_auto_validation_blocks_self_reported_reasoning_mismatch() -> None:
    result = validate_auto_action(
        live_issues=sample_issues(),
        selected_issue=sample_issues()[0],
        recommendation={
            **sample_recommendation(),
            'confidence': 0.92,
            'charter_alignment_score': 92,
            'red_line_triggered': False,
            'reasoning_matches_action': False,
        },
        ai_step_statuses=[
            {'step': 'issue_selection', 'status': 'ok'},
            {'step': 'recommendation_generation', 'status': 'ok'},
        ],
        draft_dispatch=False,
        draft_factbook=False,
        minimum_confidence=0.8,
    )

    assert result.passed is False
    assert any('reasoning_matches_action=false' in reason for reason in result.reasons)


def test_auto_validation_ignores_missing_reasoning_matches_action() -> None:
    """Cached/older recommendations predate this field; absence is not a block."""
    result = validate_auto_action(
        live_issues=sample_issues(),
        selected_issue=sample_issues()[0],
        recommendation={
            **sample_recommendation(),
            'confidence': 0.92,
            'charter_alignment_score': 92,
            'red_line_triggered': False,
        },
        ai_step_statuses=[
            {'step': 'issue_selection', 'status': 'ok'},
            {'step': 'recommendation_generation', 'status': 'ok'},
        ],
        draft_dispatch=False,
        draft_factbook=False,
        minimum_confidence=0.8,
    )

    assert result.passed is True


def test_valid_auto_enact_calls_action_endpoint(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile_path = write_auto_profile(tmp_path)

    class FakeNationStatesClient:
        user_agent = 'NSAI-Test/0.1 contact:test@example.com nation:Oringrad'
        api_version = None

        def __init__(self) -> None:
            self.answer_calls = []

        def public_nation(self, nation, shards):
            return ET.fromstring('<NATION id="oringrad"><FULLNAME>Oringrad</FULLNAME></NATION>')

        def issues(self, nation):
            return ET.fromstring(
                '''
                <NATION>
                  <ISSUES>
                    <ISSUE id="123">
                      <TITLE>Robot Teachers</TITLE>
                      <TEXT>Schools want robot teachers.</TEXT>
                      <OPTION id="1">Ban them.</OPTION>
                      <OPTION id="2">Regulate them.</OPTION>
                    </ISSUE>
                  </ISSUES>
                </NATION>
                '''
            )

        def answer_issue(self, nation, issue_id, option_id):
            self.answer_calls.append((nation, issue_id, option_id))
            return ET.fromstring('<NATION><ISSUE><OK>1</OK></ISSUE></NATION>')

    class FakeGovernor:
        def __init__(self, *, base_url=None, model=None, api_key=None):
            self.base_url = base_url or 'http://localhost:1234/v1'
            self.model = model or 'test-model'

        def advise(self, *, live_issues, **kwargs):
            return {
                **sample_recommendation(),
                'issue_id': '123',
                'option_id': '2',
                'confidence': 0.95,
                'charter_alignment_score': 95,
                'red_line_triggered': False,
                'model': self.model,
            }

    fake_ns = FakeNationStatesClient()
    monkeypatch.setattr(
        live.NationStatesClient,
        'from_env',
        classmethod(lambda cls, nation_config=None: fake_ns),
    )
    monkeypatch.setattr(live, 'LocalGovernor', FakeGovernor)

    run_advise(advise_args(tmp_path, profile_path=profile_path))

    output = capsys.readouterr().out
    assert 'Decision Summary' in output
    assert 'NationStates accepted the issue action.' in output
    assert fake_ns.answer_calls == [('Oringrad', '123', '2')]
    _, audit_entry = load_audit_log_records(tmp_path / 'audit.sqlite3')[0]
    assert audit_entry['action'] == 'auto_enact'
    assert audit_entry['action_applied'] is True
    assert audit_entry['blocked'] is False


def test_decision_summary_can_be_disabled(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile_path = write_auto_profile(tmp_path)

    class FakeNationStatesClient:
        user_agent = 'NSAI-Test/0.1 contact:test@example.com nation:Oringrad'
        api_version = None

        def public_nation(self, nation, shards):
            return ET.fromstring('<NATION id="oringrad"><FULLNAME>Oringrad</FULLNAME></NATION>')

        def issues(self, nation):
            return ET.fromstring(
                '''
                <NATION>
                  <ISSUES>
                    <ISSUE id="123">
                      <TITLE>Robot Teachers</TITLE>
                      <TEXT>Schools want robot teachers.</TEXT>
                      <OPTION id="1">Ban them.</OPTION>
                      <OPTION id="2">Regulate them.</OPTION>
                    </ISSUE>
                  </ISSUES>
                </NATION>
                '''
            )

    class FakeGovernor:
        def __init__(self, *, base_url=None, model=None, api_key=None):
            self.base_url = base_url or 'http://localhost:1234/v1'
            self.model = model or 'test-model'

        def advise(self, *, live_issues, **kwargs):
            return {
                **sample_recommendation(),
                'issue_id': '123',
                'option_id': '2',
                'model': self.model,
            }

    monkeypatch.setattr(
        live.NationStatesClient,
        'from_env',
        classmethod(lambda cls, nation_config=None: FakeNationStatesClient()),
    )
    monkeypatch.setattr(live, 'LocalGovernor', FakeGovernor)

    args = advise_args(tmp_path, profile_path=profile_path)
    args.auto = False
    args.decision_summary = False

    run_advise(args)

    output = capsys.readouterr().out
    assert 'Advisor mode only. No issue action was submitted.' in output
    assert 'Decision Summary' not in output


def test_all_issues_uses_cached_order_plan(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile_path = write_auto_profile(tmp_path)

    class FakeNationStatesClient:
        user_agent = 'NSAI-Test/0.1 contact:test@example.com nation:Oringrad'
        api_version = None

        def public_nation(self, nation, shards):
            return ET.fromstring('<NATION id="oringrad"><FULLNAME>Oringrad</FULLNAME></NATION>')

        def issues(self, nation):
            return ET.fromstring(
                '''
                <NATION>
                  <ISSUES>
                    <ISSUE id="123">
                      <TITLE>Robot Teachers</TITLE>
                      <TEXT>Schools want robot teachers.</TEXT>
                      <OPTION id="1">Ban them.</OPTION>
                      <OPTION id="2">Regulate them.</OPTION>
                    </ISSUE>
                    <ISSUE id="456">
                      <TITLE>Orbital Farms</TITLE>
                      <TEXT>Farmers want orbital hydroponics grants.</TEXT>
                      <OPTION id="1">Fund them.</OPTION>
                    </ISSUE>
                  </ISSUES>
                </NATION>
                '''
            )

    class FakeGovernor:
        plan_calls = 0
        advise_issue_ids: list[str] = []

        def __init__(self, *, base_url=None, model=None, api_key=None):
            self.base_url = base_url or 'http://localhost:1234/v1'
            self.model = model or 'test-model'

        def plan_issue_order(self, **kwargs):
            FakeGovernor.plan_calls += 1
            return {
                'ordered_issue_ids': ['456', '123'],
                'reasons': {
                    '456': 'Food security is most urgent.',
                    '123': 'Education policy can follow.',
                },
                'model': self.model,
                'token_usage': {'total_tokens': 12},
            }

        def advise(self, *, live_issues, **kwargs):
            issue = live_issues[0]
            issue_id = str(issue['issue_id'])
            FakeGovernor.advise_issue_ids.append(issue_id)
            return {
                **sample_recommendation(),
                'issue_id': issue_id,
                'option_id': str(issue['options'][0]['option_id']),
                'headline': f'Handle issue {issue_id}',
                'model': self.model,
            }

    monkeypatch.setattr(
        live.NationStatesClient,
        'from_env',
        classmethod(lambda cls, nation_config=None: FakeNationStatesClient()),
    )
    monkeypatch.setattr(live, 'LocalGovernor', FakeGovernor)

    args = advise_args(tmp_path, profile_path=profile_path)
    args.all_issues = True
    args.auto = False
    args.refresh_advice = False
    args.flag_display = 'none'

    run_advise(args)
    first_output = capsys.readouterr().out

    assert FakeGovernor.plan_calls == 1
    assert FakeGovernor.advise_issue_ids == ['456', '123']
    assert 'All-Issues Plan' in first_output
    assert '1. Orbital Farms (456)' in first_output
    assert first_output.count('Decision Summary') == 2

    run_advise(args)
    second_output = capsys.readouterr().out

    assert FakeGovernor.plan_calls == 1
    assert FakeGovernor.advise_issue_ids == ['456', '123']
    assert 'AI step skipped: reused cached all-issues order plan.' in second_output


def test_all_issues_reuses_cached_covering_order_plan(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile_path = write_auto_profile(tmp_path)
    previous_issues = [
        {
            'issue_id': '789',
            'title': 'Already Resolved',
            'text': '',
            'options': [{'option_id': '1', 'text': 'Act.'}],
        },
        {
            'issue_id': '456',
            'title': 'Orbital Farms',
            'text': 'Farmers want orbital hydroponics grants.',
            'options': [{'option_id': '1', 'text': 'Fund them.'}],
        },
        {
            'issue_id': '123',
            'title': 'Robot Teachers',
            'text': 'Schools want robot teachers.',
            'options': [{'option_id': '1', 'text': 'Regulate them.'}],
        },
    ]
    live.AdviceCache().save_issue_plan(
        nation='Oringrad',
        live_issues=previous_issues,
        ordered_issue_ids=['789', '456', '123'],
        reasons={
            '789': 'This was first before it was resolved.',
            '456': 'Food security remains next.',
            '123': 'Education can follow.',
        },
        source='ai',
        token_usage={'total_tokens': 12},
    )

    class FakeNationStatesClient:
        user_agent = 'NSAI-Test/0.1 contact:test@example.com nation:Oringrad'
        api_version = None

        def public_nation(self, nation, shards):
            return ET.fromstring('<NATION id="oringrad"><FULLNAME>Oringrad</FULLNAME></NATION>')

        def issues(self, nation):
            return ET.fromstring(
                '''
                <NATION>
                  <ISSUES>
                    <ISSUE id="123">
                      <TITLE>Robot Teachers</TITLE>
                      <TEXT>Schools want robot teachers.</TEXT>
                      <OPTION id="1">Regulate them.</OPTION>
                    </ISSUE>
                    <ISSUE id="456">
                      <TITLE>Orbital Farms</TITLE>
                      <TEXT>Farmers want orbital hydroponics grants.</TEXT>
                      <OPTION id="1">Fund them.</OPTION>
                    </ISSUE>
                  </ISSUES>
                </NATION>
                '''
            )

    class FakeGovernor:
        advise_issue_ids: list[str] = []

        def __init__(self, *, base_url=None, model=None, api_key=None):
            self.base_url = base_url or 'http://localhost:1234/v1'
            self.model = model or 'test-model'

        def plan_issue_order(self, **kwargs):
            raise AssertionError('remaining issues should reuse the cached plan')

        def advise(self, *, live_issues, **kwargs):
            issue = live_issues[0]
            issue_id = str(issue['issue_id'])
            FakeGovernor.advise_issue_ids.append(issue_id)
            return {
                **sample_recommendation(),
                'issue_id': issue_id,
                'option_id': str(issue['options'][0]['option_id']),
                'headline': f'Handle issue {issue_id}',
                'model': self.model,
            }

    monkeypatch.setattr(
        live.NationStatesClient,
        'from_env',
        classmethod(lambda cls, nation_config=None: FakeNationStatesClient()),
    )
    monkeypatch.setattr(live, 'LocalGovernor', FakeGovernor)

    args = advise_args(tmp_path, profile_path=profile_path)
    args.all_issues = True
    args.auto = False
    args.refresh_advice = False
    args.flag_display = 'none'

    run_advise(args)
    output = capsys.readouterr().out

    assert FakeGovernor.advise_issue_ids == ['456', '123']
    assert 'AI step skipped: reused cached all-issues order plan for remaining live issues.' in output
    assert '1. Orbital Farms (456)' in output


def test_all_issues_reuses_cached_fallback_order_plan_when_ai_available(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile_path = write_auto_profile(tmp_path)
    live.AdviceCache().save_issue_plan(
        nation='Oringrad',
        live_issues=[
            {
                'issue_id': '456',
                'title': 'Orbital Farms',
                'text': 'Farmers want orbital hydroponics grants.',
                'options': [{'option_id': '1', 'text': 'Fund them.'}],
            },
            {
                'issue_id': '123',
                'title': 'Robot Teachers',
                'text': 'Schools want robot teachers.',
                'options': [{'option_id': '1', 'text': 'Regulate them.'}],
            },
        ],
        ordered_issue_ids=['456', '123'],
        reasons={
            '456': 'Deterministic fallback kept the live issue order.',
            '123': 'Deterministic fallback kept the live issue order.',
        },
        source='fallback',
        token_usage={},
    )

    class FakeNationStatesClient:
        user_agent = 'NSAI-Test/0.1 contact:test@example.com nation:Oringrad'
        api_version = None

        def public_nation(self, nation, shards):
            return ET.fromstring('<NATION id="oringrad"><FULLNAME>Oringrad</FULLNAME></NATION>')

        def issues(self, nation):
            return ET.fromstring(
                '''
                <NATION>
                  <ISSUES>
                    <ISSUE id="456">
                      <TITLE>Orbital Farms</TITLE>
                      <TEXT>Farmers want orbital hydroponics grants.</TEXT>
                      <OPTION id="1">Fund them.</OPTION>
                    </ISSUE>
                    <ISSUE id="123">
                      <TITLE>Robot Teachers</TITLE>
                      <TEXT>Schools want robot teachers.</TEXT>
                      <OPTION id="1">Regulate them.</OPTION>
                    </ISSUE>
                  </ISSUES>
                </NATION>
                '''
            )

    class FakeGovernor:
        advise_issue_ids: list[str] = []

        def __init__(self, *, base_url=None, model=None, api_key=None):
            self.base_url = base_url or 'http://localhost:1234/v1'
            self.model = model or 'test-model'

        def plan_issue_order(self, **kwargs):
            raise AssertionError('cached fallback plan should be reused')

        def advise(self, *, live_issues, **kwargs):
            issue = live_issues[0]
            issue_id = str(issue['issue_id'])
            FakeGovernor.advise_issue_ids.append(issue_id)
            return {
                **sample_recommendation(),
                'issue_id': issue_id,
                'option_id': str(issue['options'][0]['option_id']),
                'headline': f'Handle issue {issue_id}',
                'model': self.model,
            }

    monkeypatch.setattr(
        live.NationStatesClient,
        'from_env',
        classmethod(lambda cls, nation_config=None: FakeNationStatesClient()),
    )
    monkeypatch.setattr(live, 'LocalGovernor', FakeGovernor)

    args = advise_args(tmp_path, profile_path=profile_path)
    args.all_issues = True
    args.auto = False
    args.refresh_advice = False
    args.flag_display = 'none'

    run_advise(args)
    output = capsys.readouterr().out

    assert FakeGovernor.advise_issue_ids == ['456', '123']
    assert 'AI step skipped: reused cached all-issues order plan.' in output
    assert 'Cached deterministic fallback all-issues order plan ignored' not in output


def test_ai_ordering_ignores_cached_arrival_order_plan(
    tmp_path,
    capsys,
) -> None:
    cache = live.AdviceCache(tmp_path / 'advice.sqlite3')
    issues = [
        {
            'issue_id': '456',
            'title': 'Orbital Farms',
            'text': 'Farmers want orbital hydroponics grants.',
            'options': [{'option_id': '1', 'text': 'Fund them.'}],
        },
        {
            'issue_id': '123',
            'title': 'Robot Teachers',
            'text': 'Schools want robot teachers.',
            'options': [{'option_id': '1', 'text': 'Regulate them.'}],
        },
    ]
    cache.save_issue_plan(
        nation='Oringrad',
        live_issues=issues,
        ordered_issue_ids=['456', '123'],
        reasons={
            '456': 'AI issue ordering was disabled.',
            '123': 'AI issue ordering was disabled.',
        },
        source='arrival',
        token_usage={},
    )

    class FakeGovernor:
        plan_calls = 0

        def plan_issue_order(self, **kwargs):
            FakeGovernor.plan_calls += 1
            return {
                'ordered_issue_ids': ['123', '456'],
                'reasons': {
                    '123': 'Education should be reviewed first.',
                    '456': 'Food security can follow.',
                },
                'model': 'test-model',
                'token_usage': {'total_tokens': 10},
            }

    args = SimpleNamespace(issue_order='ai', refresh_advice=False, no_ai=False)

    plan = live.get_or_create_issue_order_plan(
        args=args,
        cache=cache,
        nation='Oringrad',
        live_issues=issues,
        strategy='keep things stable',
        profile=None,
        nation_snapshot_xml='<NATION />',
        get_governor=FakeGovernor,
    )
    output = capsys.readouterr().out

    assert FakeGovernor.plan_calls == 1
    assert plan['ordered_issue_ids'] == ['123', '456']
    assert plan['source'] == 'ai'
    assert 'Cached non-AI all-issues order plan ignored' in output


def test_ai_ordering_ignores_cached_covering_id_order_plan(
    tmp_path,
    capsys,
) -> None:
    cache = live.AdviceCache(tmp_path / 'advice.sqlite3')
    previous_issues = [
        {
            'issue_id': '789',
            'title': 'Already Resolved',
            'text': '',
            'options': [{'option_id': '1', 'text': 'Act.'}],
        },
        {
            'issue_id': '456',
            'title': 'Orbital Farms',
            'text': 'Farmers want orbital hydroponics grants.',
            'options': [{'option_id': '1', 'text': 'Fund them.'}],
        },
        {
            'issue_id': '123',
            'title': 'Robot Teachers',
            'text': 'Schools want robot teachers.',
            'options': [{'option_id': '1', 'text': 'Regulate them.'}],
        },
    ]
    current_issues = previous_issues[1:]
    cache.save_issue_plan(
        nation='Oringrad',
        live_issues=previous_issues,
        ordered_issue_ids=['123', '456', '789'],
        reasons={
            '123': 'Issue ID ordering was requested.',
            '456': 'Issue ID ordering was requested.',
            '789': 'Issue ID ordering was requested.',
        },
        source='id',
        token_usage={},
    )

    class FakeGovernor:
        plan_calls = 0

        def plan_issue_order(self, **kwargs):
            FakeGovernor.plan_calls += 1
            return {
                'ordered_issue_ids': ['456', '123'],
                'reasons': {
                    '456': 'Food security should be reviewed first.',
                    '123': 'Education can follow.',
                },
                'model': 'test-model',
                'token_usage': {'total_tokens': 11},
            }

    args = SimpleNamespace(issue_order='ai', refresh_advice=False, no_ai=False)

    plan = live.get_or_create_issue_order_plan(
        args=args,
        cache=cache,
        nation='Oringrad',
        live_issues=current_issues,
        strategy='keep things stable',
        profile=None,
        nation_snapshot_xml='<NATION />',
        get_governor=FakeGovernor,
    )
    output = capsys.readouterr().out

    assert FakeGovernor.plan_calls == 1
    assert plan['ordered_issue_ids'] == ['456', '123']
    assert plan['source'] == 'ai'
    assert 'Cached non-AI all-issues order plan ignored' in output


def test_all_issues_can_skip_ai_ordering_with_arrival_order(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile_path = write_auto_profile(tmp_path)

    class FakeNationStatesClient:
        user_agent = 'NSAI-Test/0.1 contact:test@example.com nation:Oringrad'
        api_version = None

        def public_nation(self, nation, shards):
            return ET.fromstring('<NATION id="oringrad"><FULLNAME>Oringrad</FULLNAME></NATION>')

        def issues(self, nation):
            return ET.fromstring(
                '''
                <NATION>
                  <ISSUES>
                    <ISSUE id="456">
                      <TITLE>Orbital Farms</TITLE>
                      <TEXT>Farmers want orbital hydroponics grants.</TEXT>
                      <OPTION id="1">Fund them.</OPTION>
                    </ISSUE>
                    <ISSUE id="123">
                      <TITLE>Robot Teachers</TITLE>
                      <TEXT>Schools want robot teachers.</TEXT>
                      <OPTION id="1">Regulate them.</OPTION>
                    </ISSUE>
                  </ISSUES>
                </NATION>
                '''
            )

    class FakeGovernor:
        advise_issue_ids: list[str] = []

        def __init__(self, *, base_url=None, model=None, api_key=None):
            self.base_url = base_url or 'http://localhost:1234/v1'
            self.model = model or 'test-model'

        def plan_issue_order(self, **kwargs):
            raise AssertionError('arrival ordering should not ask AI to order')

        def advise(self, *, live_issues, **kwargs):
            issue = live_issues[0]
            issue_id = str(issue['issue_id'])
            FakeGovernor.advise_issue_ids.append(issue_id)
            return {
                **sample_recommendation(),
                'issue_id': issue_id,
                'option_id': str(issue['options'][0]['option_id']),
                'headline': f'Handle issue {issue_id}',
                'model': self.model,
            }

    monkeypatch.setattr(
        live.NationStatesClient,
        'from_env',
        classmethod(lambda cls, nation_config=None: FakeNationStatesClient()),
    )
    monkeypatch.setattr(live, 'LocalGovernor', FakeGovernor)

    args = advise_args(tmp_path, profile_path=profile_path)
    args.all_issues = True
    args.auto = False
    args.refresh_advice = False
    args.flag_display = 'none'
    args.issue_order = 'arrival'

    run_advise(args)
    output = capsys.readouterr().out

    assert FakeGovernor.advise_issue_ids == ['456', '123']
    assert 'AI step skipped: issue ordering is disabled' in output
    assert '1. Orbital Farms (456)' in output


def test_all_issues_can_sort_by_issue_id_without_ai_ordering(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile_path = write_auto_profile(tmp_path)

    class FakeNationStatesClient:
        user_agent = 'NSAI-Test/0.1 contact:test@example.com nation:Oringrad'
        api_version = None

        def public_nation(self, nation, shards):
            return ET.fromstring('<NATION id="oringrad"><FULLNAME>Oringrad</FULLNAME></NATION>')

        def issues(self, nation):
            return ET.fromstring(
                '''
                <NATION>
                  <ISSUES>
                    <ISSUE id="456">
                      <TITLE>Orbital Farms</TITLE>
                      <TEXT>Farmers want orbital hydroponics grants.</TEXT>
                      <OPTION id="1">Fund them.</OPTION>
                    </ISSUE>
                    <ISSUE id="123">
                      <TITLE>Robot Teachers</TITLE>
                      <TEXT>Schools want robot teachers.</TEXT>
                      <OPTION id="1">Regulate them.</OPTION>
                    </ISSUE>
                  </ISSUES>
                </NATION>
                '''
            )

    class FakeGovernor:
        advise_issue_ids: list[str] = []

        def __init__(self, *, base_url=None, model=None, api_key=None):
            self.base_url = base_url or 'http://localhost:1234/v1'
            self.model = model or 'test-model'

        def plan_issue_order(self, **kwargs):
            raise AssertionError('ID ordering should not ask AI to order')

        def advise(self, *, live_issues, **kwargs):
            issue = live_issues[0]
            issue_id = str(issue['issue_id'])
            FakeGovernor.advise_issue_ids.append(issue_id)
            return {
                **sample_recommendation(),
                'issue_id': issue_id,
                'option_id': str(issue['options'][0]['option_id']),
                'headline': f'Handle issue {issue_id}',
                'model': self.model,
            }

    monkeypatch.setattr(
        live.NationStatesClient,
        'from_env',
        classmethod(lambda cls, nation_config=None: FakeNationStatesClient()),
    )
    monkeypatch.setattr(live, 'LocalGovernor', FakeGovernor)

    args = advise_args(tmp_path, profile_path=profile_path)
    args.all_issues = True
    args.auto = False
    args.refresh_advice = False
    args.flag_display = 'none'
    args.issue_order = 'id'

    run_advise(args)
    output = capsys.readouterr().out

    assert FakeGovernor.advise_issue_ids == ['123', '456']
    assert 'AI step skipped: --issue-order id is set; sorting issues by ID.' in output
    assert '1. Robot Teachers (123)' in output


def test_all_issues_child_runs_reuse_loaded_nationstates_data(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile_path = write_auto_profile(tmp_path)

    class FakeNationStatesClient:
        user_agent = 'NSAI-Test/0.1 contact:test@example.com nation:Oringrad'
        api_version = None

        def __init__(self) -> None:
            self.public_calls = 0
            self.issue_calls = 0

        def public_nation(self, nation, shards):
            self.public_calls += 1
            return ET.fromstring('<NATION id="oringrad"><FULLNAME>Oringrad</FULLNAME></NATION>')

        def issues(self, nation):
            self.issue_calls += 1
            return ET.fromstring(
                '''
                <NATION>
                  <ISSUES>
                    <ISSUE id="123">
                      <TITLE>Robot Teachers</TITLE>
                      <TEXT>Schools want robot teachers.</TEXT>
                      <OPTION id="1">Regulate them.</OPTION>
                    </ISSUE>
                    <ISSUE id="456">
                      <TITLE>Orbital Farms</TITLE>
                      <TEXT>Farmers want orbital hydroponics grants.</TEXT>
                      <OPTION id="1">Fund them.</OPTION>
                    </ISSUE>
                  </ISSUES>
                </NATION>
                '''
            )

    class FakeGovernor:
        advise_issue_ids: list[str] = []

        def __init__(self, *, base_url=None, model=None, api_key=None):
            self.base_url = base_url or 'http://localhost:1234/v1'
            self.model = model or 'test-model'

        def plan_issue_order(self, **kwargs):
            return {
                'ordered_issue_ids': ['456', '123'],
                'reasons': {
                    '456': 'Food security is most urgent.',
                    '123': 'Education policy can follow.',
                },
                'model': self.model,
            }

        def advise(self, *, live_issues, **kwargs):
            issue = live_issues[0]
            issue_id = str(issue['issue_id'])
            FakeGovernor.advise_issue_ids.append(issue_id)
            return {
                **sample_recommendation(),
                'issue_id': issue_id,
                'option_id': str(issue['options'][0]['option_id']),
                'headline': f'Handle issue {issue_id}',
                'model': self.model,
            }

    fake_ns = FakeNationStatesClient()
    from_env_calls = 0

    def fake_from_env(cls, nation_config=None):
        nonlocal from_env_calls
        from_env_calls += 1
        return fake_ns

    monkeypatch.setattr(
        live.NationStatesClient,
        'from_env',
        classmethod(fake_from_env),
    )
    monkeypatch.setattr(live, 'LocalGovernor', FakeGovernor)

    args = advise_args(tmp_path, profile_path=profile_path)
    args.all_issues = True
    args.auto = False
    args.refresh_advice = False
    args.flag_display = 'none'

    run_advise(args)
    capsys.readouterr()

    assert from_env_calls == 1
    assert fake_ns.public_calls == 1
    assert fake_ns.issue_calls == 1
    assert FakeGovernor.advise_issue_ids == ['456', '123']


def test_all_issues_single_issue_order_is_not_auto_blocking_fallback(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile_path = write_auto_profile(tmp_path)
    stale_live_issues = [
        {
            'issue_id': '123',
            'title': 'Robot Teachers',
            'text': 'Schools want robot teachers.',
            'options': [
                {'option_id': '1', 'text': 'Ban them.'},
                {'option_id': '2', 'text': 'Regulate them.'},
            ],
        }
    ]
    live.AdviceCache().save_issue_plan(
        nation='Oringrad',
        live_issues=stale_live_issues,
        ordered_issue_ids=['123'],
        reasons={'123': 'Deterministic fallback kept the live issue order.'},
        source='fallback',
        token_usage={},
    )

    class FakeNationStatesClient:
        user_agent = 'NSAI-Test/0.1 contact:test@example.com nation:Oringrad'
        api_version = None

        def __init__(self) -> None:
            self.answer_calls = []

        def public_nation(self, nation, shards):
            return ET.fromstring('<NATION id="oringrad"><FULLNAME>Oringrad</FULLNAME></NATION>')

        def issues(self, nation):
            return ET.fromstring(
                '''
                <NATION>
                  <ISSUES>
                    <ISSUE id="123">
                      <TITLE>Robot Teachers</TITLE>
                      <TEXT>Schools want robot teachers.</TEXT>
                      <OPTION id="1">Ban them.</OPTION>
                      <OPTION id="2">Regulate them.</OPTION>
                    </ISSUE>
                  </ISSUES>
                </NATION>
                '''
            )

        def answer_issue(self, nation, issue_id, option_id):
            self.answer_calls.append((nation, issue_id, option_id))
            return ET.fromstring('<NATION><ISSUE><OK>1</OK></ISSUE></NATION>')

    class FakeGovernor:
        advise_issue_ids: list[str] = []

        def __init__(self, *, base_url=None, model=None, api_key=None):
            self.base_url = base_url or 'http://localhost:1234/v1'
            self.model = model or 'test-model'

        def plan_issue_order(self, **kwargs):
            raise AssertionError('single all-issues run should not ask AI to order')

        def advise(self, *, live_issues, **kwargs):
            issue = live_issues[0]
            issue_id = str(issue['issue_id'])
            FakeGovernor.advise_issue_ids.append(issue_id)
            return {
                **sample_recommendation(),
                'issue_id': issue_id,
                'option_id': '2',
                'confidence': 0.95,
                'charter_alignment_score': 95,
                'red_line_triggered': False,
                'model': self.model,
            }

    fake_ns = FakeNationStatesClient()
    monkeypatch.setattr(
        live.NationStatesClient,
        'from_env',
        classmethod(lambda cls, nation_config=None: fake_ns),
    )
    monkeypatch.setattr(live, 'LocalGovernor', FakeGovernor)

    args = advise_args(tmp_path, profile_path=profile_path)
    args.all_issues = True
    args.auto = True
    args.refresh_advice = False
    args.flag_display = 'none'
    args.parallel_requests = 4

    run_advise(args)
    output = capsys.readouterr().out

    assert FakeGovernor.advise_issue_ids == ['123']
    assert fake_ns.answer_calls == [('Oringrad', '123', '2')]
    assert 'Cached deterministic fallback all-issues order plan ignored' not in output
    assert 'AI step skipped: reused cached all-issues order plan.' in output
    assert 'AI step skipped: reused cached recommendation.' in output
    assert 'Cached advice for issue 123 cannot be reused' not in output
    assert 'deterministic fallback was used for issue_selection' not in output


def test_all_issues_parallel_requests_prefetch_missing_advice(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile_path = write_auto_profile(tmp_path)

    class FakeNationStatesClient:
        user_agent = 'NSAI-Test/0.1 contact:test@example.com nation:Oringrad'
        api_version = None

        def public_nation(self, nation, shards):
            return ET.fromstring('<NATION id="oringrad"><FULLNAME>Oringrad</FULLNAME></NATION>')

        def issues(self, nation):
            return ET.fromstring(
                '''
                <NATION>
                  <ISSUES>
                    <ISSUE id="123">
                      <TITLE>Robot Teachers</TITLE>
                      <TEXT>Schools want robot teachers.</TEXT>
                      <OPTION id="1">Regulate them.</OPTION>
                    </ISSUE>
                    <ISSUE id="456">
                      <TITLE>Orbital Farms</TITLE>
                      <TEXT>Farmers want orbital hydroponics grants.</TEXT>
                      <OPTION id="1">Fund them.</OPTION>
                    </ISSUE>
                  </ISSUES>
                </NATION>
                '''
            )

    class FakeGovernor:
        plan_calls = 0
        model_detection_inits = 0
        advise_issue_ids: list[str] = []
        advise_pulse_labels: list[str | None] = []

        def __init__(self, *, base_url=None, model=None, api_key=None):
            self.base_url = base_url or 'http://localhost:1234/v1'
            self.model = model or 'test-model'
            if model is None:
                FakeGovernor.model_detection_inits += 1

        def plan_issue_order(self, **kwargs):
            FakeGovernor.plan_calls += 1
            return {
                'ordered_issue_ids': ['456', '123'],
                'reasons': {
                    '456': 'Food security is most urgent.',
                    '123': 'Education policy can follow.',
                },
                'model': self.model,
            }

        def advise(self, *, live_issues, **kwargs):
            issue = live_issues[0]
            issue_id = str(issue['issue_id'])
            FakeGovernor.advise_issue_ids.append(issue_id)
            FakeGovernor.advise_pulse_labels.append(kwargs.get('pulse_label'))
            return {
                **sample_recommendation(),
                'issue_id': issue_id,
                'option_id': str(issue['options'][0]['option_id']),
                'headline': f'Handle issue {issue_id}',
                'model': self.model,
            }

    monkeypatch.setattr(
        live.NationStatesClient,
        'from_env',
        classmethod(lambda cls, nation_config=None: FakeNationStatesClient()),
    )
    monkeypatch.setattr(live, 'LocalGovernor', FakeGovernor)

    args = advise_args(tmp_path, profile_path=profile_path)
    args.all_issues = True
    args.auto = False
    args.refresh_advice = False
    args.flag_display = 'none'
    args.parallel_requests = 2
    args.model = None

    run_advise(args)
    output = capsys.readouterr().out

    assert FakeGovernor.plan_calls == 1
    assert FakeGovernor.model_detection_inits == 1
    assert sorted(FakeGovernor.advise_issue_ids) == ['123', '456']
    assert FakeGovernor.advise_pulse_labels == [None, None]
    assert output.count('AI step skipped: reused cached recommendation.') == 2
    assert 'Prefetching missing advice: 2/2 issue(s), 2 worker(s)' in output


def test_all_issues_parallel_prefetch_reports_cached_advice(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile_path = write_auto_profile(tmp_path)
    cached_issue = {
        'issue_id': '123',
        'title': 'Robot Teachers',
        'text': 'Schools want robot teachers.',
        'options': [{'option_id': '1', 'text': 'Regulate them.'}],
    }
    live.AdviceCache().save_advice(
        nation='Oringrad',
        live_issue=cached_issue,
        recommendation={
            **sample_recommendation(),
            'issue_id': '123',
            'option_id': '1',
            'headline': 'Handle cached issue',
            'model': 'test-model',
        },
        source='ai',
    )

    class FakeNationStatesClient:
        user_agent = 'NSAI-Test/0.1 contact:test@example.com nation:Oringrad'
        api_version = None

        def public_nation(self, nation, shards):
            return ET.fromstring('<NATION id="oringrad"><FULLNAME>Oringrad</FULLNAME></NATION>')

        def issues(self, nation):
            return ET.fromstring(
                '''
                <NATION>
                  <ISSUES>
                    <ISSUE id="123">
                      <TITLE>Robot Teachers</TITLE>
                      <TEXT>Schools want robot teachers.</TEXT>
                      <OPTION id="1">Regulate them.</OPTION>
                    </ISSUE>
                    <ISSUE id="456">
                      <TITLE>Orbital Farms</TITLE>
                      <TEXT>Farmers want orbital hydroponics grants.</TEXT>
                      <OPTION id="1">Fund them.</OPTION>
                    </ISSUE>
                  </ISSUES>
                </NATION>
                '''
            )

    class FakeGovernor:
        advise_issue_ids: list[str] = []
        advise_pulse_labels: list[str | None] = []

        def __init__(self, *, base_url=None, model=None, api_key=None):
            self.base_url = base_url or 'http://localhost:1234/v1'
            self.model = model or 'test-model'

        def plan_issue_order(self, **kwargs):
            return {
                'ordered_issue_ids': ['456', '123'],
                'reasons': {
                    '456': 'Food security is most urgent.',
                    '123': 'Education policy can follow.',
                },
                'model': self.model,
            }

        def advise(self, *, live_issues, **kwargs):
            issue = live_issues[0]
            issue_id = str(issue['issue_id'])
            FakeGovernor.advise_issue_ids.append(issue_id)
            FakeGovernor.advise_pulse_labels.append(kwargs.get('pulse_label'))
            return {
                **sample_recommendation(),
                'issue_id': issue_id,
                'option_id': str(issue['options'][0]['option_id']),
                'headline': f'Handle issue {issue_id}',
                'model': self.model,
            }

    monkeypatch.setattr(
        live.NationStatesClient,
        'from_env',
        classmethod(lambda cls, nation_config=None: FakeNationStatesClient()),
    )
    monkeypatch.setattr(live, 'LocalGovernor', FakeGovernor)

    args = advise_args(tmp_path, profile_path=profile_path)
    args.all_issues = True
    args.auto = False
    args.refresh_advice = False
    args.flag_display = 'none'
    args.parallel_requests = 2

    run_advise(args)
    output = capsys.readouterr().out

    assert FakeGovernor.advise_issue_ids == ['456']
    assert FakeGovernor.advise_pulse_labels == [None]
    assert 'Prefetching missing advice: 1/2 issue(s), 1 worker(s)' in output
    assert '1 cached, requested 2' in output


def test_all_issues_progress_uses_stable_timer_layout() -> None:
    progress = live.make_all_issues_progress(Console(record=True))
    transient_progress = live.make_all_issues_progress(
        Console(record=True),
        transient=True,
    )

    assert [type(column).__name__ for column in progress.columns] == [
        'TextColumn',
        'BarColumn',
        'TextColumn',
        'TimeElapsedColumn',
    ]
    assert progress.live.refresh_per_second == live.ALL_ISSUES_PROGRESS_REFRESH_PER_SECOND
    assert progress.live.transient is False
    assert transient_progress.live.transient is True


def test_parallel_prefetch_respects_existing_escape_cancel(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile_path = write_auto_profile(tmp_path)

    class CancelMonitor:
        def cancel_requested(self) -> bool:
            return True

    args = advise_args(tmp_path, profile_path=profile_path)
    args.refresh_advice = False
    args.parallel_requests = 2
    args._cancel_monitor = CancelMonitor()

    cache = live.AdviceCache()
    issues = sample_issues()

    with pytest.raises(live.AdvisorCancelled):
        live.prefetch_issue_advice(
            args=args,
            nation='Oringrad',
            ordered_issue_ids=['123'],
            live_issues=issues,
            cache=cache,
            strategy='keep things stable',
            profile=None,
            nation_snapshot_xml='<NATION />',
            lm_base_url='http://localhost:1234/v1',
            lm_model='test-model',
            lm_api_key='not-needed',
            draft_dispatch=False,
            draft_factbook=False,
            valid_options=collect_issue_option_ids(issues),
            console=Console(record=True, color_system=None),
        )


def test_parallel_prefetch_failure_does_not_cache_fallback_advice(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile_path = write_auto_profile(tmp_path)
    seen_kwargs: list[dict[str, object]] = []

    class FakeGovernor:
        def __init__(self, *, base_url=None, model=None, api_key=None):
            self.base_url = base_url or 'http://localhost:1234/v1'
            self.model = model or 'test-model'

        def advise(self, *, live_issues, **kwargs):
            seen_kwargs.append(kwargs)
            raise RuntimeError('prefetch model request failed')

    monkeypatch.setattr(live, 'LocalGovernor', FakeGovernor)

    args = advise_args(tmp_path, profile_path=profile_path)
    args.refresh_advice = False
    args.parallel_requests = 2
    cache = live.AdviceCache()
    issues = sample_issues()
    console = Console(record=True, color_system=None)

    live.prefetch_issue_advice(
        args=args,
        nation='Oringrad',
        ordered_issue_ids=['123'],
        live_issues=issues,
        cache=cache,
        strategy='keep things stable',
        profile=None,
        nation_snapshot_xml='<NATION />',
        lm_base_url='http://localhost:1234/v1',
        lm_model='test-model',
        lm_api_key='not-needed',
        draft_dispatch=False,
        draft_factbook=False,
        valid_options=collect_issue_option_ids(issues),
        console=console,
    )

    assert seen_kwargs
    assert seen_kwargs[0]['retry_text_mode'] is False
    assert seen_kwargs[0]['fallback_on_failure'] is False
    assert cache.get_advice('Oringrad', '123') is None
    assert 'Sequential all-issues processing will retry that issue.' in console.export_text()


def test_escape_monitor_starts_posix_watcher(monkeypatch) -> None:
    started: dict[str, object] = {}

    class FakeThread:
        def __init__(self, *, target, name, daemon):
            started['target'] = target
            started['name'] = name
            started['daemon'] = daemon

        def start(self):
            started['started'] = True

        def join(self, *, timeout=None):
            started['join_timeout'] = timeout

    monkeypatch.setattr(live.os, 'name', 'posix')
    monkeypatch.setattr(
        live.sys,
        'stdin',
        SimpleNamespace(isatty=lambda: True),
    )
    monkeypatch.setattr(live.threading, 'Thread', FakeThread)

    with live.EscapeCancelMonitor() as monitor:
        assert started['started'] is True
        assert started['name'] == 'NSAIAllIssuesEscapeMonitor'
        assert started['daemon'] is True
        target = started['target']
        assert target.__self__ is monitor
        assert target.__name__ == '_watch_posix_escape'

    assert started['join_timeout'] == 0.2


def test_all_issues_escape_monitor_is_active_during_prefetch(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile_path = write_auto_profile(tmp_path)
    seen_monitor = {'active': False}

    def fake_prefetch_issue_advice(**kwargs):
        monitor = getattr(kwargs['args'], '_cancel_monitor', None)
        seen_monitor['active'] = isinstance(monitor, live.EscapeCancelMonitor)
        raise live.AdvisorCancelled('Cancelled by Escape.')

    monkeypatch.setattr(live, 'prefetch_issue_advice', fake_prefetch_issue_advice)

    args = advise_args(tmp_path, profile_path=profile_path)
    args.issue_order = 'arrival'
    args.refresh_advice = False
    args.parallel_requests = 4
    issues = [
        {
            'issue_id': '123',
            'title': 'Robot Teachers',
            'text': 'Schools want robot teachers.',
            'options': [{'option_id': '1', 'text': 'Regulate them.'}],
        },
        {
            'issue_id': '456',
            'title': 'Orbital Farms',
            'text': 'Farmers want orbital hydroponics grants.',
            'options': [{'option_id': '1', 'text': 'Fund them.'}],
        },
    ]

    live.run_all_issues(
        args,
        nation='Oringrad',
        ns=object(),  # type: ignore[arg-type]
        nation_root=ET.fromstring('<NATION id="oringrad" />'),
        live_issues=issues,
        cache=live.AdviceCache(),
        strategy='keep things stable',
        profile=None,
        nation_snapshot_xml='<NATION />',
        get_governor=lambda: pytest.fail('arrival order should not need a governor'),
        lm_base_url='http://localhost:1234/v1',
        lm_model='test-model',
        lm_api_key='not-needed',
        draft_dispatch=False,
        draft_factbook=False,
        valid_options=collect_issue_option_ids(issues),
    )

    output = capsys.readouterr().out
    assert seen_monitor['active'] is True
    assert not hasattr(args, '_cancel_monitor')
    assert 'All-issues run cancelled by Escape.' in output
    assert 'Processed:       0/2 issue(s)' in output


def test_cooldown_state_can_seed_from_recent_audit_log(tmp_path) -> None:
    audit_log = tmp_path / 'audit.sqlite3'
    AuditStore(audit_log).append({
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'nation': 'Oringrad',
        'action_applied': True,
        'publication_results': [
            {'kind': 'dispatch', 'status': 'posted'},
        ],
    })
    publication_state = {'cooldown_hit': False, 'last_post_at': None}
    issue_state = {'last_action_at': None}

    live.seed_cooldown_state_from_audit(
        publication_state=publication_state,
        issue_state=issue_state,
        audit_log=audit_log,
        nation='oringrad',
    )

    now = time.monotonic()
    assert isinstance(issue_state['last_action_at'], float)
    assert isinstance(publication_state['last_post_at'], float)
    assert 0 <= now - issue_state['last_action_at'] < 2
    assert 0 <= now - publication_state['last_post_at'] < 2


def test_publication_cooldown_seed_counts_failed_attempts(tmp_path) -> None:
    audit_log = tmp_path / 'audit.sqlite3'
    AuditStore(audit_log).append({
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'nation': 'Oringrad',
        'action_applied': True,
        'publication_results': [
            {
                'kind': 'dispatch',
                'status': 'failed',
                'error': (
                    'Your nation is attempting to issue many announcements '
                    'in a short period of time. Please wait for the international '
                    'press to catch their breath, then try again.'
                ),
            },
        ],
    })
    publication_state = {'cooldown_hit': False, 'last_post_at': None}
    issue_state = {'last_action_at': None}

    live.seed_cooldown_state_from_audit(
        publication_state=publication_state,
        issue_state=issue_state,
        audit_log=audit_log,
        nation='Oringrad',
    )

    assert isinstance(publication_state['last_post_at'], float)
    assert time.monotonic() - publication_state['last_post_at'] < 2


def test_all_issues_paces_issue_actions_and_publications(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile_path = write_auto_profile(tmp_path)
    sleeps: list[tuple[float, str]] = []

    def fake_sleep_with_progress(seconds, *, description, console):
        sleeps.append((seconds, description))

    monkeypatch.setattr(live, 'sleep_with_progress', fake_sleep_with_progress)

    class FakeNationStatesClient:
        user_agent = 'NSAI-Test/0.1 contact:test@example.com nation:Oringrad'
        api_version = None

        def __init__(self) -> None:
            self.answer_calls = []
            self.publish_calls = []

        def public_nation(self, nation, shards):
            return ET.fromstring('<NATION id="oringrad"><FULLNAME>Oringrad</FULLNAME></NATION>')

        def issues(self, nation):
            return ET.fromstring(
                '''
                <NATION>
                  <ISSUES>
                    <ISSUE id="123">
                      <TITLE>Robot Teachers</TITLE>
                      <TEXT>Schools want robot teachers.</TEXT>
                      <OPTION id="1">Regulate them.</OPTION>
                    </ISSUE>
                    <ISSUE id="456">
                      <TITLE>Orbital Farms</TITLE>
                      <TEXT>Farmers want orbital hydroponics grants.</TEXT>
                      <OPTION id="1">Fund them.</OPTION>
                    </ISSUE>
                  </ISSUES>
                </NATION>
                '''
            )

        def answer_issue(self, nation, issue_id, option_id):
            self.answer_calls.append((nation, issue_id, option_id))
            return ET.fromstring('<NATION><ISSUE><OK>1</OK></ISSUE></NATION>')

        def create_dispatch(self, nation, *, title, text, category, subcategory):
            self.publish_calls.append((nation, title, text, category, subcategory))
            return ET.fromstring('<NATION><SUCCESS>Created.</SUCCESS></NATION>')

    class FakeGovernor:
        advise_issue_ids: list[str] = []

        def __init__(self, *, base_url=None, model=None, api_key=None):
            self.base_url = base_url or 'http://localhost:1234/v1'
            self.model = model or 'test-model'

        def plan_issue_order(self, **kwargs):
            raise AssertionError('arrival order should skip AI ordering')

        def advise(self, *, live_issues, **kwargs):
            issue = live_issues[0]
            issue_id = str(issue['issue_id'])
            FakeGovernor.advise_issue_ids.append(issue_id)
            return {
                **sample_recommendation(),
                'issue_id': issue_id,
                'option_id': '1',
                'action': 'enact',
                'confidence': 0.95,
                'charter_alignment_score': 95,
                'red_line_triggered': False,
                'headline': f'Handle issue {issue_id}',
                'model': self.model,
                'dispatch_draft': {
                    'requested': True,
                    'title': f'Issue {issue_id} Update',
                    'text': f'Policy update for issue {issue_id}.',
                    'category_hint': 'Bulletin',
                    'subcategory_hint': 'News',
                },
            }

    fake_ns = FakeNationStatesClient()
    monkeypatch.setattr(
        live.NationStatesClient,
        'from_env',
        classmethod(lambda cls, nation_config=None: fake_ns),
    )
    monkeypatch.setattr(live, 'LocalGovernor', FakeGovernor)

    args = advise_args(
        tmp_path,
        profile_path=profile_path,
        draft_dispatch=True,
    )
    args.all_issues = True
    args.auto = True
    args.refresh_advice = True
    args.flag_display = 'none'
    args.issue_order = 'arrival'
    args.issue_cooldown_seconds = 7.0
    args.publication_cooldown_seconds = 11.0

    run_advise(args)
    capsys.readouterr()

    assert FakeGovernor.advise_issue_ids == ['123', '456']
    assert fake_ns.answer_calls == [
        ('Oringrad', '123', '1'),
        ('Oringrad', '456', '1'),
    ]
    assert [call[1] for call in fake_ns.publish_calls] == [
        'Issue 123 Update',
        'Issue 456 Update',
    ]
    assert len(sleeps) == 2
    assert sleeps[0][0] > 0
    assert sleeps[0][1] == 'NationStates issue cooldown for Oringrad'
    assert sleeps[1][0] > 0
    assert sleeps[1][1] == 'NationStates publication cooldown for Oringrad'


def test_all_issues_skips_later_publications_after_cooldown_error(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile_path = write_auto_profile(tmp_path)

    class FakeNationStatesClient:
        user_agent = 'NSAI-Test/0.1 contact:test@example.com nation:Oringrad'
        api_version = None

        def __init__(self) -> None:
            self.answer_calls = []
            self.publish_calls = []

        def public_nation(self, nation, shards):
            return ET.fromstring('<NATION id="oringrad"><FULLNAME>Oringrad</FULLNAME></NATION>')

        def issues(self, nation):
            return ET.fromstring(
                '''
                <NATION>
                  <ISSUES>
                    <ISSUE id="123">
                      <TITLE>Robot Teachers</TITLE>
                      <TEXT>Schools want robot teachers.</TEXT>
                      <OPTION id="1">Regulate them.</OPTION>
                    </ISSUE>
                    <ISSUE id="456">
                      <TITLE>Orbital Farms</TITLE>
                      <TEXT>Farmers want orbital hydroponics grants.</TEXT>
                      <OPTION id="1">Fund them.</OPTION>
                    </ISSUE>
                  </ISSUES>
                </NATION>
                '''
            )

        def answer_issue(self, nation, issue_id, option_id):
            self.answer_calls.append((nation, issue_id, option_id))
            return ET.fromstring('<NATION><ISSUE><OK>1</OK></ISSUE></NATION>')

        def create_dispatch(self, nation, *, title, text, category, subcategory):
            self.publish_calls.append((nation, title, text, category, subcategory))
            return ET.fromstring(
                '<NATION><ERROR>Your nation is attempting to issue many '
                'announcements in a short period of time. Please wait for the '
                'international press to catch their breath, then try again.'
                '</ERROR></NATION>'
            )

    class FakeGovernor:
        def __init__(self, *, base_url=None, model=None, api_key=None):
            self.base_url = base_url or 'http://localhost:1234/v1'
            self.model = model or 'test-model'

        def plan_issue_order(self, **kwargs):
            raise AssertionError('arrival order should skip AI ordering')

        def advise(self, *, live_issues, **kwargs):
            issue = live_issues[0]
            issue_id = str(issue['issue_id'])
            return {
                **sample_recommendation(),
                'issue_id': issue_id,
                'option_id': '1',
                'action': 'enact',
                'confidence': 0.95,
                'charter_alignment_score': 95,
                'red_line_triggered': False,
                'headline': f'Handle issue {issue_id}',
                'model': self.model,
                'dispatch_draft': {
                    'requested': True,
                    'title': f'Issue {issue_id} Update',
                    'text': f'Policy update for issue {issue_id}.',
                    'category_hint': 'Bulletin',
                    'subcategory_hint': 'News',
                },
            }

    fake_ns = FakeNationStatesClient()
    monkeypatch.setattr(
        live.NationStatesClient,
        'from_env',
        classmethod(lambda cls, nation_config=None: fake_ns),
    )
    monkeypatch.setattr(live, 'LocalGovernor', FakeGovernor)

    args = advise_args(
        tmp_path,
        profile_path=profile_path,
        draft_dispatch=True,
    )
    args.all_issues = True
    args.auto = True
    args.refresh_advice = True
    args.flag_display = 'none'
    args.issue_order = 'arrival'

    run_advise(args)

    output = capsys.readouterr().out
    assert fake_ns.answer_calls == [
        ('Oringrad', '123', '1'),
        ('Oringrad', '456', '1'),
    ]
    assert [call[1] for call in fake_ns.publish_calls] == ['Issue 123 Update']
    assert 'press to catch their breath' in output

    audit_records = [
        record
        for _, record in load_audit_log_records(tmp_path / 'audit.sqlite3')
    ]
    assert [
        record['recommendation']['issue_id']
        for record in audit_records
    ] == ['123', '456']
    publication_results = [
        record['publication_results'][0]
        for record in audit_records
    ]
    assert [result['status'] for result in publication_results] == ['failed', 'blocked']
    assert 'press to catch their breath' in publication_results[0]['error']
    assert publication_results[1]['error'] == live.PUBLICATION_COOLDOWN_SKIP_MESSAGE


def test_auto_reuses_unsafe_cached_advice_but_blocks_action(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile_path = write_auto_profile(tmp_path)
    live_issue = sample_issues()[0]
    live.AdviceCache().save_advice(
        nation='Oringrad',
        live_issue=live_issue,
        recommendation={
            **sample_recommendation(),
            'issue_id': '123',
            'action': 'dismiss',
            'option_id': '-1',
            'confidence': 0.95,
            'charter_alignment_score': 0,
            'reasoning': 'Dismissing it with Option 2 resolves the immediate threat.',
            'model': 'bad-cache',
        },
        source='ai',
    )

    class FakeNationStatesClient:
        user_agent = 'NSAI-Test/0.1 contact:test@example.com nation:Oringrad'
        api_version = None

        def __init__(self) -> None:
            self.answer_calls = []

        def public_nation(self, nation, shards):
            return ET.fromstring('<NATION id="oringrad"><FULLNAME>Oringrad</FULLNAME></NATION>')

        def issues(self, nation):
            return ET.fromstring(
                '''
                <NATION>
                  <ISSUES>
                    <ISSUE id="123">
                      <TITLE>Robot Teachers</TITLE>
                      <TEXT>Schools want robot teachers.</TEXT>
                      <OPTION id="1">Ban them.</OPTION>
                      <OPTION id="2">Regulate them.</OPTION>
                    </ISSUE>
                  </ISSUES>
                </NATION>
                '''
            )

        def answer_issue(self, nation, issue_id, option_id):
            self.answer_calls.append((nation, issue_id, option_id))
            return ET.fromstring('<NATION><ISSUE><OK>1</OK></ISSUE></NATION>')

    class FakeGovernor:
        advise_calls = 0

        def __init__(self, *, base_url=None, model=None, api_key=None):
            self.base_url = base_url or 'http://localhost:1234/v1'
            self.model = model or 'test-model'

        def advise(self, *, live_issues, **kwargs):
            FakeGovernor.advise_calls += 1
            return {
                **sample_recommendation(),
                'issue_id': '123',
                'option_id': '2',
                'confidence': 0.95,
                'charter_alignment_score': 95,
                'red_line_triggered': False,
                'model': self.model,
            }

    fake_ns = FakeNationStatesClient()
    monkeypatch.setattr(
        live.NationStatesClient,
        'from_env',
        classmethod(lambda cls, nation_config=None: fake_ns),
    )
    monkeypatch.setattr(live, 'LocalGovernor', FakeGovernor)

    args = advise_args(tmp_path, profile_path=profile_path)
    args.refresh_advice = False
    run_advise(args)

    output = capsys.readouterr().out
    assert 'Using cached advice for issue 123.' in output
    assert 'AI step skipped: reused cached recommendation.' in output
    assert 'Cached advice for issue 123 cannot be reused' not in output
    assert 'AUTO ACTION BLOCKED' in output
    assert FakeGovernor.advise_calls == 0
    assert fake_ns.answer_calls == []

    _, audit_entry = load_audit_log_records(tmp_path / 'audit.sqlite3')[0]
    assert audit_entry['action'] == 'requires_review'
    assert audit_entry['action_applied'] is False
    assert audit_entry['blocked'] is True


def test_auto_validation_passes_for_valid_dismiss_without_publication() -> None:
    recommendation = {
        **sample_recommendation(),
        'action': 'dismiss',
        'option_id': '-1',
        'confidence': 0.91,
        'charter_alignment_score': 75,
        'reasoning': 'All live options conflict with the profile; dismissal is the safest choice.',
        'dispatch_draft': {
            'requested': False,
            'title': '',
            'text': '',
            'category_hint': '',
            'subcategory_hint': '',
        },
        'factbook_draft': {
            'requested': False,
            'pertinent': False,
            'title': '',
            'text': '',
            'category_hint': '',
            'subcategory_hint': '',
            'reason': '',
        },
    }

    result = validate_auto_action(
        live_issues=sample_issues(),
        selected_issue=sample_issues()[0],
        recommendation=recommendation,
        ai_step_statuses=[
            {'step': 'issue_selection', 'status': 'ok'},
            {'step': 'recommendation_generation', 'status': 'ok'},
        ],
        draft_dispatch=False,
        draft_factbook=False,
        minimum_confidence=0.8,
    )

    assert result.passed is True
    assert result.reasons == []


def test_valid_auto_dismiss_calls_action_endpoint_without_publication(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile_path = write_auto_profile(tmp_path)

    class FakeNationStatesClient:
        user_agent = 'NSAI-Test/0.1 contact:test@example.com nation:Oringrad'
        api_version = None

        def __init__(self) -> None:
            self.answer_calls = []
            self.publish_calls = []

        def public_nation(self, nation, shards):
            return ET.fromstring('<NATION id="oringrad"><FULLNAME>Oringrad</FULLNAME></NATION>')

        def issues(self, nation):
            return ET.fromstring(
                '''
                <NATION>
                  <ISSUES>
                    <ISSUE id="123">
                      <TITLE>Robot Teachers</TITLE>
                      <TEXT>Schools want robot teachers.</TEXT>
                      <OPTION id="1">Ban them.</OPTION>
                      <OPTION id="2">Regulate them.</OPTION>
                    </ISSUE>
                  </ISSUES>
                </NATION>
                '''
            )

        def answer_issue(self, nation, issue_id, option_id):
            self.answer_calls.append((nation, issue_id, option_id))
            return ET.fromstring('<NATION><ISSUE><OK>1</OK></ISSUE></NATION>')

        def create_dispatch(self, nation, *, title, text, category, subcategory):
            self.publish_calls.append((nation, title, text, category, subcategory))
            return ET.fromstring('<NATION><SUCCESS>Created.</SUCCESS></NATION>')

    class FakeGovernor:
        def __init__(self, *, base_url=None, model=None, api_key=None):
            self.base_url = base_url or 'http://localhost:1234/v1'
            self.model = model or 'test-model'

        def advise(self, *, live_issues, **kwargs):
            return {
                **sample_recommendation(),
                'issue_id': '123',
                'action': 'dismiss',
                'option_id': '-1',
                'confidence': 0.95,
                'charter_alignment_score': 75,
                'reasoning': 'All options conflict with the profile; dismissal is safest.',
                'red_line_triggered': False,
                'model': self.model,
            }

    fake_ns = FakeNationStatesClient()
    monkeypatch.setattr(
        live.NationStatesClient,
        'from_env',
        classmethod(lambda cls, nation_config=None: fake_ns),
    )
    monkeypatch.setattr(live, 'LocalGovernor', FakeGovernor)

    run_advise(advise_args(tmp_path, profile_path=profile_path))

    assert fake_ns.answer_calls == [('Oringrad', '123', '-1')]
    assert fake_ns.publish_calls == []


def test_publication_mismatch_blocks_dismiss_publication() -> None:
    recommendation = {
        **sample_recommendation(),
        'action': 'dismiss',
        'option_id': '-1',
        'confidence': 0.91,
        'charter_alignment_score': 75,
        'reasoning': 'Dismissal is safest.',
        'dispatch_draft': {
            'requested': True,
            'title': 'Recovery Program',
            'text': 'The government will rebuild the damaged districts immediately.',
            'category_hint': 'Bulletin',
            'subcategory_hint': 'News',
        },
    }

    validation = validate_recommendation_consistency(
        recommendation,
        live_issues=sample_issues(),
        selected_issue=sample_issues()[0],
        draft_dispatch=True,
        draft_factbook=False,
    )
    assert validation.passed is False
    assert any('policy being enacted' in reason for reason in validation.reasons)

    class FakeNationStates:
        def __init__(self) -> None:
            self.calls = []

        def create_dispatch(self, nation, *, title, text, category, subcategory):
            self.calls.append((nation, title, text, category, subcategory))
            return ET.fromstring('<NATION><SUCCESS>Created.</SUCCESS></NATION>')

    fake_ns = FakeNationStates()
    results = publish_publication_drafts(
        fake_ns,  # type: ignore[arg-type]
        nation='Oringrad',
        recommendation=recommendation,
        draft_dispatch=True,
        draft_factbook=False,
    )

    assert fake_ns.calls == []
    assert results[0]['status'] == 'blocked'


def test_extract_token_usage_from_openai_style_response() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
    )

    assert extract_token_usage(response) == {
        'prompt_tokens': 10,
        'completion_tokens': 5,
        'total_tokens': 15,
    }


def test_publication_draft_validation() -> None:
    recommendation = {
        **sample_recommendation(),
        'dispatch_draft': {
            'requested': True,
            'title': 'Robot Teacher Reform Approved',
            'text': '[b]Oringrad[/b] has adopted new oversight for robot teachers.',
            'category_hint': 'Bulletin',
            'subcategory_hint': 'News',
        },
        'factbook_draft': {
            'requested': True,
            'pertinent': False,
            'title': '',
            'text': '',
            'category_hint': 'Factbook',
            'subcategory_hint': 'Education',
            'reason': 'The issue is routine policy rather than durable lore.',
        },
    }

    validate_publication_drafts(
        recommendation,
        draft_dispatch=True,
        draft_factbook=True,
    )

    with pytest.raises(NationStatesError, match='dispatch_draft is missing text'):
        validate_publication_drafts(
            {
                **recommendation,
                'dispatch_draft': {
                    **recommendation['dispatch_draft'],
                    'text': '',
                },
            },
            draft_dispatch=True,
            draft_factbook=False,
        )


def test_publication_category_resolution() -> None:
    dispatch_category, dispatch_subcategory = resolve_publication_category(
        {
            'category_hint': 'Technology',
            'subcategory_hint': 'Space Program',
        },
        kind='dispatch',
    )
    assert (dispatch_category, dispatch_subcategory) == (3, 315)

    factbook_category, factbook_subcategory = resolve_publication_category(
        {
            'category_hint': 'Factbook',
            'subcategory_hint': 'Economy',
        },
        kind='factbook',
    )
    assert (factbook_category, factbook_subcategory) == (1, 108)


def test_build_default_user_agent() -> None:
    agent = live.build_default_user_agent('Oringrad')
    assert 'NSAI/' in agent
    assert 'nation:Oringrad' in agent


def test_build_default_user_agent_sanitizes_special_chars() -> None:
    agent = live.build_default_user_agent('New Oringrad!')
    nation_part = agent.split('nation:')[1]
    assert ' ' not in nation_part
    assert '!' not in nation_part


def test_from_env_auto_builds_user_agent_when_missing(monkeypatch) -> None:
    from nsai.nations import NationConfig
    monkeypatch.delenv('NS_USER_AGENT', raising=False)
    config = NationConfig(nation_name='Oringrad')
    config.user_agent = None
    client = NationStatesClient.from_env(config)
    assert 'NSAI/' in client.user_agent
    assert 'Oringrad' in client.user_agent


def test_from_env_uses_explicit_env_agent_when_set(monkeypatch) -> None:
    monkeypatch.setenv('NS_USER_AGENT', 'CustomAgent/1.0 contact:me nation:Oringrad')
    config = None
    client = NationStatesClient.from_env(config)
    assert client.user_agent == 'CustomAgent/1.0 contact:me nation:Oringrad'


def test_private_command_prepare_execute_flow() -> None:
    client = NationStatesClient(
        user_agent='NSAI-Test/0.1 contact:test@example.com nation:Oringrad',
        password='secret',
    )
    calls = []

    def fake_request_xml(params, *, private=False, method='GET', retries=2):
        calls.append((params.copy(), private, method))
        if params['mode'] == 'prepare':
            return ET.fromstring('<NATION><SUCCESS>abc123</SUCCESS></NATION>')
        return ET.fromstring('<NATION><SUCCESS>Created.</SUCCESS></NATION>')

    client.request_xml = fake_request_xml  # type: ignore[method-assign]

    result = client.create_dispatch(
        'Oringrad',
        title='Robot Teacher Reform Approved',
        text='New oversight has been adopted.',
        category=3,
        subcategory=315,
    )

    assert result.findtext('.//SUCCESS') == 'Created.'
    assert calls[0][0]['mode'] == 'prepare'
    assert calls[0][1] is True
    assert calls[0][2] == 'POST'
    assert calls[1][0]['mode'] == 'execute'
    assert calls[1][0]['token'] == 'abc123'
    assert calls[1][0]['dispatch'] == 'add'


def test_nationstates_api_trace_prints_redacted_exchange(capsys) -> None:
    class FakeResponse:
        status_code = 200
        ok = True
        headers = {
            'X-Pin': 'pin-secret',
            'RateLimit-Remaining': '9',
        }
        text = '<NATION><SUCCESS>abc123</SUCCESS><DATA>ok</DATA></NATION>'

    class FakeSession:
        def get(self, url, *, params, headers, timeout):
            assert headers['X-Password'] == 'secret'
            return FakeResponse()

    client = NationStatesClient(
        user_agent='NSAI-Test/0.1 contact:test@example.com nation:Oringrad',
        password='secret',
        api_trace=True,
    )
    client.session = FakeSession()  # type: ignore[assignment]

    root = client.request_xml({'nation': 'Oringrad', 'q': 'issues'}, private=True)

    output = capsys.readouterr().out
    assert root.findtext('.//DATA') == 'ok'
    assert 'API TRACE [NationStates]' in output
    assert '"X-Password": "<redacted>"' in output
    assert '"X-Pin": "<redacted>"' in output
    assert 'pin-secret' not in output
    assert 'secret' not in output
    assert 'abc123' not in output


def test_local_model_api_trace_prints_request_and_response(capsys) -> None:
    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"ok": true}'),
                    ),
                ],
                usage=SimpleNamespace(total_tokens=3),
            )

    governor = LocalGovernor(
        base_url='http://localhost:1234/v1',
        model='test-model',
        api_key='super-secret',
        api_trace=True,
    )
    governor.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions()),
    )

    governor._chat_completion_with_reload_retry(
        step_name='trace test',
        pulse_label=None,
        model='test-model',
        messages=[{'role': 'user', 'content': 'hello'}],
        temperature=0,
    )

    output = capsys.readouterr().out
    assert 'API TRACE [local model] chat.completions.create (trace test)' in output
    assert '"step_name": "trace test"' in output
    assert '"model": "test-model"' in output
    assert '"content": "hello"' in output
    assert '"total_tokens": 3' in output
    assert 'super-secret' not in output


def test_local_model_retries_without_temperature_when_provider_rejects_it(
    capsys,
) -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(dict(kwargs))
            if 'temperature' in kwargs:
                raise RuntimeError(
                    "Error code: 400 - {'error': {'message': "
                    "\"Unsupported value: 'temperature' does not support 0.25 "
                    'with this model. Only the default (1) value is supported."'
                    "}}"
                )

            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"ok": true}'),
                    ),
                ],
            )

    completions = FakeCompletions()
    governor = LocalGovernor(
        base_url='https://api.openai.com/v1',
        model='gpt-5.6-terra',
        api_key='test-key',
    )
    governor.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )

    governor._chat_completion_with_reload_retry(
        step_name='temperature retry',
        pulse_label=None,
        model='gpt-5.6-terra',
        messages=[{'role': 'user', 'content': 'hello'}],
        temperature=0.25,
    )

    output = capsys.readouterr().out
    assert len(completions.calls) == 2
    assert completions.calls[0]['temperature'] == 0.25
    assert 'temperature' not in completions.calls[1]
    assert 'retrying with the provider default' in output


def test_publish_publication_drafts_posts_dispatch_and_factbook() -> None:
    recommendation = {
        **sample_recommendation(),
        'dispatch_draft': {
            'requested': True,
            'title': 'Robot Teacher Reform Approved',
            'text': 'New oversight has been adopted.',
            'category_hint': 'Bulletin',
            'subcategory_hint': 'News',
        },
        'factbook_draft': {
            'requested': True,
            'pertinent': True,
            'title': 'Education Automation Framework',
            'text': 'Oringrad regulates classroom automation.',
            'category_hint': 'Factbook',
            'subcategory_hint': 'Legislation',
            'reason': 'The change creates durable education law.',
        },
    }

    class FakeNationStates:
        def __init__(self) -> None:
            self.calls = []

        def create_dispatch(self, nation, *, title, text, category, subcategory):
            self.calls.append({
                'nation': nation,
                'title': title,
                'text': text,
                'category': category,
                'subcategory': subcategory,
            })
            return ET.fromstring('<NATION><SUCCESS>Created.</SUCCESS></NATION>')

    fake_ns = FakeNationStates()
    results = publish_publication_drafts(
        fake_ns,  # type: ignore[arg-type]
        nation='Oringrad',
        recommendation=recommendation,
        draft_dispatch=True,
        draft_factbook=True,
    )

    assert [result['status'] for result in results] == ['posted', 'posted']
    assert fake_ns.calls == [
        {
            'nation': 'Oringrad',
            'title': 'Robot Teacher Reform Approved',
            'text': 'New oversight has been adopted.',
            'category': 3,
            'subcategory': 315,
        },
        {
            'nation': 'Oringrad',
            'title': 'Education Automation Framework',
            'text': 'Oringrad regulates classroom automation.',
            'category': 1,
            'subcategory': 105,
        },
    ]


def test_backfill_publish_retries_after_cooldown_without_sleep() -> None:
    class FakeNationStates:
        def __init__(self) -> None:
            self.calls = 0

        def create_dispatch(self, nation, *, title, text, category, subcategory):
            self.calls += 1
            if self.calls == 1:
                return ET.fromstring(
                    '<NATION><ERROR>Your nation is attempting to issue many '
                    'announcements in a short period of time. Please wait for '
                    'the international press to catch their breath, then try '
                    'again.</ERROR></NATION>'
                )

            return ET.fromstring('<NATION><SUCCESS>Created.</SUCCESS></NATION>')

    fake_ns = FakeNationStates()
    result = publish_backfill_draft_with_retry(
        fake_ns,  # type: ignore[arg-type]
        nation='Oringrad',
        kind='dispatch',
        draft={
            'title': 'Robot Teacher Reform Approved',
            'text': 'New oversight has been adopted.',
            'category_hint': 'Bulletin',
            'subcategory_hint': 'News',
        },
        cooldown_seconds=0,
        cooldown_retries=1,
        console=Console(record=True, width=120, color_system=None),
    )

    assert fake_ns.calls == 2
    assert result['status'] == 'posted'


def test_pending_publication_entries_only_includes_enacted_unposted_drafts() -> None:
    recommendation = {
        **sample_recommendation(),
        'dispatch_draft': {
            'requested': True,
            'title': 'Robot Teacher Reform Approved',
            'text': 'New oversight has been adopted.',
            'category_hint': 'Bulletin',
            'subcategory_hint': 'News',
        },
        'factbook_draft': {
            'requested': True,
            'pertinent': True,
            'title': 'Education Automation Framework',
            'text': 'Oringrad regulates classroom automation.',
            'category_hint': 'Factbook',
            'subcategory_hint': 'Legislation',
            'reason': 'The change creates durable education law.',
        },
    }
    enacted = {
        'timestamp': '2026-07-02T01:00:00+00:00',
        'nation': 'Oringrad',
        'action': 'auto_enact',
        'recommendation': recommendation,
        'result_xml': '<NATION><ISSUE id="123" choice="2"><OK>1</OK></ISSUE></NATION>',
    }
    advisor_only = {
        **enacted,
        'timestamp': '2026-07-02T00:00:00+00:00',
        'action': 'advisor_only',
        'result_xml': None,
    }

    pending = pending_publication_entries([
        (1, advisor_only),
        (2, enacted),
    ])

    assert len(pending) == 1
    assert pending[0]['line'] == 2
    assert pending[0]['draft_dispatch'] is True
    assert pending[0]['draft_factbook'] is True

    backfilled = {
        'timestamp': '2026-07-02T01:01:00+00:00',
        'action': 'publication_backfill',
        'nation': 'Oringrad',
        'source_timestamp': '2026-07-02T01:00:00+00:00',
        'issue_id': '123',
        'option_id': '2',
        'publication_results': [
            {
                'kind': 'dispatch',
                'title': 'Robot Teacher Reform Approved',
                'status': 'posted',
            },
            {
                'kind': 'factbook',
                'title': 'Education Automation Framework',
                'status': 'posted',
            },
        ],
    }

    assert pending_publication_entries([(2, enacted), (3, backfilled)]) == []


def test_manual_and_auto_enactment_guardrails() -> None:
    recommendation = sample_recommendation()

    manual_allowed, manual_reasons = should_manual_enact(
        recommendation=recommendation,
        enact_requested=True,
        override_red_line=False,
    )
    assert manual_allowed is True
    assert manual_reasons == ['Manual --enact requested and guardrails passed.']

    profile = {
        'enactment_mode': 'auto_enact_high_confidence',
        'minimum_confidence_to_enact': 0.9,
    }
    auto_allowed, auto_reasons = should_auto_enact(
        profile=profile,
        recommendation=recommendation,
        auto_requested=True,
    )
    assert auto_allowed is True
    assert auto_reasons == ['Profile allows high-confidence auto-enactment.']

    blocked, reasons = should_auto_enact(
        profile=profile,
        recommendation={**recommendation, 'confidence': 0.5},
        auto_requested=True,
    )
    assert blocked is False
    assert reasons == ['Confidence 0.50 is below profile minimum 0.90.']


def test_dismissal_recommendation_can_be_applied_with_guardrails() -> None:
    recommendation = {
        **sample_recommendation(),
        'action': 'dismiss',
        'option_id': '-1',
        'headline': 'Dismiss Robot Teachers',
        'confidence': 0.95,
    }

    assert recommendation_action(recommendation) == 'dismiss'
    assert is_dismiss_recommendation(recommendation) is True

    manual_allowed, manual_reasons = should_manual_enact(
        recommendation=recommendation,
        enact_requested=True,
        override_red_line=False,
    )
    assert manual_allowed is True
    assert manual_reasons == ['Manual --enact requested and guardrails passed.']

    auto_allowed, auto_reasons = should_auto_enact(
        profile={
            'enactment_mode': 'auto_enact_high_confidence',
            'minimum_confidence_to_enact': 0.9,
        },
        recommendation=recommendation,
        auto_requested=True,
    )
    assert auto_allowed is True
    assert auto_reasons == ['Profile allows high-confidence auto-enactment.']


def test_write_audit_log(tmp_path) -> None:
    path = tmp_path / 'audit.sqlite3'
    profile_path = tmp_path / 'profile.json'

    write_audit_log(
        path,
        nation='Oringrad',
        profile_path=profile_path,
        profile={'enactment_mode': 'advise_only'},
        recommendation=sample_recommendation(),
        action='advisor_only',
        action_reasons=['Manual enactment was not requested.'],
        result_xml=xml_to_string(ET.Element('OK')),
    )

    records = load_audit_log_records(path)
    assert len(records) == 1

    _, entry = records[0]
    assert entry['nation'] == 'Oringrad'
    assert entry['profile_path'] == str(profile_path)
    assert entry['action'] == 'advisor_only'
    assert entry['recommendation']['option_id'] == '2'
    assert entry['result_xml'] == '<OK />'


def test_migrate_jsonl_audit_log_imports_without_touching_source(tmp_path) -> None:
    jsonl_path = tmp_path / 'legacy_audit.jsonl'
    db_path = tmp_path / 'audit.sqlite3'
    lines = [
        json.dumps({'timestamp': '2026-01-01T00:00:00+00:00', 'nation': 'Oringrad', 'action': 'advisor_only'}),
        json.dumps({'timestamp': '2026-01-02T00:00:00+00:00', 'nation': 'Oringrad', 'action': 'manual_enact'}),
    ]
    jsonl_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    original_jsonl_text = jsonl_path.read_text(encoding='utf-8')

    count = migrate_jsonl_audit_log(jsonl_path, db_path)

    assert count == 2
    assert jsonl_path.read_text(encoding='utf-8') == original_jsonl_text

    records = load_audit_log_records(db_path)
    assert len(records) == 2
    assert [entry['action'] for _, entry in records] == ['advisor_only', 'manual_enact']


def test_save_advise_options_persists_defaults_and_managed_profile(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile = tmp_path / 'oringrad.json'
    profile.write_text('{"nation_name": "Oringrad"}', encoding='utf-8')

    _, _, saved_profile_path = save_advise_options(
        nation='Oringrad',
        existing_config=None,
        profile_path=profile,
        cli_profile_path=profile,
        user_agent='NSAI-Test/0.1 contact:test@example.com nation:Oringrad',
        api_version=12,
        strategy='Preserve the labs.',
        show_issues=True,
        show_instruction=False,
        no_ai=True,
        audit_log='custom-audit.jsonl',
        lm_base_url='http://localhost:1234/v1',
        lm_model='local-model',
        lm_api_key=None,
        draft_dispatch=True,
        draft_factbook=False,
    )

    config = load_nation_config('Oringrad')
    app_config = load_app_config()
    stored = profiles_dir() / 'oringrad' / 'oringrad.json'

    assert profile.exists()
    assert saved_profile_path == stored
    assert stored.exists()
    assert config.profile_path == str(stored)
    assert config.strategy == 'Preserve the labs.'
    assert config.show_issues is True
    assert config.show_instruction is False
    assert config.no_ai is True
    assert config.audit_log == 'custom-audit.jsonl'
    assert config.lm_base_url == 'http://localhost:1234/v1'
    assert config.lm_model == 'local-model'
    assert config.lm_api_key_credential_key is None
    assert config.draft_dispatch is True
    assert config.draft_factbook is False
    assert app_config.default_nation == 'Oringrad'

    output = capsys.readouterr().out
    assert 'Safety flags were not saved' in output
