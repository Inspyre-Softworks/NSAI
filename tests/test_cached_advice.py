from __future__ import annotations

import asyncio
from argparse import Namespace

from rich.console import Console
from textual.widgets import Button, Collapsible

from nsai.advisor.cache import AdviceCache
from nsai.advisor.cached_advice import (
    ENACT_ALL_ACTION,
    CachedAdviceApp,
    ConfirmAdviceActionScreen,
    RecentEnactmentEffectsScreen,
    cumulative_enactment_effects_table,
    display_reason,
    enactment_outcome_table,
    render_cached_advice_report,
    run_advisor_tui,
    website_option_label,
)


def populated_cache(tmp_path) -> AdviceCache:
    cache = AdviceCache(tmp_path / 'advice.sqlite3')
    issue = {
        'issue_id': '1777',
        'title': 'Peace, Love, and Oringrad',
        'text': 'Choose an international aid policy.',
        'options': [
            {'option_id': '0', 'text': 'Fund direct aid.'},
            {'option_id': '1', 'text': 'Use private partners.'},
            {'option_id': '3', 'text': 'Expand intelligence operations.'},
            {'option_id': '4', 'text': 'Prioritize domestic welfare.'},
        ],
    }
    cache.save_issue_plan(
        nation='Oringrad',
        live_issues=[issue],
        ordered_issue_ids=['1777'],
        reasons={'1777': 'Resolve this issue first.'},
        source='ai',
    )
    cache.save_advice(
        nation='Oringrad',
        live_issue=issue,
        recommendation={
            'issue_id': '1777',
            'option_id': '4',
            'action': 'enact',
            'reasoning': (
                'Option 4 is preferable to Option 3. The enacted option '
                'prioritizes domestic welfare.'
            ),
            'confidence': 0.91,
            'model': 'local-model',
        },
        source='ai',
    )
    return cache


def add_second_issue(cache: AdviceCache, *, include_first: bool = True) -> None:
    first = cache.get_advice('Oringrad', '1777')
    assert first is not None
    second = {
        'issue_id': '1888',
        'title': 'The Next Unanswered Issue',
        'text': 'Choose the next policy.',
        'options': [
            {'option_id': '0', 'text': 'Adopt the next policy.'},
        ],
    }
    live_issues = [first.live_issue, second] if include_first else [second]
    ordered_ids = ['1777', '1888'] if include_first else ['1888']
    cache.save_issue_plan(
        nation='Oringrad',
        live_issues=live_issues,
        ordered_issue_ids=ordered_ids,
        reasons={issue_id: 'Follow the plan.' for issue_id in ordered_ids},
        source='ai',
    )
    cache.save_advice(
        nation='Oringrad',
        live_issue=second,
        recommendation={
            'issue_id': '1888',
            'option_id': '0',
            'action': 'enact',
            'reasoning': 'This is the next recommendation.',
            'confidence': 0.93,
            'model': 'local-model',
        },
        source='ai',
    )


def test_sparse_option_ids_map_to_website_positions(tmp_path) -> None:
    cache = populated_cache(tmp_path)
    cached = cache.get_advice('Oringrad', '1777')

    assert cached is not None
    assert website_option_label(cached) == (
        'Option 4',
        'bold green',
        'Prioritize domestic welfare.',
    )
    assert display_reason(cached) == (
        'Option 4 is preferable to Option 3. The chosen option '
        'prioritizes domestic welfare.'
    )


def test_new_reason_labels_are_not_translated_twice(tmp_path) -> None:
    cache = populated_cache(tmp_path)
    cached = cache.get_advice('Oringrad', '1777')
    assert cached is not None

    cached.recommendation['option_label_basis'] = 'website_position'
    cached.recommendation['reasoning'] = 'Option 4 is the best website choice.'

    assert display_reason(cached) == 'Option 4 is the best website choice.'


def test_plain_cached_advice_report_is_detailed(tmp_path) -> None:
    cache = populated_cache(tmp_path)
    console = Console(record=True, width=120)

    render_cached_advice_report('Oringrad', cache=cache, console=console)
    output = console.export_text()

    assert 'Cached Advisor Decisions' in output
    assert 'Peace, Love, and Oringrad' in output
    assert 'Option 4' in output
    assert 'Prioritize domestic welfare.' in output
    assert 'confidence 91%' in output
    assert 'model local-model' in output


def test_textual_cached_advice_cards_can_collapse(tmp_path) -> None:
    async def exercise() -> None:
        app = CachedAdviceApp('Oringrad', cache=populated_cache(tmp_path))
        async with app.run_test() as pilot:
            cards = list(app.query(Collapsible))
            assert len(cards) == 1
            assert cards[0].collapsed is False

            await pilot.press('c')
            assert cards[0].collapsed is True

            await pilot.press('e')
            assert cards[0].collapsed is False

            await pilot.press('1')
            assert cards[0].collapsed is True

    asyncio.run(exercise())


def test_textual_action_button_requires_confirmation(tmp_path) -> None:
    async def exercise() -> None:
        app = CachedAdviceApp(
            'Oringrad',
            cache=populated_cache(tmp_path),
            allow_actions=True,
        )
        async with app.run_test() as pilot:
            button = app.query_one('#issue-action-0', Button)
            assert button.label.plain == 'Enact Option 4'

            button.focus()
            await pilot.pause()
            await pilot.press('enter')
            assert isinstance(app.screen, ConfirmAdviceActionScreen)

            await pilot.click('#cancel-action')
            assert not isinstance(app.screen, ConfirmAdviceActionScreen)

    asyncio.run(exercise())


def test_answered_tui_card_renders_stats_instead_of_another_action(tmp_path) -> None:
    cache = populated_cache(tmp_path)
    cache.record_enactment(
        nation='Oringrad',
        issue_id='1777',
        option_id='4',
        action='enact',
        result_xml='''
            <NATION id="oringrad">
              <ISSUE id="1777" choice="4">
                <OK>1</OK>
                <RANKINGS>
                  <RANK id="1">
                    <SCORE>99.17</SCORE>
                    <CHANGE>0.02</CHANGE>
                    <PCHANGE>0.020171</PCHANGE>
                  </RANK>
                </RANKINGS>
                <HEADLINES>
                  <HEADLINE>Domestic Welfare Expansion Begins</HEADLINE>
                </HEADLINES>
              </ISSUE>
            </NATION>
        ''',
    )
    cached = cache.get_advice('Oringrad', '1777')
    assert cached is not None

    console = Console(record=True, width=100)
    console.print(enactment_outcome_table(cached))
    output = console.export_text()
    assert 'Answered successfully' in output
    assert 'Economy (#1)' in output
    assert '99.17' in output
    assert '0.02' in output
    assert '0.020171' in output
    assert 'Domestic Welfare Expansion Begins' in output
    assert 'prioritizes domestic welfare' not in output

    async def exercise() -> None:
        app = CachedAdviceApp('Oringrad', cache=cache, allow_actions=True)
        async with app.run_test():
            card = app.query_one(Collapsible)
            assert str(card.title).startswith('Answered')
            assert card.has_class('answered')
            assert len(app.query('.issue-action')) == 0

    asyncio.run(exercise())


def test_tui_reload_expands_next_unanswered_issue(tmp_path) -> None:
    cache = populated_cache(tmp_path)
    cache.record_enactment(
        nation='Oringrad',
        issue_id='1777',
        option_id='4',
        action='enact',
        result_xml='<NATION><ISSUE id="1777" choice="4"><OK>1</OK></ISSUE></NATION>',
    )
    add_second_issue(cache, include_first=False)

    async def exercise() -> None:
        app = CachedAdviceApp(
            'Oringrad',
            cache=cache,
            allow_actions=True,
            outcome_issue_ids=['1777'],
        )
        async with app.run_test():
            cards = list(app.query(Collapsible))
            assert len(cards) == 2
            assert str(cards[0].title).startswith('Answered')
            assert cards[0].collapsed is True
            assert 'The Next Unanswered Issue' in str(cards[1].title)
            assert cards[1].collapsed is False
            assert app.query_one('#enact-all', Button).label.plain == (
                'Enact all recommendations (1)'
            )

    asyncio.run(exercise())


def test_enact_all_button_requires_confirmation_and_returns_bulk_action(
    tmp_path,
) -> None:
    cache = populated_cache(tmp_path)
    add_second_issue(cache)

    async def exercise() -> None:
        app = CachedAdviceApp('Oringrad', cache=cache, allow_actions=True)
        async with app.run_test() as pilot:
            button = app.query_one('#enact-all', Button)
            assert button.label.plain == 'Enact all recommendations (2)'
            button.focus()
            await pilot.press('enter')
            assert isinstance(app.screen, ConfirmAdviceActionScreen)
            assert app.screen.bulk_count == 2

            await pilot.click('#confirm-action')
            assert app.return_value == ENACT_ALL_ACTION

    asyncio.run(exercise())


def test_tab_shows_effects_for_all_recent_session_enactments(tmp_path) -> None:
    cache = populated_cache(tmp_path)
    add_second_issue(cache)
    cache.record_enactment(
        nation='Oringrad',
        issue_id='1777',
        option_id='4',
        action='enact',
        result_xml='''
            <NATION><ISSUE id="1777" choice="4">
              <RANKINGS><RANK id="1"><SCORE>99</SCORE><CHANGE>2</CHANGE>
              <PCHANGE>2.06</PCHANGE></RANK></RANKINGS>
              <HEADLINES><HEADLINE>Domestic welfare rises</HEADLINE></HEADLINES>
            </ISSUE></NATION>
        ''',
    )
    cache.record_enactment(
        nation='Oringrad',
        issue_id='1888',
        option_id='0',
        action='enact',
        result_xml='''
            <NATION><ISSUE id="1888" choice="0">
              <RANKINGS>
                <RANK id="1"><SCORE>98</SCORE><CHANGE>-1</CHANGE>
                <PCHANGE>-2.06</PCHANGE></RANK>
                <RANK id="2"><SCORE>51</SCORE><CHANGE>-1</CHANGE>
                <PCHANGE>-1.92</PCHANGE></RANK>
              </RANKINGS>
              <HEADLINES><HEADLINE>The next policy takes effect</HEADLINE></HEADLINES>
            </ISSUE></NATION>
        ''',
    )
    enactments = [
        cache.get_advice('Oringrad', issue_id)
        for issue_id in ('1777', '1888')
    ]
    assert all(enactments)
    console = Console(record=True, width=120)
    console.print(cumulative_enactment_effects_table(enactments))
    cumulative_output = console.export_text()
    assert 'Economy (#1)' in cumulative_output
    assert 'Political Freedoms (#2)' in cumulative_output
    assert 'Latest score' in cumulative_output
    assert 'Total change' in cumulative_output
    assert 'Combined % change' in cumulative_output
    assert '+1' in cumulative_output
    assert '-0.042436%' in cumulative_output

    async def exercise() -> None:
        app = CachedAdviceApp(
            'Oringrad',
            cache=cache,
            allow_actions=True,
            outcome_issue_ids=['1777', '1888'],
        )
        async with app.run_test() as pilot:
            await pilot.press('tab')
            assert isinstance(app.screen, RecentEnactmentEffectsScreen)
            assert [cached.issue_id for cached in app.screen.enactments] == [
                '1777',
                '1888',
            ]
            assert len(app.screen.query('.cumulative-effect')) == 1
            assert len(app.screen.query('.recent-effect')) == 2

            await pilot.press('tab')
            assert not isinstance(app.screen, RecentEnactmentEffectsScreen)

    asyncio.run(exercise())


def test_advisor_tui_routes_requested_issue_through_manual_enactment(
    monkeypatch,
) -> None:
    calls = []
    requested_issues = iter(['1777', None])

    class FakeApp:
        instances = []

        def __init__(
            self,
            nation,
            *,
            allow_actions=False,
            outcome_issue_ids=None,
        ):
            assert nation == 'Oringrad'
            assert allow_actions is True
            self.outcome_issue_ids = list(outcome_issue_ids or [])
            self.instances.append(self)

        def run(self):
            return next(requested_issues)

    def fake_run_advise(args) -> None:
        calls.append(args)

    monkeypatch.setattr('nsai.advisor.cached_advice.CachedAdviceApp', FakeApp)
    monkeypatch.setattr('nsai.advisor.live.run_advise', fake_run_advise)

    run_advisor_tui(Namespace(
        nation='Oringrad',
        profile=None,
        enact=False,
        auto=False,
        tui=True,
        dev=False,
        refresh_advice=False,
    ))

    assert len(calls) == 3
    assert calls[0].all_issues is True
    assert calls[0].enact is False
    assert calls[1].all_issues is False
    assert calls[1].enact is True
    assert calls[1]._target_issue_id == '1777'
    assert calls[1]._target_issue_source == 'advice_tui'
    assert calls[2].all_issues is True
    assert FakeApp.instances[0].outcome_issue_ids == []
    assert FakeApp.instances[1].outcome_issue_ids == ['1777']


def test_advisor_tui_enact_all_routes_each_issue_through_normal_path(
    monkeypatch,
) -> None:
    calls = []
    selections = iter([ENACT_ALL_ACTION, None])

    class FakeApp:
        def __init__(
            self,
            nation,
            *,
            allow_actions=False,
            outcome_issue_ids=None,
        ):
            assert nation == 'Oringrad'
            assert allow_actions is True

        def run(self):
            return next(selections)

        def actionable_issue_ids(self):
            return ['1777', '1888']

    monkeypatch.setattr('nsai.advisor.cached_advice.CachedAdviceApp', FakeApp)
    monkeypatch.setattr(
        'nsai.advisor.live.run_advise',
        lambda args: calls.append(args),
    )

    run_advisor_tui(Namespace(
        nation='Oringrad',
        profile=None,
        enact=False,
        auto=False,
        tui=True,
        dev=False,
        refresh_advice=False,
    ))

    assert len(calls) == 4
    first_action, second_action = calls[1], calls[2]
    assert first_action._target_issue_id == '1777'
    assert second_action._target_issue_id == '1888'
    assert first_action._target_issue_source == 'advice_tui_enact_all'
    assert second_action._target_issue_source == 'advice_tui_enact_all'
    assert first_action._issue_cooldown_required is False
    assert second_action._issue_cooldown_required is True
    assert first_action._issue_state is second_action._issue_state
    assert first_action._publication_state is second_action._publication_state
