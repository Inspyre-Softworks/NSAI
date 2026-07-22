from __future__ import annotations

import asyncio
from argparse import Namespace

from rich.console import Console
from textual.widgets import Button, Collapsible

from nsai.advisor.cache import AdviceCache
from nsai.advisor.cached_advice import (
    CachedAdviceApp,
    ConfirmAdviceActionScreen,
    display_reason,
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


def test_advisor_tui_routes_requested_issue_through_manual_enactment(
    monkeypatch,
) -> None:
    calls = []
    requested_issues = iter(['1777', None])

    class FakeApp:
        def __init__(self, nation, *, allow_actions=False):
            assert nation == 'Oringrad'
            assert allow_actions is True

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
