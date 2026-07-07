from __future__ import annotations

from nsai.advisor.cache import (
    AdviceCache,
    extract_enactment_outcome,
    issue_set_signature,
)


def sample_live_issues() -> list[dict[str, object]]:
    return [
        {
            'issue_id': '123',
            'title': 'Robot Teachers',
            'text': 'Schools want robot teachers.',
            'options': [
                {'option_id': '1', 'text': 'Ban them.'},
                {'option_id': '2', 'text': 'Regulate them.'},
            ],
        },
        {
            'issue_id': '456',
            'title': 'Orbital Farms',
            'text': 'Farmers want orbital hydroponics grants.',
            'options': [
                {'option_id': '1', 'text': 'Fund them.'},
            ],
        },
    ]


def sample_recommendation() -> dict[str, object]:
    return {
        'issue_id': '123',
        'option_id': '2',
        'action': 'enact',
        'headline': 'Regulate Robot Teachers',
        'confidence': 0.91,
        'model': 'local-model',
        'token_usage': {
            'prompt_tokens': 100,
            'completion_tokens': 25,
            'total_tokens': 125,
        },
    }


def test_issue_set_signature_only_depends_on_issue_ids() -> None:
    issues = sample_live_issues()
    reversed_issues = list(reversed(issues))

    assert issue_set_signature(issues) == issue_set_signature(reversed_issues)
    assert issue_set_signature(issues) != issue_set_signature(issues[:1])


def test_cache_stores_issue_choice_and_advice_separately(tmp_path) -> None:
    cache = AdviceCache(tmp_path / 'advice.sqlite3')
    issues = sample_live_issues()

    cache.save_issue_choice(
        nation='Oringrad',
        live_issues=issues,
        selected_issue_id='123',
        why='Education policy is most urgent.',
        source='ai',
        token_usage={'prompt_tokens': 40, 'completion_tokens': 8, 'total_tokens': 48},
    )
    cache.save_advice(
        nation='Oringrad',
        live_issue=issues[0],
        recommendation=sample_recommendation(),
        source='ai',
    )

    choice = cache.get_issue_choice('Oringrad', issues)
    advice = cache.get_advice('Oringrad', '123')

    assert choice is not None
    assert choice.selected_issue_id == '123'
    assert choice.token_usage['total_tokens'] == 48
    assert advice is not None
    assert advice.recommendation['headline'] == 'Regulate Robot Teachers'
    assert advice.token_usage['total_tokens'] == 125

    new_issue_set = [
        *issues,
        {
            'issue_id': '789',
            'title': 'New Issue',
            'text': '',
            'options': [{'option_id': '1', 'text': 'Act.'}],
        },
    ]
    assert cache.get_issue_choice('Oringrad', new_issue_set) is None
    assert cache.get_advice('Oringrad', '123') is not None


def test_cache_stores_issue_order_plan_by_issue_set(tmp_path) -> None:
    cache = AdviceCache(tmp_path / 'advice.sqlite3')
    issues = sample_live_issues()

    cache.save_issue_plan(
        nation='Oringrad',
        live_issues=issues,
        ordered_issue_ids=['456', '123'],
        reasons={
            '456': 'Food security is most urgent.',
            '123': 'Education policy can follow.',
        },
        source='ai',
        token_usage={'total_tokens': 42},
    )

    plan = cache.get_issue_plan('Oringrad', issues)

    assert plan is not None
    assert plan.ordered_issue_ids == ['456', '123']
    assert plan.reasons['456'] == 'Food security is most urgent.'
    assert plan.source == 'ai'
    assert plan.token_usage['total_tokens'] == 42
    assert cache.get_issue_plan('Oringrad', issues[:1]) is None


def test_cache_records_enactment_outcomes(tmp_path) -> None:
    cache = AdviceCache(tmp_path / 'advice.sqlite3')
    issues = sample_live_issues()
    cache.save_advice(
        nation='Oringrad',
        live_issue=issues[0],
        recommendation=sample_recommendation(),
        source='ai',
    )

    result_xml = '''
    <NATION>
      <HEADLINES>
        <HEADLINE>Robot teachers flourish under new oversight.</HEADLINE>
      </HEADLINES>
      <RANKINGS>
        <RANK id="education">+3</RANK>
      </RANKINGS>
      <SCALE id="economy">+1</SCALE>
    </NATION>
    '''

    effects, headlines = cache.record_enactment(
        nation='Oringrad',
        issue_id='123',
        option_id='2',
        action='enact',
        result_xml=result_xml,
    )
    advice = cache.get_advice('Oringrad', '123')

    assert headlines == ['Robot teachers flourish under new oversight.']
    assert any(effect['tag'] == 'rankings' for effect in effects)
    assert advice is not None
    assert advice.enacted_count == 1
    assert advice.last_headlines == headlines
    assert advice.last_effects == effects


def test_extract_enactment_outcome_handles_invalid_xml() -> None:
    assert extract_enactment_outcome('not xml') == ([], [])
