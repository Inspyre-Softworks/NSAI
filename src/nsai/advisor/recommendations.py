"""Recommendation validation, enactment guardrails, and display."""

from __future__ import annotations

import textwrap
import xml.etree.ElementTree as ET
from typing import Any

from nsai.advisor.cache import CachedAdvice, live_issue_by_id
from nsai.advisor.client import NationStatesError
from nsai.advisor.governor import maybe_float
from nsai.advisor.safety import (
    ValidationResult,
    publication_mismatch_reasons,
    validate_auto_action,
    validate_recommendation_consistency,
)


DISMISS_OPTION_ID = '-1'
OUTPUT_WIDTH = 88
FIELD_LABEL_WIDTH = 16


def print_section_heading(title: str) -> None:
    print()
    print(title)
    print('=' * OUTPUT_WIDTH)


def print_subheading(title: str, *, indent: int = 2) -> None:
    prefix = ' ' * indent
    print(f'{prefix}{title}')
    print(f'{prefix}{"-" * max(12, OUTPUT_WIDTH - indent)}')


def print_field(
    label: str,
    value: Any,
    *,
    indent: int = 2,
    label_width: int = FIELD_LABEL_WIDTH,
) -> None:
    text = str(value if value is not None else '').strip()
    prefix = ' ' * indent
    label_text = f'{prefix}{label + ":":<{label_width}} '
    available_width = max(20, OUTPUT_WIDTH - len(label_text))
    wrapped_lines = textwrap.wrap(text, width=available_width) or ['']
    print(f'{label_text}{wrapped_lines[0]}')
    continuation = ' ' * len(label_text)
    for line in wrapped_lines[1:]:
        print(f'{continuation}{line}')


def print_wrapped_block(text: Any, *, indent: int = 4, width: int = OUTPUT_WIDTH) -> None:
    value = str(text if text is not None else '').strip()
    if not value:
        return

    prefix = ' ' * indent
    paragraphs = value.splitlines()
    printed = False
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            if printed:
                print()
            continue

        print(
            textwrap.fill(
                paragraph,
                width=max(20, width),
                initial_indent=prefix,
                subsequent_indent=prefix,
            )
        )
        printed = True


def print_bullets(items: list[Any], *, indent: int = 4) -> None:
    bullet_prefix = ' ' * indent + '- '
    continuation = ' ' * (indent + 2)
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        print(
            textwrap.fill(
                text,
                width=max(20, OUTPUT_WIDTH),
                initial_indent=bullet_prefix,
                subsequent_indent=continuation,
            )
        )


def extract_live_issues(issues_root: ET.Element) -> list[dict[str, Any]]:
    live_issues = []

    for issue in issues_root.findall('.//ISSUE'):
        issue_id = issue.attrib.get('id')
        if not issue_id:
            continue

        title = issue.findtext('TITLE', default='').strip()
        text = issue.findtext('TEXT', default='').strip()

        options = []
        for option in issue.findall('.//OPTION'):
            option_id = option.attrib.get('id')
            option_text = ''.join(option.itertext()).strip()

            if option_id is not None:
                options.append({
                    'option_id': str(option_id),
                    'text': option_text,
                })

        live_issues.append({
            'issue_id': str(issue_id),
            'title': title,
            'text': text,
            'options': options,
        })

    return live_issues


def collect_issue_option_ids(live_issues: list[dict[str, Any]]) -> dict[str, set[str]]:
    valid: dict[str, set[str]] = {}

    for issue in live_issues:
        issue_id = str(issue['issue_id'])
        valid[issue_id] = set()

        for option in issue.get('options', []):
            valid[issue_id].add(str(option['option_id']))

    return valid


def validate_recommendation(
    recommendation: dict[str, Any],
    valid_options: dict[str, set[str]],
) -> tuple[str, str]:
    issue_id = str(recommendation.get('issue_id', '')).strip()
    option_id = str(recommendation.get('option_id', '')).strip()
    action = recommendation_action(recommendation)

    if issue_id not in valid_options:
        raise NationStatesError(
            f'AI chose invalid issue_id={issue_id!r}. '
            f'Valid issue IDs: {sorted(valid_options)}'
        )

    if action == 'dismiss':
        if option_id and option_id != DISMISS_OPTION_ID:
            raise NationStatesError(
                f'AI chose action=dismiss but option_id={option_id!r}. '
                f'Dismissal must use option_id={DISMISS_OPTION_ID!r}.'
            )
        if not option_id:
            recommendation['structured_output_repaired'] = True
        recommendation['option_id'] = DISMISS_OPTION_ID
        return issue_id, DISMISS_OPTION_ID

    if action != 'enact':
        raise NationStatesError(
            f'AI chose invalid action={action!r}. Valid actions: enact, dismiss'
        )

    if not option_id:
        raise NationStatesError(
            f'AI chose action=enact but did not provide an option_id for issue {issue_id}.'
        )

    if option_id not in valid_options[issue_id]:
        raise NationStatesError(
            f'AI chose invalid option_id={option_id!r} for issue {issue_id}. '
            f'Valid options: {sorted(valid_options[issue_id])}'
        )

    return issue_id, option_id


def empty_dispatch_draft(*, requested: bool = False) -> dict[str, Any]:
    return {
        'requested': requested,
        'title': '',
        'text': '',
        'category_hint': '',
        'subcategory_hint': '',
    }


def empty_factbook_draft(*, requested: bool = False) -> dict[str, Any]:
    return {
        'requested': requested,
        'pertinent': False,
        'title': '',
        'text': '',
        'category_hint': '',
        'subcategory_hint': '',
        'reason': '',
    }


def validate_publication_drafts(
    recommendation: dict[str, Any],
    *,
    draft_dispatch: bool,
    draft_factbook: bool,
) -> None:
    if draft_dispatch:
        draft = recommendation.get('dispatch_draft')
        if not isinstance(draft, dict):
            raise NationStatesError('AI did not provide a dispatch_draft object.')

        if not draft.get('requested'):
            raise NationStatesError('AI did not mark dispatch_draft as requested.')

        if not str(draft.get('title', '')).strip():
            raise NationStatesError('AI dispatch_draft is missing a title.')

        if not str(draft.get('text', '')).strip():
            raise NationStatesError('AI dispatch_draft is missing text.')

    if draft_factbook:
        draft = recommendation.get('factbook_draft')
        if not isinstance(draft, dict):
            raise NationStatesError('AI did not provide a factbook_draft object.')

        if not draft.get('requested'):
            raise NationStatesError('AI did not mark factbook_draft as requested.')

        if draft.get('pertinent'):
            if not str(draft.get('title', '')).strip():
                raise NationStatesError('AI factbook_draft is missing a title.')

            if not str(draft.get('text', '')).strip():
                raise NationStatesError('AI factbook_draft is missing text.')
        elif not str(draft.get('reason', '')).strip():
            raise NationStatesError(
                'AI factbook_draft must explain why no factbook entry is pertinent.'
            )


def fallback_issue_choice(
    live_issues: list[dict[str, Any]],
    strategy: str,
) -> dict[str, Any]:
    for issue in live_issues:
        if issue.get('options'):
            return {
                'issue_id': str(issue['issue_id']),
                'why_this_issue_first': (
                    'Deterministic fallback selected the first live issue with options. '
                    f'Player strategy was: {strategy}'
                ),
                'model': 'fallback',
                'fallback_issue_selection_used': True,
            }

    raise NationStatesError('No live issues with options were available.')


def fallback_issue_order(
    live_issues: list[dict[str, Any]],
    strategy: str,
) -> dict[str, Any]:
    ordered_issue_ids: list[str] = []
    reasons: dict[str, str] = {}

    for issue in live_issues:
        if not issue.get('options'):
            continue

        issue_id = str(issue['issue_id'])
        ordered_issue_ids.append(issue_id)
        reasons[issue_id] = (
            'Deterministic fallback kept the live issue order because AI ordering '
            f'was unavailable. Player strategy was: {strategy}'
        )

    if not ordered_issue_ids:
        raise NationStatesError('No live issues with options were available.')

    return {
        'ordered_issue_ids': ordered_issue_ids,
        'reasons': reasons,
        'model': 'fallback',
        'fallback_issue_order_used': True,
        'requires_review': True,
    }


def fallback_recommendation(
    live_issues: list[dict[str, Any]],
    strategy: str,
    *,
    draft_dispatch: bool = False,
    draft_factbook: bool = False,
) -> dict[str, Any]:
    for issue in live_issues:
        options = issue.get('options') or []
        if options:
            return {
                'issue_id': str(issue['issue_id']),
                'option_id': DISMISS_OPTION_ID,
                'action': 'dismiss',
                'issue_summary': (
                    f'{issue.get("title", "Untitled issue")}: '
                    f'{issue.get("text", "No issue text was provided.")}'
                ),
                'option_summaries': [
                    {
                        'option_id': str(option['option_id']),
                        'summary': str(option.get('text', '')),
                        'expected_effect': (
                            'Not evaluated because the local AI recommendation failed.'
                        ),
                    }
                    for option in options
                ],
                'dispatch_draft': {
                    **empty_dispatch_draft(requested=draft_dispatch),
                    'title': (
                        f'Administrative Notice: {issue.get("title", "NationStates Issue")}'
                        if draft_dispatch
                        else ''
                    ),
                    'text': (
                        'The local AI failed before it could draft a reviewed dispatch. '
                        'Do not publish this placeholder without manual rewriting.'
                        if draft_dispatch
                        else ''
                    ),
                    'category_hint': 'Bulletin',
                    'subcategory_hint': 'News',
                },
                'factbook_draft': {
                    **empty_factbook_draft(requested=draft_factbook),
                    'reason': (
                        'The local AI failed before it could evaluate whether a factbook '
                        'entry is pertinent.'
                        if draft_factbook
                        else ''
                    ),
                },
                'confidence': 0.25,
                'charter_alignment_score': 0,
                'red_line_triggered': True,
                'red_line_notes': [
                    'Fallback recommendation was used because the local AI failed.',
                ],
                'headline': 'Fallback Recommendation Selected',
                'why_this_issue_first': (
                    'This is a deterministic technical fallback, not a real policy judgment.'
                ),
                'reasoning': (
                    'The local AI failed to return a valid actionable recommendation. '
                    'The fallback summarizes the first issue with available options and '
                    'marks dismissal as the safest technical fallback. Review manually.'
                ),
                'expected_tradeoffs': [
                    'This is a safe technical fallback, not a thoughtful political recommendation.',
                    f'Player strategy was: {strategy}',
                ],
                'do_not_enact_if': [
                    'Do not dismiss or enact automatically until the AI returns a valid recommendation.',
                ],
                'audit_summary': (
                    f'Fallback selected dismissal for issue {issue["issue_id"]}.'
                ),
                'model': 'fallback',
                'option_label_basis': 'website_position',
                'fallback_recommendation_used': True,
                'requires_review': True,
            }

    raise NationStatesError('No live issues with options were available.')


def is_cached_advice_usable(
    cached_advice: CachedAdvice,
    *,
    valid_options: dict[str, set[str]],
    live_issues: list[dict[str, Any]] | None = None,
    selected_issue: dict[str, Any] | None = None,
    draft_dispatch: bool,
    draft_factbook: bool,
    no_ai: bool,
    auto_requested: bool = False,
    ai_step_statuses: list[dict[str, Any]] | None = None,
    minimum_confidence: float = 0.0,
    allow_fallback_auto: bool = False,
) -> tuple[bool, str]:
    recommendation = dict(cached_advice.recommendation)

    try:
        validate_recommendation(recommendation, valid_options)
        validate_publication_drafts(
            recommendation,
            draft_dispatch=draft_dispatch,
            draft_factbook=draft_factbook,
        )
    except NationStatesError as exc:
        return False, str(exc)

    return True, ''


def print_live_issues(live_issues: list[dict[str, Any]]) -> None:
    print_section_heading('Live NationStates Issues')

    for issue in live_issues:
        print_subheading(
            f'Issue {issue["issue_id"]}: {issue.get("title", "")}',
            indent=2,
        )

        issue_text = issue.get('text') or ''
        if issue_text:
            print_wrapped_block(issue_text, indent=4)
            print()

        for option in issue.get('options', []):
            print(f'    Option {option["option_id"]}')
            print_wrapped_block(option.get('text', ''), indent=6)
            print()


def print_recommendation(recommendation: dict[str, Any]) -> None:
    print_section_heading('AI Governor Recommendation')
    print_subheading('Decision')
    print_field('Headline', recommendation.get('headline', ''))
    print_field('Action', recommendation_action(recommendation))
    print_field('Issue ID', recommendation.get('issue_id', ''))
    option_id = str(recommendation.get('option_id', '')).strip()
    print_field('Option ID', option_id)
    print_field('Confidence', recommendation.get('confidence', ''))
    print_field('Alignment Score', recommendation.get('charter_alignment_score', ''))
    print_field('Red Line Hit', recommendation.get('red_line_triggered', ''))
    print_field('Model', recommendation.get('model', ''))

    issue_summary = recommendation.get('issue_summary', '')
    if issue_summary:
        print()
        print_subheading('Issue')
        print_field('Summary', issue_summary)

    why_first = recommendation.get('why_this_issue_first', '')
    reasoning = recommendation.get('reasoning', '')
    if why_first or reasoning:
        print()
        print_subheading('Reasoning')
        if why_first:
            print_field('Priority', why_first)
        if reasoning:
            print_field('Rationale', reasoning)

    option_summaries = recommendation.get('option_summaries') or []
    if option_summaries:
        print()
        print_subheading('Options')
        for item in option_summaries:
            if isinstance(item, dict):
                option = item.get('option_id', '')
                summary = item.get('summary', '')
                expected = item.get('expected_effect', '')
                print(f'    Option {option}')
                print_field('Summary', summary, indent=6)
                if expected:
                    print_field('Effect', expected, indent=6)
            else:
                print_bullets([item], indent=4)

    tradeoffs = recommendation.get('expected_tradeoffs') or []
    red_line_notes = recommendation.get('red_line_notes') or []
    blockers = recommendation.get('do_not_enact_if') or []
    if tradeoffs or red_line_notes or blockers:
        print()
        print_subheading('Guardrails')

    if tradeoffs:
        print('    Expected tradeoffs')
        print_bullets(tradeoffs, indent=6)
    if red_line_notes:
        print()
        print('    Red-line notes')
        print_bullets(red_line_notes, indent=6)
    if blockers:
        print()
        print('    Do not enact if')
        print_bullets(blockers, indent=6)

    print_publication_drafts(recommendation)


def print_publication_drafts(recommendation: dict[str, Any]) -> None:
    dispatch = recommendation.get('dispatch_draft')
    if isinstance(dispatch, dict) and dispatch.get('requested'):
        print_section_heading('Dispatch Draft')
        print_subheading('Metadata')
        print_field('Title', dispatch.get('title', ''))
        print_field('Category', dispatch.get('category_hint', ''))
        print_field('Subcategory', dispatch.get('subcategory_hint', ''))
        print()
        print_subheading('Text')
        print_wrapped_block(dispatch.get('text', ''), indent=4)

    factbook = recommendation.get('factbook_draft')
    if isinstance(factbook, dict) and factbook.get('requested'):
        print_section_heading('Factbook Draft')
        print_subheading('Metadata')
        print_field('Pertinent', factbook.get('pertinent', False))
        reason = str(factbook.get('reason', '')).strip()
        if reason:
            print_field('Reason', reason)

        if factbook.get('pertinent'):
            print_field('Title', factbook.get('title', ''))
            print_field('Category', factbook.get('category_hint', ''))
            print_field('Subcategory', factbook.get('subcategory_hint', ''))
            print()
            print_subheading('Text')
            print_wrapped_block(factbook.get('text', ''), indent=4)


def get_profile_min_confidence(profile: dict[str, Any] | None) -> float:
    if not profile:
        return 1.0

    value = profile.get('minimum_confidence_to_enact', 1.0)
    return max(0.0, min(1.0, maybe_float(value, 1.0)))


def get_profile_mode(profile: dict[str, Any] | None) -> str:
    if not profile:
        return 'advise_only'

    return str(profile.get('enactment_mode', 'advise_only')).strip()


def is_fallback_recommendation(recommendation: dict[str, Any]) -> bool:
    return (
        recommendation.get('headline') == 'Fallback Recommendation Selected'
        or recommendation.get('model') == 'fallback'
    )


def recommendation_action(recommendation: dict[str, Any]) -> str:
    action = str(recommendation.get('action', '')).strip().lower()
    if action in {'enact', 'dismiss'}:
        return action

    option_id = str(recommendation.get('option_id', '')).strip()
    return 'dismiss' if option_id == DISMISS_OPTION_ID else 'enact'


def is_dismiss_recommendation(recommendation: dict[str, Any]) -> bool:
    return recommendation_action(recommendation) == 'dismiss'


def should_auto_enact(
    *,
    profile: dict[str, Any] | None,
    recommendation: dict[str, Any],
    auto_requested: bool,
) -> tuple[bool, list[str]]:
    reasons = []

    if not auto_requested:
        return False, ['Auto mode was not requested.']

    if not profile:
        return False, ['Auto mode requires a governance profile.']

    if is_fallback_recommendation(recommendation):
        return False, ['Fallback recommendations may not be auto-enacted.']

    mode = get_profile_mode(profile)
    confidence = maybe_float(recommendation.get('confidence'), 0.0)
    minimum = get_profile_min_confidence(profile)
    red_line_triggered = bool(recommendation.get('red_line_triggered', False))

    if mode == 'advise_only':
        return False, ['Profile enactment mode is advise_only.']

    if red_line_triggered:
        return False, ['Recommendation triggered a red line.']

    if confidence < minimum:
        return False, [
            f'Confidence {confidence:.2f} is below profile minimum {minimum:.2f}.'
        ]

    if mode == 'auto_enact_high_confidence':
        required = max(minimum, 0.85)
        if confidence < required:
            return False, [
                f'auto_enact_high_confidence requires at least {required:.2f} confidence.'
            ]
        return True, ['Profile allows high-confidence auto-enactment.']

    if mode == 'auto_enact_unless_red_line':
        return True, ['Profile allows auto-enactment unless a red line is triggered.']

    if mode == 'fully_autonomous':
        return True, ['Profile is fully autonomous and validation passed.']

    return False, [f'Unknown enactment mode: {mode!r}.']


def should_manual_enact(
    *,
    recommendation: dict[str, Any],
    enact_requested: bool,
    override_red_line: bool,
) -> tuple[bool, list[str]]:
    if not enact_requested:
        return False, ['Manual enactment was not requested.']

    if is_fallback_recommendation(recommendation):
        return False, ['Fallback recommendations may not be enacted.']

    red_line_triggered = bool(recommendation.get('red_line_triggered', False))
    if red_line_triggered and not override_red_line:
        return False, [
            'Recommendation triggered a red line. Use --override-red-line if you truly mean it.'
        ]

    return True, ['Manual --enact requested and guardrails passed.']


__all__ = ['extract_live_issues', 'collect_issue_option_ids', 'validate_recommendation', 'empty_dispatch_draft', 'empty_factbook_draft', 'validate_publication_drafts', 'fallback_issue_choice', 'fallback_issue_order', 'fallback_recommendation', 'is_cached_advice_usable', 'print_live_issues', 'print_recommendation', 'print_publication_drafts', 'get_profile_min_confidence', 'get_profile_mode', 'is_fallback_recommendation', 'recommendation_action', 'is_dismiss_recommendation', 'should_auto_enact', 'should_manual_enact', 'validate_auto_action']
