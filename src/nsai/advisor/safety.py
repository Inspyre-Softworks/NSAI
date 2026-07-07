"""Fail-safe validation for live advisor auto actions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


DISMISS_OPTION_ID = '-1'
ALLOWED_ACTIONS = {'enact', 'dismiss'}
SUSPICIOUS_CONFIDENCE_THRESHOLD = 0.95


@dataclass(frozen=True)
class ValidationResult:
    """Structured validation outcome with all blocking reasons preserved."""

    passed: bool
    reasons: list[str]


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


def _raw_action(recommendation: dict[str, Any]) -> str:
    return _clean_text(recommendation.get('action')).lower()


def _issue_by_id(live_issues: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        _clean_text(issue.get('issue_id')): issue
        for issue in live_issues
        if _clean_text(issue.get('issue_id'))
    }


def _option_ids(issue: dict[str, Any] | None) -> set[str]:
    if not issue:
        return set()

    return {
        _clean_text(option.get('option_id'))
        for option in issue.get('options', [])
        if isinstance(option, dict) and _clean_text(option.get('option_id'))
    }


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y'}

    return bool(value)


def _parse_unit_interval(value: Any) -> tuple[float | None, str | None]:
    if value is None or isinstance(value, bool):
        return None, 'confidence is missing or malformed.'

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None, 'confidence is missing or malformed.'

    if not 0.0 <= parsed <= 1.0:
        return None, f'confidence {parsed!r} is outside 0.0..1.0.'

    return parsed, None


def _parse_score(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _alignment_score(recommendation: dict[str, Any]) -> float | None:
    if 'alignment_score' in recommendation:
        return _parse_score(recommendation.get('alignment_score'))

    return _parse_score(recommendation.get('charter_alignment_score'))


def _text_parts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]

    if isinstance(value, dict):
        parts: list[str] = []
        for item in value.values():
            parts.extend(_text_parts(item))
        return parts

    if isinstance(value, list | tuple | set):
        parts = []
        for item in value:
            parts.extend(_text_parts(item))
        return parts

    return []


def _recommendation_reasoning_text(recommendation: dict[str, Any]) -> str:
    keys = (
        'reasoning',
        'audit_summary',
        'summary',
        'headline',
        'why_this_issue_first',
        'expected_tradeoffs',
        'do_not_enact_if',
    )
    parts: list[str] = []
    for key in keys:
        parts.extend(_text_parts(recommendation.get(key)))
    return '\n'.join(parts).lower()


DISMISS_CONTRADICTION_PATTERNS = (
    re.compile(
        r'\bdismiss(?:ing|al|ed)?\b.{0,80}\b'
        r'(?:with|by|via|through|using)\s+option\s+[\w.-]+\b',
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r'\boption\s+[\w.-]+\b.{0,120}\b('
        r'best\s+aligns?|should\s+enact|best\s+option|'
        r'recommended\s+option|recommend(?:ed|s)?\s+enact'
        r')\b',
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r'\b('
        r'best\s+aligns?|should\s+enact|best\s+option|'
        r'recommended\s+option|recommend(?:ed|s)?\s+enact'
        r')\b.{0,120}\boption\s+[\w.-]+\b',
        re.IGNORECASE | re.DOTALL,
    ),
)


POLICY_ENACTMENT_PATTERNS = (
    re.compile(
        r'\b(?:government|nation|administration|ministry|state)\s+'
        r'(?:will|shall)\s+(?!not\b)\w+',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:will|shall)\s+(?!not\b)'
        r'(?:rebuild|implement|adopt|enact|approve|establish|create|fund|'
        r'regulate|ban|mandate|legalize|prohibit|launch|introduce|pass)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:has|have|had|was)\s+'
        r'(?:adopted|enacted|approved|implemented|established|created|passed)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:policy|program|framework|law|legislation|reform)\s+'
        r'(?:has been|will be|was)\s+'
        r'(?:adopted|enacted|implemented|approved|established|created|passed)\b',
        re.IGNORECASE,
    ),
)


def publication_mismatch_reasons(
    recommendation: dict[str, Any],
    *,
    draft_dispatch: bool,
    draft_factbook: bool,
) -> list[str]:
    """Return reasons why publication drafts contradict the action."""

    if _raw_action(recommendation) != 'dismiss':
        return []

    reasons = []
    requested_drafts: list[tuple[str, dict[str, Any]]] = []
    dispatch = recommendation.get('dispatch_draft')
    factbook = recommendation.get('factbook_draft')

    if draft_dispatch and isinstance(dispatch, dict) and dispatch.get('requested'):
        requested_drafts.append(('dispatch', dispatch))

    if (
        draft_factbook
        and isinstance(factbook, dict)
        and factbook.get('requested')
        and factbook.get('pertinent')
    ):
        requested_drafts.append(('factbook', factbook))

    for kind, draft in requested_drafts:
        text = '\n'.join(
            _clean_text(draft.get(key))
            for key in ('title', 'text', 'reason')
            if _clean_text(draft.get(key))
        )
        if any(pattern.search(text) for pattern in POLICY_ENACTMENT_PATTERNS):
            reasons.append(
                f'{kind} draft describes a concrete policy being enacted while '
                'the recommendation action is dismiss.'
            )

    return reasons


def validate_recommendation_consistency(
    recommendation: dict[str, Any],
    *,
    live_issues: list[dict[str, Any]],
    selected_issue: dict[str, Any] | None,
    draft_dispatch: bool = False,
    draft_factbook: bool = False,
) -> ValidationResult:
    """Validate a final recommendation before it is trusted for automation."""

    reasons: list[str] = []
    live_by_id = _issue_by_id(live_issues)
    issue_id = _clean_text(recommendation.get('issue_id'))
    selected_issue_id = _clean_text(selected_issue.get('issue_id')) if selected_issue else ''
    action = _raw_action(recommendation)
    option_value = recommendation.get('option_id')
    option_missing = option_value is None or _clean_text(option_value) == ''
    option_id = _clean_text(option_value)

    if action not in ALLOWED_ACTIONS:
        reasons.append(
            f'action {action!r} is not valid; expected one of {sorted(ALLOWED_ACTIONS)}.'
        )

    if issue_id not in live_by_id:
        reasons.append(
            f'selected issue_id {issue_id!r} is not one of the live issues: '
            f'{sorted(live_by_id)}.'
        )

    if selected_issue_id and issue_id and issue_id != selected_issue_id:
        reasons.append(
            f'recommendation issue_id {issue_id!r} does not match selected live '
            f'issue {selected_issue_id!r}.'
        )

    issue_for_options = live_by_id.get(issue_id) or selected_issue
    live_option_ids = _option_ids(issue_for_options)

    if action == 'enact':
        if option_missing:
            reasons.append('action enact requires a live option_id.')
        elif option_id not in live_option_ids:
            reasons.append(
                f'option_id {option_id!r} is not valid for selected issue '
                f'{issue_id!r}; live options are {sorted(live_option_ids)}.'
            )

    if action == 'dismiss' and not option_missing and option_id != DISMISS_OPTION_ID:
        reasons.append(
            f'action dismiss requires option_id {DISMISS_OPTION_ID!r} or None, '
            f'not {option_id!r}.'
        )

    confidence, confidence_error = _parse_unit_interval(recommendation.get('confidence'))
    if confidence_error:
        reasons.append(confidence_error)

    alignment = _alignment_score(recommendation)
    if alignment is not None and confidence is not None:
        if alignment <= 0 and confidence >= SUSPICIOUS_CONFIDENCE_THRESHOLD:
            reasons.append(
                f'confidence {confidence:.2f} is suspiciously high while '
                f'alignment_score is {alignment:g}.'
            )

    red_line_hit = (
        _truthy(recommendation.get('red_line_hit'))
        or _truthy(recommendation.get('red_line_triggered'))
    )
    if red_line_hit:
        reasons.append('red_line_hit is true; automatic action requires manual review.')

    if action == 'dismiss':
        reasoning_text = _recommendation_reasoning_text(recommendation)
        if any(pattern.search(reasoning_text) for pattern in DISMISS_CONTRADICTION_PATTERNS):
            reasons.append(
                'recommendation action is dismiss, but the reasoning recommends '
                'or praises a specific option.'
            )

    reasons.extend(
        publication_mismatch_reasons(
            recommendation,
            draft_dispatch=draft_dispatch,
            draft_factbook=draft_factbook,
        )
    )

    return ValidationResult(passed=not reasons, reasons=reasons)


def validate_auto_action(
    *,
    live_issues: list[dict[str, Any]],
    selected_issue: dict[str, Any] | None,
    recommendation: dict[str, Any],
    ai_step_statuses: list[dict[str, Any]],
    draft_dispatch: bool,
    draft_factbook: bool,
    minimum_confidence: float,
    allow_fallback_auto: bool = False,
) -> ValidationResult:
    """Validate every precondition before auto mode touches NationStates."""

    reasons: list[str] = []

    for status in ai_step_statuses:
        step = _clean_text(status.get('step')) or 'unknown'
        if status.get('status') == 'failed' and not allow_fallback_auto:
            detail = _clean_text(status.get('error'))
            suffix = f': {detail}' if detail else ''
            reasons.append(f'AI step failed during {step}{suffix}.')

        if status.get('fallback_used') and not allow_fallback_auto:
            reasons.append(
                f'deterministic fallback was used for {step}; auto action requires manual review.'
            )

        if status.get('structured_output_repaired') and not allow_fallback_auto:
            reasons.append(
                f'structured output was repaired or defaulted during {step}; '
                'auto action requires manual review.'
            )

    if recommendation.get('structured_output_repaired') and not allow_fallback_auto:
        reasons.append(
            'recommendation structured output was repaired or defaulted; '
            'auto action requires manual review.'
        )

    if recommendation.get('requires_review'):
        reasons.append('recommendation is marked requires_review.')

    consistency = validate_recommendation_consistency(
        recommendation,
        live_issues=live_issues,
        selected_issue=selected_issue,
        draft_dispatch=draft_dispatch,
        draft_factbook=draft_factbook,
    )
    reasons.extend(consistency.reasons)

    alignment = _alignment_score(recommendation)
    if _raw_action(recommendation) == 'enact' and alignment is not None and alignment <= 0:
        reasons.append(
            'auto action requires a positive alignment score for enacted options.'
        )

    confidence, confidence_error = _parse_unit_interval(recommendation.get('confidence'))
    if confidence_error is None and confidence is not None and confidence < minimum_confidence:
        reasons.append(
            f'confidence {confidence:.2f} is below profile minimum {minimum_confidence:.2f}.'
        )

    return ValidationResult(passed=not reasons, reasons=reasons)
