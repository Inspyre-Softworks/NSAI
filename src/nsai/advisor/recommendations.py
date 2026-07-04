"""Recommendation validation and enactment guardrails."""

from __future__ import annotations

from nsai.advisor.live import (
    collect_issue_option_ids,
    extract_live_issues,
    fallback_recommendation,
    get_profile_min_confidence,
    get_profile_mode,
    is_dismiss_recommendation,
    is_fallback_recommendation,
    recommendation_action,
    print_publication_drafts,
    print_live_issues,
    print_recommendation,
    validate_publication_drafts,
    should_auto_enact,
    should_manual_enact,
    validate_recommendation,
)
from nsai.advisor.safety import (
    ValidationResult,
    publication_mismatch_reasons,
    validate_auto_action,
    validate_recommendation_consistency,
)

__all__ = [
    'collect_issue_option_ids',
    'extract_live_issues',
    'fallback_recommendation',
    'get_profile_min_confidence',
    'get_profile_mode',
    'is_dismiss_recommendation',
    'is_fallback_recommendation',
    'recommendation_action',
    'print_publication_drafts',
    'print_live_issues',
    'print_recommendation',
    'validate_publication_drafts',
    'should_auto_enact',
    'should_manual_enact',
    'validate_recommendation',
    'ValidationResult',
    'publication_mismatch_reasons',
    'validate_auto_action',
    'validate_recommendation_consistency',
]
