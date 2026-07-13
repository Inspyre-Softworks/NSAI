"""
NationStates live AI governor/advisor with governance profile support.

This module serves as the primary orchestration layer. Domain logic is organized in:
- nsai.advisor.client — NationStates API client and XML helpers
- nsai.advisor.governor — Local LLM governor, prompt construction, and utilities
- nsai.advisor.recommendations — Issue extraction, validation, and display
- nsai.advisor.audit — Audit log read/write and publication tracking
- nsai.advisor.cli — CLI argument definitions and publication backfill

Author: Taylor B. | Inspyre-Softworks.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from rich.console import Console

from nsai.advisor.cache import AdviceCache, CachedAdvice, live_issue_by_id  # noqa: F401
from nsai.advisor.client import (  # noqa: F401
    NS_API_URL,
    NationStatesClient,
    NationStatesError,
    RateLimitState,
    build_default_user_agent,
    extract_private_command_token,
    maybe_int,
    xml_error_text,
    xml_to_string,
)
from nsai.advisor.governor import (  # noqa: F401
    DEFAULT_LM_API_KEY,
    DEFAULT_LM_BASE_URL,
    LOCAL_MODEL_RELOAD_MAX_ATTEMPTS,
    PROMPT_LIST_LIMIT,
    PROMPT_LONG_TEXT_LIMIT,
    PROMPT_TEXT_LIMIT,
    LocalGovernor,
    StatusPulse,
    build_governor_instruction,
    compact_live_issues_for_ai,
    compact_nation_context,
    compact_profile_for_ai,
    compact_prompt_list,
    compact_prompt_mapping,
    compact_prompt_text,
    extract_token_usage,
    get_completion_text,
    is_model_reload_error,
    jsonable,
    load_profile,
    maybe_float,
    parse_json_object,
    parse_json_object_with_repair,
    unique_reasons,
    wrapped,
    xml_local_name,
)
from nsai.advisor.recommendations import (  # noqa: F401
    DISMISS_OPTION_ID,
    collect_issue_option_ids,
    empty_dispatch_draft,
    empty_factbook_draft,
    extract_live_issues,
    fallback_issue_choice,
    fallback_recommendation,
    get_profile_min_confidence,
    get_profile_mode,
    is_cached_advice_usable,
    is_dismiss_recommendation,
    is_fallback_recommendation,
    print_live_issues,
    print_publication_drafts,
    print_recommendation,
    recommendation_action,
    should_auto_enact,
    should_manual_enact,
    validate_publication_drafts,
    validate_recommendation,
)
from nsai.advisor.audit import (  # noqa: F401
    append_publication_backfill_log,
    audit_issue_action_succeeded,
    load_audit_log_records,
    migrate_jsonl_audit_log,
    pending_publication_entries,
    posted_publication_keys,
    publication_source_key,
    write_audit_log,
)
from nsai.advisor.cli import (  # noqa: F401
    DEFAULT_AUDIT_LOG,
    DEFAULT_PUBLICATION_COOLDOWN_SECONDS,
    DEFAULT_STRATEGY,
    DISPATCH_CATEGORY_IDS,
    DISPATCH_SUBCATEGORY_IDS,
    FLAG_ASCII_RAMP,
    FLAG_ASCII_WIDTH,
    FLAG_DISPLAY_MODES,
    PUBLICATION_DEFAULTS,
    add_advise_arguments,
    add_publications_arguments,
    iter_pending_publication_drafts,
    make_progress,
    normalize_publication_hint,
    nation_flag_url,
    pending_publication_count,
    print_pending_publication_entries,
    print_nation_flag,
    print_publication_results,
    publication_category_name,
    publication_cooldown_hit,
    publication_hint_id,
    publication_result_record,
    publish_backfill_draft_with_retry,
    publish_one_publication_draft,
    publish_publication_drafts,
    render_ascii_flag,
    render_banner,
    resolve_bool_option,
    resolve_nation_name,
    resolve_publication_category,
    run_migrate_audit_log,
    run_publication_backfill,
    sleep_with_progress,
)
from nsai.advisor.safety import (  # noqa: F401
    ValidationResult,
    publication_mismatch_reasons,
    validate_auto_action,
    validate_recommendation_consistency,
)
from nsai.nations import (  # noqa: F401
    NationConfig,
    advice_cache_path,
    config_path_for,
    load_app_config,
    lm_api_key_credential_key_for,
    maybe_load_nation_config,
    normalize_nation_key,
    save_app_config,
    save_nation_config,
    saved_default_nation_config,
    store_profile_for_nation,
)
from nsai.secure_store import (  # noqa: F401
    SECRET_BACKENDS,
    SECRET_BACKEND_KEYRING,
    SecureStoreError,
    default_secret_backend,
    get_secret,
    set_secret,
)


LM_BASE_URL = os.environ.get('LM_STUDIO_BASE_URL', DEFAULT_LM_BASE_URL)
LM_MODEL = os.environ.get('LM_STUDIO_MODEL')


def run_advise(args: argparse.Namespace) -> None:
    save_opts = bool(getattr(args, 'save_opts', False))
    profile_path = Path(args.profile).expanduser().resolve() if args.profile else None
    cli_profile_path = profile_path
    profile = load_profile(profile_path) if profile_path else None
    nation_config = None
    automatic_nation_source = None

    try:
        nation = resolve_nation_name(cli_nation=args.nation, profile=profile)
    except SystemExit:
        if args.no_nation_config:
            raise

        nation_config, automatic_nation_source = saved_default_nation_config()
        if not nation_config:
            if automatic_nation_source:
                raise SystemExit(
                    f'Provide --nation because {automatic_nation_source}.'
                ) from None
            raise

        nation = nation_config.nation_name

    if not args.no_nation_config and nation_config is None:
        nation_config = maybe_load_nation_config(nation)

    if automatic_nation_source:
        print(f'Using saved nation {nation!r} from {automatic_nation_source}.')

    if nation_config:
        print(f'Using saved nation config: {config_path_for(nation_config.nation_name)}')

    if nation_config and not profile_path and nation_config.profile_path:
        profile_path = Path(nation_config.profile_path).expanduser().resolve()
        profile = load_profile(profile_path)
        print(f'Using saved profile for {nation}: {profile_path}')

    strategy = args.strategy or (nation_config.strategy if nation_config else None) or DEFAULT_STRATEGY
    show_issues = resolve_bool_option(
        args.show_issues,
        nation_config.show_issues if nation_config else None,
    )
    show_instruction = resolve_bool_option(
        args.show_instruction,
        nation_config.show_instruction if nation_config else None,
    )
    no_ai = resolve_bool_option(
        args.no_ai,
        nation_config.no_ai if nation_config else None,
    )
    audit_log = args.audit_log or (nation_config.audit_log if nation_config else None) or DEFAULT_AUDIT_LOG
    lm_base_url, lm_model, lm_api_key = resolve_lm_settings(args, nation_config)

    draft_dispatch = resolve_draft_request(
        args.draft_dispatch,
        nation_config.draft_dispatch if nation_config else None,
    )
    draft_factbook = resolve_draft_request(
        args.draft_factbook,
        nation_config.draft_factbook if nation_config else None,
    )
    flag_display = str(getattr(args, 'flag_display', 'ascii'))

    if show_instruction:
        print()
        print('Governor Instruction')
        print('=' * 88)
        print(build_governor_instruction(profile, strategy))
        print()

    ns = NationStatesClient.from_env(nation_config)

    if save_opts:
        _, _, saved_profile_path = save_advise_options(
            nation=nation,
            existing_config=nation_config,
            profile_path=profile_path,
            cli_profile_path=cli_profile_path,
            user_agent=ns.user_agent,
            api_version=ns.api_version,
            strategy=strategy,
            show_issues=show_issues,
            show_instruction=show_instruction,
            no_ai=no_ai,
            audit_log=audit_log,
            lm_base_url=lm_base_url,
            lm_model=lm_model,
            lm_api_key=args.lm_api_key,
            draft_dispatch=draft_dispatch,
            draft_factbook=draft_factbook,
            secret_backend=args.secret_backend,
        )
        if saved_profile_path:
            profile_path = saved_profile_path
        nation_config = maybe_load_nation_config(nation)

    with StatusPulse('NationStates: loading public nation data'):
        public_shards = [
            'fullname',
            'motto',
            'category',
            'region',
            'population',
            'freedom',
            'gdp',
            'tax',
            'crime',
            'govtdesc',
            'policies',
            'legislation',
        ]
        if flag_display != 'none':
            public_shards.insert(1, 'flag')
        nation_root = ns.public_nation(nation, public_shards)

    print_nation_flag(nation_root, flag_display=flag_display)

    with StatusPulse('NationStates: loading live issues'):
        issues_root = ns.issues(nation)
    live_issues = extract_live_issues(issues_root)

    if not live_issues:
        raise SystemExit('No live issues found for this nation.')

    valid_options = collect_issue_option_ids(live_issues)

    if show_issues:
        print_live_issues(live_issues)

    cache = AdviceCache()
    governor: LocalGovernor | None = None
    nation_snapshot_xml = xml_to_string(nation_root)

    def get_governor() -> LocalGovernor:
        nonlocal governor
        if governor is None:
            governor = LocalGovernor(
                base_url=lm_base_url,
                model=lm_model,
                api_key=lm_api_key,
            )
            print(f'Using local model: {governor.model} at {governor.base_url}')
        return governor

    if args.refresh_advice:
        print(f'Refreshing advice cache for this run: {advice_cache_path()}')

    selected_issue_id = ''
    selected_issue_reason = ''
    selected_issue_source = ''
    selected_issue_token_usage: dict[str, Any] = {}
    recommendation: dict[str, Any] | None = None
    ai_step_statuses: list[dict[str, Any]] = []
    fallback_issue_selection_used = False

    def record_ai_step(
        step: str,
        status: str,
        *,
        fallback_used: bool = False,
        error: str | None = None,
        structured_output_repaired: bool = False,
        source: str | None = None,
    ) -> None:
        ai_step_statuses.append({
            'step': step,
            'status': status,
            'fallback_used': fallback_used,
            'error': error,
            'structured_output_repaired': structured_output_repaired,
            'source': source,
        })

    if not args.refresh_advice:
        cached_choice = cache.get_issue_choice(nation, live_issues)
        if cached_choice and live_issue_by_id(live_issues, cached_choice.selected_issue_id):
            selected_issue_id = cached_choice.selected_issue_id
            selected_issue_reason = cached_choice.why
            selected_issue_source = 'cache'
            cached_fallback = cached_choice.source == 'fallback'
            fallback_issue_selection_used = cached_fallback
            print(
                'Using cached issue choice for current issue set: '
                f'{selected_issue_id}'
            )
            print('AI step skipped: reused cached issue selection.')
            if cached_fallback:
                print('fallback_issue_selection_used: true')
                print(
                    'Cached issue selection came from deterministic fallback. '
                    'Auto action disabled; manual review required.'
                )
            record_ai_step(
                'issue_selection',
                'skipped',
                fallback_used=cached_fallback,
                source='cache',
            )

    if not selected_issue_id:
        if len(live_issues) == 1:
            selected_issue_id = str(live_issues[0]['issue_id'])
            selected_issue_reason = 'Only one live issue is present.'
            selected_issue_source = 'single_issue'
            print('AI step skipped: only one live issue is present.')
            record_ai_step('issue_selection', 'skipped', source='single_issue')
        elif no_ai:
            print('AI step skipped: --no-ai is set; using deterministic issue selection.')
            selection = fallback_issue_choice(live_issues, strategy)
            selected_issue_id = str(selection['issue_id'])
            selected_issue_reason = str(selection.get('why_this_issue_first', ''))
            selected_issue_source = 'fallback'
            selected_issue_token_usage = dict(selection.get('token_usage') or {})
            fallback_issue_selection_used = True
            record_ai_step(
                'issue_selection',
                'skipped',
                fallback_used=True,
                source='no_ai',
            )
        else:
            try:
                selection = get_governor().select_issue(
                    nation_snapshot_xml=nation_snapshot_xml,
                    live_issues=live_issues,
                    strategy=strategy,
                    profile=profile,
                )
            except Exception as exc:
                print(f'[Local AI issue selection failed. Using fallback: {exc}]')
                selection = fallback_issue_choice(live_issues, strategy)
                selection.update({
                    'ai_step_failed': True,
                    'ai_step_error': str(exc),
                    'fallback_issue_selection_used': True,
                    'requires_review': True,
                })

            selected_issue_id = str(selection['issue_id'])
            selected_issue_reason = str(selection.get('why_this_issue_first', ''))
            selected_issue_token_usage = dict(selection.get('token_usage') or {})
            selected_issue_source = (
                'fallback'
                if str(selection.get('model', '')).lower() == 'fallback'
                else 'ai'
            )
            fallback_issue_selection_used = (
                selected_issue_source == 'fallback'
                or bool(selection.get('fallback_issue_selection_used'))
            )
            if fallback_issue_selection_used:
                print('fallback_issue_selection_used: true')
                if selection.get('ai_step_failed'):
                    print(
                        'Issue selection fallback used due to model failure. '
                        'Auto action disabled; manual review required.'
                    )

            record_ai_step(
                'issue_selection',
                'failed' if selection.get('ai_step_failed') else 'ok',
                fallback_used=fallback_issue_selection_used,
                error=(
                    str(selection.get('ai_step_error'))
                    if selection.get('ai_step_error')
                    else None
                ),
                structured_output_repaired=bool(
                    selection.get('structured_output_repaired')
                ),
                source=selected_issue_source,
            )

        cache.save_issue_choice(
            nation=nation,
            live_issues=live_issues,
            selected_issue_id=selected_issue_id,
            why=selected_issue_reason,
            source=selected_issue_source,
            token_usage=selected_issue_token_usage,
        )
        print(f'Saved issue choice for current issue set: {selected_issue_id}')

    selected_issue = live_issue_by_id(live_issues, selected_issue_id)
    if selected_issue is None:
        raise NationStatesError(
            f'Cached or selected issue {selected_issue_id!r} is not live anymore.'
        )

    if not args.refresh_advice:
        cached_advice = cache.get_advice(nation, selected_issue_id)
        if cached_advice:
            usable, reason = is_cached_advice_usable(
                cached_advice,
                valid_options=valid_options,
                live_issues=live_issues,
                selected_issue=selected_issue,
                draft_dispatch=draft_dispatch,
                draft_factbook=draft_factbook,
                no_ai=no_ai,
                auto_requested=args.auto,
                ai_step_statuses=ai_step_statuses,
                minimum_confidence=get_profile_min_confidence(profile),
                allow_fallback_auto=bool(
                    getattr(args, 'allow_fallback_auto', False)
                ),
            )
            if usable:
                recommendation = dict(cached_advice.recommendation)
                print(f'Using cached advice for issue {selected_issue_id}.')
                print('AI step skipped: reused cached recommendation.')
                record_ai_step(
                    'recommendation_generation',
                    'skipped',
                    fallback_used=(
                        cached_advice.source == 'fallback'
                        or is_fallback_recommendation(recommendation)
                    ),
                    structured_output_repaired=bool(
                        recommendation.get('structured_output_repaired')
                    ),
                    source='cache',
                )
            else:
                print(
                    f'Cached advice for issue {selected_issue_id} cannot be reused: '
                    f'{reason}'
                )

    if recommendation is None:
        selected_live_issues = [selected_issue]
        if no_ai:
            print('AI step skipped: --no-ai is set; using deterministic recommendation.')
            recommendation = fallback_recommendation(
                selected_live_issues,
                strategy,
                draft_dispatch=draft_dispatch,
                draft_factbook=draft_factbook,
            )
            record_ai_step(
                'recommendation_generation',
                'skipped',
                fallback_used=True,
                source='no_ai',
            )
        else:
            try:
                recommendation = get_governor().advise(
                    nation_snapshot_xml=nation_snapshot_xml,
                    live_issues=selected_live_issues,
                    strategy=strategy,
                    profile=profile,
                    draft_dispatch=draft_dispatch,
                    draft_factbook=draft_factbook,
                )
            except Exception as exc:
                print(f'[Local AI failed. Using deterministic fallback: {exc}]')
                recommendation = fallback_recommendation(
                    selected_live_issues,
                    strategy,
                    draft_dispatch=draft_dispatch,
                    draft_factbook=draft_factbook,
                )
                recommendation.update({
                    'ai_step_failed': True,
                    'ai_step_error': str(exc),
                    'requires_review': True,
                })

            if not no_ai:
                record_ai_step(
                    'recommendation_generation',
                    'failed' if recommendation.get('ai_step_failed') else 'ok',
                    fallback_used=is_fallback_recommendation(recommendation),
                    error=(
                        str(recommendation.get('ai_step_error'))
                        if recommendation.get('ai_step_error')
                        else None
                    ),
                    structured_output_repaired=bool(
                        recommendation.get('structured_output_repaired')
                    ),
                    source=(
                        'fallback'
                        if is_fallback_recommendation(recommendation)
                        else 'ai'
                    ),
                )

        cache.save_advice(
            nation=nation,
            live_issue=selected_issue,
            recommendation=recommendation,
            source='fallback' if is_fallback_recommendation(recommendation) else 'ai',
        )
        print(f'Saved advice for issue {recommendation.get("issue_id")}: {cache.path}')

    issue_id, option_id = validate_recommendation(recommendation, valid_options)
    print_recommendation(recommendation)

    manual_allowed, manual_reasons = should_manual_enact(
        recommendation=recommendation,
        enact_requested=args.enact,
        override_red_line=args.override_red_line,
    )

    manual_consistency = validate_recommendation_consistency(
        recommendation,
        live_issues=live_issues,
        selected_issue=selected_issue,
        draft_dispatch=draft_dispatch,
        draft_factbook=draft_factbook,
    )
    if args.enact and not manual_consistency.passed:
        manual_consistency_reasons = manual_consistency.reasons
        if args.override_red_line:
            manual_consistency_reasons = [
                reason
                for reason in manual_consistency_reasons
                if not reason.startswith('red_line_hit is true')
            ]
        if manual_consistency_reasons:
            manual_allowed = False
            manual_reasons = unique_reasons(manual_reasons + manual_consistency_reasons)

    auto_allowed, auto_reasons = should_auto_enact(
        profile=profile,
        recommendation=recommendation,
        auto_requested=args.auto,
    )

    allow_fallback_auto = bool(getattr(args, 'allow_fallback_auto', False))
    auto_safety_result = ValidationResult(True, [])
    auto_block_reasons: list[str] = []
    if args.auto:
        auto_safety_result = validate_auto_action(
            live_issues=live_issues,
            selected_issue=selected_issue,
            recommendation=recommendation,
            ai_step_statuses=ai_step_statuses,
            draft_dispatch=draft_dispatch,
            draft_factbook=draft_factbook,
            minimum_confidence=get_profile_min_confidence(profile),
            allow_fallback_auto=allow_fallback_auto,
        )
        if not auto_safety_result.passed:
            auto_allowed = False
            auto_block_reasons = unique_reasons(auto_safety_result.reasons)
            auto_reasons = unique_reasons(auto_reasons + auto_block_reasons)
            recommendation['requires_review'] = True
            recommendation['review_reasons'] = auto_block_reasons
            cache.save_advice(
                nation=nation,
                live_issue=selected_issue,
                recommendation=recommendation,
                source='fallback' if is_fallback_recommendation(recommendation) else 'ai',
            )

    should_enact = manual_allowed or auto_allowed

    if manual_allowed:
        action_reasons = manual_reasons
        action_mode = 'manual_enact'
    elif auto_allowed:
        action_reasons = auto_reasons
        action_mode = 'auto_enact'
    elif args.enact and args.auto:
        action_reasons = manual_reasons + auto_reasons
        action_mode = 'advisor_only'
    elif args.enact:
        action_reasons = manual_reasons
        action_mode = 'advisor_only'
    elif args.auto:
        action_reasons = auto_reasons
        action_mode = 'requires_review' if auto_block_reasons else 'advisor_only'
    else:
        action_reasons = manual_reasons + auto_reasons
        action_mode = 'advisor_only'

    action_reasons = unique_reasons(action_reasons)

    print()
    print('Action Decision')
    print('=' * 88)

    if should_enact:
        print(f'Will apply recommendation via: {action_mode}')
    else:
        print('Advisor mode only. No issue action was submitted.')

    for reason in action_reasons:
        print(f' - {reason}')

    if auto_block_reasons and not should_enact:
        print()
        print('AUTO ACTION BLOCKED')
        print('=' * 88)
        for reason in auto_block_reasons:
            print(f' - {reason}')
        print('Final decision: requires_review.')
        print('No NationStates issue action or publication will be submitted.')

    result_xml = None
    publication_results: list[dict[str, Any]] = []
    action_applied = False

    if should_enact:
        result = ns.answer_issue(nation, issue_id, option_id)
        result_xml = xml_to_string(result)
        result_error = xml_error_text(result)
        effects, headlines = cache.record_enactment(
            nation=nation,
            issue_id=issue_id,
            option_id=option_id,
            action=recommendation_action(recommendation),
            result_xml=result_xml,
        )

        print()
        if recommendation_action(recommendation) == 'dismiss':
            print('Issue dismissed.')
        else:
            print('Issue enacted.')
        print('=' * 88)
        print(result_xml)
        print(
            f'Cached enactment outcome: {len(effects)} effect record(s), '
            f'{len(headlines)} headline(s).'
        )

        if result_error:
            print()
            print(
                'Publication pages were not posted because NationStates returned '
                f'an issue-action error: {result_error}'
            )
        else:
            action_applied = True

        if action_applied and (draft_dispatch or draft_factbook):
            publication_results = publish_publication_drafts(
                ns,
                nation=nation,
                recommendation=recommendation,
                draft_dispatch=draft_dispatch,
                draft_factbook=draft_factbook,
                max_posts=1,
            )
            print_publication_results(publication_results)
    else:
        print()
        if is_fallback_recommendation(recommendation):
            print(
                'Fallback recommendations are review-only; use --refresh-advice '
                'with AI enabled before enacting.'
            )
        elif args.enact:
            print('Manual enactment was requested, but the guardrails above blocked it.')
        else:
            print('To manually apply this exact recommendation, run again with --enact.')

        if args.auto:
            print(
                'Auto mode was requested, but automatic action was blocked by '
                'the guardrails above.'
            )
        elif profile:
            print('To allow profile-controlled autonomy, run with --auto.')
            if save_opts:
                print(
                    '--save-opts does not persist --auto; pass --auto on each '
                    'run that should allow automatic action.'
                )
        if draft_dispatch or draft_factbook:
            print(
                'Publication drafts were not posted because no issue action was '
                'submitted.'
            )

    write_audit_log(
        Path(audit_log),
        nation=nation,
        profile_path=profile_path,
        profile=profile,
        recommendation=recommendation,
        action=action_mode,
        action_reasons=action_reasons,
        result_xml=result_xml,
        publication_results=publication_results,
        action_applied=action_applied,
        blocked=bool(auto_block_reasons and not should_enact),
        block_reasons=auto_block_reasons if not should_enact else [],
        ai_step_statuses=ai_step_statuses,
        fallback_issue_selection_used=fallback_issue_selection_used,
    )

    print()
    print(f'Audit log updated: {audit_log}')


def save_advise_options(
    *,
    nation: str,
    existing_config: NationConfig | None,
    profile_path: Path | None,
    cli_profile_path: Path | None,
    user_agent: str,
    api_version: int | None,
    strategy: str,
    show_issues: bool,
    show_instruction: bool,
    no_ai: bool,
    audit_log: str,
    lm_base_url: str,
    lm_model: str | None,
    lm_api_key: str | None,
    draft_dispatch: bool,
    draft_factbook: bool,
    secret_backend: str | None = None,
) -> tuple[Path, Path, Path | None]:
    saved_profile_path = profile_path
    if cli_profile_path:
        saved_profile_path, profile_action = store_profile_for_nation(
            nation,
            cli_profile_path,
            move=False,
        )
        if profile_action == 'already-managed':
            print(f'Profile already managed by NSAI: {saved_profile_path}')
        else:
            print(f'Profile {profile_action} into NSAI storage: {saved_profile_path}')

    config = NationConfig(
        nation_name=nation,
        user_agent=user_agent,
        api_version=api_version,
        profile_path=str(saved_profile_path) if saved_profile_path else None,
        strategy=strategy,
        show_issues=show_issues,
        show_instruction=show_instruction,
        no_ai=no_ai,
        audit_log=audit_log,
        lm_base_url=lm_base_url,
        lm_model=lm_model,
        lm_api_key_credential_key=(
            lm_api_key_credential_key_for(nation)
            if lm_api_key is not None
            else (existing_config.lm_api_key_credential_key if existing_config else None)
        ),
        lm_api_key_backend=(
            secret_backend or default_secret_backend()
            if lm_api_key is not None
            else (existing_config.lm_api_key_backend if existing_config else None)
        ),
        draft_dispatch=draft_dispatch,
        draft_factbook=draft_factbook,
        auth_kind=existing_config.auth_kind if existing_config else None,
        credential_key=existing_config.credential_key if existing_config else None,
        credential_backend=existing_config.credential_backend if existing_config else None,
    )
    if lm_api_key is not None and config.lm_api_key_credential_key:
        set_secret(
            config.lm_api_key_credential_key,
            lm_api_key,
            backend=config.lm_api_key_backend,
            reason=f'Store LM API key for {nation}',
        )

    nation_config_path = save_nation_config(config)

    app_config = load_app_config()
    app_config.default_nation = nation
    program_config_path = save_app_config(app_config)

    print(f'Saved advisor options for {nation}: {nation_config_path}')
    print(f'Saved default nation in program config: {program_config_path}')
    print(
        'Safety flags were not saved; pass --enact or --auto on each run that '
        'should submit an issue action.'
    )
    return nation_config_path, program_config_path, saved_profile_path


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_advise(args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='AI-assisted NationStates live governor/advisor.'
    )
    add_advise_arguments(parser)
    return parser


def resolve_draft_request(
    cli_value: bool | None,
    nation_value: bool | None,
) -> bool:
    if cli_value is not None:
        return cli_value

    return bool(nation_value)


def resolve_lm_settings(
    args: argparse.Namespace,
    nation_config: NationConfig | None,
) -> tuple[str, str | None, str]:
    base_url = (
        args.base_url
        or os.environ.get('LM_STUDIO_BASE_URL')
        or (nation_config.lm_base_url if nation_config else None)
        or DEFAULT_LM_BASE_URL
    )
    model = (
        args.model
        or os.environ.get('LM_STUDIO_MODEL')
        or (nation_config.lm_model if nation_config else None)
    )
    api_key = args.lm_api_key or os.environ.get('LM_STUDIO_API_KEY')

    if (
        not api_key
        and nation_config
        and nation_config.lm_api_key_credential_key
    ):
        try:
            api_key = get_secret(
                nation_config.lm_api_key_credential_key,
                backend=nation_config.lm_api_key_backend or SECRET_BACKEND_KEYRING,
                reason=f'Unlock LM API key for {nation_config.nation_name}',
            )
        except SecureStoreError as exc:
            raise NationStatesError(str(exc)) from exc

        if not api_key:
            raise NationStatesError(
                f'Saved config for {nation_config.nation_name} references '
                'a missing LM API key secret: '
                f'{nation_config.lm_api_key_credential_key}'
            )

    return base_url, model, api_key or DEFAULT_LM_API_KEY
