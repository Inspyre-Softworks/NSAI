"""Live advisor command-line argument wiring, options resolution, and orchestration."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image, UnidentifiedImageError
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from nsai.advisor.audit import (
    append_publication_backfill_log,
    load_audit_log_records,
    pending_publication_entries,
    write_audit_log,
)
from nsai.advisor.cache import AdviceCache, CachedAdvice, live_issue_by_id
from nsai.advisor.client import (
    NationStatesClient,
    NationStatesError,
    xml_error_text,
    xml_to_string,
)
from nsai.advisor.governor import (
    DEFAULT_LM_API_KEY,
    DEFAULT_LM_BASE_URL,
    LocalGovernor,
    StatusPulse,
    build_governor_instruction,
    extract_token_usage,
    load_profile,
    unique_reasons,
)
from nsai.advisor.recommendations import (
    DISMISS_OPTION_ID,
    collect_issue_option_ids,
    empty_dispatch_draft,
    empty_factbook_draft,
    extract_live_issues,
    fallback_issue_choice,
    fallback_recommendation,
    get_profile_min_confidence,
    get_profile_mode,
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
from nsai.advisor.safety import (
    ValidationResult,
    publication_mismatch_reasons,
    validate_auto_action,
    validate_recommendation_consistency,
)
from nsai.nations import (
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
from nsai.secure_store import (
    SECRET_BACKENDS,
    SECRET_BACKEND_KEYRING,
    SecureStoreError,
    default_secret_backend,
    get_secret,
    set_secret,
)


DEFAULT_STRATEGY = (
    'Keep the nation prosperous, socially stable, technologically advanced, '
    'not authoritarian, and avoid absurdly destructive policies.'
)
DEFAULT_AUDIT_LOG = 'ns_governor_audit.jsonl'
DEFAULT_PUBLICATION_COOLDOWN_SECONDS = 300.0
FLAG_DISPLAY_MODES = {'ascii', 'banner', 'none'}
FLAG_ASCII_WIDTH = 42
FLAG_ASCII_RAMP = '@%#*+=-:. '
DISPATCH_CATEGORY_IDS = {
    'factbook': 1,
    'bulletin': 3,
    'account': 5,
    'meta': 8,
}
DISPATCH_SUBCATEGORY_IDS = {
    'factbook': {
        'overview': 100,
        'history': 101,
        'geography': 102,
        'culture': 103,
        'politics': 104,
        'legislation': 105,
        'law': 105,
        'laws': 105,
        'legal': 105,
        'religion': 106,
        'military': 107,
        'defense': 107,
        'defence': 107,
        'economy': 108,
        'economic': 108,
        'finance': 108,
        'international': 109,
        'foreign affairs': 109,
        'diplomacy': 109,
        'trivia': 110,
        'miscellaneous': 111,
        'misc': 111,
    },
    'bulletin': {
        'policy': 305,
        'news': 315,
        'opinion': 325,
        'campaign': 335,
    },
    'account': {
        'military': 505,
    },
    'meta': {},
}
PUBLICATION_DEFAULTS = {
    'dispatch': ('bulletin', 315),
    'factbook': ('factbook', 111),
}


def normalize_publication_hint(value: Any) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', str(value or '').lower()).strip()


def publication_category_name(category_id: int, fallback: str) -> str:
    for name, value in DISPATCH_CATEGORY_IDS.items():
        if value == category_id:
            return name

    return fallback


def publication_hint_id(value: Any) -> int | None:
    text = str(value or '').strip()
    if text.isdigit():
        return int(text)

    return None


def resolve_publication_category(
    draft: dict[str, Any],
    *,
    kind: str,
) -> tuple[int, int]:
    default_category_name, default_subcategory = PUBLICATION_DEFAULTS[kind]
    category_hint = draft.get('category_hint')
    subcategory_hint = draft.get('subcategory_hint')
    combined_hint = normalize_publication_hint(
        f'{category_hint or ""} {subcategory_hint or ""}'
    )

    category_id = publication_hint_id(category_hint)
    if category_id is None:
        normalized_category = normalize_publication_hint(category_hint)
        if normalized_category in DISPATCH_CATEGORY_IDS:
            category_name = normalized_category
            category_id = DISPATCH_CATEGORY_IDS[category_name]
        else:
            category_name = default_category_name
            category_id = DISPATCH_CATEGORY_IDS[category_name]
    else:
        category_name = publication_category_name(category_id, default_category_name)

    if not category_name:
        category_name = default_category_name

    subcategory_id = publication_hint_id(subcategory_hint)
    if subcategory_id is not None:
        return category_id, subcategory_id

    subcategories = DISPATCH_SUBCATEGORY_IDS.get(category_name, {})
    normalized_subcategory = normalize_publication_hint(subcategory_hint)
    if normalized_subcategory in subcategories:
        return category_id, subcategories[normalized_subcategory]

    for name, value in subcategories.items():
        if name in combined_hint:
            return category_id, value

    return category_id, default_subcategory


def publication_result_record(
    *,
    kind: str,
    draft: dict[str, Any],
    category: int,
    subcategory: int,
    status: str,
    result_xml: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        'kind': kind,
        'title': str(draft.get('title', '')).strip(),
        'category': category,
        'subcategory': subcategory,
        'status': status,
        'result_xml': result_xml,
        'error': error,
    }


def publish_one_publication_draft(
    ns: NationStatesClient,
    *,
    nation: str,
    kind: str,
    draft: dict[str, Any],
) -> dict[str, Any]:
    category, subcategory = resolve_publication_category(draft, kind=kind)

    try:
        root = ns.create_dispatch(
            nation,
            title=str(draft.get('title', '')).strip(),
            text=str(draft.get('text', '')).strip(),
            category=category,
            subcategory=subcategory,
        )
        result_xml = xml_to_string(root)
        error = xml_error_text(root)
        if error:
            return publication_result_record(
                kind=kind,
                draft=draft,
                category=category,
                subcategory=subcategory,
                status='failed',
                result_xml=result_xml,
                error=error,
            )

        return publication_result_record(
            kind=kind,
            draft=draft,
            category=category,
            subcategory=subcategory,
            status='posted',
            result_xml=result_xml,
        )
    except Exception as exc:
        return publication_result_record(
            kind=kind,
            draft=draft,
            category=category,
            subcategory=subcategory,
            status='failed',
            error=str(exc),
        )


def publication_cooldown_hit(result: dict[str, Any]) -> bool:
    error = str(result.get('error') or '').lower()
    return (
        'many announcements' in error
        or 'press to catch their breath' in error
    )


def publish_publication_drafts(
    ns: NationStatesClient,
    *,
    nation: str,
    recommendation: dict[str, Any],
    draft_dispatch: bool,
    draft_factbook: bool,
    max_posts: int | None = None,
) -> list[dict[str, Any]]:
    results = []
    posted_count = 0
    mismatch_reasons = publication_mismatch_reasons(
        recommendation,
        draft_dispatch=draft_dispatch,
        draft_factbook=draft_factbook,
    )

    if mismatch_reasons:
        error = '; '.join(mismatch_reasons)
        if draft_dispatch:
            dispatch = recommendation.get('dispatch_draft')
            if isinstance(dispatch, dict) and dispatch.get('requested'):
                category, subcategory = resolve_publication_category(
                    dispatch,
                    kind='dispatch',
                )
                results.append(publication_result_record(
                    kind='dispatch',
                    draft=dispatch,
                    category=category,
                    subcategory=subcategory,
                    status='blocked',
                    error=error,
                ))

        if draft_factbook:
            factbook = recommendation.get('factbook_draft')
            if (
                isinstance(factbook, dict)
                and factbook.get('requested')
                and factbook.get('pertinent')
            ):
                category, subcategory = resolve_publication_category(
                    factbook,
                    kind='factbook',
                )
                results.append(publication_result_record(
                    kind='factbook',
                    draft=factbook,
                    category=category,
                    subcategory=subcategory,
                    status='blocked',
                    error=error,
                ))

        return results

    if draft_dispatch:
        dispatch = recommendation.get('dispatch_draft')
        if isinstance(dispatch, dict) and dispatch.get('requested'):
            result = (
                publish_one_publication_draft(
                    ns,
                    nation=nation,
                    kind='dispatch',
                    draft=dispatch,
                )
            )
            results.append(result)
            if result.get('status') == 'posted':
                posted_count += 1
            if publication_cooldown_hit(result) or (
                max_posts is not None and posted_count >= max_posts
            ):
                return results

    if draft_factbook:
        factbook = recommendation.get('factbook_draft')
        if (
            isinstance(factbook, dict)
            and factbook.get('requested')
            and factbook.get('pertinent')
        ):
            result = (
                publish_one_publication_draft(
                    ns,
                    nation=nation,
                    kind='factbook',
                    draft=factbook,
                )
            )
            results.append(result)

    return results


def print_publication_results(
    results: list[dict[str, Any]],
    *,
    console: Console | None = None,
) -> None:
    if not results:
        return

    output = console or Console()
    output.print()
    output.print('Publication Results')
    output.print('=' * 88)
    for result in results:
        output.print(
            f'{str(result["kind"]).title()}: {result["status"]} '
            f'({result["category"]}/{result["subcategory"]})'
        )
        output.print(f'Title: {result["title"]}')
        if result.get('error'):
            output.print(f'Error: {result["error"]}')


def pending_publication_count(entry: dict[str, Any]) -> int:
    return int(bool(entry.get('draft_dispatch'))) + int(bool(entry.get('draft_factbook')))


def iter_pending_publication_drafts(
    entry: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    recommendation = entry['record'].get('recommendation') or {}
    drafts: list[tuple[str, dict[str, Any]]] = []

    if entry.get('draft_dispatch'):
        dispatch = recommendation.get('dispatch_draft')
        if isinstance(dispatch, dict):
            drafts.append(('dispatch', dispatch))

    if entry.get('draft_factbook'):
        factbook = recommendation.get('factbook_draft')
        if isinstance(factbook, dict):
            drafts.append(('factbook', factbook))

    return drafts


def make_progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn('{task.description}'),
        BarColumn(),
        TextColumn('{task.completed:.0f}/{task.total:.0f}'),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def sleep_with_progress(
    seconds: float,
    *,
    description: str,
    console: Console,
) -> None:
    if seconds <= 0:
        return

    total = max(1, int(round(seconds)))
    with make_progress(console) as progress:
        task = progress.add_task(description, total=total)
        for _ in range(total):
            time.sleep(1)
            progress.advance(task)


def publish_backfill_draft_with_retry(
    ns: NationStatesClient,
    *,
    nation: str,
    kind: str,
    draft: dict[str, Any],
    cooldown_seconds: float,
    cooldown_retries: int,
    console: Console,
) -> dict[str, Any]:
    attempts = 0

    while True:
        result = publish_one_publication_draft(
            ns,
            nation=nation,
            kind=kind,
            draft=draft,
        )
        if not publication_cooldown_hit(result) or attempts >= cooldown_retries:
            return result

        attempts += 1
        sleep_with_progress(
            cooldown_seconds,
            description=(
                f'NationStates cooldown for {nation}; retrying '
                f'{kind}'
            ),
            console=console,
        )


def print_pending_publication_entries(entries: list[dict[str, Any]]) -> None:
    if not entries:
        print('No pending enacted publication drafts found.')
        return

    print('Pending Publication Backfill')
    print('=' * 88)
    for entry in entries:
        record = entry['record']
        recommendation = record.get('recommendation') or {}
        kinds = []
        if entry['draft_dispatch']:
            kinds.append('dispatch')
        if entry['draft_factbook']:
            kinds.append('factbook')

        print(
            f'Line {entry["line"]}: {record.get("nation")} '
            f'issue {recommendation.get("issue_id")} option {recommendation.get("option_id")} '
            f'[{", ".join(kinds)}]'
        )
        print(f'Headline: {recommendation.get("headline", "")}')
        if entry['draft_dispatch']:
            print(f'Dispatch: {(recommendation.get("dispatch_draft") or {}).get("title", "")}')
        if entry['draft_factbook']:
            print(f'Factbook: {(recommendation.get("factbook_draft") or {}).get("title", "")}')


def run_publication_backfill(args: argparse.Namespace) -> None:
    audit_path = Path(args.audit_log).expanduser().resolve()
    records = load_audit_log_records(audit_path)
    pending = pending_publication_entries(records, nation=args.nation)

    if args.limit is not None:
        pending = pending[:args.limit]

    cooldown_seconds = max(0.0, float(args.cooldown_seconds))
    cooldown_retries = max(0, int(args.cooldown_retries))

    print_pending_publication_entries(pending)

    if not pending:
        return

    if not args.execute:
        print()
        print('Preview only. Re-run with --execute to publish these pages.')
        return

    print()
    print('Publishing pending pages...')
    print('=' * 88)
    console = Console()
    clients: dict[str, NationStatesClient] = {}
    last_post_attempt_at: dict[str, float] = {}
    total_pages = sum(pending_publication_count(entry) for entry in pending)

    with make_progress(console) as progress:
        task = progress.add_task('Publication backfill', total=total_pages)
        for entry in pending:
            record = entry['record']
            nation = str(record.get('nation') or '').strip()
            drafts = iter_pending_publication_drafts(entry)

            if not nation:
                console.print(f'Line {entry["line"]}: skipped; audit record has no nation.')
                progress.advance(task, len(drafts))
                continue

            nation_key = normalize_nation_key(nation)
            if nation_key not in clients:
                nation_config = maybe_load_nation_config(nation)
                clients[nation_key] = NationStatesClient.from_env(nation_config)
            ns = clients[nation_key]

            for kind, draft in drafts:
                last_attempt = last_post_attempt_at.get(nation_key)
                if last_attempt is not None:
                    elapsed = time.monotonic() - last_attempt
                    wait_seconds = max(0.0, cooldown_seconds - elapsed)
                    if wait_seconds > 0:
                        progress.stop()
                        sleep_with_progress(
                            wait_seconds,
                            description=f'NationStates cooldown for {nation}',
                            console=console,
                        )
                        progress.start()

                progress.update(
                    task,
                    description=(
                        f'Publishing {kind} for {nation} '
                        f'(audit line {entry["line"]})'
                    ),
                )
                result = publish_backfill_draft_with_retry(
                    ns,
                    nation=nation,
                    kind=kind,
                    draft=draft,
                    cooldown_seconds=cooldown_seconds,
                    cooldown_retries=cooldown_retries,
                    console=console,
                )
                last_post_attempt_at[nation_key] = time.monotonic()
                progress.advance(task)
                print_publication_results([result], console=console)
                append_publication_backfill_log(
                    audit_path,
                    source_entry=record,
                    source_line=entry['line'],
                    publication_results=[result],
                )

        progress.update(task, description='Publication backfill complete')


def resolve_nation_name(
    *,
    cli_nation: str | None,
    profile: dict[str, Any] | None,
) -> str:
    if cli_nation:
        return cli_nation

    if profile and profile.get('nation_name'):
        return str(profile['nation_name'])

    env_nation = os.environ.get('NS_NATION')
    if env_nation:
        return env_nation

    raise SystemExit('Provide --nation, set NS_NATION, or use a profile with nation_name.')


def nation_flag_url(nation_root: ET.Element) -> str:
    return (nation_root.findtext('FLAG') or '').strip()


def render_ascii_flag(flag_url: str, *, width: int = FLAG_ASCII_WIDTH) -> str:
    response = requests.get(flag_url, timeout=15)
    response.raise_for_status()

    with Image.open(BytesIO(response.content)) as image:
        image = image.convert('RGBA')
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert('L')
        aspect = image.height / max(1, image.width)
        target_width = max(8, width)
        target_height = max(1, int(target_width * aspect * 0.5))
        image = image.resize((target_width, target_height))
        pixels = list(image.getdata())

    ramp = FLAG_ASCII_RAMP
    ramp_size = len(ramp) - 1
    rows = [
        ''.join(
            ramp[pixel * ramp_size // 255]
            for pixel in pixels[index:index + target_width]
        )
        for index in range(0, len(pixels), target_width)
    ]
    return '\n'.join(rows)


def render_banner(name: str) -> str:
    title = f' {name} '
    width = max(40, len(title) + 8)
    border = '=' * width
    return '\n'.join([border, title.center(width), border])


def print_nation_flag(
    nation_root: ET.Element,
    *,
    flag_display: str = 'ascii',
) -> None:
    if flag_display == 'none':
        return

    nation_name = (
        nation_root.findtext('FULLNAME') or nation_root.get('id') or 'Unknown nation'
    ).strip()
    flag_url = nation_flag_url(nation_root)
    if flag_display == 'banner':
        print()
        print(f'Found nation on NationStates: {nation_name}')
        print(render_banner(nation_name))
        if flag_url:
            print(f'Flag source: {flag_url}')
        return

    if not flag_url:
        return

    print()
    print(f'Found nation on NationStates: {nation_name}')
    try:
        print(render_ascii_flag(flag_url))
    except (
        requests.RequestException,
        OSError,
        UnidentifiedImageError,
        ValueError,
    ) as exc:
        print(f'Could not render nation flag as ASCII: {exc}')
        print(render_banner(nation_name))
        if flag_url:
            print(f'Flag source: {flag_url}')


def resolve_bool_option(
    cli_value: bool | None,
    nation_value: bool | None,
    default: bool = False,
) -> bool:
    if cli_value is not None:
        return cli_value

    if nation_value is not None:
        return bool(nation_value)

    return default


def add_advise_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--nation',
        default=None,
        help='Nation name. Overrides profile nation_name and NS_NATION.',
    )

    parser.add_argument(
        '--profile',
        help='Path to governance profile JSON from ns_profile_tui.py.',
    )

    parser.add_argument(
        '--strategy',
        default=None,
        help='Fallback strategy when no profile is provided.',
    )

    parser.add_argument(
        '--base-url',
        default=None,
        help='OpenAI-compatible local model base URL. Overrides saved config and LM_STUDIO_BASE_URL.',
    )

    parser.add_argument(
        '--model',
        default=None,
        help='OpenAI-compatible local model name. Overrides saved config and LM_STUDIO_MODEL.',
    )

    parser.add_argument(
        '--lm-api-key',
        default=None,
        help='OpenAI-compatible local model API key for this run. Saved with --save-opts using --secret-backend.',
    )

    parser.add_argument(
        '--secret-backend',
        choices=SECRET_BACKENDS,
        default=None,
        help=(
            'Secret backend for newly stored secrets with --save-opts. '
            'Defaults to windows-hello on Windows, keyring elsewhere.'
        ),
    )

    parser.add_argument(
        '--show-issues',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Print live issues before recommendation.',
    )

    parser.add_argument(
        '--show-instruction',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Print the final governor instruction sent to the AI.',
    )

    parser.add_argument(
        '--flag-display',
        choices=sorted(FLAG_DISPLAY_MODES),
        default='ascii',
        help='How to display the nation flag after loading the NationStates nation.',
    )

    parser.add_argument(
        '--draft-dispatch',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Ask the AI to draft dispatch title/text for the recommended action.',
    )

    parser.add_argument(
        '--draft-factbook',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Ask the AI to draft a factbook entry when the action is pertinent.',
    )

    parser.add_argument(
        '--no-ai',
        dest='no_ai',
        action='store_true',
        default=None,
        help='Skip local AI and use deterministic fallback recommendation.',
    )
    parser.add_argument(
        '--use-ai',
        dest='no_ai',
        action='store_false',
        help='Use local AI even if saved advisor defaults disable it.',
    )

    parser.add_argument(
        '--enact',
        action='store_true',
        help='Manually apply the recommended issue action after validation.',
    )

    parser.add_argument(
        '--auto',
        action='store_true',
        help='Allow profile-controlled auto-enactment according to enactment_mode.',
    )

    parser.add_argument(
        '--allow-fallback-auto',
        action='store_true',
        help=(
            'Dangerous: allow auto mode to continue after deterministic AI '
            'fallbacks or repaired structured output if all other safety checks pass.'
        ),
    )

    parser.add_argument(
        '--override-red-line',
        action='store_true',
        help='Allow manual --enact even when the AI marks a red-line violation.',
    )

    parser.add_argument(
        '--audit-log',
        default=None,
        help='Path to JSONL audit log.',
    )

    parser.add_argument(
        '--no-nation-config',
        action='store_true',
        help='Do not load saved per-nation config or secure credentials.',
    )

    parser.add_argument(
        '--refresh-advice',
        action='store_true',
        help='Ignore cached issue choice/advice and ask the advisor again.',
    )

    parser.add_argument(
        '--save-opts',
        action='store_true',
        default=argparse.SUPPRESS,
        help='Save current advisor options as this nation default; stores --lm-api-key only when provided.',
    )


def add_publications_arguments(subparsers: argparse._SubParsersAction) -> None:
    publications_parser = subparsers.add_parser(
        'publications',
        help='Publish missing dispatch/factbook pages from enacted audit records.',
    )
    publication_subparsers = publications_parser.add_subparsers(
        dest='publications_command',
    )

    backfill_parser = publication_subparsers.add_parser(
        'backfill',
        help='Find enacted recommendations with unposted publication drafts.',
    )
    backfill_parser.add_argument(
        '--audit-log',
        default=DEFAULT_AUDIT_LOG,
        help='Audit JSONL file to scan. Defaults to ns_governor_audit.jsonl.',
    )
    backfill_parser.add_argument(
        '--nation',
        default=None,
        help='Only backfill publications for this nation.',
    )
    backfill_parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Only process the first N pending audit entries.',
    )
    backfill_parser.add_argument(
        '--cooldown-seconds',
        type=float,
        default=DEFAULT_PUBLICATION_COOLDOWN_SECONDS,
        help=(
            'Seconds to wait between publication posts for the same nation. '
            f'Default: {DEFAULT_PUBLICATION_COOLDOWN_SECONDS:.0f}.'
        ),
    )
    backfill_parser.add_argument(
        '--cooldown-retries',
        type=int,
        default=1,
        help='How many times to retry a page after a NationStates publication cooldown error.',
    )
    backfill_parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually create the missing pages. Without this, only preview.',
    )
    backfill_parser.set_defaults(func=run_publication_backfill)


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


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_advise(args)


__all__ = [
    'FLAG_ASCII_RAMP',
    'FLAG_ASCII_WIDTH',
    'FLAG_DISPLAY_MODES',
    'add_advise_arguments',
    'add_publications_arguments',
    'build_arg_parser',
    'nation_flag_url',
    'main',
    'print_nation_flag',
    'render_ascii_flag',
    'render_banner',
    'resolve_nation_name',
    'run_advise',
    'run_publication_backfill',
    'save_advise_options',
    'publish_publication_drafts',
    'publish_backfill_draft_with_retry',
    'resolve_publication_category',
    'extract_token_usage',
]
