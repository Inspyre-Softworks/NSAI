"""Auto-profile builder: derive a governance profile from public nation stats."""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nsai.nations import (
    maybe_load_nation_config,
    normalize_nation_key,
    saved_default_nation_config,
)
from nsai.profile.storage import write_profile_json


AUTO_PROFILE_SHARDS = [
    'name',
    'fullname',
    'motto',
    'category',
    'govtdesc',
    'freedomscores',
    'govt',
    'policies',
    'sensibilities',
]

FREEDOM_SCORE_POLICY_AREAS = (
    ('CIVILRIGHTS', 'civil_rights'),
    ('ECONOMY', 'economy'),
    ('POLITICALFREEDOM', 'political_freedom'),
)

GOVT_SPENDING_POLICY_AREAS = {
    'COMMERCE': 'economy',
    'DEFENCE': 'military',
    'EDUCATION': 'education',
    'ENVIRONMENT': 'environment',
    'HEALTHCARE': 'healthcare',
    'INTERNATIONALAID': 'international_reputation',
    'LAWANDORDER': 'public_order',
    'PUBLICTRANSPORT': 'technology',
    'SOCIALEQUALITY': 'civil_rights',
    'SPIRITUALITY': 'religion',
    'WELFARE': 'worker_rights',
}

FREEDOM_PROTECTION_RED_LINES = {
    'civil_rights': 'Do not gut civil rights; they are part of the national character.',
    'economy': 'Do not strangle the existing strong economy with reckless controls.',
    'political_freedom': 'Do not undermine political freedom or suppress dissent.',
}
FREEDOM_PROTECTION_THRESHOLD = 60.0


def _quote_cli_arg(value: str | Path) -> str:
    text = str(value)
    if not text:
        return "''"
    if any(char.isspace() for char in text) or "'" in text:
        return "'" + text.replace("'", "''") + "'"
    return text


def _profile_file_arg_error(value: str) -> str | None:
    path = Path(value).expanduser()
    if path.suffix.lower() != '.json':
        return None

    display_path = _quote_cli_arg(path)
    if path.exists():
        return (
            '`nsai profile auto` builds a new profile from a NationStates '
            f'nation name, but {value!r} looks like an existing profile file.\n\n'
            'To inspect it, run:\n'
            f'  nsai profile preview {display_path}\n'
            'To enrich it, run:\n'
            f'  nsai profile enrich {display_path}\n'
            'To attach it to a saved nation, run:\n'
            f'  nsai nation set <nation> --profile {display_path}'
        )

    return (
        '`nsai profile auto` expects a NationStates nation name, not a profile '
        'JSON path.\n\n'
        'To choose where the generated profile is written, run:\n'
        f'  nsai profile auto <nation> --output {display_path}'
    )


def _text(root: ET.Element, tag: str) -> str:
    return (root.findtext(tag) or '').strip()


def _maybe_float(value: str | None) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def extract_freedom_scores(nation_root: ET.Element) -> dict[str, float]:
    node = nation_root.find('FREEDOMSCORES')
    if node is None:
        return {}

    scores = {}
    for tag, area in FREEDOM_SCORE_POLICY_AREAS:
        value = _maybe_float(node.findtext(tag))
        if value is not None:
            scores[area] = value

    return scores


def extract_spending_by_policy_area(nation_root: ET.Element) -> dict[str, float]:
    node = nation_root.find('GOVT')
    if node is None:
        return {}

    spending: dict[str, float] = {}
    for tag, area in GOVT_SPENDING_POLICY_AREAS.items():
        value = _maybe_float(node.findtext(tag))
        if value is not None and value > 0:
            spending[area] = spending.get(area, 0.0) + value

    return spending


def extract_policy_names(nation_root: ET.Element) -> list[str]:
    names = []
    for policy in nation_root.findall('.//POLICY'):
        name = _text(policy, 'NAME')
        if name:
            names.append(name)

    return names


def freedom_score_weight(score: float) -> int:
    return _clamp(round((score - 50.0) / 10.0), -5, 5)


def spending_weight(percent: float) -> int:
    return _clamp(round(percent / 4.0), 1, 5)


def build_auto_profile(
    nation_root: ET.Element,
    *,
    nation_fallback: str = '',
) -> dict[str, Any]:
    """Build a governance profile dict from a public nation snapshot."""

    name = _text(nation_root, 'NAME') or nation_fallback
    fullname = _text(nation_root, 'FULLNAME') or name
    category = _text(nation_root, 'CATEGORY')
    motto = _text(nation_root, 'MOTTO')
    govtdesc = _text(nation_root, 'GOVTDESC')
    sensibilities = _text(nation_root, 'SENSIBILITIES')

    freedom_scores = extract_freedom_scores(nation_root)
    spending = extract_spending_by_policy_area(nation_root)
    policy_names = extract_policy_names(nation_root)

    ranked_areas = [
        area
        for area, _ in sorted(spending.items(), key=lambda item: item[1], reverse=True)
    ]
    if not ranked_areas:
        ranked_areas = [
            area
            for area, _ in sorted(
                freedom_scores.items(), key=lambda item: item[1], reverse=True
            )
        ] or ['economy', 'public_order']

    top_priorities = ranked_areas[:5]
    secondary_priorities = ranked_areas[5:10]

    scoring_weights: dict[str, int] = {}
    for area in top_priorities + secondary_priorities:
        if area in spending:
            scoring_weights[area] = spending_weight(spending[area])
    for area, score in freedom_scores.items():
        scoring_weights[area] = freedom_score_weight(score)
    scoring_weights.setdefault('weirdness', -2)

    red_lines = [
        'Do not intentionally collapse the economy.',
        'Do not enact obviously joke-only policies.',
        'Do not sacrifice long-term stability for tiny short-term approval gains.',
    ]
    for area, score in freedom_scores.items():
        if score >= FREEDOM_PROTECTION_THRESHOLD:
            red_lines.append(FREEDOM_PROTECTION_RED_LINES[area])

    premise_parts = [f'{fullname} is a {category}.' if category else f'{fullname}.']
    if govtdesc:
        premise_parts.append(govtdesc)
    if motto:
        premise_parts.append(f'National motto: "{motto}".')
    if policy_names:
        premise_parts.append(f'Notable national policies: {", ".join(policy_names[:8])}.')

    top_priority_text = ', '.join(top_priorities) or 'its current strengths'
    lead_priority = top_priorities[0] if top_priorities else 'national stability'

    tone = 'direct and honest about tradeoffs'
    if sensibilities:
        tone = (
            f'measured, consistent with a {sensibilities.lower()} national '
            'character, and honest about tradeoffs'
        )

    generated_at = datetime.now(timezone.utc).isoformat()

    profile_name = f'Auto Profile: {category or name}'
    national_vision = (
        'Continue along the current national trajectory: strengthen '
        f'{top_priority_text} while preserving the established character '
        'of the nation.'
    )
    governing_style = (
        f'steady custodian of a {category.lower()}'
        if category
        else 'steady custodian of the current national character'
    )

    preferred_tradeoffs = [
        'Accept modest costs in lower-priority areas to protect '
        f'{lead_priority}.',
        'Accept incremental change that matches the existing national '
        'direction.',
    ]
    unacceptable_tradeoffs = [
        f'Do not sacrifice {lead_priority} for short-term gains elsewhere.',
        'Do not abruptly reverse the national character this profile was '
        'generated from.',
    ]

    return {
        'profile_name': profile_name,
        'nation_name': name,
        'created_at': generated_at,
        'roleplay_premise': ' '.join(premise_parts),
        'national_vision': national_vision,
        'governing_style': governing_style,
        'top_priorities': top_priorities,
        'secondary_priorities': secondary_priorities,
        'red_lines': red_lines,
        'preferred_tradeoffs': preferred_tradeoffs,
        'unacceptable_tradeoffs': unacceptable_tradeoffs,
        'risk_tolerance': 'balanced',
        'enactment_mode': 'advise_only',
        'minimum_confidence_to_enact': 0.85,
        'issue_selection_strategy': 'most_aligned_with_priorities',
        'tone': tone,
        'custom_instruction': (
            'This profile was generated automatically from public nation '
            'statistics. Govern the nation as it already is; when in doubt, '
            'prefer choices that reinforce its current identity.'
        ),
        'scoring_weights': scoring_weights,
        'auto_generated': {
            'source': 'nationstates-public-stats',
            'generated_at': generated_at,
            'category': category,
            'freedom_scores': freedom_scores,
            'spending_percentages': spending,
        },
    }


def add_auto_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        'nation',
        nargs='?',
        help=(
            'Nation to build the profile from. Defaults to NS_NATION or the '
            'saved default nation.'
        ),
    )
    parser.add_argument(
        '-o',
        '--output',
        help='Output path. Defaults to <nation>_auto_profile.json.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite the output file if it already exists.',
    )


def run_auto(args: argparse.Namespace) -> None:
    # Late import so the profile package does not pull advisor dependencies
    # unless auto-profile is actually used.
    from nsai.advisor.client import NationStatesClient

    if args.nation:
        profile_file_error = _profile_file_arg_error(args.nation)
        if profile_file_error:
            raise SystemExit(profile_file_error)

    nation = args.nation or os.environ.get('NS_NATION')
    nation_config = None
    if not nation:
        nation_config, source = saved_default_nation_config()
        if nation_config:
            nation = nation_config.nation_name
            print(f'Using saved nation {nation!r} from {source}.')

    if not nation:
        raise SystemExit(
            'Provide a nation name, set NS_NATION, or save a default nation first.'
        )

    if nation_config is None:
        nation_config = maybe_load_nation_config(nation)

    ns = NationStatesClient.from_env(nation_config)
    nation_root = ns.public_nation(nation, AUTO_PROFILE_SHARDS)
    profile = build_auto_profile(nation_root, nation_fallback=nation)

    output = (
        Path(args.output).expanduser()
        if args.output
        else Path(f'{normalize_nation_key(nation)}_auto_profile.json')
    )
    if output.exists() and not args.force:
        raise SystemExit(f'Output already exists (use --force to overwrite): {output}')

    write_profile_json(output, profile)
    print(f'Saved auto profile for {profile["nation_name"]}: {output}')
    output_arg = _quote_cli_arg(output)
    nation_arg = _quote_cli_arg(profile['nation_name'])
    print('Next steps:')
    print(f'  nsai profile preview {output_arg}')
    print(f'  nsai profile enrich {output_arg}')
    print(f'  nsai nation set {nation_arg} --profile {output_arg}')
