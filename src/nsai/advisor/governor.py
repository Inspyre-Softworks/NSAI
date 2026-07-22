"""Local LLM governor for AI-assisted NationStates issue recommendations."""

from __future__ import annotations

import json
import os
import re
import textwrap
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from openai import OpenAI
from rich.console import Console

from nsai.advisor.client import (
    NationStatesError,
    xml_to_string,
)
from nsai.advisor.trace import print_api_trace


DEFAULT_LM_BASE_URL = 'http://localhost:1234/v1'
DEFAULT_LM_API_KEY = 'lm-studio'
DEFAULT_LM_REQUEST_TIMEOUT_SECONDS = 1800.0
LOCAL_MODEL_RELOAD_MAX_ATTEMPTS = 2
PROMPT_TEXT_LIMIT = 900
PROMPT_LONG_TEXT_LIMIT = 1800
PROMPT_LIST_LIMIT = 8


class StatusPulse:
    """Show a Rich spinner for blocking local/API calls."""

    def __init__(self, label: str, *, console: Console | None = None) -> None:
        self.label = label
        self.console = console or Console()
        self.started_at = 0.0
        self._status: Any = None

    def __enter__(self) -> 'StatusPulse':
        self.started_at = time.monotonic()
        if self.console.is_terminal:
            self._status = self.console.status(
                f'{self.label}...',
                spinner='dots',
            )
            self._status.start()
        else:
            self.console.print(f'{self.label}...')

        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._status:
            self._status.stop()

        elapsed = time.monotonic() - self.started_at
        if exc_type is None:
            self.console.print(f'{self.label} complete in {elapsed:.1f}s.')
        else:
            self.console.print(f'{self.label} failed after {elapsed:.1f}s.')


def maybe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def local_model_request_timeout() -> float:
    return max(
        1.0,
        maybe_float(
            os.environ.get('LM_STUDIO_TIMEOUT_SECONDS'),
            DEFAULT_LM_REQUEST_TIMEOUT_SECONDS,
        ),
    )


def get_completion_text(response: Any) -> str:
    """Extract assistant text from normal content or LM Studio reasoning fields."""
    message = response.choices[0].message

    content = getattr(message, 'content', None)
    if isinstance(content, str) and content.strip():
        return content.strip()

    reasoning_content = getattr(message, 'reasoning_content', None)
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return reasoning_content.strip()

    reasoning = getattr(message, 'reasoning', None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    model_extra = getattr(message, 'model_extra', None)
    if isinstance(model_extra, dict):
        for key in ('content', 'reasoning_content', 'reasoning'):
            value = model_extra.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    if hasattr(message, 'model_dump'):
        dumped = message.model_dump()
        for key in ('content', 'reasoning_content', 'reasoning'):
            value = dumped.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    if hasattr(response, 'model_dump'):
        dumped_response = response.model_dump()
        try:
            dumped_message = dumped_response['choices'][0]['message']
            for key in ('content', 'reasoning_content', 'reasoning'):
                value = dumped_message.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        except (KeyError, IndexError, TypeError):
            pass

    return ''




def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): jsonable(item)
            for key, item in value.items()
            if item is not None
        }

    if hasattr(value, 'model_dump'):
        return jsonable(value.model_dump(mode='json', exclude_none=True))

    if hasattr(value, 'to_dict'):
        return jsonable(value.to_dict())

    if hasattr(value, '__dict__'):
        return jsonable({
            key: item
            for key, item in vars(value).items()
            if not key.startswith('_')
        })

    return str(value)




def extract_token_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, 'usage', None)
    if usage is None and isinstance(response, dict):
        usage = response.get('usage')

    if usage is None:
        return {}

    normalized = jsonable(usage)
    if isinstance(normalized, dict):
        return normalized

    return {'usage': normalized}




def parse_json_object(text: str) -> dict[str, Any]:
    data, _ = parse_json_object_with_repair(text)
    return data


def parse_json_object_with_repair(text: str) -> tuple[dict[str, Any], bool]:
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0)), True




def load_profile(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'Profile file does not exist: {path}')

    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'Profile file is not valid JSON: {path}') from exc

    if not isinstance(data, dict):
        raise ValueError('Profile JSON root must be an object.')

    return data



def compact_prompt_text(value: Any, *, limit: int = PROMPT_TEXT_LIMIT) -> str:
    text = ' '.join(str(value or '').split())
    if len(text) <= limit:
        return text

    return text[: max(0, limit - 3)].rstrip() + '...'


def compact_prompt_list(
    values: Any,
    *,
    item_limit: int = PROMPT_LIST_LIMIT,
    text_limit: int = PROMPT_TEXT_LIMIT,
) -> list[Any]:
    if not isinstance(values, list | tuple | set):
        return []

    compacted = []
    for item in list(values)[:item_limit]:
        if isinstance(item, dict):
            compacted.append(compact_prompt_mapping(item, text_limit=text_limit))
        else:
            compacted.append(compact_prompt_text(item, limit=text_limit))

    return compacted


def compact_prompt_mapping(
    values: Any,
    *,
    key_limit: int = PROMPT_LIST_LIMIT,
    text_limit: int = PROMPT_TEXT_LIMIT,
) -> dict[str, Any]:
    if not isinstance(values, dict):
        return {}

    compacted: dict[str, Any] = {}
    for key, value in list(values.items())[:key_limit]:
        if value in (None, '', [], {}):
            continue
        if isinstance(value, dict):
            compacted[str(key)] = compact_prompt_mapping(
                value,
                key_limit=key_limit,
                text_limit=text_limit,
            )
        elif isinstance(value, list | tuple | set):
            compacted[str(key)] = compact_prompt_list(
                value,
                item_limit=key_limit,
                text_limit=text_limit,
            )
        elif isinstance(value, int | float | bool):
            compacted[str(key)] = value
        else:
            compacted[str(key)] = compact_prompt_text(value, limit=text_limit)

    return compacted


def compact_profile_for_ai(profile: dict[str, Any] | None) -> dict[str, Any] | None:
    if not profile:
        return None

    ai_generated = profile.get('ai_generated') or {}
    constitution = ai_generated.get('short_constitution') or {}
    summary: dict[str, Any] = {}

    scalar_fields = (
        'nation_name',
        'profile_name',
        'roleplay_premise',
        'national_vision',
        'governing_style',
        'risk_tolerance',
        'enactment_mode',
        'minimum_confidence_to_enact',
        'issue_selection_strategy',
        'tone',
        'custom_instruction',
    )
    for field in scalar_fields:
        value = profile.get(field)
        if value not in (None, '', [], {}):
            if isinstance(value, int | float | bool):
                summary[field] = value
            else:
                summary[field] = compact_prompt_text(value)

    for field in (
        'top_priorities',
        'secondary_priorities',
        'red_lines',
        'preferred_tradeoffs',
        'unacceptable_tradeoffs',
    ):
        values = compact_prompt_list(profile.get(field))
        if values:
            summary[field] = values

    scoring_weights = compact_prompt_mapping(profile.get('scoring_weights'))
    if scoring_weights:
        summary['scoring_weights'] = scoring_weights

    vision_description = compact_prompt_text(
        ai_generated.get('vision_description'),
        limit=PROMPT_LONG_TEXT_LIMIT,
    )
    if vision_description:
        summary['vision_description'] = vision_description

    compact_constitution = compact_prompt_mapping(
        constitution,
        key_limit=12,
        text_limit=PROMPT_LONG_TEXT_LIMIT,
    )
    if compact_constitution:
        summary['short_constitution'] = compact_constitution

    return summary




def xml_local_name(tag: str) -> str:
    return tag.rsplit('}', 1)[-1].lower()


def compact_nation_context(nation_snapshot_xml: str) -> dict[str, Any]:
    try:
        root = ET.fromstring(nation_snapshot_xml)
    except ET.ParseError:
        return {
            'raw_xml_excerpt': compact_prompt_text(
                nation_snapshot_xml,
                limit=PROMPT_LONG_TEXT_LIMIT,
            )
        }

    wanted_tags = {
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
    }
    context: dict[str, Any] = {}
    for element in root.iter():
        key = xml_local_name(element.tag)
        if key not in wanted_tags:
            continue
        text = compact_prompt_text(
            ''.join(element.itertext()),
            limit=PROMPT_LONG_TEXT_LIMIT,
        )
        if text:
            context[key] = text

    return context


def compact_live_issues_for_ai(
    live_issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compacted = []
    for issue in live_issues:
        compacted.append({
            'issue_id': str(issue.get('issue_id', '')),
            'title': compact_prompt_text(issue.get('title'), limit=180),
            'text': compact_prompt_text(
                issue.get('text'),
                limit=PROMPT_LONG_TEXT_LIMIT,
            ),
            'options': [
                {
                    'option_id': str(option.get('option_id', '')),
                    'option_label': f'Option {position}',
                    'text': compact_prompt_text(
                        option.get('text'),
                        limit=PROMPT_LONG_TEXT_LIMIT,
                    ),
                }
                for position, option in enumerate(
                    issue.get('options', []),
                    start=1,
                )
                if isinstance(option, dict)
            ],
        })

    return compacted




def build_governor_instruction(
    profile: dict[str, Any] | None,
    fallback_strategy: str,
) -> str:
    if not profile:
        return f"""
You are governing from this simple strategy:

{compact_prompt_text(fallback_strategy, limit=PROMPT_LONG_TEXT_LIMIT)}

Choose policies that best satisfy this strategy while avoiding absurdly destructive outcomes.
If you choose a listed option, action must be "enact" with that option_id.
If every option is a poor fit, dismiss the issue instead with action "dismiss" and option_id "-1".
"""

    profile_summary = compact_profile_for_ai(profile) or {}

    return f"""
You are the AI governor for the NationStates nation {profile.get("nation_name", "unknown")}.

You must govern according to this user-approved governance profile.

Profile summary:
{json.dumps(profile_summary, indent=2, ensure_ascii=False)}

Binding rules:
- The user is sovereign.
- The profile and constitution are binding.
- Do not ignore red lines.
- Do not invent issue IDs or option IDs.
- Choose the policy that best satisfies the profile.
- If you choose any listed option, action must be "enact" with that option_id.
- If every option is bad, dismiss the issue with action "dismiss" and option_id "-1".
- Never describe dismissing an issue "with Option N"; that means action "enact".
- Confidence and charter_alignment_score must be honest and mutually consistent.
- charter_alignment_score is required and should be an intentional 0-100 integer,
  not a placeholder default.
- If the score would be 0, the recommendation should probably be dismissed or
  reconsidered.
"""




def is_model_reload_error(exc: BaseException) -> bool:
    return 'model reloaded.' in str(exc).lower()


def is_unsupported_temperature_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        'temperature' in message
        and 'unsupported value' in message
        and 'only the default (1)' in message
    )


class LocalGovernor:
    """Uses LM Studio/OpenAI-compatible local server to recommend issue choices."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        api_trace: bool = False,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get('LM_STUDIO_BASE_URL')
            or DEFAULT_LM_BASE_URL
        )
        self.api_trace = api_trace
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=(
                api_key
                or os.environ.get('LM_STUDIO_API_KEY')
                or DEFAULT_LM_API_KEY
            ),
            max_retries=0,
            timeout=local_model_request_timeout(),
        )
        self.model = model or os.environ.get('LM_STUDIO_MODEL') or self._detect_model()

    def close(self) -> None:
        close = getattr(self.client, 'close', None)
        if callable(close):
            close()

    def _chat_completion_with_reload_retry(
        self,
        *,
        step_name: str,
        pulse_label: str | None,
        **kwargs: Any,
    ) -> Any:
        request_kwargs = dict(kwargs)
        attempt = 1
        temperature_retry_used = False
        while attempt <= LOCAL_MODEL_RELOAD_MAX_ATTEMPTS:
            try:
                if pulse_label:
                    with StatusPulse(pulse_label):
                        response = self.client.chat.completions.create(
                            **request_kwargs
                        )
                else:
                    response = self.client.chat.completions.create(**request_kwargs)

                if self.api_trace:
                    print_api_trace(
                        api='local model',
                        operation=f'chat.completions.create ({step_name})',
                        request={
                            'base_url': self.base_url,
                            'attempt': attempt,
                            'step_name': step_name,
                            **request_kwargs,
                        },
                        response=jsonable(response),
                    )
                return response
            except Exception as exc:
                if self.api_trace:
                    print_api_trace(
                        api='local model',
                        operation=f'chat.completions.create ({step_name})',
                        request={
                            'base_url': self.base_url,
                            'attempt': attempt,
                            'step_name': step_name,
                            **request_kwargs,
                        },
                        error=exc,
                    )
                if (
                    is_unsupported_temperature_error(exc)
                    and 'temperature' in request_kwargs
                    and not temperature_retry_used
                ):
                    temperature_retry_used = True
                    temperature = request_kwargs.pop('temperature')
                    print(
                        f'[Model does not support custom temperature {temperature}; '
                        'retrying with the provider default.]'
                    )
                    continue

                if is_model_reload_error(exc) and attempt < LOCAL_MODEL_RELOAD_MAX_ATTEMPTS:
                    print(
                        f'[Local model reloaded during {step_name}; retrying '
                        f'AI step ({attempt + 1}/{LOCAL_MODEL_RELOAD_MAX_ATTEMPTS}): {exc}]'
                    )
                    attempt += 1
                    continue

                raise

        raise RuntimeError(f'{step_name} failed after model reload retries.')

    def _detect_model(self) -> str:
        try:
            with StatusPulse('AI setup: detecting loaded local model'):
                models = self.client.models.list()
        except Exception as exc:
            if self.api_trace:
                print_api_trace(
                    api='local model',
                    operation='models.list',
                    request={'base_url': self.base_url},
                    error=exc,
                )
            raise

        if self.api_trace:
            print_api_trace(
                api='local model',
                operation='models.list',
                request={'base_url': self.base_url},
                response=jsonable(models),
            )
        if not models.data:
            raise RuntimeError('No local Studio model is loaded.')
        return models.data[0].id

    def select_issue(
        self,
        *,
        nation_snapshot_xml: str,
        live_issues: list[dict[str, Any]],
        strategy: str,
        profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Late import to avoid circular dependency (recommendations imports from governor)
        from nsai.advisor.recommendations import fallback_issue_choice

        valid_issue_ids = [str(issue['issue_id']) for issue in live_issues]
        governor_instruction = build_governor_instruction(profile, strategy)

        system = """
You are an AI governor/advisor for a NationStates player.

Return strict JSON only. No markdown.

Choose the single most important issue to handle first from live_issues.

Rules:
- issue_id must be copied exactly from the provided live_issues list.
- Do not choose an issue that is not present.
- Prefer the user's governance profile over your own politics.
- Choose the issue whose decision is most urgent or most consequential for the profile.
- If all issues are routine, choose the one most aligned with the user's issue strategy.
- Keep why_this_issue_first to one or two sentences.
"""

        user = {
            'governor_instruction': governor_instruction,
            'profile_summary': compact_profile_for_ai(profile),
            'fallback_strategy': compact_prompt_text(strategy),
            'nation_context': compact_nation_context(nation_snapshot_xml),
            'live_issues': compact_live_issues_for_ai(live_issues),
        }

        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': json.dumps(user, indent=2, ensure_ascii=False)},
        ]

        selection_schema = {
            'type': 'object',
            'properties': {
                'issue_id': {
                    'type': 'string',
                    'enum': valid_issue_ids,
                },
                'why_this_issue_first': {
                    'type': 'string',
                },
            },
            'required': [
                'issue_id',
                'why_this_issue_first',
            ],
            'additionalProperties': False,
        }

        try:
            response = self._chat_completion_with_reload_retry(
                step_name='issue selection',
                pulse_label=f'AI step: selecting issue from {len(live_issues)} live issues',
                model=self.model,
                messages=messages,
                temperature=0.15,
                response_format={
                    'type': 'json_schema',
                    'json_schema': {
                        'name': 'NationStatesIssueSelection',
                        'schema': selection_schema,
                        'strict': True,
                    },
                },
            )
        except Exception as exc:
            print(f'[issue selection failed. Using deterministic selection: {exc}]')
            selection = fallback_issue_choice(live_issues, strategy)
            selection.update({
                'ai_step_failed': True,
                'ai_step_error': str(exc),
                'fallback_issue_selection_used': True,
                'requires_review': True,
            })
            return selection

        text = get_completion_text(response)
        if not text:
            print('[AI returned no issue selection. Using deterministic selection.]')
            selection = fallback_issue_choice(live_issues, strategy)
            selection['token_usage'] = extract_token_usage(response)
            selection['ai_step_failed'] = True
            selection['ai_step_error'] = 'AI returned no issue selection.'
            selection['structured_output_repaired'] = True
            selection['fallback_issue_selection_used'] = True
            selection['requires_review'] = True
            return selection

        try:
            selection, repaired = parse_json_object_with_repair(text)
        except Exception as exc:
            print('[AI returned unparsable issue selection. Using deterministic selection.]')
            print('[Raw AI text follows]')
            print(text)
            selection = fallback_issue_choice(live_issues, strategy)
            selection['token_usage'] = extract_token_usage(response)
            selection['ai_step_failed'] = True
            selection['ai_step_error'] = str(exc)
            selection['structured_output_repaired'] = True
            selection['structured_output_error'] = str(exc)
            selection['fallback_issue_selection_used'] = True
            selection['requires_review'] = True
            return selection

        issue_id = str(selection.get('issue_id', '')).strip()
        if issue_id not in valid_issue_ids:
            print(
                f'[AI chose invalid issue_id={issue_id!r}. '
                'Using deterministic selection.]'
            )
            selection = fallback_issue_choice(live_issues, strategy)
            selection['token_usage'] = extract_token_usage(response)
            selection['ai_step_failed'] = True
            selection['ai_step_error'] = f'AI chose invalid issue_id={issue_id!r}.'
            selection['structured_output_repaired'] = True
            selection['fallback_issue_selection_used'] = True
            selection['requires_review'] = True
            return selection

        selection['issue_id'] = issue_id
        selection['model'] = self.model
        selection['token_usage'] = extract_token_usage(response)
        if repaired:
            selection['structured_output_repaired'] = True
        return selection

    def plan_issue_order(
        self,
        *,
        nation_snapshot_xml: str,
        live_issues: list[dict[str, Any]],
        strategy: str,
        profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Late import to avoid circular dependency (recommendations imports from governor)
        from nsai.advisor.recommendations import fallback_issue_order

        valid_issue_ids = [str(issue['issue_id']) for issue in live_issues]
        governor_instruction = build_governor_instruction(profile, strategy)

        system = """
You are an AI governor/advisor for a NationStates player.

Return strict JSON only. No markdown.

Sort all live_issues into the order they should be resolved.

Rules:
- Include every live issue exactly once.
- issue_id values must be copied exactly from the provided live_issues list.
- Do not invent issue IDs.
- Prefer the user's governance profile over your own politics.
- Put urgent, high-impact, red-line-sensitive, or strategically consequential issues first.
- Keep each reason to one short sentence.
"""

        user = {
            'governor_instruction': governor_instruction,
            'profile_summary': compact_profile_for_ai(profile),
            'fallback_strategy': compact_prompt_text(strategy),
            'nation_context': compact_nation_context(nation_snapshot_xml),
            'live_issues': compact_live_issues_for_ai(live_issues),
        }

        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': json.dumps(user, indent=2, ensure_ascii=False)},
        ]

        order_schema = {
            'type': 'object',
            'properties': {
                'issue_order': {
                    'type': 'array',
                    'minItems': len(valid_issue_ids),
                    'maxItems': len(valid_issue_ids),
                    'items': {
                        'type': 'object',
                        'properties': {
                            'issue_id': {
                                'type': 'string',
                                'enum': valid_issue_ids,
                            },
                            'why_this_issue_now': {
                                'type': 'string',
                            },
                        },
                        'required': ['issue_id', 'why_this_issue_now'],
                        'additionalProperties': False,
                    },
                },
            },
            'required': ['issue_order'],
            'additionalProperties': False,
        }

        try:
            response = self._chat_completion_with_reload_retry(
                step_name='issue ordering',
                pulse_label=f'AI step: ordering {len(live_issues)} live issues',
                model=self.model,
                messages=messages,
                temperature=0.1,
                response_format={
                    'type': 'json_schema',
                    'json_schema': {
                        'name': 'NationStatesIssueOrder',
                        'schema': order_schema,
                        'strict': True,
                    },
                },
            )
        except Exception as exc:
            print(f'[issue ordering failed. Using deterministic order: {exc}]')
            plan = fallback_issue_order(live_issues, strategy)
            plan.update({
                'ai_step_failed': True,
                'ai_step_error': str(exc),
            })
            return plan

        text = get_completion_text(response)
        if not text:
            print('[AI returned no issue order. Using deterministic order.]')
            plan = fallback_issue_order(live_issues, strategy)
            plan['token_usage'] = extract_token_usage(response)
            plan['ai_step_failed'] = True
            plan['ai_step_error'] = 'AI returned no issue order.'
            plan['structured_output_repaired'] = True
            return plan

        try:
            payload, repaired = parse_json_object_with_repair(text)
        except Exception as exc:
            print('[AI returned unparsable issue order. Using deterministic order.]')
            print('[Raw AI text follows]')
            print(text)
            plan = fallback_issue_order(live_issues, strategy)
            plan['token_usage'] = extract_token_usage(response)
            plan['ai_step_failed'] = True
            plan['ai_step_error'] = str(exc)
            plan['structured_output_repaired'] = True
            plan['structured_output_error'] = str(exc)
            return plan

        order_entries = payload.get('issue_order')
        if not isinstance(order_entries, list):
            print('[AI issue order was malformed. Using deterministic order.]')
            plan = fallback_issue_order(live_issues, strategy)
            plan['token_usage'] = extract_token_usage(response)
            plan['structured_output_repaired'] = True
            return plan

        ordered_issue_ids: list[str] = []
        reasons: dict[str, str] = {}
        for entry in order_entries:
            if not isinstance(entry, dict):
                continue

            issue_id = str(entry.get('issue_id', '')).strip()
            if issue_id in valid_issue_ids and issue_id not in ordered_issue_ids:
                ordered_issue_ids.append(issue_id)
                reasons[issue_id] = str(entry.get('why_this_issue_now', '')).strip()

        if set(ordered_issue_ids) != set(valid_issue_ids):
            print('[AI issue order missed or duplicated live issues. Using deterministic order.]')
            plan = fallback_issue_order(live_issues, strategy)
            plan['token_usage'] = extract_token_usage(response)
            plan['structured_output_repaired'] = True
            return plan

        plan = {
            'ordered_issue_ids': ordered_issue_ids,
            'reasons': reasons,
            'model': self.model,
            'token_usage': extract_token_usage(response),
        }
        if repaired:
            plan['structured_output_repaired'] = True
        return plan

    def advise(
        self,
        *,
        nation_snapshot_xml: str,
        live_issues: list[dict[str, Any]],
        strategy: str,
        profile: dict[str, Any] | None = None,
        draft_dispatch: bool = False,
        draft_factbook: bool = False,
        pulse_label: str | None = 'AI step: generating recommendation for selected issue',
        retry_text_mode: bool = True,
        fallback_on_failure: bool = True,
    ) -> dict[str, Any]:
        # Late import to avoid circular dependency (recommendations imports from governor)
        from nsai.advisor.recommendations import (
            collect_issue_option_ids,
            DISMISS_OPTION_ID,
            fallback_issue_choice,
            fallback_recommendation,
            validate_publication_drafts,
            validate_recommendation,
        )

        valid_issue_ids = [issue['issue_id'] for issue in live_issues]
        valid_option_ids = sorted({
            option['option_id']
            for issue in live_issues
            for option in issue['options']
        })

        governor_instruction = build_governor_instruction(profile, strategy)

        system = """
You are an AI governor/advisor for a NationStates player.

Return strict JSON only. No markdown.

You will receive:
- a concise governance charter or strategy
- a compact public nation context
- a clean list of live NationStates issues
- publication draft requests for dispatch/factbook text

Choose exactly one issue from the provided live_issues data, summarize that issue
and each of its options, then decide whether to enact an option or dismiss the issue.

Rules:
- action must be "enact" when you recommend answering an issue.
- action must be "dismiss" when no option is acceptable, when more information is needed,
  or when dismissing the issue best satisfies the profile.
- issue_id must be copied exactly from the provided live_issues list.
- For action "enact", option_id must be copied exactly from the selected issue's options.
- For action "dismiss", option_id must be "-1".
- If you prefer any listed option, action is "enact"; never call that a dismissal.
- Never write that the nation is dismissing, accepting, or resolving an issue "with Option N".
- Do not invent IDs.
- Do not leave issue_id blank.
- Option identity is strict: copy option_id exactly for machine actions. The
  human-facing option_label is the option's numbered position on the NationStates
  website and may differ from option_id when raw IDs contain gaps. Whenever you
  mention an option in prose, use its listed option_label, never its raw option_id.
- option_summaries must use the exact option_id values from the selected issue,
  in the same order the options are listed.
- issue_summary must summarize the selected issue in plain language.
- option_summaries must summarize every available option for the selected issue.
- If publication_requests.dispatch is true, dispatch_draft must contain a usable
  BBCode-friendly title and body text about the recommended issue action.
  Use category_hint "Bulletin" and a subcategory_hint of "News", "Policy",
  "Opinion", or "Campaign".
- dispatch_draft must read like a release from the nation's state media: a
  formal news bulletin announcing the government's action or position. Where
  the issue text or its options quote named individuals arguing a side, quote
  those individuals (with attribution) in the dispatch, preferring voices that
  support the chosen action and optionally one dissenting voice for balance.
- If publication_requests.factbook is true, decide whether the issue action is
  pertinent to durable national lore, institutions, laws, or statistics. If yes,
  factbook_draft must contain a usable title and body text. If no, set pertinent
  false and explain why in reason.
  Use category_hint "Factbook" and one of these subcategory hints when possible:
  "Overview", "History", "Geography", "Culture", "Politics", "Legislation",
  "Religion", "Military", "Economy", "International", "Trivia", or
  "Miscellaneous".
- factbook_draft must read like an objective reference text: a neutral,
  textbook-like third-person record of the nation's institutions, laws, or
  history. No rhetoric, no promotion, no first-person voice, no quotes.
- Publication drafts are in-world documents for NationStates readers. Never
  mention the issue ID, option IDs or option numbers, this advisory process,
  or that an "issue" was chosen, enacted, or dismissed. Describe the policy or
  event itself, not the game mechanics behind it.
- Publication drafts may be posted through the NationStates API after NSAI
  successfully enacts or dismisses the issue. Do not claim they were posted
  inside the draft text.
- Prefer the user's governance profile over your own politics.
- If the profile includes a constitution, treat it as binding.
- If an option violates a red line, mark red_line_triggered true.
- If all options are bad, use action "dismiss" and explain why.
- Set confidence and charter_alignment_score honestly.
- charter_alignment_score is required: use a deliberate 0-100 integer that
  reflects how well the chosen action fits the profile and constitution.
- Do not leave charter_alignment_score at 0 by default; only use 0 when the
  recommendation truly has no meaningful alignment.
- If you cannot give the action a meaningful alignment score above 0, lower
  confidence or dismiss instead of forcing enactment.
- Keep reasoning, expected_tradeoffs, and do_not_enact_if concise.
- Put the final JSON in the normal assistant content field if possible.
"""

        user = {
            'governor_instruction': governor_instruction,
            'profile_summary': compact_profile_for_ai(profile),
            'fallback_strategy': compact_prompt_text(strategy),
            'nation_context': compact_nation_context(nation_snapshot_xml),
            'live_issues': compact_live_issues_for_ai(live_issues),
            'publication_requests': {
                'dispatch': draft_dispatch,
                'factbook': draft_factbook,
            },
        }

        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': json.dumps(user, indent=2, ensure_ascii=False)},
        ]

        advisor_schema = {
            'type': 'object',
            'properties': {
                'issue_id': {
                    'type': 'string',
                    'enum': valid_issue_ids,
                },
                'option_id': {
                    'type': 'string',
                    'enum': [DISMISS_OPTION_ID, *valid_option_ids],
                },
                'action': {
                    'type': 'string',
                    'enum': ['enact', 'dismiss'],
                },
                'issue_summary': {
                    'type': 'string',
                },
                'option_summaries': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'option_id': {
                                'type': 'string',
                                'enum': valid_option_ids,
                            },
                            'summary': {
                                'type': 'string',
                            },
                            'expected_effect': {
                                'type': 'string',
                            },
                        },
                        'required': [
                            'option_id',
                            'summary',
                            'expected_effect',
                        ],
                        'additionalProperties': False,
                    },
                },
                'dispatch_draft': {
                    'type': 'object',
                    'properties': {
                        'requested': {'type': 'boolean'},
                        'title': {'type': 'string'},
                        'text': {'type': 'string'},
                        'category_hint': {'type': 'string'},
                        'subcategory_hint': {'type': 'string'},
                    },
                    'required': [
                        'requested',
                        'title',
                        'text',
                        'category_hint',
                        'subcategory_hint',
                    ],
                    'additionalProperties': False,
                },
                'factbook_draft': {
                    'type': 'object',
                    'properties': {
                        'requested': {'type': 'boolean'},
                        'pertinent': {'type': 'boolean'},
                        'title': {'type': 'string'},
                        'text': {'type': 'string'},
                        'category_hint': {'type': 'string'},
                        'subcategory_hint': {'type': 'string'},
                        'reason': {'type': 'string'},
                    },
                    'required': [
                        'requested',
                        'pertinent',
                        'title',
                        'text',
                        'category_hint',
                        'subcategory_hint',
                        'reason',
                    ],
                    'additionalProperties': False,
                },
                'confidence': {
                    'type': 'number',
                    'minimum': 0,
                    'maximum': 1,
                },
                'charter_alignment_score': {
                    'type': 'integer',
                    'description': (
                        'Required 0-100 alignment score; do not use 0 as a '
                        'placeholder.'
                    ),
                    'minimum': 0,
                    'maximum': 100,
                },
                'red_line_triggered': {
                    'type': 'boolean',
                },
                'red_line_notes': {
                    'type': 'array',
                    'items': {'type': 'string'},
                },
                'headline': {
                    'type': 'string',
                },
                'why_this_issue_first': {
                    'type': 'string',
                },
                'reasoning': {
                    'type': 'string',
                },
                'expected_tradeoffs': {
                    'type': 'array',
                    'items': {'type': 'string'},
                },
                'do_not_enact_if': {
                    'type': 'array',
                    'items': {'type': 'string'},
                },
                'audit_summary': {
                    'type': 'string',
                },
            },
            'required': [
                'issue_id',
                'option_id',
                'action',
                'issue_summary',
                'option_summaries',
                'dispatch_draft',
                'factbook_draft',
                'confidence',
                'charter_alignment_score',
                'red_line_triggered',
                'red_line_notes',
                'headline',
                'why_this_issue_first',
                'reasoning',
                'expected_tradeoffs',
                'do_not_enact_if',
                'audit_summary',
            ],
            'additionalProperties': False,
        }

        structured_output_repaired = False
        try:
            response = self._chat_completion_with_reload_retry(
                step_name='recommendation generation',
                pulse_label=pulse_label,
                model=self.model,
                messages=messages,
                temperature=0.25,
                response_format={
                    'type': 'json_schema',
                    'json_schema': {
                        'name': 'NationStatesGovernorRecommendation',
                        'schema': advisor_schema,
                        'strict': True,
                    },
                },
            )
        except Exception as exc:
            if is_model_reload_error(exc):
                if not fallback_on_failure:
                    raise
                print(
                    '[Local AI recommendation failed after model reload retries. '
                    f'Using deterministic fallback: {exc}]'
                )
                recommendation = fallback_recommendation(
                    live_issues,
                    strategy,
                    draft_dispatch=draft_dispatch,
                    draft_factbook=draft_factbook,
                )
                recommendation.update({
                    'ai_step_failed': True,
                    'ai_step_error': str(exc),
                    'requires_review': True,
                })
                return recommendation

            if not retry_text_mode:
                raise

            print(f'[json_schema failed, falling back to text mode: {exc}]')
            structured_output_repaired = True

            try:
                response = self._chat_completion_with_reload_retry(
                    step_name='recommendation text-mode retry',
                    pulse_label=(
                        'AI step: retrying recommendation in text mode'
                        if pulse_label
                        else None
                    ),
                    model=self.model,
                    messages=messages,
                    temperature=0.25,
                    response_format={'type': 'text'},
                )
            except Exception as retry_exc:
                if not fallback_on_failure:
                    raise
                print(
                    '[Local AI text-mode recommendation failed. '
                    f'Using deterministic fallback: {retry_exc}]'
                )
                recommendation = fallback_recommendation(
                    live_issues,
                    strategy,
                    draft_dispatch=draft_dispatch,
                    draft_factbook=draft_factbook,
                )
                recommendation.update({
                    'ai_step_failed': True,
                    'ai_step_error': str(retry_exc),
                    'requires_review': True,
                })
                return recommendation

        text = get_completion_text(response)

        if not text:
            if not fallback_on_failure:
                raise ValueError('AI returned no visible content.')
            print('[AI returned no visible content. Using deterministic fallback.]')
            recommendation = fallback_recommendation(
                live_issues,
                strategy,
                draft_dispatch=draft_dispatch,
                draft_factbook=draft_factbook,
            )
            recommendation['token_usage'] = extract_token_usage(response)
            recommendation['ai_step_failed'] = True
            recommendation['ai_step_error'] = 'AI returned no visible content.'
            recommendation['structured_output_repaired'] = True
            recommendation['requires_review'] = True
            return recommendation

        try:
            recommendation, repaired = parse_json_object_with_repair(text)
            structured_output_repaired = structured_output_repaired or repaired
        except Exception as exc:
            if not fallback_on_failure:
                raise
            print('[AI returned unparsable output. Using deterministic fallback.]')
            print('[Raw AI text follows]')
            print(text)
            recommendation = fallback_recommendation(
                live_issues,
                strategy,
                draft_dispatch=draft_dispatch,
                draft_factbook=draft_factbook,
            )
            recommendation['token_usage'] = extract_token_usage(response)
            recommendation['ai_step_failed'] = True
            recommendation['ai_step_error'] = str(exc)
            recommendation['structured_output_repaired'] = True
            recommendation['requires_review'] = True
            return recommendation

        try:
            validate_recommendation(recommendation, collect_issue_option_ids(live_issues))
            validate_publication_drafts(
                recommendation,
                draft_dispatch=draft_dispatch,
                draft_factbook=draft_factbook,
            )
        except NationStatesError as exc:
            if not fallback_on_failure:
                raise
            print(f'[AI returned invalid recommendation. Using deterministic fallback: {exc}]')
            print('[Raw AI text follows]')
            print(text)
            recommendation = fallback_recommendation(
                live_issues,
                strategy,
                draft_dispatch=draft_dispatch,
                draft_factbook=draft_factbook,
            )
            recommendation['token_usage'] = extract_token_usage(response)
            recommendation['ai_step_failed'] = True
            recommendation['ai_step_error'] = str(exc)
            recommendation['structured_output_repaired'] = True
            recommendation['requires_review'] = True
            return recommendation

        recommendation['model'] = self.model
        recommendation['token_usage'] = extract_token_usage(response)
        recommendation['option_label_basis'] = 'website_position'
        if structured_output_repaired:
            recommendation['structured_output_repaired'] = True
        return recommendation




def wrapped(text: str, width: int = 88) -> str:
    return textwrap.fill(str(text), width=width)


def unique_reasons(reasons: list[str]) -> list[str]:
    seen = set()
    unique = []
    for reason in reasons:
        if reason and reason not in seen:
            unique.append(reason)
            seen.add(reason)
    return unique




__all__ = [
    'DEFAULT_LM_BASE_URL',
    'DEFAULT_LM_API_KEY',
    'LOCAL_MODEL_RELOAD_MAX_ATTEMPTS',
    'LocalGovernor',
    'StatusPulse',
    'build_governor_instruction',
    'compact_live_issues_for_ai',
    'compact_nation_context',
    'compact_profile_for_ai',
    'compact_prompt_list',
    'compact_prompt_mapping',
    'compact_prompt_text',
    'extract_token_usage',
    'get_completion_text',
    'is_model_reload_error',
    'jsonable',
    'load_profile',
    'maybe_float',
    'parse_json_object',
    'parse_json_object_with_repair',
    'unique_reasons',
    'wrapped',
    'xml_local_name',
]
