"""
NationStates live AI governor/advisor with governance profile support.

Author: Taylor B. | Inspyre-Softworks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI
from PIL import Image, UnidentifiedImageError

from nsai import __version__ as _NSAI_VERSION
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from nsai.advisor.cache import AdviceCache, CachedAdvice, live_issue_by_id
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
from nsai.secure_store import SecureStoreError, get_secret, set_secret
from nsai.secure_store import (
    SECRET_BACKENDS,
    SECRET_BACKEND_KEYRING,
    default_secret_backend,
)


NS_API_URL = 'https://www.nationstates.net/cgi-bin/api.cgi'
DEFAULT_LM_BASE_URL = 'http://localhost:1234/v1'
LM_BASE_URL = os.environ.get('LM_STUDIO_BASE_URL', DEFAULT_LM_BASE_URL)
LM_MODEL = os.environ.get('LM_STUDIO_MODEL')
DEFAULT_LM_API_KEY = 'lm-studio'
DISMISS_OPTION_ID = '-1'
DEFAULT_STRATEGY = (
    'Keep the nation prosperous, socially stable, technologically advanced, '
    'not authoritarian, and avoid absurdly destructive policies.'
)
DEFAULT_AUDIT_LOG = 'ns_governor_audit.jsonl'
DEFAULT_PUBLICATION_COOLDOWN_SECONDS = 300.0
LOCAL_MODEL_RELOAD_MAX_ATTEMPTS = 2
PROMPT_TEXT_LIMIT = 900
PROMPT_LONG_TEXT_LIMIT = 1800
PROMPT_LIST_LIMIT = 8
FLAG_DISPLAY_MODES = {'ascii', 'banner'}
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


class NationStatesError(RuntimeError):
    """Raised when the NationStates API or validation fails."""


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


@dataclass
class RateLimitState:
    limit: int | None = None
    remaining: int | None = None
    reset_seconds: int | None = None


def build_default_user_agent(nation: str) -> str:
    """Build a NationStates-compliant User-Agent for the given nation.

    Used as an automatic fallback when neither NS_USER_AGENT nor a saved
    per-nation user_agent is available.  The format follows NationStates
    API etiquette (tool/version contact:script nation:name).
    """
    safe_nation = re.sub(r'[^a-zA-Z0-9_-]', '_', nation.strip()) or 'unknown'
    return f'NSAI/{_NSAI_VERSION} contact:script nation:{safe_nation}'


class NationStatesClient:
    """Small NationStates API client with User-Agent and rate-limit handling."""

    def __init__(
        self,
        user_agent: str,
        api_version: int | None = None,
        password: str | None = None,
        autologin: str | None = None,
        pin: str | None = None,
        min_delay_seconds: float = 0.75,
    ) -> None:
        if not user_agent or len(user_agent.strip()) < 8:
            raise ValueError(
                'NS_USER_AGENT must be informative, for example:\n'
                'InspyreSoftworksNationGM/0.1 contact:you@example.com nation:Oringrad'
            )

        self.user_agent = user_agent.strip()
        self.api_version = api_version
        self.password = password
        self.autologin = autologin
        self.pin = pin
        self.min_delay_seconds = min_delay_seconds
        self.last_request_at = 0.0
        self.rate_limit = RateLimitState()
        self.session = requests.Session()

    @classmethod
    def from_env(cls, nation_config: NationConfig | None = None) -> 'NationStatesClient':
        user_agent = os.environ.get('NS_USER_AGENT') or (
            nation_config.user_agent if nation_config else None
        )
        if not user_agent:
            nation_name = nation_config.nation_name if nation_config else 'unknown'
            user_agent = build_default_user_agent(nation_name)
            print(
                f'NS_USER_AGENT not set; using auto-generated agent: {user_agent}\n'
                'Set NS_USER_AGENT or run `nsai nation set <nation> --user-agent ...` '
                'to use a custom agent.'
            )

        version_raw = os.environ.get('NS_API_VERSION')
        api_version = (
            int(version_raw)
            if version_raw
            else (nation_config.api_version if nation_config else None)
        )

        password = os.environ.get('NS_PASSWORD')
        autologin = os.environ.get('NS_AUTOLOGIN')
        pin = os.environ.get('NS_PIN')

        if nation_config and not any([password, autologin, pin]):
            credential_key = nation_config.credential_key
            if nation_config.auth_kind and credential_key:
                try:
                    secret = get_secret(
                        credential_key,
                        backend=nation_config.credential_backend or SECRET_BACKEND_KEYRING,
                        reason=f'Unlock NationStates secret for {nation_config.nation_name}',
                    )
                except SecureStoreError as exc:
                    raise NationStatesError(str(exc)) from exc

                if not secret:
                    raise NationStatesError(
                        f'Saved config for {nation_config.nation_name} references '
                        f'a missing credential-store secret: {credential_key}'
                    )

                if nation_config.auth_kind == 'password':
                    password = secret
                elif nation_config.auth_kind == 'autologin':
                    autologin = secret
                elif nation_config.auth_kind == 'pin':
                    pin = secret

        return cls(
            user_agent=user_agent,
            api_version=api_version,
            password=password,
            autologin=autologin,
            pin=pin,
        )

    def _headers(self, private: bool = False) -> dict[str, str]:
        headers = {
            'User-Agent': self.user_agent,
            'Accept': 'application/xml,text/xml,*/*',
        }

        if private:
            if self.pin:
                headers['X-Pin'] = self.pin
            elif self.autologin:
                headers['X-Autologin'] = self.autologin
            elif self.password:
                headers['X-Password'] = self.password
            else:
                raise NationStatesError(
                    'Private API request needs NS_PASSWORD, NS_AUTOLOGIN, or NS_PIN.'
                )

        return headers

    def _sleep_if_needed(self) -> None:
        elapsed = time.monotonic() - self.last_request_at

        if elapsed < self.min_delay_seconds:
            time.sleep(self.min_delay_seconds - elapsed)

        if (
            self.rate_limit.remaining is not None
            and self.rate_limit.remaining <= 1
            and self.rate_limit.reset_seconds is not None
        ):
            time.sleep(max(1, self.rate_limit.reset_seconds + 1))

    def _record_headers(self, response: requests.Response) -> None:
        headers = response.headers

        self.pin = headers.get('X-Pin', self.pin)
        self.autologin = headers.get('X-Autologin', self.autologin)

        self.rate_limit.limit = maybe_int(headers.get('RateLimit-Limit'))
        self.rate_limit.remaining = maybe_int(headers.get('RateLimit-Remaining'))
        self.rate_limit.reset_seconds = maybe_int(headers.get('RateLimit-Reset'))

    def get_stream(
        self,
        url: str,
        *,
        private: bool = False,
        accept: str = '*/*',
        timeout: float = 120,
        retries: int = 2,
    ) -> requests.Response:
        """Open a streamed GET using this client's session and rate-limit handling."""

        for attempt in range(retries + 1):
            self._sleep_if_needed()

            headers = self._headers(private=private)
            headers['Accept'] = accept

            response = self.session.get(
                url,
                headers=headers,
                stream=True,
                timeout=timeout,
            )

            self.last_request_at = time.monotonic()
            self._record_headers(response)

            if response.status_code == 429:
                response.close()
                retry_after = maybe_int(response.headers.get('Retry-After')) or 5
                if attempt < retries:
                    time.sleep(retry_after + 1)
                    continue

            if response.status_code == 403:
                response.close()
                raise NationStatesError(
                    'NationStates returned 403 Forbidden. Check your NS_USER_AGENT.'
                )

            if not response.ok:
                body_preview = response.text[:1000] if response.content else ''
                status_code = response.status_code
                response.close()
                raise NationStatesError(
                    f'NationStates download error {status_code}:\n'
                    f'{body_preview}'
                )

            return response

        raise NationStatesError('Stream request failed after retries.')

    def request_xml(
        self,
        params: dict[str, Any],
        *,
        private: bool = False,
        method: str = 'GET',
        retries: int = 2,
    ) -> ET.Element:
        clean_params = {
            key: value
            for key, value in params.items()
            if value is not None
        }

        if self.api_version is not None:
            clean_params['v'] = self.api_version

        for attempt in range(retries + 1):
            self._sleep_if_needed()

            if method.upper() == 'POST':
                response = self.session.post(
                    NS_API_URL,
                    data=clean_params,
                    headers=self._headers(private=private),
                    timeout=30,
                )
            else:
                response = self.session.get(
                    NS_API_URL,
                    params=clean_params,
                    headers=self._headers(private=private),
                    timeout=30,
                )

            self.last_request_at = time.monotonic()
            self._record_headers(response)

            if response.status_code == 429:
                retry_after = maybe_int(response.headers.get('Retry-After')) or 5
                if attempt < retries:
                    time.sleep(retry_after + 1)
                    continue

            if response.status_code == 403:
                raise NationStatesError(
                    'NationStates returned 403 Forbidden. Check your NS_USER_AGENT.'
                )

            if response.status_code == 409:
                raise NationStatesError(
                    'NationStates returned 409 Conflict. You may be logging in too '
                    'often with password/autologin. Use NS_PIN if you have one.'
                )

            if not response.ok:
                raise NationStatesError(
                    f'NationStates API error {response.status_code}:\n'
                    f'{response.text[:1000]}'
                )

            try:
                return ET.fromstring(response.text)
            except ET.ParseError as exc:
                raise NationStatesError(
                    f'Could not parse NationStates XML: {exc}\n\n'
                    f'{response.text[:1000]}'
                ) from exc

        raise NationStatesError('Request failed after retries.')

    def public_nation(self, nation: str, shards: list[str]) -> ET.Element:
        return self.request_xml({
            'nation': nation,
            'q': '+'.join(shards),
        })

    def issues(self, nation: str) -> ET.Element:
        return self.request_xml({
            'nation': nation,
            'q': 'issues',
        }, private=True)

    def answer_issue(self, nation: str, issue_id: str, option_id: str) -> ET.Element:
        return self.request_xml({
            'nation': nation,
            'c': 'issue',
            'issue': issue_id,
            'option': option_id,
        }, private=True, method='POST')

    def private_command(self, params: dict[str, Any]) -> ET.Element:
        prepare = {
            **params,
            'mode': 'prepare',
        }
        prepared = self.request_xml(prepare, private=True, method='POST')
        token = extract_private_command_token(prepared)

        execute = {
            **params,
            'mode': 'execute',
            'token': token,
        }
        return self.request_xml(execute, private=True, method='POST')

    def create_dispatch(
        self,
        nation: str,
        *,
        title: str,
        text: str,
        category: int,
        subcategory: int,
    ) -> ET.Element:
        return self.private_command({
            'nation': nation,
            'c': 'dispatch',
            'dispatch': 'add',
            'title': title,
            'text': text,
            'category': category,
            'subcategory': subcategory,
        })


def is_model_reload_error(exc: BaseException) -> bool:
    return 'model reloaded.' in str(exc).lower()


class LocalGovernor:
    """Uses LM Studio/OpenAI-compatible local server to recommend issue choices."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get('LM_STUDIO_BASE_URL')
            or DEFAULT_LM_BASE_URL
        )
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=(
                api_key
                or os.environ.get('LM_STUDIO_API_KEY')
                or DEFAULT_LM_API_KEY
            ),
        )
        self.model = model or os.environ.get('LM_STUDIO_MODEL') or self._detect_model()

    def _chat_completion_with_reload_retry(
        self,
        *,
        step_name: str,
        pulse_label: str,
        **kwargs: Any,
    ) -> Any:
        for attempt in range(1, LOCAL_MODEL_RELOAD_MAX_ATTEMPTS + 1):
            try:
                with StatusPulse(pulse_label):
                    return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                if is_model_reload_error(exc) and attempt < LOCAL_MODEL_RELOAD_MAX_ATTEMPTS:
                    print(
                        f'[Local model reloaded during {step_name}; retrying '
                        f'AI step ({attempt + 1}/{LOCAL_MODEL_RELOAD_MAX_ATTEMPTS}): {exc}]'
                    )
                    continue

                raise

        raise RuntimeError(f'{step_name} failed after model reload retries.')

    def _detect_model(self) -> str:
        with StatusPulse('AI setup: detecting loaded local model'):
            models = self.client.models.list()
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

    def advise(
        self,
        *,
        nation_snapshot_xml: str,
        live_issues: list[dict[str, Any]],
        strategy: str,
        profile: dict[str, Any] | None = None,
        draft_dispatch: bool = False,
        draft_factbook: bool = False,
    ) -> dict[str, Any]:
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
- issue_summary must summarize the selected issue in plain language.
- option_summaries must summarize every available option for the selected issue.
- If publication_requests.dispatch is true, dispatch_draft must contain a usable
  BBCode-friendly title and body text about the recommended issue action.
  Use category_hint "Bulletin" and a subcategory_hint of "News", "Policy",
  "Opinion", or "Campaign".
- If publication_requests.factbook is true, decide whether the issue action is
  pertinent to durable national lore, institutions, laws, or statistics. If yes,
  factbook_draft must contain a usable title and body text. If no, set pertinent
  false and explain why in reason.
  Use category_hint "Factbook" and one of these subcategory hints when possible:
  "Overview", "History", "Geography", "Culture", "Politics", "Legislation",
  "Religion", "Military", "Economy", "International", "Trivia", or
  "Miscellaneous".
- Publication drafts may be posted through the NationStates API after NSAI
  successfully enacts or dismisses the issue. Do not claim they were posted
  inside the draft text.
- Prefer the user's governance profile over your own politics.
- If the profile includes a constitution, treat it as binding.
- If an option violates a red line, mark red_line_triggered true.
- If all options are bad, use action "dismiss" and explain why.
- Set confidence and charter_alignment_score honestly.
- charter_alignment_score is required: use a deliberate 0-100 integer that reflects how well the chosen action fits the profile and constitution.
- Do not leave charter_alignment_score at 0 by default; only use 0 when the recommendation truly has no meaningful alignment.
- If you cannot give the action a meaningful alignment score above 0, lower confidence or dismiss instead of forcing enactment.
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
                    'minimum': 0,
                    'maximum': 100,
                    'description': (
                        'Required 0-100 alignment score; do not use 0 as a placeholder.'
                    ),
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
                pulse_label='AI step: generating recommendation for selected issue',
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

            print(f'[json_schema failed, falling back to text mode: {exc}]')
            structured_output_repaired = True

            try:
                response = self._chat_completion_with_reload_retry(
                    step_name='recommendation text-mode retry',
                    pulse_label='AI step: retrying recommendation in text mode',
                    model=self.model,
                    messages=messages,
                    temperature=0.25,
                    response_format={'type': 'text'},
                )
            except Exception as retry_exc:
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
        if structured_output_repaired:
            recommendation['structured_output_repaired'] = True
        return recommendation


def maybe_int(value: str | None) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return None


def maybe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def xml_to_string(root: ET.Element) -> str:
    return ET.tostring(root, encoding='unicode')


def xml_error_text(root: ET.Element) -> str:
    return (root.findtext('.//ERROR') or '').strip()


def extract_private_command_token(root: ET.Element) -> str:
    for element in root.iter():
        if element.tag.upper() == 'TOKEN' and element.text and element.text.strip():
            return element.text.strip()

        token = element.attrib.get('token') or element.attrib.get('TOKEN')
        if token:
            return token.strip()

    success = root.findtext('.//SUCCESS')
    if success and success.strip():
        return success.strip()

    text = xml_to_string(root)
    match = re.search(r'token[=:]\s*([A-Za-z0-9_-]+)', text, re.IGNORECASE)
    if match:
        return match.group(1)

    error = xml_error_text(root)
    if error:
        raise NationStatesError(
            f'NationStates refused to prepare private command: {error}'
        )

    raise NationStatesError(
        'NationStates did not return a private-command token during prepare mode.'
    )


def wrapped(text: str, width: int = 88) -> str:
    return textwrap.fill(str(text), width=width)


def nation_flag_url(nation_root: ET.Element) -> str:
    return (nation_root.findtext('FLAG') or '').strip()


def render_ascii_flag(flag_url: str, *, width: int = FLAG_ASCII_WIDTH) -> str:
    response = requests.get(flag_url, timeout=15)
    response.raise_for_status()

    with Image.open(BytesIO(response.content)) as image:
        image = image.convert('RGBA')
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert('L')

        target_width = max(8, width)
        aspect = image.height / image.width if image.width else 1.0
        target_height = max(1, round(target_width * aspect * 0.5))
        image = image.resize((target_width, target_height))

        pixels = list(image.getdata())
        ramp = FLAG_ASCII_RAMP
        ramp_size = len(ramp) - 1
        rows = [
            ''.join(ramp[pixel * ramp_size // 255] for pixel in pixels[index:index + target_width])
            for index in range(0, len(pixels), target_width)
        ]
        return '\n'.join(rows)


def render_banner(name: str) -> str:
    title = f' {name} '
    width = max(40, len(title) + 8)
    border = '=' * width
    return '\n'.join([
        border,
        title.center(width),
        border,
    ])


def print_nation_flag(nation_root: ET.Element, *, flag_display: str = 'ascii') -> None:
    nation_name = (nation_root.findtext('FULLNAME') or nation_root.get('id') or 'Unknown nation').strip()
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
    except (requests.RequestException, OSError, UnidentifiedImageError, ValueError) as exc:
        print(f'Could not render nation flag as ASCII: {exc}')
        print(render_banner(nation_name))
        if flag_url:
            print(f'Flag source: {flag_url}')


def unique_reasons(reasons: list[str]) -> list[str]:
    seen = set()
    unique = []
    for reason in reasons:
        if reason and reason not in seen:
            unique.append(reason)
            seen.add(reason)
    return unique


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
                    'text': compact_prompt_text(
                        option.get('text'),
                        limit=PROMPT_LONG_TEXT_LIMIT,
                    ),
                }
                for option in issue.get('options', [])
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
- charter_alignment_score is required and should be an intentional 0-100 integer, not a placeholder default.
- If the score would be 0, the recommendation should probably be dismissed or reconsidered.
"""


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

    if is_fallback_recommendation(recommendation) and not no_ai:
        return False, 'cached fallback advice is not reused when AI is enabled'

    try:
        validate_recommendation(recommendation, valid_options)
        validate_publication_drafts(
            recommendation,
            draft_dispatch=draft_dispatch,
            draft_factbook=draft_factbook,
        )
    except NationStatesError as exc:
        return False, str(exc)

    if auto_requested:
        validation = validate_auto_action(
            live_issues=live_issues or [],
            selected_issue=selected_issue,
            recommendation=recommendation,
            ai_step_statuses=ai_step_statuses or [],
            draft_dispatch=draft_dispatch,
            draft_factbook=draft_factbook,
            minimum_confidence=minimum_confidence,
            allow_fallback_auto=allow_fallback_auto,
        )
        if not validation.passed:
            return (
                False,
                'cached advice is not safe for auto mode: '
                + '; '.join(validation.reasons),
            )

    return True, ''


def print_live_issues(live_issues: list[dict[str, Any]]) -> None:
    print()
    print('Live NationStates Issues')
    print('=' * 88)

    for issue in live_issues:
        print()
        print(f'Issue {issue["issue_id"]}: {issue.get("title", "")}')
        print('-' * 88)

        issue_text = issue.get('text') or ''
        if issue_text:
            print(wrapped(issue_text))
            print()

        for option in issue.get('options', []):
            print(f'  Option {option["option_id"]}:')
            print(wrapped(option.get('text', ''), width=82))
            print()


def print_recommendation(recommendation: dict[str, Any]) -> None:
    print()
    print('AI Governor Recommendation')
    print('=' * 88)
    print(f'Headline:        {recommendation.get("headline", "")}')
    print(f'Action:          {recommendation_action(recommendation)}')
    print(f'Issue ID:        {recommendation.get("issue_id", "")}')
    option_id = str(recommendation.get('option_id', '')).strip()
    print(f'Option ID:       {option_id}')
    print(f'Confidence:      {recommendation.get("confidence", "")}')
    print(f'Alignment Score: {recommendation.get("charter_alignment_score", "")}')
    print(f'Red Line Hit:    {recommendation.get("red_line_triggered", "")}')
    print(f'Model:           {recommendation.get("model", "")}')
    print()

    issue_summary = recommendation.get('issue_summary', '')
    if issue_summary:
        print('Issue summary:')
        print(wrapped(str(issue_summary)))
        print()

    option_summaries = recommendation.get('option_summaries') or []
    if option_summaries:
        print('Option summaries:')
        for item in option_summaries:
            if isinstance(item, dict):
                option = item.get('option_id', '')
                summary = item.get('summary', '')
                expected = item.get('expected_effect', '')
                print(f' - Option {option}: {summary}')
                if expected:
                    print(f'   Expected effect: {expected}')
            else:
                print(f' - {item}')
        print()

    why_first = recommendation.get('why_this_issue_first', '')
    if why_first:
        print('Why this issue first:')
        print(wrapped(why_first))
        print()

    reasoning = recommendation.get('reasoning', '')
    if reasoning:
        print('Reasoning:')
        print(wrapped(reasoning))
        print()

    tradeoffs = recommendation.get('expected_tradeoffs') or []
    if tradeoffs:
        print('Expected tradeoffs:')
        for item in tradeoffs:
            print(f' - {item}')

    red_line_notes = recommendation.get('red_line_notes') or []
    if red_line_notes:
        print()
        print('Red-line notes:')
        for item in red_line_notes:
            print(f' - {item}')

    blockers = recommendation.get('do_not_enact_if') or []
    if blockers:
        print()
        print('Do not enact if:')
        for item in blockers:
            print(f' - {item}')

    print_publication_drafts(recommendation)


def print_publication_drafts(recommendation: dict[str, Any]) -> None:
    dispatch = recommendation.get('dispatch_draft')
    if isinstance(dispatch, dict) and dispatch.get('requested'):
        print()
        print('Dispatch draft')
        print('=' * 88)
        print(f'Title:       {dispatch.get("title", "")}')
        print(f'Category:    {dispatch.get("category_hint", "")}')
        print(f'Subcategory: {dispatch.get("subcategory_hint", "")}')
        print()
        print(wrapped(dispatch.get('text', '')))

    factbook = recommendation.get('factbook_draft')
    if isinstance(factbook, dict) and factbook.get('requested'):
        print()
        print('Factbook draft')
        print('=' * 88)
        print(f'Pertinent:   {factbook.get("pertinent", False)}')
        reason = str(factbook.get('reason', '')).strip()
        if reason:
            print(f'Reason:      {reason}')

        if factbook.get('pertinent'):
            print(f'Title:       {factbook.get("title", "")}')
            print(f'Category:    {factbook.get("category_hint", "")}')
            print(f'Subcategory: {factbook.get("subcategory_hint", "")}')
            print()
            print(wrapped(factbook.get('text', '')))


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


def write_audit_log(
    path: Path,
    *,
    nation: str,
    profile_path: Path | None,
    profile: dict[str, Any] | None,
    recommendation: dict[str, Any],
    action: str,
    action_reasons: list[str],
    result_xml: str | None = None,
    publication_results: list[dict[str, Any]] | None = None,
    action_applied: bool | None = None,
    blocked: bool = False,
    block_reasons: list[str] | None = None,
    ai_step_statuses: list[dict[str, Any]] | None = None,
    fallback_issue_selection_used: bool = False,
) -> None:
    record = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'nation': nation,
        'profile_path': str(profile_path) if profile_path else None,
        'profile_name': profile.get('profile_name') if profile else None,
        'enactment_mode': get_profile_mode(profile),
        'action': action,
        'action_reasons': action_reasons,
        'recommendation': recommendation,
        'result_xml': result_xml,
        'publication_results': publication_results or [],
        'action_applied': action_applied,
        'blocked': blocked,
        'block_reasons': block_reasons or [],
        'ai_step_statuses': ai_step_statuses or [],
        'fallback_issue_selection_used': fallback_issue_selection_used,
    }

    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def append_publication_backfill_log(
    path: Path,
    *,
    source_entry: dict[str, Any],
    source_line: int,
    publication_results: list[dict[str, Any]],
) -> None:
    recommendation = source_entry.get('recommendation') or {}
    record = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'action': 'publication_backfill',
        'nation': source_entry.get('nation'),
        'source_audit_line': source_line,
        'source_timestamp': source_entry.get('timestamp'),
        'source_action': source_entry.get('action'),
        'issue_id': recommendation.get('issue_id'),
        'option_id': recommendation.get('option_id'),
        'publication_results': publication_results,
    }

    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def load_audit_log_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records = []
    if not path.exists():
        return records

    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise NationStatesError(
                f'Audit log {path} has invalid JSON on line {line_number}: {exc}'
            ) from exc

        if isinstance(record, dict):
            records.append((line_number, record))

    return records


def audit_issue_action_succeeded(record: dict[str, Any]) -> bool:
    if record.get('action') not in {'manual_enact', 'auto_enact'}:
        return False

    result_xml = record.get('result_xml')
    if not isinstance(result_xml, str) or not result_xml.strip():
        return False

    try:
        root = ET.fromstring(result_xml)
    except ET.ParseError:
        return False

    return (root.findtext('.//OK') or '').strip() == '1'


def publication_source_key(record: dict[str, Any], kind: str, title: str) -> tuple[str, str, str, str, str, str]:
    recommendation = record.get('recommendation') or {}
    nation = str(record.get('nation') or '')
    timestamp = str(record.get('source_timestamp') or record.get('timestamp') or '')
    issue_id = str(record.get('issue_id') or recommendation.get('issue_id') or '')
    option_id = str(record.get('option_id') or recommendation.get('option_id') or '')
    return (
        normalize_nation_key(nation),
        timestamp,
        issue_id,
        option_id,
        kind,
        title.strip().lower(),
    )


def posted_publication_keys(
    records: list[tuple[int, dict[str, Any]]],
) -> set[tuple[str, str, str, str, str, str]]:
    keys: set[tuple[str, str, str, str, str, str]] = set()

    for _, record in records:
        for result in record.get('publication_results') or []:
            if not isinstance(result, dict) or result.get('status') != 'posted':
                continue

            kind = str(result.get('kind') or '').strip()
            title = str(result.get('title') or '').strip()
            if kind and title:
                keys.add(publication_source_key(record, kind, title))

    return keys


def pending_publication_entries(
    records: list[tuple[int, dict[str, Any]]],
    *,
    nation: str | None = None,
) -> list[dict[str, Any]]:
    posted_keys = posted_publication_keys(records)
    pending = []
    nation_key = normalize_nation_key(nation) if nation else None

    for line_number, record in records:
        if not audit_issue_action_succeeded(record):
            continue

        record_nation = str(record.get('nation') or '')
        if nation_key and normalize_nation_key(record_nation) != nation_key:
            continue

        recommendation = record.get('recommendation')
        if not isinstance(recommendation, dict):
            continue

        if publication_mismatch_reasons(
            recommendation,
            draft_dispatch=True,
            draft_factbook=True,
        ):
            continue

        dispatch = recommendation.get('dispatch_draft')
        factbook = recommendation.get('factbook_draft')
        publish_dispatch = False
        publish_factbook = False

        if isinstance(dispatch, dict) and dispatch.get('requested'):
            title = str(dispatch.get('title') or '').strip()
            if title and publication_source_key(record, 'dispatch', title) not in posted_keys:
                publish_dispatch = True

        if (
            isinstance(factbook, dict)
            and factbook.get('requested')
            and factbook.get('pertinent')
        ):
            title = str(factbook.get('title') or '').strip()
            if title and publication_source_key(record, 'factbook', title) not in posted_keys:
                publish_factbook = True

        if publish_dispatch or publish_factbook:
            pending.append({
                'line': line_number,
                'record': record,
                'draft_dispatch': publish_dispatch,
                'draft_factbook': publish_factbook,
            })

    return pending


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
        nation_root = ns.public_nation(nation, [
            'fullname',
            'flag',
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
        ])

    print_nation_flag(
        nation_root,
        flag_display=str(getattr(args, 'flag_display', 'ascii')),
    )

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


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled.')
        sys.exit(130)
    except Exception as exc:
        print(f'\nERROR: {exc}')
        sys.exit(1)
