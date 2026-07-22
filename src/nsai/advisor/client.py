"""NationStates API client, error types, and XML helpers."""

from __future__ import annotations

import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import requests

from nsai import __version__ as _NSAI_VERSION
from nsai.nations import NationConfig
from nsai.secure_store import (
    SECRET_BACKEND_KEYRING,
    SecureStoreError,
    get_secret,
)


NS_API_URL = 'https://www.nationstates.net/cgi-bin/api.cgi'


def build_default_user_agent(nation: str) -> str:
    """Build a NationStates-compliant User-Agent for the given nation."""

    safe_nation = re.sub(r'[^a-zA-Z0-9_-]', '_', nation.strip()) or 'unknown'
    return f'NSAI/{_NSAI_VERSION} nation:{safe_nation}'


class NationStatesError(RuntimeError):
    """Raised when the NationStates API or validation fails."""


@dataclass
class RateLimitState:
    limit: int | None = None
    remaining: int | None = None
    reset_seconds: int | None = None


def maybe_int(value: str | None) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return None


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
                (
                    'WARNING: NS_USER_AGENT not set; using auto-generated placeholder '
                    f'agent: {user_agent}\n'
                    'For proper API etiquette, set NS_USER_AGENT to a User-Agent with a '
                    'real contact address, or run '
                    '`nsai nation set <nation> --user-agent ...` to configure a custom '
                    'agent.\n'
                ),
                file=sys.stderr,
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
                    # Use live module's get_secret to support monkeypatching in tests
                    _live = sys.modules.get('nsai.advisor.live')
                    _get_secret = getattr(_live, 'get_secret', get_secret) if _live else get_secret
                    secret = _get_secret(
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

                if nation_config.pin_credential_key:
                    try:
                        pin = _get_secret(
                            nation_config.pin_credential_key,
                            backend=nation_config.credential_backend or SECRET_BACKEND_KEYRING,
                            reason=f'Unlock NationStates PIN for {nation_config.nation_name}',
                        ) or pin
                    except SecureStoreError as exc:
                        raise NationStatesError(str(exc)) from exc

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

    def establish_session(self, nation: str) -> None:
        """Authenticate and capture the session credentials returned by NationStates."""

        self.request_xml({
            'nation': nation,
            'q': 'unread',
        }, private=True, retries=0)

        if not self.pin:
            raise NationStatesError(
                'NationStates authenticated the request but did not return an X-Pin.'
            )

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


__all__ = [
    'NS_API_URL',
    'NationStatesClient',
    'NationStatesError',
    'RateLimitState',
    'build_default_user_agent',
    'extract_private_command_token',
    'maybe_int',
    'xml_error_text',
    'xml_to_string',
]
