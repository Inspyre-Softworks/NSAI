from __future__ import annotations

import argparse
import io
import json

import pytest

import nsai.advisor.live as live
import nsai.nations as nations
from nsai.advisor.live import NationStatesClient
from nsai.nations import (
    AppConfig,
    NationConfig,
    app_config_path,
    config_path_for,
    credential_key_for,
    load_nation_config,
    profiles_dir,
    nation_config_summary,
    run_nation_set,
    save_app_config,
    saved_default_nation_config,
)


@pytest.fixture()
def isolated_config_home(tmp_path, monkeypatch):
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path))
    return tmp_path


def test_nation_config_stores_secret_reference_not_plaintext(
    isolated_config_home,
    monkeypatch,
    tmp_path,
) -> None:
    secrets: dict[str, str] = {}
    profile = tmp_path / 'oringrad.json'
    profile.write_text('{}', encoding='utf-8')

    monkeypatch.setattr(
        nations,
        'set_secret',
        lambda key, value, **kwargs: secrets.__setitem__(key, value),
    )
    monkeypatch.setattr(nations, 'get_secret', lambda key, **kwargs: secrets.get(key))
    monkeypatch.setattr(nations.sys, 'stdin', io.StringIO('super-secret-password\n'))

    args = argparse.Namespace(
        nation='Oringrad',
        user_agent='NSAI-Test/0.1 contact:test@example.com nation:Oringrad',
        api_version=12,
        profile=str(profile),
        move_profile=False,
        base_url='http://localhost:1234/v1',
        model='local-model',
        lm_api_key=False,
        lm_api_key_stdin=False,
        clear_lm_api_key=False,
        secret_backend='keyring',
        draft_dispatch=True,
        draft_factbook=False,
        auth_kind='password',
        password=False,
        password_stdin=True,
        clear_auth=False,
    )

    run_nation_set(args)

    path = config_path_for('Oringrad')
    text = path.read_text(encoding='utf-8')
    data = json.loads(text)
    config = load_nation_config('Oringrad')

    assert 'super-secret-password' not in text
    assert data['auth_kind'] == 'password'
    assert data['draft_dispatch'] is True
    assert data['draft_factbook'] is False
    assert data['lm_base_url'] == 'http://localhost:1234/v1'
    assert data['lm_model'] == 'local-model'
    assert data['credential_backend'] == 'keyring'
    assert data['credential_key'] == credential_key_for('Oringrad', 'password')
    assert secrets[data['credential_key']] == 'super-secret-password'
    assert profile.exists()
    assert config.profile_path == str(profiles_dir() / 'oringrad' / 'oringrad.json')
    assert (profiles_dir() / 'oringrad' / 'oringrad.json').read_text(encoding='utf-8') == '{}'
    assert config.draft_dispatch is True
    assert config.draft_factbook is False
    assert config.lm_base_url == 'http://localhost:1234/v1'
    assert config.lm_model == 'local-model'
    assert config.credential_backend == 'keyring'

    summary = nation_config_summary(config)
    assert summary['secret_configured'] is True
    assert summary['secret_available'] is True
    assert summary['draft_dispatch'] is True
    assert summary['draft_factbook'] is False
    assert summary['lm_base_url'] == 'http://localhost:1234/v1'
    assert summary['lm_model'] == 'local-model'
    assert summary['credential_backend'] == 'keyring'


def test_nation_set_can_move_profile_into_managed_storage(
    isolated_config_home,
    monkeypatch,
    tmp_path,
) -> None:
    profile = tmp_path / 'move-me.json'
    profile.write_text('{"nation_name": "Oringrad"}', encoding='utf-8')
    monkeypatch.setattr(nations, 'get_secret', lambda key, **kwargs: None)

    args = argparse.Namespace(
        nation='Oringrad',
        user_agent=None,
        api_version=None,
        profile=str(profile),
        move_profile=True,
        base_url=None,
        model=None,
        lm_api_key=False,
        lm_api_key_stdin=False,
        clear_lm_api_key=False,
        secret_backend=None,
        draft_dispatch=None,
        draft_factbook=None,
        auth_kind=None,
        password=False,
        password_stdin=False,
        clear_auth=False,
    )

    run_nation_set(args)
    config = load_nation_config('Oringrad')
    stored = profiles_dir() / 'oringrad' / 'move-me.json'

    assert not profile.exists()
    assert stored.exists()
    assert config.profile_path == str(stored)


def test_saved_default_nation_prefers_program_config(
    isolated_config_home,
    monkeypatch,
) -> None:
    monkeypatch.setattr(nations, 'get_secret', lambda key, **kwargs: None)
    save_app_config(AppConfig(default_nation='Oringrad'))
    nations.save_nation_config(NationConfig(nation_name='Otheria'))
    nations.save_nation_config(NationConfig(nation_name='Oringrad'))

    config, source = saved_default_nation_config()

    assert config is not None
    assert config.nation_name == 'Oringrad'
    assert str(app_config_path()) in str(source)


def test_client_loads_secure_secret_from_nation_config(monkeypatch) -> None:
    config = NationConfig(
        nation_name='Oringrad',
        user_agent='NSAI-Test/0.1 contact:test@example.com nation:Oringrad',
        api_version=12,
        auth_kind='password',
        credential_key=credential_key_for('Oringrad', 'password'),
    )

    monkeypatch.delenv('NS_USER_AGENT', raising=False)
    monkeypatch.delenv('NS_API_VERSION', raising=False)
    monkeypatch.delenv('NS_PASSWORD', raising=False)
    monkeypatch.delenv('NS_AUTOLOGIN', raising=False)
    monkeypatch.delenv('NS_PIN', raising=False)
    monkeypatch.setattr(live, 'get_secret', lambda key, **kwargs: 'stored-password')

    client = NationStatesClient.from_env(config)

    assert client.user_agent == config.user_agent
    assert client.api_version == 12
    assert client.password == 'stored-password'
    assert client.autologin is None
    assert client.pin is None


def test_environment_secret_overrides_saved_secret(monkeypatch) -> None:
    config = NationConfig(
        nation_name='Oringrad',
        user_agent='saved-user-agent',
        auth_kind='password',
        credential_key=credential_key_for('Oringrad', 'password'),
    )

    monkeypatch.setenv('NS_USER_AGENT', 'env-user-agent')
    monkeypatch.setenv('NS_PASSWORD', 'env-password')
    monkeypatch.setattr(
        live,
        'get_secret',
        lambda key, **kwargs: pytest.fail('saved secret should not be read when NS_PASSWORD is set'),
    )

    client = NationStatesClient.from_env(config)

    assert client.user_agent == 'env-user-agent'
    assert client.password == 'env-password'
