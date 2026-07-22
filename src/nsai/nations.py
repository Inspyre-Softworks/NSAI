"""Per-nation configuration and secure credential commands."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nsai.secure_store import (
    SECRET_BACKENDS,
    SECRET_BACKEND_KEYRING,
    default_secret_backend,
    SecureStoreError,
    delete_secret,
    get_secret,
    set_secret,
)


CONFIG_VERSION = 1
AUTH_KINDS = ('password', 'autologin', 'pin')


@dataclass
class NationConfig:
    nation_name: str
    user_agent: str | None = None
    api_version: int | None = None
    profile_path: str | None = None
    strategy: str | None = None
    show_issues: bool | None = None
    show_instruction: bool | None = None
    no_ai: bool | None = None
    audit_log: str | None = None
    lm_base_url: str | None = None
    lm_model: str | None = None
    lm_api_key_credential_key: str | None = None
    draft_dispatch: bool | None = None
    draft_factbook: bool | None = None
    auth_kind: str | None = None
    credential_key: str | None = None
    pin_credential_key: str | None = None
    credential_backend: str | None = None
    lm_api_key_backend: str | None = None
    config_version: int = CONFIG_VERSION
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def secret_configured(self) -> bool:
        return bool(self.auth_kind and self.credential_key)

    def to_json_data(self) -> dict[str, Any]:
        data = asdict(self)
        data['updated_at'] = datetime.now(timezone.utc).isoformat()
        return data


@dataclass
class AppConfig:
    default_nation: str | None = None
    profile_enrich_lm_base_url: str | None = None
    profile_enrich_lm_model: str | None = None
    profile_enrich_lm_api_key_credential_key: str | None = None
    profile_enrich_lm_api_key_backend: str | None = None
    config_version: int = CONFIG_VERSION
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json_data(self) -> dict[str, Any]:
        data = asdict(self)
        data['updated_at'] = datetime.now(timezone.utc).isoformat()
        return data


def config_root() -> Path:
    override = os.environ.get('NSAI_CONFIG_HOME')
    if override:
        return Path(override).expanduser().resolve()

    appdata = os.environ.get('APPDATA')
    if os.name == 'nt' and appdata:
        return Path(appdata) / 'NSAI'

    return Path.home() / '.config' / 'nsai'


def nations_dir() -> Path:
    return config_root() / 'nations'


def profiles_dir() -> Path:
    return config_root() / 'profiles'


def advice_cache_path() -> Path:
    return config_root() / 'advice.sqlite3'


def app_config_path() -> Path:
    return config_root() / 'config.json'


def normalize_nation_key(nation_name: str) -> str:
    key = re.sub(r'[^a-z0-9]+', '-', nation_name.strip().lower())
    return key.strip('-') or 'nation'


def credential_key_for(nation_name: str, auth_kind: str) -> str:
    if auth_kind not in AUTH_KINDS:
        raise ValueError(f'Unsupported auth kind: {auth_kind!r}')

    return f'nation:{normalize_nation_key(nation_name)}:{auth_kind}'


def lm_api_key_credential_key_for(nation_name: str) -> str:
    return f'nation:{normalize_nation_key(nation_name)}:lm-api-key'


def config_path_for(nation_name: str) -> Path:
    return nations_dir() / f'{normalize_nation_key(nation_name)}.json'


def load_app_config() -> AppConfig:
    path = app_config_path()
    if not path.exists():
        return AppConfig()

    data = json.loads(path.read_text(encoding='utf-8'))
    return AppConfig(
        default_nation=data.get('default_nation'),
        profile_enrich_lm_base_url=data.get('profile_enrich_lm_base_url'),
        profile_enrich_lm_model=data.get('profile_enrich_lm_model'),
        profile_enrich_lm_api_key_credential_key=data.get(
            'profile_enrich_lm_api_key_credential_key'
        ),
        profile_enrich_lm_api_key_backend=data.get(
            'profile_enrich_lm_api_key_backend'
        ),
        config_version=int(data.get('config_version', CONFIG_VERSION)),
        updated_at=str(data.get('updated_at') or datetime.now(timezone.utc).isoformat()),
    )


def save_app_config(config: AppConfig) -> Path:
    path = app_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.to_json_data(), indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    return path


def save_nation_config(config: NationConfig) -> Path:
    path = config_path_for(config.nation_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.to_json_data(), indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    return path


def load_nation_config(nation_name: str) -> NationConfig:
    path = config_path_for(nation_name)
    if not path.exists():
        raise FileNotFoundError(f'Nation config does not exist: {path}')

    data = json.loads(path.read_text(encoding='utf-8'))
    return NationConfig(
        nation_name=str(data.get('nation_name') or nation_name),
        user_agent=data.get('user_agent'),
        api_version=data.get('api_version'),
        profile_path=data.get('profile_path'),
        strategy=data.get('strategy'),
        show_issues=data.get('show_issues'),
        show_instruction=data.get('show_instruction'),
        no_ai=data.get('no_ai'),
        audit_log=data.get('audit_log'),
        lm_base_url=data.get('lm_base_url'),
        lm_model=data.get('lm_model'),
        lm_api_key_credential_key=data.get('lm_api_key_credential_key'),
        draft_dispatch=data.get('draft_dispatch'),
        draft_factbook=data.get('draft_factbook'),
        auth_kind=data.get('auth_kind'),
        credential_key=data.get('credential_key'),
        pin_credential_key=data.get('pin_credential_key'),
        credential_backend=data.get('credential_backend'),
        lm_api_key_backend=data.get('lm_api_key_backend'),
        config_version=int(data.get('config_version', CONFIG_VERSION)),
        updated_at=str(data.get('updated_at') or datetime.now(timezone.utc).isoformat()),
    )


def maybe_load_nation_config(nation_name: str | None) -> NationConfig | None:
    if not nation_name:
        return None

    try:
        return load_nation_config(nation_name)
    except FileNotFoundError:
        return None


def list_nation_configs() -> list[NationConfig]:
    directory = nations_dir()
    if not directory.exists():
        return []

    configs = []
    for path in sorted(directory.glob('*.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            configs.append(load_nation_config(str(data.get('nation_name') or path.stem)))
        except (OSError, ValueError, json.JSONDecodeError):
            continue

    return configs


def saved_default_nation_config() -> tuple[NationConfig | None, str | None]:
    app_config = load_app_config()
    if app_config.default_nation:
        config = maybe_load_nation_config(app_config.default_nation)
        if config:
            return config, f'default nation in {app_config_path()}'

    configs = list_nation_configs()
    if len(configs) == 1:
        return configs[0], 'only saved nation config'

    if len(configs) > 1:
        return None, f'multiple saved nation configs in {nations_dir()}'

    return None, None


def store_profile_for_nation(
    nation_name: str,
    profile_path: str | Path,
    *,
    move: bool = False,
) -> tuple[Path, str]:
    source = Path(profile_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f'Profile does not exist: {source}')

    destination_dir = profiles_dir() / normalize_nation_key(nation_name)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = (destination_dir / source.name).resolve()

    if source == destination:
        return destination, 'already-managed'

    if move:
        if destination.exists():
            destination.unlink()
        shutil.move(str(source), str(destination))
        return destination, 'moved'

    shutil.copy2(source, destination)
    return destination, 'copied'


def read_secret_from_args(args: argparse.Namespace, auth_kind: str) -> str | None:
    if args.password and args.password_stdin:
        raise ValueError('Use either --password or --password-stdin, not both.')

    if args.password_stdin:
        return sys.stdin.read().strip()

    if args.password:
        return getpass.getpass(f'{auth_kind} for {args.nation}: ')

    return None


def read_lm_api_key_from_args(args: argparse.Namespace) -> str | None:
    if getattr(args, 'lm_api_key', False) and getattr(args, 'lm_api_key_stdin', False):
        raise ValueError('Use either --lm-api-key or --lm-api-key-stdin, not both.')

    if getattr(args, 'lm_api_key_stdin', False):
        return sys.stdin.read().strip()

    if getattr(args, 'lm_api_key', False):
        return getpass.getpass(f'LM API key for {args.nation}: ')

    return None


def add_nation_arguments(subparsers: argparse._SubParsersAction) -> None:
    nation_parser = subparsers.add_parser(
        'nation',
        help='Manage saved NationStates nation configs and secure auth secrets.',
    )
    nation_subparsers = nation_parser.add_subparsers(dest='nation_command')

    set_parser = nation_subparsers.add_parser(
        'set',
        help='Create or update a nation config.',
    )
    set_parser.add_argument('nation', help='NationStates nation name.')
    set_parser.add_argument(
        '--user-agent',
        help='Informative NationStates API User-Agent for this nation.',
    )
    set_parser.add_argument(
        '--api-version',
        type=int,
        help='Optional NationStates API version.',
    )
    set_parser.add_argument(
        '--profile',
        help='Default governance profile JSON path to copy into NSAI storage for this nation.',
    )
    set_parser.add_argument(
        '--move-profile',
        action='store_true',
        help='Move --profile into NSAI storage instead of copying it.',
    )
    set_parser.add_argument(
        '--base-url',
        help='Default OpenAI-compatible local model base URL for advisor calls.',
    )
    set_parser.add_argument(
        '--model',
        help='Default OpenAI-compatible local model name for advisor calls.',
    )
    set_parser.add_argument(
        '--lm-api-key',
        action='store_true',
        help='Prompt securely and store the advisor LM API key in the OS credential store.',
    )
    set_parser.add_argument(
        '--lm-api-key-stdin',
        action='store_true',
        help='Read the advisor LM API key from stdin and store it in the OS credential store.',
    )
    set_parser.add_argument(
        '--secret-backend',
        choices=SECRET_BACKENDS,
        default=None,
        help=(
            'Secret backend for newly stored NationStates and LM API secrets. '
            'Defaults to windows-hello on Windows, keyring elsewhere.'
        ),
    )
    set_parser.add_argument(
        '--clear-lm-api-key',
        action='store_true',
        help='Remove the saved advisor LM API key reference and stored secret if present.',
    )
    set_parser.add_argument(
        '--draft-dispatch',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Default whether advise asks the AI for a dispatch draft.',
    )
    set_parser.add_argument(
        '--draft-factbook',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Default whether advise asks the AI for a pertinent factbook draft.',
    )
    set_parser.add_argument(
        '--auth-kind',
        choices=AUTH_KINDS,
        default=None,
        help='Which private API credential kind to use.',
    )
    set_parser.add_argument(
        '--password',
        action='store_true',
        help='Prompt securely and store the private API secret in the OS credential store.',
    )
    set_parser.add_argument(
        '--password-stdin',
        action='store_true',
        help='Read the private API secret from stdin and store it in the OS credential store.',
    )
    set_parser.add_argument(
        '--clear-auth',
        action='store_true',
        help='Remove the saved auth reference and delete the stored secret if present.',
    )
    set_parser.set_defaults(func=run_nation_set)

    login_parser = nation_subparsers.add_parser(
        'login',
        help='Authenticate once and securely cache X-Pin plus X-Autologin.',
    )
    login_parser.add_argument('nation', help='NationStates nation name.')
    login_parser.add_argument(
        '--user-agent',
        help='Informative NationStates API User-Agent to save for this nation.',
    )
    login_parser.add_argument(
        '--password',
        action='store_true',
        help='Prompt securely for the nation password; the password is not saved.',
    )
    login_parser.add_argument(
        '--password-stdin',
        action='store_true',
        help='Read the nation password from stdin; the password is not saved.',
    )
    login_parser.add_argument(
        '--secret-backend',
        choices=SECRET_BACKENDS,
        default=None,
        help=(
            'Secret backend for the returned X-Pin and X-Autologin token. '
            'Defaults to the existing backend, or windows-hello on Windows.'
        ),
    )
    login_parser.set_defaults(func=run_nation_login)

    show_parser = nation_subparsers.add_parser(
        'show',
        help='Show a saved nation config without revealing secrets.',
    )
    show_parser.add_argument('nation', help='NationStates nation name.')
    show_parser.set_defaults(func=run_nation_show)

    list_parser = nation_subparsers.add_parser(
        'list',
        help='List saved nation configs.',
    )
    list_parser.set_defaults(func=run_nation_list)

    paths_parser = nation_subparsers.add_parser(
        'paths',
        help='Show where NSAI stores nation config files.',
    )
    paths_parser.set_defaults(func=run_nation_paths)

    remove_parser = nation_subparsers.add_parser(
        'remove',
        help='Remove a saved nation config.',
    )
    remove_parser.add_argument('nation', help='NationStates nation name.')
    remove_parser.add_argument(
        '--keep-secret',
        action='store_true',
        help='Remove only the JSON config and leave any credential-store secret untouched.',
    )
    remove_parser.set_defaults(func=run_nation_remove)


def run_nation_set(args: argparse.Namespace) -> None:
    if args.clear_auth and (args.password or args.password_stdin):
        raise ValueError('Use either --clear-auth or a password storage option, not both.')

    if getattr(args, 'clear_lm_api_key', False) and (
        getattr(args, 'lm_api_key', False) or getattr(args, 'lm_api_key_stdin', False)
    ):
        raise ValueError('Use either --clear-lm-api-key or an LM API key storage option, not both.')

    if getattr(args, 'password_stdin', False) and getattr(args, 'lm_api_key_stdin', False):
        raise ValueError('Only one stdin secret can be read at a time.')

    if getattr(args, 'move_profile', False) and not args.profile:
        raise ValueError('--move-profile requires --profile.')

    existing = maybe_load_nation_config(args.nation)
    auth_kind = args.auth_kind or (existing.auth_kind if existing else None)

    if args.password or args.password_stdin:
        auth_kind = auth_kind or 'password'

    if args.clear_auth:
        if existing and existing.credential_key:
            delete_secret(
                existing.credential_key,
                backend=existing.credential_backend or SECRET_BACKEND_KEYRING,
                reason=f'Delete NationStates secret for {args.nation}',
            )
        if existing and existing.pin_credential_key:
            delete_secret(
                existing.pin_credential_key,
                backend=existing.credential_backend or SECRET_BACKEND_KEYRING,
                reason=f'Delete NationStates PIN for {args.nation}',
            )
        auth_kind = None
        credential_key = None
        pin_credential_key = None
        credential_backend = None
    else:
        credential_key = (
            credential_key_for(args.nation, auth_kind)
            if auth_kind
            else (existing.credential_key if existing else None)
        )
        pin_credential_key = existing.pin_credential_key if existing else None
        credential_backend = existing.credential_backend if existing else None

    secret = read_secret_from_args(args, auth_kind or 'password')
    if secret is not None:
        if existing and existing.pin_credential_key:
            delete_secret(
                existing.pin_credential_key,
                backend=existing.credential_backend or SECRET_BACKEND_KEYRING,
                reason=f'Delete stale NationStates PIN for {args.nation}',
            )
            pin_credential_key = None
        if not auth_kind or not credential_key:
            auth_kind = 'password'
            credential_key = credential_key_for(args.nation, auth_kind)
        credential_backend = getattr(args, 'secret_backend', None) or default_secret_backend()
        set_secret(
            credential_key,
            secret,
            backend=credential_backend,
            reason=f'Store NationStates secret for {args.nation}',
        )

    lm_api_key_credential_key = (
        existing.lm_api_key_credential_key
        if existing
        else None
    )
    if getattr(args, 'clear_lm_api_key', False):
        if lm_api_key_credential_key:
            delete_secret(
                lm_api_key_credential_key,
                backend=existing.lm_api_key_backend if existing else SECRET_BACKEND_KEYRING,
                reason=f'Delete LM API key for {args.nation}',
            )
        lm_api_key_credential_key = None
        lm_api_key_backend = None
    else:
        lm_api_key_backend = existing.lm_api_key_backend if existing else None

    lm_api_key = read_lm_api_key_from_args(args)
    if lm_api_key is not None:
        lm_api_key_credential_key = lm_api_key_credential_key_for(args.nation)
        lm_api_key_backend = getattr(args, 'secret_backend', None) or default_secret_backend()
        set_secret(
            lm_api_key_credential_key,
            lm_api_key,
            backend=lm_api_key_backend,
            reason=f'Store LM API key for {args.nation}',
        )

    profile_path = args.profile
    profile_action = None
    if profile_path:
        stored_path, profile_action = store_profile_for_nation(
            args.nation,
            profile_path,
            move=getattr(args, 'move_profile', False),
        )
        profile_path = str(stored_path)
    elif existing:
        profile_path = existing.profile_path

    config = NationConfig(
        nation_name=args.nation,
        user_agent=args.user_agent if args.user_agent is not None else (existing.user_agent if existing else None),
        api_version=args.api_version if args.api_version is not None else (existing.api_version if existing else None),
        profile_path=profile_path,
        strategy=existing.strategy if existing else None,
        show_issues=existing.show_issues if existing else None,
        show_instruction=existing.show_instruction if existing else None,
        no_ai=existing.no_ai if existing else None,
        audit_log=existing.audit_log if existing else None,
        lm_base_url=(
            args.base_url
            if getattr(args, 'base_url', None) is not None
            else (existing.lm_base_url if existing else None)
        ),
        lm_model=(
            args.model
            if getattr(args, 'model', None) is not None
            else (existing.lm_model if existing else None)
        ),
        lm_api_key_credential_key=lm_api_key_credential_key,
        draft_dispatch=(
            args.draft_dispatch
            if getattr(args, 'draft_dispatch', None) is not None
            else (existing.draft_dispatch if existing else None)
        ),
        draft_factbook=(
            args.draft_factbook
            if getattr(args, 'draft_factbook', None) is not None
            else (existing.draft_factbook if existing else None)
        ),
        auth_kind=auth_kind,
        credential_key=credential_key,
        pin_credential_key=pin_credential_key,
        credential_backend=credential_backend,
        lm_api_key_backend=lm_api_key_backend,
    )
    path = save_nation_config(config)

    print(f'Saved nation config: {path}')
    if profile_path and profile_action:
        print(f'Profile {profile_action} into NSAI storage: {profile_path}')
    if config.secret_configured:
        print(
            'Secret stored in OS credential store as: '
            f'{config.credential_key} ({config.credential_backend or SECRET_BACKEND_KEYRING})'
        )
    else:
        print('No private API secret is configured for this nation.')
    if config.lm_api_key_credential_key:
        print(
            'LM API key stored in OS credential store as: '
            f'{config.lm_api_key_credential_key} '
            f'({config.lm_api_key_backend or SECRET_BACKEND_KEYRING})'
        )


def run_nation_login(args: argparse.Namespace) -> None:
    """Create a NationStates session and securely persist its returned credentials."""

    from nsai.advisor.client import (
        NationStatesClient,
        NationStatesError,
        build_default_user_agent,
    )

    if args.password and args.password_stdin:
        raise ValueError('Use either --password or --password-stdin, not both.')

    existing = maybe_load_nation_config(args.nation)
    password = read_secret_from_args(args, 'password')
    user_agent = (
        args.user_agent
        or (existing.user_agent if existing else None)
        or os.environ.get('NS_USER_AGENT')
        or build_default_user_agent(args.nation)
    )

    if password is not None:
        client = NationStatesClient(
            user_agent=user_agent,
            api_version=existing.api_version if existing else None,
            password=password,
        )
    else:
        lookup_config = existing or NationConfig(
            nation_name=args.nation,
            user_agent=user_agent,
        )
        client = NationStatesClient.from_env(lookup_config)

    client.establish_session(args.nation)
    if not client.autologin:
        raise NationStatesError(
            'NationStates returned an X-Pin but no reusable X-Autologin token. '
            'Run this command with --password to create one.'
        )

    backend = (
        args.secret_backend
        or (existing.credential_backend if existing else None)
        or default_secret_backend()
    )
    credential_key = credential_key_for(args.nation, 'autologin')
    pin_credential_key = credential_key_for(args.nation, 'pin')
    set_secret(
        credential_key,
        client.autologin,
        backend=backend,
        reason=f'Store NationStates autologin token for {args.nation}',
    )
    set_secret(
        pin_credential_key,
        client.pin,
        backend=backend,
        reason=f'Store NationStates PIN for {args.nation}',
    )

    if existing:
        config = replace(
            existing,
            user_agent=user_agent,
            auth_kind='autologin',
            credential_key=credential_key,
            pin_credential_key=pin_credential_key,
            credential_backend=backend,
        )
    else:
        config = NationConfig(
            nation_name=args.nation,
            user_agent=user_agent,
            auth_kind='autologin',
            credential_key=credential_key,
            pin_credential_key=pin_credential_key,
            credential_backend=backend,
        )

    save_nation_config(config)

    old_key = existing.credential_key if existing else None
    if old_key and old_key != credential_key:
        delete_secret(
            old_key,
            backend=existing.credential_backend or SECRET_BACKEND_KEYRING,
            reason=f'Delete replaced NationStates secret for {args.nation}',
        )

    print(f'NationStates login succeeded for {args.nation}.')
    print('A fresh X-Pin was obtained and securely cached; it was not printed.')
    print(
        'Saved the reusable X-Autologin token in the OS credential store as: '
        f'{credential_key} ({backend})'
    )
    print(
        'NSAI will reuse the cached PIN across commands and retain X-Autologin '
        'for the next explicit login refresh.'
    )


def nation_config_summary(config: NationConfig) -> dict[str, Any]:
    secret_available = False
    if config.credential_key:
        try:
            secret_available = bool(
                get_secret(
                    config.credential_key,
                    backend=config.credential_backend or SECRET_BACKEND_KEYRING,
                    reason=f'Check NationStates secret for {config.nation_name}',
                )
            )
        except SecureStoreError:
            secret_available = False

    lm_api_key_available = False
    if config.lm_api_key_credential_key:
        try:
            lm_api_key_available = bool(
                get_secret(
                    config.lm_api_key_credential_key,
                    backend=config.lm_api_key_backend or SECRET_BACKEND_KEYRING,
                    reason=f'Check LM API key for {config.nation_name}',
                )
            )
        except SecureStoreError:
            lm_api_key_available = False

    return {
        'nation_name': config.nation_name,
        'user_agent': config.user_agent,
        'api_version': config.api_version,
        'profile_path': config.profile_path,
        'strategy': config.strategy,
        'show_issues': config.show_issues,
        'show_instruction': config.show_instruction,
        'no_ai': config.no_ai,
        'audit_log': config.audit_log,
        'lm_base_url': config.lm_base_url,
        'lm_model': config.lm_model,
        'lm_api_key_configured': bool(config.lm_api_key_credential_key),
        'lm_api_key_available': lm_api_key_available,
        'draft_dispatch': config.draft_dispatch,
        'draft_factbook': config.draft_factbook,
        'auth_kind': config.auth_kind,
        'pin_configured': bool(config.pin_credential_key),
        'credential_backend': config.credential_backend,
        'lm_api_key_backend': config.lm_api_key_backend,
        'secret_configured': config.secret_configured,
        'secret_available': secret_available,
        'config_path': str(config_path_for(config.nation_name)),
        'updated_at': config.updated_at,
    }


def run_nation_show(args: argparse.Namespace) -> None:
    config = load_nation_config(args.nation)
    print(json.dumps(nation_config_summary(config), indent=2, ensure_ascii=False))


def run_nation_list(args: argparse.Namespace) -> None:
    configs = list_nation_configs()
    if not configs:
        print(f'No nation configs found in {nations_dir()}')
        return

    for config in configs:
        secret = 'secret=yes' if config.secret_configured else 'secret=no'
        profile = config.profile_path or '-'
        print(f'{config.nation_name}\t{secret}\tprofile={profile}')


def run_nation_paths(args: argparse.Namespace) -> None:
    print(f'Config root: {config_root()}')
    print(f'Program config: {app_config_path()}')
    print(f'Nation configs: {nations_dir()}')
    print(f'Managed profiles: {profiles_dir()}')
    print(f'Advice cache: {advice_cache_path()}')


def run_nation_remove(args: argparse.Namespace) -> None:
    config = maybe_load_nation_config(args.nation)
    path = config_path_for(args.nation)

    if config and config.credential_key and not args.keep_secret:
        delete_secret(
            config.credential_key,
            backend=config.credential_backend or SECRET_BACKEND_KEYRING,
            reason=f'Delete NationStates secret for {args.nation}',
        )
    if config and config.pin_credential_key and not args.keep_secret:
        delete_secret(
            config.pin_credential_key,
            backend=config.credential_backend or SECRET_BACKEND_KEYRING,
            reason=f'Delete NationStates PIN for {args.nation}',
        )
    if config and config.lm_api_key_credential_key and not args.keep_secret:
        delete_secret(
            config.lm_api_key_credential_key,
            backend=config.lm_api_key_backend or SECRET_BACKEND_KEYRING,
            reason=f'Delete LM API key for {args.nation}',
        )

    if path.exists():
        path.unlink()
        print(f'Removed nation config: {path}')
    else:
        print(f'No nation config existed at: {path}')
