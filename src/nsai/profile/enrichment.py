"""Profile AI enrichment – addendum generation and validation."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI


DEFAULT_LM_BASE_URL = 'http://localhost:1234/v1'
DEFAULT_LM_API_KEY = 'lm-studio'
LM_BASE_URL = DEFAULT_LM_BASE_URL
LM_MODEL = None


def get_local_model_name(client: OpenAI, *, model: str | None = None) -> str:
    if model:
        return model

    env_model = os.environ.get('LM_STUDIO_MODEL')
    if env_model:
        return env_model

    models = client.models.list()

    if not models.data:
        raise RuntimeError('No local Studio model is loaded.')

    return models.data[0].id


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


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def fallback_ai_governance_addendum(profile_data: dict[str, Any]) -> dict[str, Any]:
    """Used if the local AI is unavailable or returns malformed output."""
    nation_name = profile_data.get('nation_name', 'the nation')
    profile_name = profile_data.get('profile_name', 'Governance Profile')
    vision = profile_data.get('national_vision', '')
    priorities = profile_data.get('top_priorities', [])

    priority_text = ', '.join(priorities) if priorities else 'national stability'

    return {
        'vision_description': (
            f'{nation_name} shall be governed under the {profile_name} doctrine: '
            f'a state guided by {priority_text}, with policy decisions measured against '
            f'the long-term national vision. {vision}'
        ),
        'short_constitution': {
            'title': f'Compact Charter of {nation_name}',
            'preamble': (
                f'We establish this charter to guide the AI governor of {nation_name} '
                'toward stable, consistent, and user-approved rule.'
            ),
            'articles': [
                {
                    'number': 1,
                    'title': 'Purpose of Government',
                    'text': 'The government shall pursue the national vision defined by the user-approved governance profile.',
                },
                {
                    'number': 2,
                    'title': 'Policy Priorities',
                    'text': f'The AI governor shall prioritize: {priority_text}.',
                },
                {
                    'number': 3,
                    'title': 'Limits on Authority',
                    'text': 'The AI governor shall respect all red lines and reject policies that violate unacceptable tradeoffs.',
                },
                {
                    'number': 4,
                    'title': 'Risk and Autonomy',
                    'text': 'The AI governor shall act only within the selected risk tolerance and enactment mode.',
                },
                {
                    'number': 5,
                    'title': 'Auditability',
                    'text': 'Every autonomous decision should be explainable, logged, and reviewable by the user.',
                },
            ],
        },
        'generation_source': 'fallback',
    }


def validate_addendum(addendum: dict[str, Any]) -> None:
    if not isinstance(addendum.get('vision_description'), str):
        raise ValueError('AI addendum missing vision_description string.')

    constitution = addendum.get('short_constitution')
    if not isinstance(constitution, dict):
        raise ValueError('AI addendum missing short_constitution object.')

    if not isinstance(constitution.get('title'), str):
        raise ValueError('Constitution missing title.')

    if not isinstance(constitution.get('preamble'), str):
        raise ValueError('Constitution missing preamble.')

    articles = constitution.get('articles')
    if not isinstance(articles, list) or len(articles) != 5:
        raise ValueError('Constitution must contain exactly 5 articles.')

    for index, article in enumerate(articles, start=1):
        if not isinstance(article, dict):
            raise ValueError(f'Article {index} is not an object.')

        if not isinstance(article.get('number'), int):
            raise ValueError(f'Article {index} missing integer number.')

        if not isinstance(article.get('title'), str):
            raise ValueError(f'Article {index} missing title.')

        if not isinstance(article.get('text'), str):
            raise ValueError(f'Article {index} missing text.')


def generate_ai_governance_addendum(
    profile_data: dict[str, Any],
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Send the completed interview/profile JSON to the local AI and ask it to append
    a vision description and short constitution.
    """
    client = OpenAI(
        base_url=(
            base_url
            or os.environ.get('LM_STUDIO_BASE_URL')
            or DEFAULT_LM_BASE_URL
        ),
        api_key=api_key or os.environ.get('LM_STUDIO_API_KEY') or DEFAULT_LM_API_KEY,
    )

    model = get_local_model_name(client, model=model)

    system = """
You are a political worldbuilding assistant for a NationStates AI governor.

You will receive a JSON governance profile created from a user interview.

Create a polished but concise governance addendum.

Return strict JSON only. No markdown.

Required JSON shape:
{
  "vision_description": "string",
  "short_constitution": {
    "title": "string",
    "preamble": "string",
    "articles": [
      {
        "number": 1,
        "title": "string",
        "text": "string"
      }
    ]
  }
}

Rules:
- Keep the vision_description to 1 or 2 paragraphs.
- Create exactly 5 constitution articles.
- The constitution must reflect the user's priorities, red lines, risk tolerance, and enactment mode.
- Do not override the user's profile.
- Do not add new political goals that are not implied by the profile.
- Keep it usable as instructions for an AI governor.
- If the profile is silly, theatrical, authoritarian, libertarian, religious, corporate, or utopian, preserve that chosen flavor.
- The user is sovereign; the AI governor is bounded by this charter.
"""

    user = {
        'task': 'Generate vision_description and short_constitution for this AI governor profile.',
        'profile': profile_data,
    }

    schema = {
        'type': 'object',
        'properties': {
            'vision_description': {
                'type': 'string',
            },
            'short_constitution': {
                'type': 'object',
                'properties': {
                    'title': {'type': 'string'},
                    'preamble': {'type': 'string'},
                    'articles': {
                        'type': 'array',
                        'minItems': 5,
                        'maxItems': 5,
                        'items': {
                            'type': 'object',
                            'properties': {
                                'number': {'type': 'integer'},
                                'title': {'type': 'string'},
                                'text': {'type': 'string'},
                            },
                            'required': ['number', 'title', 'text'],
                            'additionalProperties': False,
                        },
                    },
                },
                'required': ['title', 'preamble', 'articles'],
                'additionalProperties': False,
            },
        },
        'required': ['vision_description', 'short_constitution'],
        'additionalProperties': False,
    }

    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': json.dumps(user, indent=2)},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.45,
            response_format={
                'type': 'json_schema',
                'json_schema': {
                    'name': 'GovernanceAddendum',
                    'schema': schema,
                    'strict': True,
                },
            },
        )
    except Exception:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.45,
            response_format={'type': 'text'},
        )

    text = get_completion_text(response)

    if not text:
        raise ValueError('Local AI returned no content.')

    addendum = parse_json_object(text)
    validate_addendum(addendum)

    addendum['generation_source'] = 'local_ai'
    addendum['generated_at'] = datetime.now(timezone.utc).isoformat()
    addendum['model'] = model

    return addendum


def append_ai_generated_governance(
    profile_data: dict[str, Any],
    *,
    force: bool = False,
    use_fallback: bool = True,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Append AI-generated vision/constitution material to the profile JSON.
    """
    import sys

    enriched = dict(profile_data)

    if 'ai_generated' in enriched and not force:
        return enriched

    # Use builder module's binding if available (supports monkeypatching in tests)
    _builder = sys.modules.get('nsai.profile.builder')
    _generate = getattr(_builder, 'generate_ai_governance_addendum', generate_ai_governance_addendum) if _builder else generate_ai_governance_addendum
    generation_kwargs = {
        key: value
        for key, value in {
            'base_url': base_url,
            'model': model,
            'api_key': api_key,
        }.items()
        if value is not None
    }

    try:
        addendum = _generate(profile_data, **generation_kwargs)
    except Exception as exc:
        if not use_fallback:
            raise

        addendum = fallback_ai_governance_addendum(profile_data)
        addendum['generation_warning'] = str(exc)
        addendum['generated_at'] = datetime.now(timezone.utc).isoformat()

    enriched['ai_generated'] = addendum
    enriched['updated_at'] = datetime.now(timezone.utc).isoformat()

    return enriched
