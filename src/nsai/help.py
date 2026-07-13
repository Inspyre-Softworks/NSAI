"""Shared command-line help formatting for NSAI."""

from __future__ import annotations

import argparse
from typing import Any

try:
    from rich_argparse import RichHelpFormatter as _BaseHelpFormatter
except ImportError:  # pragma: no cover - dependency fallback for source checkouts
    _BaseHelpFormatter = argparse.HelpFormatter


class NSAIHelpFormatter(_BaseHelpFormatter):
    """Colorful argparse help formatter with a plain argparse fallback."""

    if hasattr(_BaseHelpFormatter, 'styles'):
        styles = {
            **_BaseHelpFormatter.styles,
            'argparse.args': 'cyan',
            'argparse.groups': 'bold blue',
            'argparse.help': 'default',
            'argparse.metavar': 'bold yellow',
            'argparse.prog': 'bold cyan',
            'argparse.syntax': 'bold',
        }


class NSAIArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that keeps Rich help formatting across subcommands."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault('formatter_class', NSAIHelpFormatter)
        super().__init__(*args, **kwargs)
