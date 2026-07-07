"""Run the NSAI command-line app with ``python -m nsai``."""

from __future__ import annotations

import sys

from nsai.cli import main


if __name__ == '__main__':
    sys.exit(main())
