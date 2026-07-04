"""Compatibility wrapper for the packaged NSAI live advisor."""

from __future__ import annotations

import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))

from nsai.advisor.live import main


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled.')
        sys.exit(130)
    except Exception as exc:
        print(f'\nERROR: {exc}')
        sys.exit(1)
