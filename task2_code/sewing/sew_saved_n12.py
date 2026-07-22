"""Backward-compatible entry point for the generalized block sewing module.

The implementation moved to ``task2_code.sewing.block_sewing`` because the
helpers are not n=12-specific.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from task2_code.sewing.block_sewing import *  # noqa: F403
from task2_code.sewing.block_sewing import main


if __name__ == "__main__":
    main()
