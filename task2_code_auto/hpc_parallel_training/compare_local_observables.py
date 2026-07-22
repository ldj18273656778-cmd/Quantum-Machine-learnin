"""Compatibility wrapper for the parallel local-observable comparison script.

The implementation lives in
``task2_code.hpc_parallel_training.local_observable_parallel.compare_local_observables``.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task2_code_auto.hpc_parallel_training.local_observable_parallel.compare_local_observables import main


if __name__ == "__main__":
    main()
