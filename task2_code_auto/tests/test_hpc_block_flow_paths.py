"""Path contracts for task2_code_auto experiment helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

from task2_code_auto.hpc_parallel_training import create_parallel_experiment


class HpcBlockFlowPathTests(unittest.TestCase):
    def test_create_parallel_experiment_defaults_inside_task2_code_auto(self) -> None:
        with mock.patch(
            "sys.argv",
            ["create_parallel_experiment.py", "--preset", "n4_single_block"],
        ):
            args = create_parallel_experiment.parse_args()

        self.assertEqual(
            args.output_dir,
            Path("task2_code_auto/hpc_parallel_training/data"),
        )

    def test_explicit_output_dir_remains_unchanged(self) -> None:
        explicit = Path("report/custom_phase4")
        with mock.patch(
            "sys.argv",
            [
                "create_parallel_experiment.py",
                "--preset",
                "n4_single_block",
                "--output-dir",
                str(explicit),
            ],
        ):
            args = create_parallel_experiment.parse_args()

        self.assertEqual(args.output_dir, explicit)


if __name__ == "__main__":
    unittest.main()
