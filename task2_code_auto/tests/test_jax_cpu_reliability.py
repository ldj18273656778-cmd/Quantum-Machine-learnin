"""Reliability contract tests for the JAX CPU launcher."""

from __future__ import annotations

from pathlib import Path
import unittest


class JaxCpuReliabilityTests(unittest.TestCase):
    def test_cpu_launcher_sets_thread_limits_before_python_invocation(self) -> None:
        # Given: the repository-level JAX CPU launcher contract.
        launcher_path = Path(__file__).resolve().parents[1] / "hpc_parallel_training" / "run_jax_cpu_training.ps1"

        # When: the launcher script is present on disk.
        self.assertTrue(launcher_path.is_file(), f"missing launcher: {launcher_path}")

        # Then: its content sets all native thread limits before invoking QML Python.
        content = launcher_path.read_text(encoding="utf-8")
        expected_variables = (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
        for variable in expected_variables:
            with self.subTest(variable=variable):
                self.assertIn(f"$env:{variable} = '1'", content)

        python_call = "C:\\ProgramData\\anaconda3\\envs\\QML\\python.exe"
        self.assertIn(python_call, content)
        self.assertIn("ValueFromRemainingArguments", content)
        self.assertIn("Join-Path $PSScriptRoot 'train_block.py'", content)
        self.assertIn("$RemainingArgs", content)
        self.assertIn("$LASTEXITCODE", content)
        self.assertNotIn("XLA_FLAGS", content)

        env_positions = [content.index(f"$env:{variable} = '1'") for variable in expected_variables]
        python_position = content.index(python_call)
        self.assertLess(max(env_positions), python_position)


if __name__ == "__main__":
    _ = unittest.main()
