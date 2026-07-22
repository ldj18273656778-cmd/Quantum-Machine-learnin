"""Phase 4 contracts for the legacy training backend selector."""

from __future__ import annotations

import unittest
import io
import contextlib
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from task2_code_auto import module_e_training as training
from task2_code_auto.jax_backend.heisenberg import heisenberg_pauli_loss
from task2_code_auto.loss_registry import set_active_loss_function
from task2_code_auto.tests.test_jax_backend_parity import deterministic_random_theta, n4_context
from task2_code_auto.hpc_parallel_training import train_block


@dataclass(frozen=True, slots=True)
class FakeParallelConfig:
    loss_function: str = "heisenberg_pauli"
    block_count: int = 1
    blocks: tuple[tuple[int, ...], ...] = ((0, 1),)
    target_bits: tuple[int, ...] = (0,)
    n_qubits: int = 2
    radius: int = 1
    target_seed: int = 123
    time_k: int = 1
    lightcone_mode: str = "block"
    loss_mode: str = "per_bit"
    ansatz: str = "test"
    block_only_ansatz: bool = True
    superoperator: str = "cnot"
    iterations: int = 2
    lr: float = 0.05
    max_restarts: int = 1
    success_threshold: float = 0.0

    def training_seed_for_block(self, _block_index: int) -> int:
        return 456


@dataclass(frozen=True, slots=True)
class FakeObjectiveContext:
    lightcone_qubits: tuple[int, ...] = (0, 1)
    theta_size: int = 2
    ansatz_qubits: int = 2
    ansatz: str = "test"
    block_only_ansatz: bool = True


@dataclass(frozen=True, slots=True)
class FakeAdamResult:
    loss_history: np.ndarray
    best_params: np.ndarray
    best_loss: float
    best_iteration: int
    failed: bool = False
    failure_reason: str = ""


@dataclass(frozen=True, slots=True)
class FakeMultiRestartResult:
    restart_results: tuple[FakeAdamResult, ...]
    best_restart: int
    best_iteration: int
    best_loss: float
    best_params: np.ndarray


class Phase4GradientBackendTests(unittest.TestCase):
    def test_adam_optimize_defaults_to_finite_difference(self) -> None:
        theta = np.array([0.2, -0.3], dtype=np.float64)
        calls = 0

        def loss_fn(candidate: np.ndarray) -> float:
            return float(np.sum(candidate * candidate))

        result = training.adam_optimize(
            loss_fn,
            theta,
            training.AdamConfig(iterations=1, wrap_angles=False),
        )

        self.assertFalse(result.failed, result.failure_reason)
        np.testing.assert_array_equal(result.initial_params, theta)

    def test_adam_optimize_rejects_unknown_gradient_backend(self) -> None:
        invalid_backend = "unknown"
        with self.assertRaisesRegex(ValueError, "gradient_backend"):
            training.adam_optimize(
                lambda candidate: float(np.sum(candidate * candidate)),
                np.array([0.2, -0.3], dtype=np.float64),
                training.AdamConfig(iterations=0),
                gradient_backend=invalid_backend,
            )

    def test_adam_optimize_jax_requires_context_aware_adapter(self) -> None:
        with self.assertRaisesRegex(ValueError, "objective_context"):
            training.adam_optimize(
                lambda candidate: float(np.sum(candidate * candidate)),
                np.array([0.2, -0.3], dtype=np.float64),
                training.AdamConfig(iterations=0),
                gradient_backend="jax",
            )

    def test_adam_optimize_jax_returns_legacy_numpy_result_and_callback_history(self) -> None:
        context = n4_context()
        theta = deterministic_random_theta()
        snapshots: list[tuple[int, np.ndarray, float]] = []

        result = training.adam_optimize(
            lambda candidate: float(heisenberg_pauli_loss(candidate, context)),
            theta,
            training.AdamConfig(iterations=1, lr=0.05, wrap_angles=False),
            gradient_backend="jax",
            objective_context=context,
            jax_loss_fn=heisenberg_pauli_loss,
            step_callback=lambda step, params, loss: snapshots.append(
                (step, np.asarray(params, dtype=np.float64), float(loss))
            ),
        )
        self.assertFalse(result.failed, result.failure_reason)
        self.assertIsInstance(result.initial_params, np.ndarray)
        self.assertEqual(result.loss_history.shape, (2,))
        self.assertEqual([step for step, _params, _loss in snapshots], [0, 1])
        np.testing.assert_array_equal(snapshots[0][1], theta)
        np.testing.assert_allclose(snapshots[-1][1], result.final_params)

    def test_legacy_and_jax_objectives_match_when_active_loss_is_explicit(self) -> None:
        context = n4_context()
        theta = deterministic_random_theta()
        set_active_loss_function("heisenberg_pauli")
        legacy_loss = training.sum_block_loss(theta, context)
        jax_loss = float(heisenberg_pauli_loss(theta, context))
        self.assertAlmostEqual(legacy_loss, jax_loss, places=12)

    def test_train_block_parser_defaults_to_finite_difference_and_accepts_jax(self) -> None:
        with mock.patch("sys.argv", ["train_block.py", "--experiment-root", "run", "--block-index", "1"]):
            self.assertEqual(train_block.parse_args().gradient_backend, "finite-difference")
        with mock.patch(
            "sys.argv",
            [
                "train_block.py",
                "--experiment-root",
                "run",
                "--block-index",
                "1",
                "--gradient-backend",
                "jax",
            ],
        ):
            self.assertEqual(train_block.parse_args().gradient_backend, "jax")

    def test_train_block_parser_rejects_telemetry_path_when_parent_is_missing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            missing_parent_path = Path(temporary_directory) / "missing" / "memory.jsonl"
            with mock.patch(
                "sys.argv",
                [
                    "train_block.py",
                    "--experiment-root",
                    "run",
                    "--block-index",
                    "1",
                    "--memory-telemetry-path",
                    str(missing_parent_path),
                ],
            ):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit):
                        train_block.parse_args()
                self.assertIn("parent directory", stderr.getvalue())

    def test_train_block_writes_jax_memory_telemetry_after_each_completed_iteration(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            experiment_root = Path(temporary_directory) / "experiment"
            telemetry_path = Path(temporary_directory) / "memory.jsonl"
            argv = [
                "train_block.py",
                "--experiment-root",
                str(experiment_root),
                "--block-index",
                "1",
                "--gradient-backend",
                "jax",
                "--memory-telemetry-path",
                str(telemetry_path),
                "--no-progress",
            ]

            def fake_multi_restart_train(*_args, step_callback, **_kwargs):
                step_callback(0, np.asarray([0.0, 0.0], dtype=np.float64), 4.0)
                step_callback(1, np.asarray([0.1, 0.2], dtype=np.float64), 3.0)
                step_callback(2, np.asarray([0.3, 0.4], dtype=np.float64), 2.0)
                result = FakeAdamResult(
                    loss_history=np.asarray([3.0, 2.0], dtype=np.float64),
                    best_params=np.asarray([0.3, 0.4], dtype=np.float64),
                    best_loss=2.0,
                    best_iteration=2,
                )
                return FakeMultiRestartResult(
                    restart_results=(result,),
                    best_restart=0,
                    best_iteration=2,
                    best_loss=2.0,
                    best_params=np.asarray([0.3, 0.4], dtype=np.float64),
                )

            with mock.patch("sys.argv", argv):
                with mock.patch.object(train_block, "load_manifest", return_value={"preset": "n4_single_block_heisenberg"}):
                    with mock.patch.object(train_block, "config_from_manifest", return_value=FakeParallelConfig()):
                        with mock.patch.object(train_block, "build_target_objective_context", return_value=(FakeObjectiveContext(), {})):
                            with mock.patch.object(train_block, "multi_restart_train", side_effect=fake_multi_restart_train):
                                with mock.patch.object(train_block, "active_loss_breakdown", return_value={"0": 2.0}):
                                    with mock.patch.object(train_block, "residual_operator_for_context", return_value=(None, (0,))):
                                        train_block.main()

            records = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["iteration"] for record in records], [1, 2])
            self.assertEqual([record["loss"] for record in records], [3.0, 2.0])
            self.assertEqual([record["best_loss"] for record in records], [3.0, 2.0])
            for record in records:
                self.assertGreaterEqual(record["elapsed_step_seconds"], 0.0)
                self.assertGreater(record["process_rss_bytes"], 0)
                self.assertGreaterEqual(record["available_physical_memory_bytes"], 0)

    def test_train_block_ignores_telemetry_path_for_finite_difference_backend(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            experiment_root = Path(temporary_directory) / "experiment"
            telemetry_path = Path(temporary_directory) / "memory.jsonl"
            argv = [
                "train_block.py",
                "--experiment-root",
                str(experiment_root),
                "--block-index",
                "1",
                "--memory-telemetry-path",
                str(telemetry_path),
                "--no-progress",
            ]

            def fake_multi_restart_train(*_args, step_callback, **_kwargs):
                step_callback(0, np.asarray([0.0, 0.0], dtype=np.float64), 4.0)
                step_callback(1, np.asarray([0.1, 0.2], dtype=np.float64), 3.0)
                result = FakeAdamResult(
                    loss_history=np.asarray([3.0], dtype=np.float64),
                    best_params=np.asarray([0.1, 0.2], dtype=np.float64),
                    best_loss=3.0,
                    best_iteration=1,
                )
                return FakeMultiRestartResult(
                    restart_results=(result,),
                    best_restart=0,
                    best_iteration=1,
                    best_loss=3.0,
                    best_params=np.asarray([0.1, 0.2], dtype=np.float64),
                )

            with mock.patch("sys.argv", argv):
                with mock.patch.object(train_block, "load_manifest", return_value={"preset": "n4_single_block_heisenberg"}):
                    with mock.patch.object(train_block, "config_from_manifest", return_value=FakeParallelConfig()):
                        with mock.patch.object(train_block, "build_target_objective_context", return_value=(FakeObjectiveContext(), {})):
                            with mock.patch.object(train_block, "multi_restart_train", side_effect=fake_multi_restart_train):
                                with mock.patch.object(train_block, "active_loss_breakdown", return_value={"0": 3.0}):
                                    with mock.patch.object(train_block, "residual_operator_for_context", return_value=(None, (0,))):
                                        train_block.main()

            self.assertFalse(telemetry_path.exists())

    def test_jax_training_emits_progress_when_requested(self) -> None:
        context = n4_context()
        theta = deterministic_random_theta()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = training.adam_optimize(
                lambda candidate: float(heisenberg_pauli_loss(candidate, context)),
                theta,
                training.AdamConfig(iterations=1, lr=0.05, wrap_angles=False),
                gradient_backend="jax",
                objective_context=context,
                jax_loss_fn=heisenberg_pauli_loss,
                show_progress=True,
            )

        self.assertFalse(result.failed, result.failure_reason)
        self.assertIn("1/1", output.getvalue())


if __name__ == "__main__":
    unittest.main()
