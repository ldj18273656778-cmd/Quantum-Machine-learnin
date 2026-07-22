"""Contracts for explicit JAX memory-mode selection."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol
from unittest import TestCase, main, mock

import numpy as np

from task2_code_auto import module_e_training as training
from task2_code_auto.hpc_parallel_training import train_block


class StepCallback(Protocol):
    def __call__(self, step: int, theta: np.ndarray, loss: float) -> None: ...


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


class JaxMemoryModeTests(TestCase):
    def test_adam_optimize_jax_rematerialized_matches_standard_outer_jit_for_one_and_five_steps(self) -> None:
        from task2_code_auto.jax_backend.heisenberg import heisenberg_pauli_loss, heisenberg_pauli_loss_rematerialized
        from task2_code_auto.tests.test_jax_backend_parity import deterministic_random_theta, n4_context

        context = n4_context()
        theta = deterministic_random_theta()

        for iterations in (1, 5):
            with self.subTest(iterations=iterations):
                standard_result = training.adam_optimize(
                    lambda _candidate: 0.0,
                    theta,
                    training.AdamConfig(iterations=iterations, lr=0.05, wrap_angles=False),
                    gradient_backend="jax",
                    objective_context=context,
                    jax_loss_fn=heisenberg_pauli_loss,
                    jax_memory_mode="standard",
                )
                rematerialized_result = training.adam_optimize(
                    lambda _candidate: 0.0,
                    theta,
                    training.AdamConfig(iterations=iterations, lr=0.05, wrap_angles=False),
                    gradient_backend="jax",
                    objective_context=context,
                    jax_loss_fn=heisenberg_pauli_loss_rematerialized,
                    jax_memory_mode="rematerialized",
                )

                self.assertFalse(standard_result.failed, standard_result.failure_reason)
                self.assertFalse(rematerialized_result.failed, rematerialized_result.failure_reason)
                np.testing.assert_allclose(rematerialized_result.loss_history, standard_result.loss_history, rtol=0.0, atol=1e-12)
                np.testing.assert_allclose(rematerialized_result.best_loss_history, standard_result.best_loss_history, rtol=0.0, atol=1e-12)
                np.testing.assert_allclose(rematerialized_result.grad_norm_history, standard_result.grad_norm_history, rtol=0.0, atol=1e-10)
                np.testing.assert_allclose(rematerialized_result.final_params, standard_result.final_params, rtol=0.0, atol=1e-10)
                np.testing.assert_allclose(rematerialized_result.best_params, standard_result.best_params, rtol=0.0, atol=1e-10)

    def test_train_block_parser_defaults_to_standard_jax_memory_mode(self) -> None:
        with mock.patch("sys.argv", ["train_block.py", "--experiment-root", "run", "--block-index", "1"]):
            self.assertEqual(train_block.parse_args().jax_memory_mode, "standard")

    def test_train_block_parser_rejects_rematerialized_memory_mode_without_jax_backend(self) -> None:
        with mock.patch(
            "sys.argv",
            ["train_block.py", "--experiment-root", "run", "--block-index", "1", "--jax-memory-mode", "rematerialized"],
        ):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit):
                    train_block.parse_args()
            self.assertIn("--jax-memory-mode rematerialized requires --gradient-backend jax", stderr.getvalue())

    def test_train_block_passes_rematerialized_loss_to_jax_training_without_result_schema_change(self) -> None:
        from task2_code_auto.jax_backend.heisenberg import heisenberg_pauli_loss_rematerialized

        with TemporaryDirectory() as temporary_directory:
            experiment_root = Path(temporary_directory) / "experiment"
            argv = [
                "train_block.py",
                "--experiment-root",
                str(experiment_root),
                "--block-index",
                "1",
                "--gradient-backend",
                "jax",
                "--jax-memory-mode",
                "rematerialized",
                "--no-progress",
            ]
            seen_loss_functions: list[object] = []

            def fake_multi_restart_train(
                *_args: object,
                jax_loss_fn: object,
                step_callback: StepCallback,
                **_kwargs: object,
            ) -> FakeMultiRestartResult:
                seen_loss_functions.append(jax_loss_fn)
                step_callback(0, np.asarray([0.0, 0.0], dtype=np.float64), 4.0)
                step_callback(1, np.asarray([0.1, 0.2], dtype=np.float64), 3.0)
                result = FakeAdamResult(
                    loss_history=np.asarray([4.0, 3.0], dtype=np.float64),
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

            payload = json.loads((experiment_root / "blocks" / "block_01" / "result.json").read_text(encoding="utf-8"))
            self.assertIs(seen_loss_functions[0], heisenberg_pauli_loss_rematerialized)
            self.assertEqual(payload["gradient_backend"], "jax")
            self.assertNotIn("jax_memory_mode", payload)


if __name__ == "__main__":
    main()
