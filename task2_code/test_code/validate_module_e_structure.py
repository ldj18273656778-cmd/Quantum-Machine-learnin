"""Structural and smoke validation for Module E training.

Run from repository root:
    python task2_code/test_code/validate_module_e_structure.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from numpy.typing import NDArray

import task2_code.module_e_training as training
from task2_code.ansatz import ansatz_unitary, build_ansatz, cz_pairs_for_layer, random_theta, theta_count
from task2_code.module_e_training import (
    AdamConfig,
    adam_optimize,
    build_target_objective_context,
    finite_difference_gradient,
    make_objective_context,
    multi_restart_train,
    save_training_artifacts,
    target_bit_loss,
)
from task2_code.superoperator import (
    per_bit_losses_from_V,
    reduced_bit_superoperator_from_V,
    superoperator,
)
from task2_code.superoperator_registry import get_active_superop, set_active_superop


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _random_unitary(dim: int, seed: int) -> NDArray[np.complex128]:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(matrix)
    phases = np.diag(r) / np.abs(np.diag(r))
    return np.asarray(q * phases.conj(), dtype=complex)


def validate_per_bit_convention() -> None:
    U1 = _random_unitary(2, 101)
    R1 = reduced_bit_superoperator_from_V(U1, 0, [0], [0])
    _assert(np.allclose(R1, superoperator(U1), atol=1e-10), "one-qubit column-stacking mismatch")

    U_local = _random_unitary(2, 102)
    U_env = _random_unitary(2, 103)
    U2 = np.kron(U_local, U_env)
    actual_q0 = reduced_bit_superoperator_from_V(U2, 0, [0, 1], [0, 1])
    actual_q1 = reduced_bit_superoperator_from_V(U2, 1, [0, 1], [0, 1])
    _assert(np.allclose(actual_q0, superoperator(U_local), atol=1e-10), "factorized q0 channel mismatch")
    _assert(np.allclose(actual_q1, superoperator(U_env), atol=1e-10), "factorized q1 channel mismatch")

    losses = per_bit_losses_from_V(np.eye(16, dtype=complex), [0, 1, 2, 3], [0, 1, 2, 3])
    _assert(all(value < 1e-12 for value in losses.values()), "identity residual should have zero per-bit losses")
    print("per-bit convention checks passed")


def validate_dynamic_ansatz_api() -> None:
    _assert(theta_count() == 60, "default theta_count must remain 4-qubit compatible")
    _assert(theta_count(5) == 75, "dynamic theta_count must be 15*n")
    _assert(random_theta(np.random.default_rng(1)).shape == (60,), "default random_theta shape changed")
    _assert(random_theta(np.random.default_rng(2), n_qubits=5).shape == (75,), "dynamic random_theta shape mismatch")
    _assert(cz_pairs_for_layer(5, 0) == [(0, 1), (2, 3)], "even CZ layer mismatch")
    _assert(cz_pairs_for_layer(5, 1) == [(1, 2), (3, 4)], "odd CZ layer mismatch")

    theta = np.zeros(theta_count(5), dtype=float)
    unitary = ansatz_unitary(theta, n_qubits=5)
    _assert(unitary.shape == (32, 32), "dynamic ansatz unitary shape mismatch")
    _assert(np.allclose(unitary.conj().T @ unitary, np.eye(32), atol=1e-10), "dynamic ansatz not unitary")

    try:
        _ = build_ansatz(np.zeros(theta_count(4)), n_qubits=5)
    except ValueError:
        pass
    else:
        raise AssertionError("theta/n_qubits conflict should raise")
    print("dynamic ansatz API checks passed")


def validate_objective_and_gradient() -> None:
    rng = np.random.default_rng(202)
    n_cone = 5
    theta_true = random_theta(rng, n_qubits=n_cone)
    target = ansatz_unitary(theta_true, n_qubits=n_cone)
    context = make_objective_context(target, [1, 2, 3, 4], [0, 1, 2, 3, 4], target_bit=2)
    perfect_loss = target_bit_loss(theta_true, context)
    _assert(perfect_loss < 1e-9, f"perfect orientation loss should be zero, got {perfect_loss}")

    call_count = 0
    called_target_bits: list[int] | None = None
    original = get_active_superop()

    def spy(V, block_qubits, lightcone_qubits, target_bits=None):
        nonlocal call_count, called_target_bits
        call_count += 1
        called_target_bits = list(target_bits) if target_bits is not None else None
        return original(V, block_qubits, lightcone_qubits, target_bits=target_bits)

    set_active_superop(spy)
    try:
        loss_value = target_bit_loss(theta_true, context)
    finally:
        set_active_superop(original)
    _assert(loss_value < 1e-9, "spy objective changed perfect loss")
    _assert(call_count == 1, "objective must call production per_bit_losses_from_V exactly once")
    _assert(called_target_bits is None, "objective must request all block bits for max loss")

    theta_probe = theta_true.copy()
    grad = finite_difference_gradient(lambda theta: target_bit_loss(theta, context), theta_probe, fd_eps=1e-5)
    _assert(grad.shape == (theta_count(n_cone),), f"gradient shape mismatch: {grad.shape}")
    _assert(bool(np.all(np.isfinite(grad))), "gradient contains non-finite values")
    _assert(np.allclose(theta_probe, theta_true), "finite_difference_gradient mutated input theta")
    print("objective and gradient checks passed")


def validate_optimizer_and_artifacts(output_dir: Path) -> None:
    target = np.full(theta_count(), 0.25, dtype=float)

    def quadratic(theta: NDArray[np.float64]) -> float:
        return float(np.sum((np.asarray(theta, dtype=float) - target) ** 2))

    initial = np.zeros(theta_count(), dtype=float)
    result = adam_optimize(
        quadratic,
        initial,
        AdamConfig(iterations=5, lr=0.05, fd_eps=1e-6, wrap_angles=False),
    )
    _assert(not result.failed, f"quadratic ADAM failed: {result.failure_reason}")
    _assert(result.loss_history[0] > result.best_loss, "ADAM should improve quadratic loss")
    _assert(np.allclose(initial, np.zeros(theta_count())), "adam_optimize mutated initial theta")

    rng = np.random.default_rng(303)
    multi = multi_restart_train(
        quadratic,
        restarts=2,
        rng=rng,
        config=AdamConfig(iterations=1, lr=0.01, fd_eps=1e-6, wrap_angles=False),
        init_low=0.0,
        init_high=0.1,
    )
    arrays_path, metadata_path = save_training_artifacts(
        multi,
        output_dir / "quadratic_artifacts",
        {"validation": "quadratic"},
    )
    arrays = np.load(arrays_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _assert(arrays["best_params"].shape == (theta_count(),), "saved best_params shape mismatch")
    _assert(metadata["best_restart"] == multi.best_restart, "metadata best_restart mismatch")
    print("optimizer and artifact checks passed")


def validate_n4_training(output_dir: Path) -> None:
    context, metadata = build_target_objective_context(
        n_qubits=4,
        block_qubits=[0, 1, 2, 3],
        target_bit=1,
        radius=0,
        target_seed=42,
        time_k=1,
        require_unitary=False,
        max_n_qubits=12,
        max_hilbert_dim=4096,
    )
    loss_fn = lambda theta: target_bit_loss(theta, context)
    rng = np.random.default_rng(404)
    result = multi_restart_train(
        loss_fn,
        restarts=1,
        rng=rng,
        config=AdamConfig(iterations=1, lr=0.005, fd_eps=1e-5),
        n_qubits=context.ansatz_qubits,
    )
    _assert(np.isfinite(result.best_loss), "n=4 best loss is non-finite")
    arrays_path, metadata_path = save_training_artifacts(
        result,
        output_dir / "n4_artifacts",
        metadata | {"training_seed": 404, "validation": "n4"},
    )
    _assert(arrays_path.exists() and metadata_path.exists(), "n=4 artifacts were not saved")
    print("n=4 smoke training passed")


def validate_circuit_lightcone_flow() -> None:
    context, metadata = build_target_objective_context(
        n_qubits=5,
        block_qubits=[0, 1, 2, 3],
        target_bit=2,
        radius=0,
        target_seed=42,
        time_k=1,
        lightcone_mode="circuit",
        require_unitary=False,
        max_n_qubits=12,
        max_hilbert_dim=4096,
    )
    _assert(context.lightcone_qubits == (0, 1, 2, 3, 4), f"unexpected circuit cone: {context.lightcone_qubits}")
    _assert(context.theta_size == theta_count(5), f"unexpected circuit theta_size: {context.theta_size}")
    _assert(context.loss_mode == "lightcone", f"unexpected loss mode: {context.loss_mode}")
    _assert(context.loss_qubits == (0, 1, 2, 3, 4), f"unexpected lightcone loss qubits: {context.loss_qubits}")
    _assert(metadata["lightcone_mode"] == "circuit", "metadata lightcone_mode mismatch")
    _assert(metadata["loss_mode"] == "lightcone", "metadata loss_mode mismatch")
    _assert(metadata["loss_semantics"] == "lightcone_residual_channel", "metadata loss_semantics mismatch")
    theta0 = random_theta(np.random.default_rng(405), n_qubits=context.ansatz_qubits)
    initial_loss = target_bit_loss(theta0, context)
    _assert(np.isfinite(initial_loss), "circuit lightcone initial loss is non-finite")

    full_context, _ = build_target_objective_context(
        n_qubits=5,
        block_qubits=[0, 1, 2, 3],
        target_bit=2,
        radius=0,
        target_seed=42,
        time_k=1,
        lightcone_mode="circuit",
        loss_mode="full_system",
        require_unitary=False,
        max_n_qubits=12,
        max_hilbert_dim=4096,
    )
    full_loss = target_bit_loss(theta0, full_context)
    _assert(abs(initial_loss - full_loss) < 1e-9, f"lightcone/full-system loss mismatch: {initial_loss} vs {full_loss}")
    print("circuit lightcone flow passed")


def validate_n12_flow(output_dir: Path, iterations: int) -> None:
    context, metadata = build_target_objective_context(
        n_qubits=12,
        block_qubits=[4, 5, 6, 7],
        target_bit=5,
        radius=2,
        target_seed=42,
        time_k=5,
        require_unitary=False,
        max_n_qubits=12,
        max_hilbert_dim=4096,
    )
    _assert(
        context.lightcone_qubits == (2, 3, 4, 5, 6, 7, 8, 9),
        f"unexpected n=12 cone: {context.lightcone_qubits}",
    )
    loss_fn = lambda theta: target_bit_loss(theta, context)
    theta0 = random_theta(np.random.default_rng(505), n_qubits=context.ansatz_qubits)
    initial_loss = loss_fn(theta0)
    _assert(np.isfinite(initial_loss), "n=12 initial target-bit loss is non-finite")

    rng = np.random.default_rng(506)
    result = multi_restart_train(
        loss_fn,
        restarts=1,
        rng=rng,
        config=AdamConfig(iterations=iterations, lr=0.005, fd_eps=1e-5),
        n_qubits=context.ansatz_qubits,
    )
    arrays_path, metadata_path = save_training_artifacts(
        result,
        output_dir / "n12_artifacts",
        metadata | {"training_seed": 506, "validation": "n12", "initial_probe_loss": initial_loss},
    )
    arrays = np.load(arrays_path)
    _assert(arrays["initial_params"].shape[1] == context.theta_size, "n=12 saved theta shape mismatch")
    _assert(json.loads(metadata_path.read_text(encoding="utf-8"))["lightcone_qubits"] == [2, 3, 4, 5, 6, 7, 8, 9], "n=12 metadata cone mismatch")
    print("n=12 flow smoke passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Module E structure and smoke flows.")
    parser.add_argument("--output-dir", type=Path, default=Path("task2_code") / "module_e_validation_output")
    parser.add_argument("--skip-n12", action="store_true")
    parser.add_argument("--n12-iterations", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_per_bit_convention()
    validate_dynamic_ansatz_api()
    validate_objective_and_gradient()
    validate_optimizer_and_artifacts(args.output_dir)
    validate_n4_training(args.output_dir)
    validate_circuit_lightcone_flow()
    if not args.skip_n12:
        validate_n12_flow(args.output_dir, args.n12_iterations)
    print("Module E validation PASSED")


if __name__ == "__main__":
    main()
