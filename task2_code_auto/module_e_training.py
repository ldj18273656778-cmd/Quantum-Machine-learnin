"""Module E training utilities for target-bit local inversion."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import cirq
import numpy as np
from numpy.typing import NDArray

from task2_code_auto.ansatz import ansatz_scope_size
from task2_code_auto.ansatz_registry import ansatz_theta_count, random_ansatz_theta
from task2_code_auto.lightcone import (
    backward_block_lightcone_from_circuit,
    build_lightcone_target_unitary,
    extract_target_lightcone_operator,
    lightcone_qubits_for_block,
    projected_operator_diagnostics,
)
from task2_code_auto.loss_registry import get_active_loss_function, residual_operator_for_context as _registry_residual_operator_for_context
from task2_code_auto.U_target import build_u_target_circuit


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
LossFunction = Callable[[FloatArray], float]
StepCallback = Callable[[int, FloatArray, float], None]  # step, theta, scalar_loss
GradientBackend = Literal["finite-difference", "jax"]
JaxMemoryMode = Literal["standard", "rematerialized"]
JaxLossFunction = Callable[..., object]


@dataclass(frozen=True)
class ObjectiveContext:
    target_operator: ComplexArray
    block_qubits: tuple[int, ...]
    lightcone_qubits: tuple[int, ...]
    target_bit: int
    full_target_operator: ComplexArray | None = None
    system_qubits: tuple[int, ...] | None = None
    loss_mode: str = "lightcone"
    ansatz: str = "default_5layer_cz"
    block_only_ansatz: bool = False

    @property
    def ansatz_qubits(self) -> int:
        return ansatz_scope_size(
            self.lightcone_qubits,
            self.block_qubits,
            block_only_ansatz=self.block_only_ansatz,
        )

    @property
    def theta_size(self) -> int:
        return ansatz_theta_count(self.ansatz, self.ansatz_qubits)

    @property
    def loss_qubits(self) -> tuple[int, ...]:
        if self.loss_mode == "full_system" and self.full_target_operator is not None and self.system_qubits is not None:
            return self.system_qubits
        return self.lightcone_qubits


@dataclass(frozen=True)
class AdamConfig:
    iterations: int = 100
    lr: float = 0.1#学习率
    beta1: float = 0.85
    beta2: float = 0.9995
    eps: float = 1e-8
    fd_eps: float = 1e-5
    wrap_angles: bool = True


@dataclass(frozen=True)
class AdamRunResult:
    initial_params: FloatArray
    final_params: FloatArray
    best_params: FloatArray
    loss_history: FloatArray
    grad_norm_history: FloatArray
    best_loss_history: FloatArray
    best_loss: float
    best_iteration: int
    failed: bool
    failure_reason: str


@dataclass(frozen=True)
class MultiRestartResult:
    restart_results: tuple[AdamRunResult, ...]
    best_restart: int
    best_iteration: int
    best_loss: float
    best_params: FloatArray


def _as_unique_int_tuple(values: Sequence[int], name: str) -> tuple[int, ...]:
    items = tuple(int(v) for v in values)
    if len(items) != len(set(items)):
        raise ValueError(f"{name} must not contain duplicates, got {items}")
    return items


def _validate_theta(theta: object, expected_size: int) -> FloatArray:
    theta_arr = np.asarray(theta, dtype=float)
    expected = int(expected_size)
    if theta_arr.shape != (expected,):
        raise ValueError(f"theta must have shape ({expected},), got {theta_arr.shape}")
    if not np.all(np.isfinite(theta_arr)):
        raise ValueError("theta must contain only finite values")
    return theta_arr


def _validate_theta_vector(theta: object) -> FloatArray:
    theta_arr = np.asarray(theta, dtype=float)
    if theta_arr.ndim != 1:
        raise ValueError(f"theta must be one-dimensional, got shape {theta_arr.shape}")
    if theta_arr.size == 0:
        raise ValueError("theta must contain at least one parameter")
    if not np.all(np.isfinite(theta_arr)):
        raise ValueError("theta must contain only finite values")
    return theta_arr


def make_objective_context(
    target_operator: object,
    block_qubits: Sequence[int],
    lightcone_qubits: Sequence[int],
    target_bit: int,
    *,
    ansatz: str = "default_5layer_cz",
    block_only_ansatz: bool = False,
) -> ObjectiveContext:
    target = np.asarray(target_operator, dtype=complex)
    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError(f"target_operator must be square, got {target.shape}")

    block = _as_unique_int_tuple(block_qubits, "block_qubits")
    cone = _as_unique_int_tuple(lightcone_qubits, "lightcone_qubits")
    bit = int(target_bit)
    if len(block) != 4:
        raise ValueError(f"block_qubits must contain exactly 4 qubits, got {block}")
    if bit not in block:
        raise ValueError(f"target_bit must be a global label in block_qubits, got {bit}")
    missing = [q for q in block if q not in cone]
    if missing:
        raise ValueError(f"block_qubits must be contained in lightcone_qubits; missing {missing}")

    expected_dim = 1 << len(cone)
    if target.shape != (expected_dim, expected_dim):
        raise ValueError(
            f"target_operator shape must be ({expected_dim}, {expected_dim}) for cone {cone}"
        )
    return ObjectiveContext(
        target,
        block,
        cone,
        bit,
        ansatz=str(ansatz),
        block_only_ansatz=bool(block_only_ansatz),
    )


def _make_full_system_context(
    projected_operator: object,
    full_target_operator: object,
    block_qubits: Sequence[int],
    lightcone_qubits: Sequence[int],
    target_bit: int,
    n_qubits: int,
    *,
    ansatz: str = "default_5layer_cz",
    block_only_ansatz: bool = False,
) -> ObjectiveContext:
    context = make_objective_context(
        projected_operator,
        block_qubits,
        lightcone_qubits,
        target_bit,
        ansatz=ansatz,
        block_only_ansatz=block_only_ansatz,
    )
    full_target = np.asarray(full_target_operator, dtype=complex)
    expected_dim = 1 << int(n_qubits)
    if full_target.shape != (expected_dim, expected_dim):
        raise ValueError(
            f"full_target_operator shape must be ({expected_dim}, {expected_dim}) "
            + f"for n_qubits={n_qubits}, got {full_target.shape}"
        )
    return ObjectiveContext(
        target_operator=context.target_operator,
        block_qubits=context.block_qubits,
        lightcone_qubits=context.lightcone_qubits,
        target_bit=context.target_bit,
        full_target_operator=full_target,
        system_qubits=tuple(range(int(n_qubits))),
        loss_mode="full_system",
        ansatz=context.ansatz,
        block_only_ansatz=context.block_only_ansatz,
    )


def _normalise_loss_mode(loss_mode: str) -> str:
    mode = str(loss_mode).lower()
    if mode not in {"lightcone", "full_system"}:
        raise ValueError(f"loss_mode must be 'lightcone' or 'full_system', got {loss_mode!r}")
    return mode


def build_target_objective_context(
    n_qubits: int,
    block_qubits: Sequence[int],
    target_bit: int,
    radius: int,
    target_seed: int,
    time_k: int = 5,
    time: float | None = None,
    lightcone_mode: str = "radius",
    loss_mode: str = "lightcone",
    require_unitary: bool = False,
    max_n_qubits: int = 12,
    max_hilbert_dim: int = 4096,
    ansatz: str = "default_5layer_cz",
    block_only_ansatz: bool = False,
) -> tuple[ObjectiveContext, dict[str, Any]]:
    n_val = int(n_qubits)
    max_n_val = int(max_n_qubits)
    max_dim = int(max_hilbert_dim)
    if n_val <= 0:
        raise ValueError(f"n_qubits must be positive, got {n_val}")
    loss_mode_value = _normalise_loss_mode(loss_mode)
    full_hilbert_dim = 1 << n_val
    if loss_mode_value == "full_system" and (n_val > max_n_val or full_hilbert_dim > max_dim):
        raise ValueError(
            "dense target construction is guarded for small systems; "
            + f"got n_qubits={n_val}, hilbert_dim={full_hilbert_dim}"
        )

    rng = np.random.default_rng(int(target_seed))
    h_arr = rng.uniform(-1.0, 1.0, size=n_val)
    t_value = float(time if time is not None else 3.0 * math.pi / 40.0 * int(time_k) + 0.001)
    target_circuit = build_u_target_circuit(n_val, rng, h_arr, t_value)

    mode = str(lightcone_mode).lower()
    if mode == "radius":
        selected_lightcone = lightcone_qubits_for_block(block_qubits, n_val, int(radius))
    elif mode == "circuit":
        selected_lightcone = backward_block_lightcone_from_circuit(target_circuit, block_qubits)
    else:
        raise ValueError(f"lightcone_mode must be 'radius' or 'circuit', got {lightcone_mode!r}")

    lightcone_dim = 1 << len(selected_lightcone)
    if len(selected_lightcone) > max_n_val or lightcone_dim > max_dim:
        raise ValueError(
            "light-cone target construction is guarded for small cones; "
            + f"got n_C={len(selected_lightcone)}, hilbert_dim={lightcone_dim}"
        )

    if loss_mode_value == "lightcone":
        target_operator = build_lightcone_target_unitary(target_circuit, selected_lightcone)
        diagnostics = projected_operator_diagnostics(target_operator)
        if require_unitary and (diagnostics.left_unitarity_error > 1e-8 or diagnostics.right_unitarity_error > 1e-8):
            raise ValueError(
                "light-cone circuit target is not unitary within tolerance; "
                + f"left_error={diagnostics.left_unitarity_error:.3e}, "
                + f"right_error={diagnostics.right_unitarity_error:.3e}"
            )
        context = make_objective_context(
            target_operator,
            block_qubits,
            selected_lightcone,
            target_bit,
            ansatz=ansatz,
            block_only_ansatz=block_only_ansatz,
        )
        block_positions = tuple(selected_lightcone.index(int(q)) for q in block_qubits)
        outside_qubits = tuple(q for q in range(n_val) if q not in selected_lightcone)
        lightcone_semantics = "circuit_gate_restriction"
        full_target_summary = None
    else:
        qubit_order = list(cirq.LineQubit.range(n_val))
        U_full = np.asarray(target_circuit.unitary(qubit_order=qubit_order), dtype=complex)
        lightcone = extract_target_lightcone_operator(
            U_full,
            block_qubits=block_qubits,
            n_qubits=n_val,
            radius=int(radius),
            lightcone_qubits=selected_lightcone,
            require_unitary=require_unitary,
            max_n_qubits=max_n_val,
            max_hilbert_dim=max_dim,
        )
        target_operator = lightcone.operator
        diagnostics = lightcone.diagnostics
        context = _make_full_system_context(
            target_operator,
            U_full,
            block_qubits,
            lightcone.lightcone_qubits,
            target_bit,
            n_val,
            ansatz=ansatz,
            block_only_ansatz=block_only_ansatz,
        )
        selected_lightcone = list(lightcone.lightcone_qubits)
        block_positions = lightcone.block_positions
        outside_qubits = lightcone.outside_qubits
        lightcone_semantics = lightcone.semantics
        full_target_summary = target_operator_summary(U_full)

    metadata = {
        "n_qubits": n_val,
        "block_qubits": [int(q) for q in block_qubits],
        "target_bit": int(target_bit),
        "radius": int(radius),
        "lightcone_mode": mode,
        "loss_mode": loss_mode_value,
        "target_seed": int(target_seed),
        "time_k": int(time_k),
        "time": t_value,
        "h_arr": np.asarray(h_arr, dtype=float),
        "lightcone_qubits": list(selected_lightcone),
        "ansatz": context.ansatz,
        "block_only_ansatz": context.block_only_ansatz,
        "ansatz_qubits": context.ansatz_qubits,
        "theta_size": context.theta_size,
        "block_positions": list(block_positions),
        "outside_qubits": list(outside_qubits),
        "lightcone_semantics": lightcone_semantics,
        "loss_semantics": "lightcone_residual_channel" if loss_mode_value == "lightcone" else "full_system_residual_channel",
        "lightcone_diagnostics": {
            "left_unitarity_error": diagnostics.left_unitarity_error,
            "right_unitarity_error": diagnostics.right_unitarity_error,
            "max_column_leakage": diagnostics.max_column_leakage,
        },
        "target_operator_summary": target_operator_summary(target_operator),
        "full_target_operator_summary": full_target_summary,
    }
    return context, metadata


def residual_operator_for_context(theta: object, context: ObjectiveContext) -> tuple[ComplexArray, tuple[int, ...]]:
    """Return the residual operator and qubit labels used by the loss."""
    theta_arr = _validate_theta(theta, context.theta_size)
    residual, loss_qubits = _registry_residual_operator_for_context(theta_arr, context)
    return np.asarray(residual, dtype=complex), tuple(int(q) for q in loss_qubits)


def sum_block_loss(theta: object, context: ObjectiveContext) -> float:
    """Return the sum of deterministic local losses over all block qubits.

    This is the Mode 1 training objective from Eq. S.2.3:
    ``L = Σ_k ||Tr_{n\\k} L(U U_t^†) - I_4||_F^2``.
    """
    loss = float(get_active_loss_function()(theta, context))
    if not np.isfinite(loss):
        raise FloatingPointError(f"non-finite sum-block loss: {loss}")
    return loss


# backward-compatible aliases
max_block_loss = sum_block_loss       # old name still works for callers that used max
target_bit_loss = sum_block_loss      # still works for older scripts


def finite_difference_gradient(
    loss_fn: LossFunction,
    theta: object,
    fd_eps: float = 1e-5,
) -> FloatArray:
    if fd_eps <= 0:
        raise ValueError(f"fd_eps must be positive, got {fd_eps}")
    base = np.asarray(theta, dtype=float)
    if base.ndim != 1:
        raise ValueError(f"theta must be one-dimensional, got shape {base.shape}")
    if not np.all(np.isfinite(base)):
        raise ValueError("theta must contain only finite values")
    grad = np.zeros_like(base, dtype=float)
    for idx in range(base.size):
        theta_plus = base.copy()
        theta_minus = base.copy()
        theta_plus[idx] += fd_eps
        theta_minus[idx] -= fd_eps
        loss_plus = float(loss_fn(theta_plus))
        loss_minus = float(loss_fn(theta_minus))
        if not np.isfinite(loss_plus) or not np.isfinite(loss_minus):
            raise FloatingPointError(
                f"non-finite finite-difference loss at parameter {idx}: "
                f"plus={loss_plus}, minus={loss_minus}"
            )
        grad[idx] = (loss_plus - loss_minus) / (2.0 * fd_eps)
    if not np.all(np.isfinite(grad)):
        raise FloatingPointError("finite-difference gradient contains non-finite values")
    return grad


def adam_optimize(
    loss_fn: LossFunction,
    initial_theta: object,
    config: AdamConfig | None = None,
    *,
    show_progress: bool = False,
    step_callback: StepCallback | None = None,
    gradient_backend: GradientBackend = "finite-difference",
    objective_context: ObjectiveContext | None = None,
    jax_loss_fn: JaxLossFunction | None = None,
    jax_memory_mode: JaxMemoryMode = "standard",
) -> AdamRunResult:
    cfg = config or AdamConfig()
    _validate_adam_config(cfg)
    theta = _validate_theta_vector(initial_theta).copy()
    initial = theta.copy()

    if gradient_backend not in {"finite-difference", "jax"}:
        raise ValueError("gradient_backend must be 'finite-difference' or 'jax'")
    if jax_memory_mode not in {"standard", "rematerialized"}:
        raise ValueError("jax_memory_mode must be 'standard' or 'rematerialized'")
    if gradient_backend != "jax" and jax_memory_mode != "standard":
        raise ValueError("jax_memory_mode='rematerialized' requires gradient_backend='jax'")
    if gradient_backend == "jax":
        if objective_context is None or jax_loss_fn is None:
            raise ValueError("gradient_backend='jax' requires objective_context and jax_loss_fn")
        if objective_context.loss_mode != "lightcone":
            raise ValueError("gradient_backend='jax' supports only lightcone objectives")
        from task2_code_auto.jax_backend.optimizer import AdamConfig as JaxAdamConfig
        from task2_code_auto.jax_backend.optimizer import adam_optimize as jax_adam_optimize

        jax_config = JaxAdamConfig(
            iterations=cfg.iterations,
            lr=cfg.lr,
            beta1=cfg.beta1,
            beta2=cfg.beta2,
            eps=cfg.eps,
            fd_eps=cfg.fd_eps,
            wrap_angles=cfg.wrap_angles,
        )
        jax_result = jax_adam_optimize(
            loss_fn=jax_loss_fn,
            initial_theta=theta,
            context=objective_context,
            config=jax_config,
            gradient_backend="outer-jit",
            step_callback=step_callback,
            show_progress=show_progress,
        )
        return AdamRunResult(
            initial_params=np.asarray(jax_result.initial_params, dtype=float),
            final_params=np.asarray(jax_result.final_params, dtype=float),
            best_params=np.asarray(jax_result.best_params, dtype=float),
            loss_history=np.asarray(jax_result.loss_history, dtype=float),
            grad_norm_history=np.asarray(jax_result.grad_norm_history, dtype=float),
            best_loss_history=np.asarray(jax_result.best_loss_history, dtype=float),
            best_loss=float(jax_result.best_loss),
            best_iteration=int(jax_result.best_iteration),
            failed=bool(jax_result.failed),
            failure_reason=str(jax_result.failure_reason),
        )

    loss0 = float(loss_fn(theta.copy()))
    if not np.isfinite(loss0):
        raise FloatingPointError(f"initial loss is non-finite: {loss0}")

    if step_callback is not None:
        step_callback(0, theta.copy(), loss0)

    losses = [loss0]
    grad_norms = []
    best_losses = [loss0]
    best_loss = loss0
    best_theta = theta.copy()
    best_iteration = 0
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    failed = False
    failure_reason = ""

    for step in range(1, cfg.iterations + 1):
        try:
            grad = finite_difference_gradient(loss_fn, theta.copy(), cfg.fd_eps)
            grad_norm = float(np.linalg.norm(grad))
            if not np.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at step {step}")

            m = cfg.beta1 * m + (1.0 - cfg.beta1) * grad
            v = cfg.beta2 * v + (1.0 - cfg.beta2) * (grad * grad)
            m_hat = m / (1.0 - cfg.beta1 ** step)
            v_hat = v / (1.0 - cfg.beta2 ** step)
            theta = theta - cfg.lr * m_hat / (np.sqrt(v_hat) + cfg.eps)
            if cfg.wrap_angles:
                theta = np.mod(theta, 2.0 * np.pi)
            if not np.all(np.isfinite(theta)):
                raise FloatingPointError(f"non-finite theta after ADAM step {step}")

            current_loss = float(loss_fn(theta.copy()))
            if not np.isfinite(current_loss):
                raise FloatingPointError(f"non-finite loss after ADAM step {step}")
        except Exception as exc:
            failed = True
            failure_reason = f"step {step}: {exc}"
            break

        losses.append(current_loss)
        grad_norms.append(grad_norm)
        if current_loss < best_loss:
            best_loss = current_loss
            best_theta = theta.copy()
            best_iteration = step
        best_losses.append(best_loss)

        if step_callback is not None:
            step_callback(step, theta.copy(), current_loss)

        if show_progress:
            bar_width = 30
            ratio = step / cfg.iterations
            filled = int(bar_width * ratio)
            bar = "#" * filled + "·" * (bar_width - filled)
            print(
                f"\r  [{bar}] {step:3d}/{cfg.iterations}  "
                f"loss={current_loss:.6g}  best={best_loss:.6g}  "
                f"|g|={grad_norm:.4g}  ",
                end="",
                flush=True,
            )

    if show_progress:
        print()

    return AdamRunResult(
        initial_params=initial,
        final_params=theta.copy(),
        best_params=best_theta,
        loss_history=np.asarray(losses, dtype=float),
        grad_norm_history=np.asarray(grad_norms, dtype=float),
        best_loss_history=np.asarray(best_losses, dtype=float),
        best_loss=float(best_loss),
        best_iteration=int(best_iteration),
        failed=failed,
        failure_reason=failure_reason,
    )


def multi_restart_train(
    loss_fn: LossFunction,
    restarts: int,
    rng: np.random.Generator,
    config: AdamConfig | None = None,
    init_low: float = 0.0,
    init_high: float = 2.0 * np.pi,
    n_qubits: int = 4,
    ansatz: str = "default_5layer_cz",
    *,
    show_progress: bool = False,
    success_threshold: float | None = None,
    step_callback: StepCallback | None = None,
    gradient_backend: GradientBackend = "finite-difference",
    objective_context: ObjectiveContext | None = None,
    jax_loss_fn: JaxLossFunction | None = None,
    jax_memory_mode: JaxMemoryMode = "standard",
) -> MultiRestartResult:
    if restarts <= 0:
        raise ValueError(f"restarts must be positive, got {restarts}")
    results = []
    for restart_idx in range(restarts):
        theta0 = np.asarray(
            random_ansatz_theta(
                ansatz,
                rng,
                low=init_low,
                high=init_high,
                n_qubits=n_qubits,
            ),
            dtype=float,
        )
        if show_progress and restarts > 1:
            print(f"\n--- restart {restart_idx + 1}/{restarts} ---")
        result = adam_optimize(
            loss_fn,
            theta0,
            config,
            show_progress=show_progress,
            step_callback=step_callback,
            gradient_backend=gradient_backend,
            objective_context=objective_context,
            jax_loss_fn=jax_loss_fn,
            jax_memory_mode=jax_memory_mode,
        )
        results.append(result)
        if success_threshold is not None and not result.failed and result.best_loss <= success_threshold:
            if show_progress and restart_idx + 1 < restarts:
                print(f"  reached success_threshold={success_threshold:.6g}; stopping restarts")
            break

    successful = [(idx, result) for idx, result in enumerate(results) if not result.failed]
    if not successful:
        reasons = "; ".join(result.failure_reason for result in results)
        raise FloatingPointError(f"all restarts failed: {reasons}")

    best_restart, best_result = min(successful, key=lambda item: item[1].best_loss)
    return MultiRestartResult(
        restart_results=tuple(results),
        best_restart=int(best_restart),
        best_iteration=int(best_result.best_iteration),
        best_loss=float(best_result.best_loss),
        best_params=best_result.best_params.copy(),
    )


def save_training_artifacts(
    result: MultiRestartResult,
    output_dir: str | Path,
    metadata: Mapping[str, Any],
) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    initial_params = np.stack([r.initial_params for r in result.restart_results])
    final_params = np.stack([r.final_params for r in result.restart_results])
    per_restart_best_params = np.stack([r.best_params for r in result.restart_results])
    best_losses = np.asarray([r.best_loss for r in result.restart_results], dtype=float)
    best_iterations = np.asarray([r.best_iteration for r in result.restart_results], dtype=int)

    max_loss_len = max(r.loss_history.size for r in result.restart_results)
    loss_trajectories = np.full((len(result.restart_results), max_loss_len), np.nan)
    best_loss_trajectories = np.full_like(loss_trajectories, np.nan)
    max_grad_len = max(max(r.grad_norm_history.size, 1) for r in result.restart_results)
    grad_norm_trajectories = np.full((len(result.restart_results), max_grad_len), np.nan)
    for idx, run in enumerate(result.restart_results):
        loss_trajectories[idx, : run.loss_history.size] = run.loss_history
        best_loss_trajectories[idx, : run.best_loss_history.size] = run.best_loss_history
        grad_norm_trajectories[idx, : run.grad_norm_history.size] = run.grad_norm_history

    arrays_path = out_dir / "module_e_training_arrays.npz"
    np.savez(
        arrays_path,
        initial_params=initial_params,
        final_params=final_params,
        per_restart_best_params=per_restart_best_params,
        best_params=result.best_params,
        loss_trajectories=loss_trajectories,
        best_loss_trajectories=best_loss_trajectories,
        grad_norm_trajectories=grad_norm_trajectories,
        per_restart_best_losses=best_losses,
        per_restart_best_iterations=best_iterations,
    )

    failures = [r.failure_reason for r in result.restart_results]
    meta = dict(metadata)
    meta.update(
        {
            "schema_version": 1,
            "best_restart": result.best_restart,
            "best_iteration": result.best_iteration,
            "best_loss": result.best_loss,
            "restart_failed": [r.failed for r in result.restart_results],
            "failure_reasons": failures,
            "array_file": arrays_path.name,
        }
    )
    metadata_path = out_dir / "module_e_training_metadata.json"
    metadata_path.write_text(json.dumps(_json_ready(meta), indent=2), encoding="utf-8")
    return arrays_path, metadata_path


def target_operator_summary(operator: object) -> dict[str, float | str]:
    op = np.asarray(operator, dtype=complex)
    digest = hashlib.sha256(np.ascontiguousarray(op).view(np.float64).tobytes()).hexdigest()
    return {
        "shape": str(tuple(int(v) for v in op.shape)),
        "fro_norm": float(np.linalg.norm(op, ord="fro")),
        "sha256_float_view": digest,
    }


def _validate_adam_config(config: AdamConfig) -> None:
    if config.iterations < 0:
        raise ValueError(f"iterations must be nonnegative, got {config.iterations}")
    if config.lr <= 0:
        raise ValueError(f"lr must be positive, got {config.lr}")
    if not 0 <= config.beta1 < 1:
        raise ValueError(f"beta1 must satisfy 0 <= beta1 < 1, got {config.beta1}")
    if not 0 <= config.beta2 < 1:
        raise ValueError(f"beta2 must satisfy 0 <= beta2 < 1, got {config.beta2}")
    if config.eps <= 0:
        raise ValueError(f"eps must be positive, got {config.eps}")
    if config.fd_eps <= 0:
        raise ValueError(f"fd_eps must be positive, got {config.fd_eps}")


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(v) for v in value]
    return value
