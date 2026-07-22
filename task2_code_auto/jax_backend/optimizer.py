"""Pure JAX Adam optimizer for the isolated backend."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .runtime import jax

from jax import Array
from jax.errors import ConcretizationTypeError, TracerArrayConversionError
import jax.numpy as jnp


class ObjectiveContextLike(Protocol):
    target_operator: Array
    block_qubits: tuple[int, ...]
    lightcone_qubits: tuple[int, ...]


LossFunction = Callable[[Array, ObjectiveContextLike], Array]
StepCallback = Callable[[int, Array, float], None]
LossAndGrad = Callable[[Array], tuple[Array, Array]]


@dataclass(frozen=True, slots=True)
class AdamConfig:
    iterations: int = 100
    lr: float = 0.1
    beta1: float = 0.85
    beta2: float = 0.9995
    eps: float = 1e-8
    fd_eps: float = 1e-5
    wrap_angles: bool = True


@dataclass(frozen=True, slots=True)
class AdamState:
    iteration: int
    params: Array
    m: Array
    v: Array


@dataclass(frozen=True, slots=True)
class AdamRunMetadata:
    gradient_method: str
    optimizer_backend: str
    jax_x64_enabled: bool
    fd_eps: float
    fd_eps_used: bool


@dataclass(frozen=True, slots=True)
class AdamRunResult:
    initial_params: Array
    final_params: Array
    best_params: Array
    loss_history: Array
    grad_norm_history: Array
    best_loss_history: Array
    best_loss: float
    best_iteration: int
    failed: bool
    failure_reason: str
    metadata: AdamRunMetadata


def adam_optimize(
    *,
    loss_fn: LossFunction,
    initial_theta: Array,
    context: ObjectiveContextLike,
    config: AdamConfig,
    gradient_backend: str = "outer-jit",
    step_callback: StepCallback | None = None,
    show_progress: bool = False,
) -> AdamRunResult:
    _validate_adam_config(config)
    theta = _validate_theta_vector(initial_theta)
    initial = jnp.array(theta, copy=True)
    metadata = _metadata(config)
    loss_and_grad = _make_loss_and_grad(loss_fn, context, gradient_backend)

    initial_loss = _scalar_loss(loss_fn(theta, context))
    losses = [initial_loss]
    grad_norms: list[Array] = []
    best_losses = [initial_loss]
    best_loss = initial_loss
    best_theta = jnp.array(theta, copy=True)
    best_iteration = 0
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)

    def finish(*, failed: bool, failure_reason: str) -> AdamRunResult:
        return AdamRunResult(
            initial_params=jnp.array(initial, copy=True),
            final_params=jnp.array(theta, copy=True),
            best_params=jnp.array(best_theta, copy=True),
            loss_history=jnp.asarray(losses, dtype=jnp.float64),
            grad_norm_history=jnp.asarray(grad_norms, dtype=jnp.float64),
            best_loss_history=jnp.asarray(best_losses, dtype=jnp.float64),
            best_loss=float(best_loss),
            best_iteration=best_iteration,
            failed=failed,
            failure_reason=failure_reason,
            metadata=metadata,
        )

    if not bool(jnp.isfinite(initial_loss)):
        best_loss = initial_loss
        return finish(
            failed=True,
            failure_reason=f"initial loss is non-finite: {float(initial_loss)}",
        )

    if step_callback is not None:
        step_callback(0, jnp.array(theta, copy=True), float(initial_loss))

    for step in range(1, config.iterations + 1):
        try:
            _, grad = loss_and_grad(theta)
        except (ConcretizationTypeError, TracerArrayConversionError) as exc:
            return finish(
                failed=True,
                failure_reason=f"step {step}: {type(exc).__name__}: {exc}",
            )
        grad = jnp.asarray(grad, dtype=jnp.float64)
        grad_norm = jnp.linalg.norm(grad)
        if not bool(jnp.all(jnp.isfinite(grad))) or not bool(jnp.isfinite(grad_norm)):
            return finish(
                failed=True,
                failure_reason=f"step {step}: non-finite gradient",
            )

        m = config.beta1 * m + (1.0 - config.beta1) * grad
        v = config.beta2 * v + (1.0 - config.beta2) * (grad * grad)
        m_hat = m / (1.0 - config.beta1**step)
        v_hat = v / (1.0 - config.beta2**step)
        theta = theta - config.lr * m_hat / (jnp.sqrt(v_hat) + config.eps)
        if config.wrap_angles:
            theta = jnp.mod(theta, 2.0 * jnp.pi)
        if not bool(jnp.all(jnp.isfinite(theta))):
            return finish(
                failed=True,
                failure_reason=f"step {step}: non-finite theta",
            )

        current_loss = _scalar_loss(loss_fn(theta, context))
        if not bool(jnp.isfinite(current_loss)):
            return finish(
                failed=True,
                failure_reason=f"step {step}: non-finite loss",
            )

        losses.append(current_loss)
        grad_norms.append(grad_norm)
        if bool(current_loss < best_loss):
            best_loss = current_loss
            best_theta = jnp.array(theta, copy=True)
            best_iteration = step
        best_losses.append(best_loss)

        if step_callback is not None:
            step_callback(step, jnp.array(theta, copy=True), float(current_loss))

        if show_progress and config.iterations > 0:
            bar_width = 30
            ratio = step / config.iterations
            filled = int(bar_width * ratio)
            bar = "#" * filled + "-" * (bar_width - filled)
            print(
                f"\r  [{bar}] {step:3d}/{config.iterations}  "
                f"loss={float(current_loss):.6g}  best={float(best_loss):.6g}  "
                f"|g|={float(grad_norm):.4g}  ",
                end="",
                flush=True,
            )

    if show_progress and config.iterations > 0:
        print()

    return finish(
        failed=False,
        failure_reason="",
    )


def _metadata(config: AdamConfig) -> AdamRunMetadata:
    return AdamRunMetadata(
        gradient_method="jax_value_and_grad",
        optimizer_backend="jax_cpu",
        jax_x64_enabled=bool(jax.config.jax_enable_x64),
        fd_eps=float(config.fd_eps),
        fd_eps_used=False,
    )


def _make_loss_and_grad(
    loss_fn: LossFunction,
    context: ObjectiveContextLike,
    gradient_backend: str,
) -> LossAndGrad:
    loss = lambda candidate_theta: jnp.asarray(loss_fn(candidate_theta, context), dtype=jnp.float64)
    match gradient_backend:
        case "outer-jit":
            return jax.jit(jax.value_and_grad(loss))
        case "eager":
            return jax.value_and_grad(loss)
        case _:
            raise ValueError("gradient_backend must be 'outer-jit' or 'eager'")


def _validate_theta_vector(theta: Array) -> Array:
    theta_array = jnp.asarray(theta, dtype=jnp.float64)
    if theta_array.ndim != 1:
        raise ValueError(f"theta must be one-dimensional, got shape {theta_array.shape}")
    if theta_array.size == 0:
        raise ValueError("theta must contain at least one parameter")
    if not bool(jnp.all(jnp.isfinite(theta_array))):
        raise ValueError("theta must contain only finite values")
    return jnp.array(theta_array, copy=True)


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


def _scalar_loss(value: Array) -> Array:
    return jnp.asarray(value, dtype=jnp.float64)


__all__ = [
    "AdamConfig",
    "AdamRunMetadata",
    "AdamRunResult",
    "AdamState",
    "adam_optimize",
]
