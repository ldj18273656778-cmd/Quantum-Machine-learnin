"""Core gradient-statistics computation for supplied parameter batches."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from task2_code_auto.loss_registry import loss_function_uses_superoperator, set_active_loss_function
from task2_code_auto.module_e_training import ObjectiveContext, finite_difference_gradient, sum_block_loss
from task2_code_auto.superoperator_registry import set_active_superop


FloatArray = NDArray[np.float64]


def set_gradient_mode(mode: dict[str, Any]) -> None:
    """Set the active loss and superoperator registries for one mode."""
    loss_function = str(mode["loss_function"])
    set_active_loss_function(loss_function)
    if loss_function_uses_superoperator(loss_function):
        set_active_superop(str(mode["superoperator"]))


def validate_theta_batch(theta_batch: object, theta_size: int) -> FloatArray:
    """Return a finite 2D theta batch with shape ``(samples, theta_size)``."""
    arr = np.asarray(theta_batch, dtype=float)
    expected_size = int(theta_size)
    if arr.ndim != 2:
        raise ValueError(f"theta_batch must be 2D, got shape {arr.shape}")
    if arr.shape[1] != expected_size:
        raise ValueError(f"theta_batch must have width {expected_size}, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("theta_batch must contain at least one sample")
    if not np.all(np.isfinite(arr)):
        raise ValueError("theta_batch contains non-finite values")
    return arr.astype(float, copy=True)


def compute_gradient_statistics_for_batch(
    theta_batch: object,
    context: ObjectiveContext,
    modes: list[dict[str, Any]],
    fd_eps: float,
) -> dict[str, FloatArray]:
    """Compute losses and finite-difference gradients for a supplied theta batch.

    Returned arrays use the same schema as random-initialization batch files:

    - ``losses`` has shape ``(modes, samples)``
    - ``gradients`` has shape ``(modes, samples, theta_size)``
    - ``grad_sq_norms`` and ``normalized_grad_sq_norms`` have shape ``(modes, samples)``
    - ``elapsed_seconds_by_mode`` has shape ``(modes,)``
    """
    if fd_eps <= 0.0:
        raise ValueError(f"fd_eps must be positive, got {fd_eps}")
    if not modes:
        raise ValueError("modes must contain at least one mode")
    theta_values = validate_theta_batch(theta_batch, context.theta_size)
    sample_count, theta_size = theta_values.shape
    losses = np.empty((len(modes), sample_count), dtype=float)
    gradients = np.empty((len(modes), sample_count, theta_size), dtype=float)
    elapsed_by_mode = np.empty((len(modes),), dtype=float)

    for mode_index, mode in enumerate(modes):
        started = perf_counter()
        set_gradient_mode(mode)
        loss_fn = lambda theta, ctx=context: sum_block_loss(theta, ctx)
        for sample_pos in range(sample_count):
            theta = theta_values[sample_pos]
            losses[mode_index, sample_pos] = float(loss_fn(theta))
            gradients[mode_index, sample_pos] = finite_difference_gradient(loss_fn, theta, fd_eps)
        elapsed_by_mode[mode_index] = perf_counter() - started

    grad_sq_norms = np.sum(gradients * gradients, axis=2)
    normalized_grad_sq_norms = grad_sq_norms / float(theta_size)
    return {
        "losses": losses,
        "gradients": gradients,
        "grad_sq_norms": grad_sq_norms,
        "normalized_grad_sq_norms": normalized_grad_sq_norms,
        "elapsed_seconds_by_mode": elapsed_by_mode,
    }
