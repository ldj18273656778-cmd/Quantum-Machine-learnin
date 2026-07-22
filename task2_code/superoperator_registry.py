"""Superoperator / per-bit loss registry.

Maps stable string names to the loss-calculation functions used
during training so experiment configs can switch loss definitions
without editing training internals.

Each registered callable has the same signature as
``per_bit_losses_from_V``:

    (V, block_qubits, lightcone_qubits, target_bits=None) -> dict[int, float]

Usage:
    from task2_code.superoperator_registry import set_active_superop, get_active_superop

    set_active_superop("superoperator_from_mix")
    loss_fn = get_active_superop()
    losses = loss_fn(V, block_qubits, lightcone_qubits)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from task2_code.superoperator import per_bit_losses_from_V, per_bit_losses_from_V_one_env, per_bit_losses_from_V_zero_env

# ── public type ───────────────────────────────────────────────────────
PerBitLossFn = Callable[..., dict[int, float]]

# ── registry ──────────────────────────────────────────────────────────
SUPEROP_REGISTRY: dict[str, PerBitLossFn] = {
    "superoperator_from_mix": per_bit_losses_from_V,
    "superoperator_from_zero": per_bit_losses_from_V_zero_env,
    "superoperator_from_one": per_bit_losses_from_V_one_env,
}

# ── active superoperator (module-level singleton) ─────────────────────
_active: PerBitLossFn = per_bit_losses_from_V  # default at import time


def resolve_superop(name: str) -> PerBitLossFn:
    """Look up a per-bit-loss function by registry key.

    Raises ``KeyError`` with available key names for unrecognised inputs.
    """
    try:
        return SUPEROP_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown superoperator '{name}'.  Available: {sorted(SUPEROP_REGISTRY)}"
        ) from None


def register_superop(name: str, fn: PerBitLossFn) -> None:
    """Register a new per-bit-loss function under *name* (overwrites existing)."""
    if not callable(fn):
        raise TypeError(f"superoperator must be callable, got {type(fn).__name__}")
    SUPEROP_REGISTRY[name] = fn


def set_active_superop(name_or_fn) -> None:
    """Set the active superoperator used by all training internals.

    Accepts either a registry key (``str``) or a callable directly
    (useful for tests that inject spy wrappers).

    Call once before training / evaluation, typically at the top of a
    training script based on ``ExperimentConfig.superoperator``.
    """
    global _active
    if isinstance(name_or_fn, str):
        _active = resolve_superop(name_or_fn)
    else:
        _active = name_or_fn  # direct callable (e.g. test spy)


def get_active_superop() -> PerBitLossFn:
    """Return the currently active per-bit-loss function."""
    return _active
