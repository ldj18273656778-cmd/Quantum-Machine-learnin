"""Isolated JAX parity backend for light-cone parity tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .capabilities import SUPPORTED_LIGHTCONE_WIDTHS
from .runtime import jax as _jax

from jax import Array

if TYPE_CHECKING:
    from .heisenberg import HeisenbergEigvalshDiagnostics
    from .optimizer import AdamConfig, AdamRunMetadata, AdamRunResult, AdamState, adam_optimize

_ = _jax

__all__ = [
    "SUPPORTED_LIGHTCONE_WIDTHS",
    "ansatz_unitary",
    "heisenberg_pauli_eigvalsh_diagnostics",
    "heisenberg_pauli_loss",
    "heisenberg_pauli_loss_eigvalsh",
    "heisenberg_pauli_loss_eigvalsh_rematerialized",
    "heisenberg_pauli_loss_rematerialized",
    "value_and_grad_heisenberg_pauli_loss",
    "value_and_grad_heisenberg_pauli_loss_rematerialized",
    "AdamConfig",
    "AdamRunMetadata",
    "AdamRunResult",
    "AdamState",
    "adam_optimize",
]


def __getattr__(name: str):
    if name == "AdamConfig":
        from .optimizer import AdamConfig

        return AdamConfig
    if name == "AdamRunMetadata":
        from .optimizer import AdamRunMetadata

        return AdamRunMetadata
    if name == "AdamRunResult":
        from .optimizer import AdamRunResult

        return AdamRunResult
    if name == "AdamState":
        from .optimizer import AdamState

        return AdamState
    if name == "adam_optimize":
        from .optimizer import adam_optimize

        return adam_optimize
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class ObjectiveContextLike(Protocol):
    target_operator: Array
    block_qubits: tuple[int, ...]
    lightcone_qubits: tuple[int, ...]


def ansatz_unitary(theta: Array, *, n_qubits: int) -> Array:
    from .ansatz import ansatz_unitary as build_ansatz_unitary

    return build_ansatz_unitary(theta, n_qubits=n_qubits)


def heisenberg_pauli_loss(theta: Array, context: ObjectiveContextLike) -> Array:
    from .heisenberg import heisenberg_pauli_loss as evaluate_heisenberg_pauli_loss

    return evaluate_heisenberg_pauli_loss(theta, context)


def heisenberg_pauli_loss_rematerialized(theta: Array, context: ObjectiveContextLike) -> Array:
    from .heisenberg import heisenberg_pauli_loss_rematerialized as evaluate_heisenberg_pauli_loss

    return evaluate_heisenberg_pauli_loss(theta, context)


def heisenberg_pauli_loss_eigvalsh(
    theta: Array,
    target_operator: Array,
    *,
    block_qubits: tuple[int, ...],
    lightcone_qubits: tuple[int, ...],
) -> Array:
    from .heisenberg import heisenberg_pauli_loss_eigvalsh as evaluate_heisenberg_pauli_loss

    return evaluate_heisenberg_pauli_loss(
        theta,
        target_operator,
        block_qubits=block_qubits,
        lightcone_qubits=lightcone_qubits,
    )


def heisenberg_pauli_loss_eigvalsh_rematerialized(
    theta: Array,
    target_operator: Array,
    *,
    block_qubits: tuple[int, ...],
    lightcone_qubits: tuple[int, ...],
) -> Array:
    from .heisenberg import heisenberg_pauli_loss_eigvalsh_rematerialized as evaluate_heisenberg_pauli_loss

    return evaluate_heisenberg_pauli_loss(
        theta,
        target_operator,
        block_qubits=block_qubits,
        lightcone_qubits=lightcone_qubits,
    )


def heisenberg_pauli_eigvalsh_diagnostics(
    theta: Array,
    target_operator: Array,
    *,
    block_qubits: tuple[int, ...],
    lightcone_qubits: tuple[int, ...],
) -> "HeisenbergEigvalshDiagnostics":
    from .heisenberg import heisenberg_pauli_eigvalsh_diagnostics as evaluate_diagnostics

    return evaluate_diagnostics(
        theta,
        target_operator,
        block_qubits=block_qubits,
        lightcone_qubits=lightcone_qubits,
    )


def value_and_grad_heisenberg_pauli_loss(theta: Array, context: ObjectiveContextLike) -> tuple[Array, Array]:
    from .heisenberg import value_and_grad_heisenberg_pauli_loss as evaluate_value_and_grad

    return evaluate_value_and_grad(theta, context)


def value_and_grad_heisenberg_pauli_loss_rematerialized(
    theta: Array,
    context: ObjectiveContextLike,
) -> tuple[Array, Array]:
    from .heisenberg import value_and_grad_heisenberg_pauli_loss_rematerialized as evaluate_value_and_grad

    return evaluate_value_and_grad(theta, context)
