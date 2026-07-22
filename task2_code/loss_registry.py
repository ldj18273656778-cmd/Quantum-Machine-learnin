"""Training objective registry for Task 2 local-inversion losses.

This registry selects the scalar loss optimized by ADAM.  It is separate
from ``superoperator_registry``: the superoperator registry controls how a
residual unitary is converted into per-bit losses, while this registry
controls how those per-bit losses are aggregated into a scalar objective.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from task2_code.ansatz_registry import registered_ansatz_unitary_on_lightcone
from task2_code.local_loss import embed_block_unitary_in_lightcone
from task2_code.superoperator_registry import get_active_superop

ComplexArray: TypeAlias = NDArray[np.complex128]
FloatArray: TypeAlias = NDArray[np.float64]


class ObjectiveContextLike(Protocol):
    @property
    def target_operator(self) -> ComplexArray: ...

    @property
    def block_qubits(self) -> tuple[int, ...]: ...

    @property
    def lightcone_qubits(self) -> tuple[int, ...]: ...

    @property
    def target_bit(self) -> int: ...

    @property
    def full_target_operator(self) -> ComplexArray | None: ...

    @property
    def system_qubits(self) -> tuple[int, ...] | None: ...

    @property
    def loss_mode(self) -> str: ...

    @property
    def ansatz_qubits(self) -> int: ...

    @property
    def theta_size(self) -> int: ...

    @property
    def ansatz(self) -> str: ...

    @property
    def block_only_ansatz(self) -> bool: ...


ObjectiveLossFn = Callable[[object, ObjectiveContextLike], float]
PerBitObjectiveLossFn = Callable[[object, ObjectiveContextLike], dict[int, float]]


@dataclass(frozen=True)
class LossFunctionSpec:
    fn: ObjectiveLossFn
    per_bit_fn: PerBitObjectiveLossFn
    uses_superoperator: bool = False


def _normalise_loss_mode(loss_mode: str) -> str:
    mode = str(loss_mode).strip().lower()
    if mode not in {"lightcone", "full_system"}:
        raise ValueError(f"unknown loss_mode {loss_mode!r}; expected 'lightcone' or 'full_system'")
    return mode


def residual_operator_for_context(
    theta: object,
    context: ObjectiveContextLike,
) -> tuple[ComplexArray, Sequence[int]]:
    theta_arr = np.asarray(theta, dtype=float)
    trial = np.asarray(
        registered_ansatz_unitary_on_lightcone(
            context.ansatz,
            theta_arr,
            context.lightcone_qubits,
            context.block_qubits,
            block_only_ansatz=context.block_only_ansatz,
        ),
        dtype=complex,
    )
    loss_mode = _normalise_loss_mode(context.loss_mode)
    if loss_mode == "lightcone":
        return np.asarray(context.target_operator @ trial.conj().T, dtype=complex), context.lightcone_qubits

    if context.full_target_operator is None:
        return np.asarray(context.target_operator @ trial.conj().T, dtype=complex), context.lightcone_qubits
    if context.system_qubits is None:
        raise ValueError("system_qubits are required when full_target_operator is set")
    embedded_trial = embed_block_unitary_in_lightcone(
        trial,
        context.system_qubits,
        context.lightcone_qubits,
    )
    residual = context.full_target_operator @ embedded_trial.conj().T
    return np.asarray(residual, dtype=complex), context.system_qubits


def _lightcone_residual_operator_for_context(
    theta: object,
    context: ObjectiveContextLike,
) -> tuple[ComplexArray, tuple[int, ...]]:
    theta_arr = np.asarray(theta, dtype=float)
    trial = np.asarray(
        registered_ansatz_unitary_on_lightcone(
            context.ansatz,
            theta_arr,
            context.lightcone_qubits,
            context.block_qubits,
            block_only_ansatz=context.block_only_ansatz,
        ),
        dtype=complex,
    )
    residual = context.target_operator @ trial.conj().T
    return np.asarray(residual, dtype=complex), context.lightcone_qubits


PAULI_MATRICES: dict[str, ComplexArray] = {
    "X": np.asarray([[0, 1], [1, 0]], dtype=complex),
    "Y": np.asarray([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.asarray([[1, 0], [0, -1]], dtype=complex),
}


def _single_qubit_pauli_operator(pauli: str, qubit_position: int, n_qubits: int) -> ComplexArray:
    identity = np.eye(2, dtype=complex)
    factors = [PAULI_MATRICES[pauli] if pos == qubit_position else identity for pos in range(n_qubits)]
    op = factors[0]
    for factor in factors[1:]:
        op = np.kron(op, factor)
    return np.asarray(op, dtype=complex)


def edge_quantum_channel_per_bit_losses(
    theta: object,
    context: ObjectiveContextLike,
) -> dict[int, float]:
    residual, loss_qubits = residual_operator_for_context(theta, context)
    return {
        int(q): float(v)
        for q, v in get_active_superop()(residual, context.block_qubits, loss_qubits, target_bits=None).items()
    }


def edge_quantum_channel_loss(theta: object, context: ObjectiveContextLike) -> float:
    """Edge quantum channel objective: sum active-superoperator losses over block bits."""
    losses = edge_quantum_channel_per_bit_losses(theta, context)
    loss = float(sum(losses.values()))
    if not np.isfinite(loss):
        raise FloatingPointError(f"non-finite edge quantum channel loss: {loss}")
    return loss


def heisenberg_pauli_per_bit_losses(
    theta: object,
    context: ObjectiveContextLike,
) -> dict[int, float]:
    """Per-block-qubit Heisenberg Pauli loss on the block lightcone."""
    W, loss_qubits = _lightcone_residual_operator_for_context(theta, context)
    n_qubits = len(loss_qubits)
    losses: dict[int, float] = {}
    for q in context.block_qubits:
        if q not in loss_qubits:
            raise ValueError(f"block qubit {q} is not in lightcone_qubits={loss_qubits}")
        q_pos = loss_qubits.index(q)
        total = 0.0
        for pauli in ("X", "Y", "Z"):
            P = _single_qubit_pauli_operator(pauli, q_pos, n_qubits)
            delta = W.conj().T @ P @ W - P
            op_norm = float(np.linalg.norm(delta, ord=2))
            total += op_norm * op_norm
        losses[int(q)] = total
    return losses


def heisenberg_pauli_loss(theta: object, context: ObjectiveContextLike) -> float:
    """Heisenberg Pauli objective: sum_q sum_P ||W^dag P_q W - P_q||_inf^2."""
    losses = heisenberg_pauli_per_bit_losses(theta, context)
    loss = float(sum(losses.values()))
    if not np.isfinite(loss):
        raise FloatingPointError(f"non-finite Heisenberg Pauli loss: {loss}")
    return loss


LOSS_FUNCTION_REGISTRY: dict[str, LossFunctionSpec] = {
    "edge_quantum_channel": LossFunctionSpec(
        edge_quantum_channel_loss,
        edge_quantum_channel_per_bit_losses,
        uses_superoperator=True,
    ),
    "edge quantum channel": LossFunctionSpec(
        edge_quantum_channel_loss,
        edge_quantum_channel_per_bit_losses,
        uses_superoperator=True,
    ),
    "heisenberg_pauli": LossFunctionSpec(
        heisenberg_pauli_loss,
        heisenberg_pauli_per_bit_losses,
        uses_superoperator=False,
    ),
    "Heisenberg Pauli": LossFunctionSpec(
        heisenberg_pauli_loss,
        heisenberg_pauli_per_bit_losses,
        uses_superoperator=False,
    ),
    "heisenberg pauli": LossFunctionSpec(
        heisenberg_pauli_loss,
        heisenberg_pauli_per_bit_losses,
        uses_superoperator=False,
    ),
    "heisenberg-pauli": LossFunctionSpec(
        heisenberg_pauli_loss,
        heisenberg_pauli_per_bit_losses,
        uses_superoperator=False,
    ),
    "sum_block_loss": LossFunctionSpec(
        edge_quantum_channel_loss,
        edge_quantum_channel_per_bit_losses,
        uses_superoperator=True,
    ),
    "target_bit_loss": LossFunctionSpec(
        edge_quantum_channel_loss,
        edge_quantum_channel_per_bit_losses,
        uses_superoperator=True,
    ),
    "max_block_loss": LossFunctionSpec(
        edge_quantum_channel_loss,
        edge_quantum_channel_per_bit_losses,
        uses_superoperator=True,
    ),
}

_active_spec: LossFunctionSpec = LOSS_FUNCTION_REGISTRY["edge_quantum_channel"]


def resolve_loss_function(name: str) -> ObjectiveLossFn:
    """Look up a scalar objective function by registry key."""
    return resolve_loss_function_spec(name).fn


def resolve_loss_function_spec(name: str) -> LossFunctionSpec:
    """Look up a scalar objective spec by registry key."""
    try:
        return LOSS_FUNCTION_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown loss function '{name}'. Available: {sorted(LOSS_FUNCTION_REGISTRY)}"
        ) from None


def register_loss_function(
    name: str,
    fn: ObjectiveLossFn,
    *,
    per_bit_fn: PerBitObjectiveLossFn | None = None,
    uses_superoperator: bool = False,
) -> None:
    """Register a scalar objective function under *name*."""
    if per_bit_fn is None:
        per_bit_fn = lambda theta, context: {-1: fn(theta, context)}
    LOSS_FUNCTION_REGISTRY[name] = LossFunctionSpec(
        fn,
        per_bit_fn,
        uses_superoperator=uses_superoperator,
    )


def loss_function_uses_superoperator(name: str) -> bool:
    """Return whether the named loss function needs a superoperator sub-mode."""
    return resolve_loss_function_spec(name).uses_superoperator


def active_loss_breakdown(theta: object, context: ObjectiveContextLike) -> dict[int, float]:
    """Return the active loss function's per-qubit or component breakdown."""
    return _active_spec.per_bit_fn(theta, context)


def set_active_loss_function(name_or_fn: str | ObjectiveLossFn) -> None:
    """Set the active scalar objective used by training wrappers."""
    global _active_spec
    if isinstance(name_or_fn, str):
        _active_spec = resolve_loss_function_spec(name_or_fn)
    else:
        _active_spec = LossFunctionSpec(name_or_fn, lambda theta, context: {-1: name_or_fn(theta, context)})


def get_active_loss_function() -> ObjectiveLossFn:
    """Return the currently active scalar objective function."""
    return _active_spec.fn
