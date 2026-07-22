"""Light-cone Heisenberg Pauli loss in pure JAX."""

from __future__ import annotations

from typing import Final, NamedTuple, Protocol

from .runtime import jax
from .ansatz import ansatz_unitary
from .gates import pauli_operator

from jax import Array
import jax.numpy as jnp

ZERO_RESIDUAL_TOLERANCE: Final = 1e-24
NEAR_DEGENERATE_EIGENGAP: Final = 1e-10


class ObjectiveContextLike(Protocol):
    target_operator: Array
    block_qubits: tuple[int, ...]
    lightcone_qubits: tuple[int, ...]


class HeisenbergEigvalshDiagnostics(NamedTuple):
    loss: Array
    minimum_eigengap: Array
    near_degenerate: Array


def heisenberg_pauli_loss(theta: Array, context: ObjectiveContextLike) -> Array:
    return heisenberg_pauli_loss_eigvalsh(
        theta,
        context.target_operator,
        block_qubits=context.block_qubits,
        lightcone_qubits=context.lightcone_qubits,
    )


def heisenberg_pauli_loss_rematerialized(theta: Array, context: ObjectiveContextLike) -> Array:
    return heisenberg_pauli_loss_eigvalsh_rematerialized(
        theta,
        context.target_operator,
        block_qubits=context.block_qubits,
        lightcone_qubits=context.lightcone_qubits,
    )


def heisenberg_pauli_loss_eigvalsh(
    theta: Array,
    target_operator: Array,
    *,
    block_qubits: tuple[int, ...],
    lightcone_qubits: tuple[int, ...],
) -> Array:
    lightcone_qubits = tuple(int(qubit) for qubit in lightcone_qubits)
    n_qubits = len(lightcone_qubits)
    target = jnp.asarray(target_operator, dtype=jnp.complex128)
    trial = ansatz_unitary(theta, n_qubits=n_qubits)
    residual = target @ jnp.conjugate(trial.T)
    loss = jnp.asarray(0.0, dtype=jnp.float64)
    for block_qubit in block_qubits:
        qubit_position = lightcone_qubits.index(int(block_qubit))
        for pauli_name in ("X", "Y", "Z"):
            pauli = pauli_operator(pauli_name, qubit_position, n_qubits)
            loss = loss + heisenberg_pauli_term_loss(residual, pauli)
    return loss


def heisenberg_pauli_loss_eigvalsh_rematerialized(
    theta: Array,
    target_operator: Array,
    *,
    block_qubits: tuple[int, ...],
    lightcone_qubits: tuple[int, ...],
) -> Array:
    lightcone_qubits = tuple(int(qubit) for qubit in lightcone_qubits)
    n_qubits = len(lightcone_qubits)
    target = jnp.asarray(target_operator, dtype=jnp.complex128)
    trial = ansatz_unitary(theta, n_qubits=n_qubits)
    residual = target @ jnp.conjugate(trial.T)
    loss = jnp.asarray(0.0, dtype=jnp.float64)
    rematerialized_term_loss = jax.checkpoint(heisenberg_pauli_term_loss)
    for block_qubit in block_qubits:
        qubit_position = lightcone_qubits.index(int(block_qubit))
        for pauli_name in ("X", "Y", "Z"):
            pauli = pauli_operator(pauli_name, qubit_position, n_qubits)
            loss = loss + rematerialized_term_loss(residual, pauli)
    return loss


def heisenberg_pauli_eigvalsh_diagnostics(
    theta: Array,
    target_operator: Array,
    *,
    block_qubits: tuple[int, ...],
    lightcone_qubits: tuple[int, ...],
) -> HeisenbergEigvalshDiagnostics:
    lightcone_qubits = tuple(int(qubit) for qubit in lightcone_qubits)
    n_qubits = len(lightcone_qubits)
    target = jnp.asarray(target_operator, dtype=jnp.complex128)
    trial = ansatz_unitary(theta, n_qubits=n_qubits)
    residual = target @ jnp.conjugate(trial.T)
    loss = jnp.asarray(0.0, dtype=jnp.float64)
    minimum_eigengap = jnp.asarray(jnp.inf, dtype=jnp.float64)
    for block_qubit in block_qubits:
        qubit_position = lightcone_qubits.index(int(block_qubit))
        for pauli_name in ("X", "Y", "Z"):
            pauli = pauli_operator(pauli_name, qubit_position, n_qubits)
            delta = jnp.conjugate(residual.T) @ pauli @ residual - pauli
            eigenvalues = hermitian_residual_eigenvalues(delta)
            operator_norm = jnp.max(jnp.abs(eigenvalues))
            loss = loss + jnp.real(operator_norm * operator_norm)
            minimum_eigengap = jnp.minimum(minimum_eigengap, minimum_adjacent_eigengap(eigenvalues))
    return HeisenbergEigvalshDiagnostics(
        loss=loss,
        minimum_eigengap=minimum_eigengap,
        near_degenerate=minimum_eigengap <= NEAR_DEGENERATE_EIGENGAP,
    )


def squared_operator_norm(delta: Array) -> Array:
    delta_norm_squared = jnp.real(jnp.vdot(delta, delta))
    return jax.lax.cond(
        delta_norm_squared <= ZERO_RESIDUAL_TOLERANCE,
        lambda: jnp.asarray(0.0, dtype=jnp.float64),
        lambda: squared_hermitian_operator_norm(delta),
    )


def heisenberg_pauli_term_loss(residual: Array, pauli: Array) -> Array:
    delta = jnp.conjugate(residual.T) @ pauli @ residual - pauli
    return squared_operator_norm(delta)


def squared_hermitian_operator_norm(delta: Array) -> Array:
    eigenvalues = hermitian_residual_eigenvalues(delta)
    operator_norm = jnp.max(jnp.abs(eigenvalues))
    return jnp.real(operator_norm * operator_norm)


def hermitian_residual_eigenvalues(delta: Array) -> Array:
    hermitian_delta = (delta + jnp.conjugate(delta.T)) / 2.0
    return jnp.linalg.eigvalsh(hermitian_delta)


def minimum_adjacent_eigengap(eigenvalues: Array) -> Array:
    """Return the minimum sorted adjacent gap; one-dimensional spectra use 0.0."""
    if eigenvalues.shape[0] <= 1:
        return jnp.asarray(0.0, dtype=jnp.float64)
    gaps = eigenvalues[1:] - eigenvalues[:-1]
    return jnp.min(jnp.maximum(gaps, 0.0))


def value_and_grad_heisenberg_pauli_loss(theta: Array, context: ObjectiveContextLike) -> tuple[Array, Array]:
    theta_array = jnp.asarray(theta, dtype=jnp.float64)
    loss_function = lambda candidate_theta: heisenberg_pauli_loss(candidate_theta, context)
    return jax.value_and_grad(loss_function)(theta_array)


def value_and_grad_heisenberg_pauli_loss_rematerialized(
    theta: Array,
    context: ObjectiveContextLike,
) -> tuple[Array, Array]:
    theta_array = jnp.asarray(theta, dtype=jnp.float64)
    loss_function = lambda candidate_theta: heisenberg_pauli_loss_rematerialized(candidate_theta, context)
    return jax.value_and_grad(loss_function)(theta_array)
