"""Pure JAX gate matrices using Cirq-compatible basis ordering."""

from __future__ import annotations

from typing import Final

from .runtime import jax as _jax

from jax import Array
import jax.numpy as jnp

_ = _jax

IDENTITY_2: Final = jnp.eye(2, dtype=jnp.complex128)


def rx(angle: Array) -> Array:
    half = angle / 2.0
    cos_half = jnp.cos(half)
    sin_half = jnp.sin(half)
    return jnp.asarray(
        [[cos_half, -1j * sin_half], [-1j * sin_half, cos_half]],
        dtype=jnp.complex128,
    )


def ry(angle: Array) -> Array:
    half = angle / 2.0
    cos_half = jnp.cos(half)
    sin_half = jnp.sin(half)
    return jnp.asarray(
        [[cos_half, -sin_half], [sin_half, cos_half]],
        dtype=jnp.complex128,
    )


def rz(angle: Array) -> Array:
    half = angle / 2.0
    return jnp.asarray(
        [[jnp.exp(-1j * half), 0.0], [0.0, jnp.exp(1j * half)]],
        dtype=jnp.complex128,
    )


def kron_all(factors: tuple[Array, ...]) -> Array:
    result = factors[0]
    for factor in factors[1:]:
        result = jnp.kron(result, factor)
    return result


def single_qubit_operator(gate: Array, qubit_position: int, n_qubits: int) -> Array:
    factors = tuple(gate if position == qubit_position else IDENTITY_2 for position in range(n_qubits))
    return kron_all(factors)


def cz_operator(control_position: int, target_position: int, n_qubits: int) -> Array:
    dim = 1 << n_qubits
    phases = []
    for basis_index in range(dim):
        control_bit = (basis_index >> (n_qubits - 1 - control_position)) & 1
        target_bit = (basis_index >> (n_qubits - 1 - target_position)) & 1
        phases.append(-1.0 if control_bit and target_bit else 1.0)
    return jnp.diag(jnp.asarray(phases, dtype=jnp.complex128))


def pauli_operator(name: str, qubit_position: int, n_qubits: int) -> Array:
    match name:
        case "X":
            base = jnp.asarray([[0, 1], [1, 0]], dtype=jnp.complex128)
        case "Y":
            base = jnp.asarray([[0, -1j], [1j, 0]], dtype=jnp.complex128)
        case "Z":
            base = jnp.asarray([[1, 0], [0, -1]], dtype=jnp.complex128)
        case unreachable:
            raise AssertionError(f"unsupported Pauli operator: {unreachable}")
    return single_qubit_operator(base, qubit_position, n_qubits)
