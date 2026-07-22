"""default_5layer_cz light-cone ansatz implemented with pure JAX matrices."""

from __future__ import annotations

from typing import Final

from .capabilities import SUPPORTED_LIGHTCONE_WIDTHS
from .runtime import jax as _jax
from .gates import cz_operator, rx, ry, rz, single_qubit_operator

from jax import Array
import jax.numpy as jnp

_ = _jax

N_LAYERS: Final = 5
PARAMS_PER_ROTATION: Final = 3
CZ_LAYER_PARITIES: Final = (0, 1, 1, 0)


def ansatz_unitary(theta: Array, *, n_qubits: int) -> Array:
    theta_array = jnp.asarray(theta, dtype=jnp.float64)
    expected_theta_size = N_LAYERS * n_qubits * PARAMS_PER_ROTATION
    if n_qubits <= 0:
        raise ValueError(f"n_qubits must be positive, got {n_qubits}")
    if n_qubits not in SUPPORTED_LIGHTCONE_WIDTHS:
        raise ValueError(f"n_qubits must be one of {SUPPORTED_LIGHTCONE_WIDTHS}, got {n_qubits}")
    if theta_array.ndim != 1:
        raise ValueError(f"theta must be one-dimensional, got shape {theta_array.shape}")
    if theta_array.shape[0] != expected_theta_size:
        raise ValueError(f"theta must have length {expected_theta_size} for {n_qubits} qubits, got {theta_array.shape[0]}")
    dim = 1 << n_qubits
    unitary = jnp.eye(dim, dtype=jnp.complex128)
    for layer in range(N_LAYERS):
        for qubit_position in range(n_qubits):
            theta_index = (layer * n_qubits + qubit_position) * PARAMS_PER_ROTATION
            x_angle = theta_array[theta_index]
            y_angle = theta_array[theta_index + 1]
            z_angle = theta_array[theta_index + 2]
            for gate in (rx(x_angle), ry(y_angle), rz(z_angle)):
                unitary = single_qubit_operator(gate, qubit_position, n_qubits) @ unitary
        if layer < len(CZ_LAYER_PARITIES):
            parity = CZ_LAYER_PARITIES[layer]
            for left_position in range(parity, n_qubits - 1, 2):
                unitary = cz_operator(left_position, left_position + 1, n_qubits) @ unitary
    return unitary
