"""Module A: local inversion ansatz circuits.

The paper's 4-qubit ansatz is a special case of the light-cone ansatz used
here.  For ``n`` light-cone qubits the circuit has five layers of single-qubit
``R(x,y,z)=Rx(x) Ry(y) Rz(z)`` rotations and four fixed CZ layers in the order
``even, odd, odd, even``.  The trainable parameter count is ``15*n``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import cirq
import numpy as np
from numpy.typing import NDArray

from task2_code.local_loss import embed_block_unitary_in_lightcone


N_QUBITS = 4
N_LAYERS = 5
PARAMS_PER_ROTATION = 3
N_GATES = N_LAYERS * N_QUBITS
N_PARAMS = N_GATES * PARAMS_PER_ROTATION
CZ_LAYER_PARITIES = (0, 1, 1, 0)
AnsatzBuilder = Callable[..., cirq.Circuit]


def _validate_n_qubits(n_qubits: int) -> int:
    n_val = int(n_qubits)
    if n_val <= 0:
        raise ValueError(f"n_qubits must be positive, got {n_qubits}")
    return n_val


def theta_count(n_qubits: int = N_QUBITS) -> int:
    """Return the number of trainable scalar parameters for ``n_qubits``."""
    return N_LAYERS * _validate_n_qubits(n_qubits) * PARAMS_PER_ROTATION


def random_theta(
    rng: np.random.Generator | None = None,
    low: float = 0.0,
    high: float = 2.0 * np.pi,
    *,
    n_qubits: int = N_QUBITS,
) -> NDArray[np.float64]:
    """Return a random parameter vector of length ``15*n_qubits``.

    ``n_qubits`` is keyword-only so existing calls like
    ``random_theta(rng, low, high)`` keep their 4-qubit behavior.
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(low, high, size=theta_count(n_qubits))


def pack_theta(x_vec: Sequence[float], y_vec: Sequence[float], z_vec: Sequence[float]) -> NDArray[np.float64]:
    """Pack three equal-length vectors into ``(x0,y0,z0, x1,y1,z1, ...)``."""
    if not (len(x_vec) == len(y_vec) == len(z_vec)):
        raise ValueError("x_vec, y_vec, and z_vec must have the same length")
    theta = np.empty(len(x_vec) * PARAMS_PER_ROTATION, dtype=float)
    theta[0::3] = x_vec
    theta[1::3] = y_vec
    theta[2::3] = z_vec
    return theta


def _r_xyz_gate(x: float, y: float, z: float, qubit: cirq.Qid) -> list[cirq.Operation]:
    """Return operations for ``R(x,y,z)=Rx(x)Ry(y)Rz(z)``."""
    return [cirq.rx(x).on(qubit), cirq.ry(y).on(qubit), cirq.rz(z).on(qubit)]


def cz_pairs_for_layer(n_qubits: int, layer_idx: int) -> list[tuple[int, int]]:
    """Return local-index CZ pairs for one of the four fixed CZ layers."""
    n_val = _validate_n_qubits(n_qubits)
    if layer_idx < 0 or layer_idx >= len(CZ_LAYER_PARITIES):
        raise ValueError(f"layer_idx must be in [0, 3], got {layer_idx}")
    start = CZ_LAYER_PARITIES[layer_idx]
    return [(idx, idx + 1) for idx in range(start, n_val - 1, 2)]


def _cz_moment(layer_idx: int, qubits: Sequence[cirq.Qid]) -> list[cirq.Operation]:
    pairs = cz_pairs_for_layer(len(qubits), layer_idx)
    return [cirq.CZ(qubits[a], qubits[b]) for a, b in pairs]


def _cnot_moment(layer_idx: int, qubits: Sequence[cirq.Qid]) -> list[cirq.Operation]:
    pairs = cz_pairs_for_layer(len(qubits), layer_idx)
    return [cirq.CNOT(qubits[a], qubits[b]) for a, b in pairs]


def _resolve_qubits(
    theta: object,
    qubits: Sequence[cirq.Qid] | None,
    n_qubits: int | None,
) -> tuple[NDArray[np.float64], Sequence[cirq.Qid], int]:
    theta_arr = np.asarray(theta, dtype=float)
    if theta_arr.ndim != 1:
        raise ValueError(f"theta must be one-dimensional, got shape {theta_arr.shape}")
    if theta_arr.size % (N_LAYERS * PARAMS_PER_ROTATION) != 0:
        raise ValueError(
            "theta length must be divisible by 15; "
            + f"got length {theta_arr.size}"
        )

    inferred_from_theta = theta_arr.size // (N_LAYERS * PARAMS_PER_ROTATION)
    if qubits is not None:
        qubit_list = list(qubits)
        if not qubit_list:
            raise ValueError("qubits must not be empty")
        if n_qubits is not None and len(qubit_list) != int(n_qubits):
            raise ValueError(
                f"len(qubits)={len(qubit_list)} conflicts with n_qubits={n_qubits}"
            )
        n_val = len(qubit_list)
    elif n_qubits is not None:
        n_val = _validate_n_qubits(n_qubits)
        qubit_list = list(cirq.LineQubit.range(n_val))
    else:
        n_val = _validate_n_qubits(inferred_from_theta)
        qubit_list = list(cirq.LineQubit.range(n_val))

    expected = theta_count(n_val)
    if theta_arr.size != expected:
        raise ValueError(f"theta must have length {expected} for {n_val} qubits, got {theta_arr.size}")
    return theta_arr, qubit_list, n_val


def build_ansatz(
    theta: object,
    qubits: Sequence[cirq.Qid] | None = None,
    n_qubits: int | None = None,
) -> cirq.Circuit:
    """Build the dynamic light-cone local-inversion ansatz circuit."""
    theta_arr, qubit_list, n_val = _resolve_qubits(theta, qubits, n_qubits)
    circuit = cirq.Circuit()
    for layer in range(N_LAYERS):
        for q_idx, qubit in enumerate(qubit_list):
            idx = (layer * n_val + q_idx) * PARAMS_PER_ROTATION
            x, y, z = theta_arr[idx], theta_arr[idx + 1], theta_arr[idx + 2]
            circuit.append(_r_xyz_gate(float(x), float(y), float(z), qubit))
        if layer < len(CZ_LAYER_PARITIES):
            circuit.append(_cz_moment(layer, qubit_list))
    return circuit


def build_ansatz_cnot(
    theta: object,
    qubits: Sequence[cirq.Qid] | None = None,
    n_qubits: int | None = None,
) -> cirq.Circuit:
    """Build the default 5-layer ansatz using CNOT entanglers instead of CZ."""
    theta_arr, qubit_list, n_val = _resolve_qubits(theta, qubits, n_qubits)
    circuit = cirq.Circuit()
    for layer in range(N_LAYERS):
        for q_idx, qubit in enumerate(qubit_list):
            idx = (layer * n_val + q_idx) * PARAMS_PER_ROTATION
            x, y, z = theta_arr[idx], theta_arr[idx + 1], theta_arr[idx + 2]
            circuit.append(_r_xyz_gate(float(x), float(y), float(z), qubit))
        if layer < len(CZ_LAYER_PARITIES):
            circuit.append(_cnot_moment(layer, qubit_list))
    return circuit


def build_ansatz_4q(theta: object, qubits: Sequence[cirq.Qid] | None = None) -> cirq.Circuit:
    """Build the backward-compatible 4-qubit local-inversion ansatz."""
    return build_ansatz(theta, qubits=qubits, n_qubits=N_QUBITS)


def ansatz_unitary(
    theta: object,
    qubits: Sequence[cirq.Qid] | None = None,
    n_qubits: int | None = None,
) -> NDArray[np.complex128]:
    """Return the ansatz unitary in the explicit qubit order used by the circuit."""
    theta_arr, qubit_list, n_val = _resolve_qubits(theta, qubits, n_qubits)
    circuit = build_ansatz(theta_arr, qubits=qubit_list, n_qubits=n_val)
    return np.asarray(circuit.unitary(qubit_order=qubit_list), dtype=complex)


def ansatz_scope_qubits(
    lightcone_qubits: Sequence[int],
    block_qubits: Sequence[int],
    *,
    block_only_ansatz: bool = False,
) -> tuple[int, ...]:
    """Return global qubit labels the ansatz circuit should act on."""
    return tuple(int(q) for q in (block_qubits if block_only_ansatz else lightcone_qubits))


def ansatz_scope_size(
    lightcone_qubits: Sequence[int],
    block_qubits: Sequence[int],
    *,
    block_only_ansatz: bool = False,
) -> int:
    """Return the number of qubits in the selected ansatz scope."""
    return len(ansatz_scope_qubits(lightcone_qubits, block_qubits, block_only_ansatz=block_only_ansatz))


def build_scoped_ansatz_circuit(
    theta: object,
    lightcone_qubits: Sequence[int],
    block_qubits: Sequence[int],
    *,
    block_only_ansatz: bool = False,
    builder: AnsatzBuilder = build_ansatz,
) -> cirq.Circuit:
    """Build an ansatz circuit on either the full lightcone or only the block."""
    qubit_labels = ansatz_scope_qubits(
        lightcone_qubits,
        block_qubits,
        block_only_ansatz=block_only_ansatz,
    )
    qubits = [cirq.LineQubit(q) for q in qubit_labels]
    return builder(theta, qubits=qubits, n_qubits=len(qubits))


def scoped_ansatz_unitary(
    theta: object,
    lightcone_qubits: Sequence[int],
    block_qubits: Sequence[int],
    *,
    block_only_ansatz: bool = False,
    builder: AnsatzBuilder = build_ansatz,
) -> NDArray[np.complex128]:
    """Return the ansatz unitary on its own selected scope."""
    qubit_labels = ansatz_scope_qubits(
        lightcone_qubits,
        block_qubits,
        block_only_ansatz=block_only_ansatz,
    )
    qubits = [cirq.LineQubit(q) for q in qubit_labels]
    circuit = build_scoped_ansatz_circuit(
        theta,
        lightcone_qubits,
        block_qubits,
        block_only_ansatz=block_only_ansatz,
        builder=builder,
    )
    return np.asarray(circuit.unitary(qubit_order=qubits), dtype=complex)


def scoped_ansatz_unitary_on_lightcone(
    theta: object,
    lightcone_qubits: Sequence[int],
    block_qubits: Sequence[int],
    *,
    block_only_ansatz: bool = False,
    builder: AnsatzBuilder = build_ansatz,
) -> NDArray[np.complex128]:
    """Return the trial unitary represented in lightcone Hilbert-space order."""
    trial = scoped_ansatz_unitary(
        theta,
        lightcone_qubits,
        block_qubits,
        block_only_ansatz=block_only_ansatz,
        builder=builder,
    )
    if not block_only_ansatz:
        return trial
    return embed_block_unitary_in_lightcone(trial, lightcone_qubits, block_qubits)


if __name__ == "__main__":
    from task2_code.experiment_config import DEFAULT_SEED as seed

    rng = np.random.default_rng(seed)
    theta_zero = np.zeros(theta_count())
    u_zero = ansatz_unitary(theta_zero)
    dim = 1 << N_QUBITS
    print("A.6.1  zero-parameter unitary check:")
    print(f"  shape = {u_zero.shape}")
    print(f"  ||U U^dag - I|| = {np.linalg.norm(u_zero @ u_zero.conj().T - np.eye(dim)):.2e}")

    theta_rand = random_theta(rng, low=0, high=2 * np.pi)
    u_rand = ansatz_unitary(theta_rand)
    err_u = np.linalg.norm(u_rand @ u_rand.conj().T - np.eye(dim))
    print(f"\nA.6.2  random unitary check:")
    print(f"  ||U U^dag - I|| = {err_u:.2e}")
    print(f"\nA.6.3  parameter count:")
    print(f"  theta_count() = {theta_count()}")

    print("\nA.6.4  circuit diagram:")
    circ = build_ansatz_4q(theta_rand)
    print(circ)
