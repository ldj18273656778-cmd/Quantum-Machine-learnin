"""Cirq-based superoperator and partial-trace comparison utilities.

This script answers two practical questions for Module B:

1. Cirq does provide public helpers for partial trace and channel
   superoperator conversion.
2. Cirq's channel superoperator uses row-stacking order, while
   ``task2_code.superoperator`` uses column-stacking order.  The wrapper below
   converts Cirq's result before comparing.

Run from the repository root:

    python task2_code/test_code/cirq_superoperator_compare.py
"""

import os
import sys
from pathlib import Path

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import numpy as np
import cirq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from task2_code.ansatz import build_ansatz_4q, ansatz_unitary, random_theta
from task2_code.superoperator import (
    partial_trace,
    partial_trace_superoperator,
    superoperator,
)


def _validate_square_matrix(matrix, name):
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix, got shape {matrix.shape}")
    return matrix


def _validate_dims(dims):
    dims = [int(d) for d in dims]
    if not dims:
        raise ValueError("dims must contain at least one subsystem dimension")
    if any(d <= 0 for d in dims):
        raise ValueError(f"all dims must be positive, got {dims}")
    return dims


def _row_to_column_permutation(dim):
    """Return P such that vec_row(A) = P @ vec_column(A)."""
    P = np.zeros((dim * dim, dim * dim), dtype=int)
    for row in range(dim):
        for col in range(dim):
            row_index = row * dim + col
            column_index = col * dim + row
            P[row_index, column_index] = 1
    return P


def _row_to_column_superoperator(S_row, dim):
    """Convert a row-stacked superoperator to column-stacked convention."""
    P = _row_to_column_permutation(dim)
    return P.T @ S_row @ P


def cirq_superoperator_from_unitary(unitary):
    """Return Cirq's unitary-channel superoperator in column-stacked order."""
    unitary = _validate_square_matrix(unitary, "U")
    S_row = cirq.kraus_to_superoperator([unitary])
    return _row_to_column_superoperator(S_row, unitary.shape[0])


def cirq_superoperator_from_circuit(circuit):
    """Return a Cirq circuit superoperator in column-stacked order."""
    U = cirq.unitary(circuit)
    S_row = cirq.operation_to_superoperator(circuit)
    return _row_to_column_superoperator(S_row, U.shape[0])


def cirq_partial_trace(rho_total, keep_indices, dims):
    """Partial trace using ``cirq.partial_trace`` with matrix I/O."""
    rho_total = _validate_square_matrix(rho_total, "rho_total")
    dims = _validate_dims(dims)
    total_dim = int(np.prod(dims))
    if rho_total.shape != (total_dim, total_dim):
        raise ValueError(
            f"rho_total shape must be ({total_dim}, {total_dim}) for dims={dims}, "
            f"got {rho_total.shape}"
        )

    keep = [int(i) for i in keep_indices]
    if not keep:
        return np.trace(rho_total)

    reduced_tensor = cirq.partial_trace(rho_total.reshape(dims + dims), keep)
    keep_dim = int(np.prod([dims[i] for i in keep]))
    return reduced_tensor.reshape((keep_dim, keep_dim))


def cirq_partial_trace_superoperator(S_total, keep_indices, dims, normalize=True):
    """Trace physical subsystems from a column-stacked superoperator via Cirq."""
    S_total = _validate_square_matrix(S_total, "S_total")
    dims = _validate_dims(dims)
    keep = [int(i) for i in keep_indices]

    hilbert_dim = int(np.prod(dims))
    liouville_dim = hilbert_dim * hilbert_dim
    if S_total.shape != (liouville_dim, liouville_dim):
        raise ValueError(
            f"S_total shape must be ({liouville_dim}, {liouville_dim}) "
            f"for dims={dims}, got {S_total.shape}"
        )

    n_subsystems = len(dims)
    liouville_dims = [d * d for d in dims]
    tensor = S_total.reshape(dims + dims + dims + dims)

    perm = []
    for q in range(n_subsystems):
        perm.extend([q, n_subsystems + q])
    for q in range(n_subsystems):
        perm.extend([2 * n_subsystems + q, 3 * n_subsystems + q])

    grouped = np.transpose(tensor, axes=perm).reshape(
        liouville_dims + liouville_dims
    )
    reduced_tensor = cirq.partial_trace(grouped, keep)

    if not keep:
        reduced = np.asarray(reduced_tensor)
    else:
        keep_dim = int(np.prod([liouville_dims[i] for i in keep]))
        reduced = reduced_tensor.reshape((keep_dim, keep_dim))

    if normalize:
        traced_dims = [liouville_dims[i] for i in range(n_subsystems) if i not in keep]
        if traced_dims:
            reduced = reduced / int(np.prod(traced_dims))

    return reduced


def _random_unitary(dim, rng):
    X = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    Q, R = np.linalg.qr(X)
    phases = np.diag(R) / np.abs(np.diag(R))
    return Q * phases.conj()


def _report(name, actual, expected, tol=1e-10):
    diff = np.linalg.norm(actual - expected)
    print(f"{name}: ||diff|| = {diff:.3e}")
    assert diff < tol, f"{name} failed with diff {diff}"


def run_comparisons():
    rng = np.random.default_rng(1234)

    rho = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = rho @ rho.conj().T
    rho = rho / np.trace(rho)
    _report(
        "partial_trace keep [0]",
        cirq_partial_trace(rho, [0], [2, 2]),
        partial_trace(rho, [0], [2, 2]),
    )
    _report(
        "partial_trace keep [1, 0]",
        cirq_partial_trace(rho, [1, 0], [2, 2]),
        partial_trace(rho, [1, 0], [2, 2]),
    )

    U = _random_unitary(4, rng)
    _report(
        "superoperator from unitary",
        cirq_superoperator_from_unitary(U),
        superoperator(U),
    )

    theta = random_theta(rng)
    circuit = build_ansatz_4q(theta)
    U_ansatz = ansatz_unitary(theta)
    _report(
        "superoperator from ansatz circuit",
        cirq_superoperator_from_circuit(circuit),
        superoperator(U_ansatz),
    )

    S = superoperator(U_ansatz)
    for keep in ([0], [2], [1, 0]):
        _report(
            f"partial_trace_superoperator keep {keep}",
            cirq_partial_trace_superoperator(S, keep, [2, 2, 2, 2]),
            partial_trace_superoperator(S, keep, [2, 2, 2, 2]),
        )

    print("Cirq comparison checks passed.")


if __name__ == "__main__":
    run_comparisons()
