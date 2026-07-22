# task2_code/superoperator.py
# Module B: Superoperator and partial trace utilities — Cirq-native edition.
#
# Conventions:
#   - vec(A) uses column-stacking, matching NumPy order='F'.
#   - For rho' = U rho U^dag, vec(rho') = (U.conj() kron U) vec(rho).
#   - Cirq's superoperator/channel uses row-stacking internally; this module
#     converts to column-stacking before returning, preserving the repo API.
#
# Dependencies: numpy, cirq.

import numpy as np
import cirq


# ======================================================================
#  Validation helpers
# ======================================================================

def _validate_square_matrix(A, name):
    """Return A as an array after checking it is a square matrix."""
    a_arr = np.asarray(A)
    if a_arr.ndim != 2 or a_arr.shape[0] != a_arr.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix, got shape {a_arr.shape}")
    return a_arr


def _validate_dims(dims):
    dims = [int(d) for d in dims]
    if not dims:
        raise ValueError("dims must contain at least one subsystem dimension")
    if any(d <= 0 for d in dims):
        raise ValueError(f"all dims must be positive, got {dims}")
    return dims


def _validate_keep_indices(keep_indices, n_subsystems):
    keep = [int(i) for i in keep_indices]
    if len(set(keep)) != len(keep):
        raise ValueError(f"keep_indices must be unique, got {keep}")
    bad = [i for i in keep if i < 0 or i >= n_subsystems]
    if bad:
        raise ValueError(
            f"keep_indices out of range for {n_subsystems} subsystems: {bad}"
        )
    return keep


# ======================================================================
#  vectorization (no Cirq equivalent — keep NumPy)
# ======================================================================

def vec(A):
    """Column-stack a square matrix into a one-dimensional vector.

    Parameters
    ----------
    A : np.ndarray, shape (d, d)

    Returns
    -------
    np.ndarray, shape (d*d,)
    """
    a_mat = _validate_square_matrix(A, "A")
    return a_mat.reshape(-1, order="F")


def unvec(v, dim):
    """Inverse of vec for a square dim-by-dim matrix."""
    v = np.asarray(v)
    if v.ndim != 1:
        raise ValueError(f"v must be one-dimensional, got shape {v.shape}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if v.size != dim * dim:
        raise ValueError(f"v must have length {dim * dim}, got {v.size}")
    return v.reshape((dim, dim), order="F")


# ======================================================================
#  Row-to-column stacking conversion (Cirq uses row, we use column)
# ======================================================================

def _row_to_column_permutation(dim):
    """Return P such that vec_row(A) = P @ vec_column(A).

    Row-stacked vector has A[i,j] at position i*dim + j.
    Column-stacked vector has A[i,j] at position j*dim + i.
    """
    P = np.zeros((dim * dim, dim * dim), dtype=int)
    for r in range(dim):
        for c in range(dim):
            P[r * dim + c, c * dim + r] = 1
    return P


def _cirq_superoperator_to_column_stack(S_row, dim):
    """Convert Cirq's row-stacked superoperator to column-stacked convention.

    If S_row @ vec_row(rho) = vec_row(rho'), then
    S_col @ vec_column(rho) = vec_column(rho')  where
    S_col = P^T @ S_row @ P.
    """
    P = _row_to_column_permutation(dim)
    return P.T @ S_row @ P


# ======================================================================
#  superoperator — via cirq.kraus_to_superoperator
# ======================================================================

def superoperator(U):
    """Return the Liouville superoperator for rho -> U rho U^dag.

    Implemented via ``cirq.kraus_to_superoperator`` and converted to the
    repository's column-stacking convention.

    Parameters
    ----------
    U : np.ndarray, shape (d, d)

    Returns
    -------
    np.ndarray, shape (d*d, d*d)
    """
    u_mat = _validate_square_matrix(U, "U")
    S_row = cirq.kraus_to_superoperator([u_mat])
    return _cirq_superoperator_to_column_stack(S_row, u_mat.shape[0])


# ======================================================================
#  partial_trace — via cirq.partial_trace
# ======================================================================

def partial_trace(rho_total, keep_indices, dims):
    """Trace out all subsystems except keep_indices.

    Implemented via ``cirq.partial_trace``.

    Parameters
    ----------
    rho_total : np.ndarray, shape (D, D)
        Matrix on the full tensor-product Hilbert space, where D=prod(dims).
    keep_indices : iterable of int
        Subsystem indices to retain.  The output follows this order.
    dims : iterable of int
        Dimension of each subsystem, e.g. [2, 2, 2] for three qubits.

    Returns
    -------
    np.ndarray or scalar
        Reduced matrix with shape (D_keep, D_keep).  If keep_indices is empty,
        returns the scalar trace of rho_total.
    """
    rho_total = _validate_square_matrix(rho_total, "rho_total")
    dims = _validate_dims(dims)
    keep = _validate_keep_indices(keep_indices, len(dims))

    total_dim = int(np.prod(dims))
    if rho_total.shape != (total_dim, total_dim):
        raise ValueError(
            f"rho_total shape must be ({total_dim}, {total_dim}) for dims={dims}, "
            f"got {rho_total.shape}"
        )

    if not keep:
        return np.trace(rho_total)

    # cirq.partial_trace expects a tensor of shape (d0,d1,...,d_{n-1}, d0,d1,...,d_{n-1})
    tensor = rho_total.reshape(dims + dims)
    reduced_tensor = cirq.partial_trace(tensor, keep)

    keep_dim = int(np.prod([dims[i] for i in keep]))
    reduced = reduced_tensor.reshape((keep_dim, keep_dim))

    # cirq.partial_trace returns subsystems in the order specified by
    # keep_indices, so the output already follows the caller's ordering.
    return reduced


# ======================================================================
#  partial_trace_superoperator — via cirq.partial_trace
# ======================================================================

def partial_trace_superoperator(S_total, keep_indices, dims, normalize=True):
    """Trace out physical subsystems from a Liouville superoperator.

    Implemented via ``cirq.partial_trace``.

    Parameters
    ----------
    S_total : np.ndarray, shape (D*D, D*D)
        Superoperator on the full physical Hilbert space, where D=prod(dims).
    keep_indices : iterable of int
        Physical subsystem indices to retain.
    dims : iterable of int
        Physical Hilbert-space dimensions, e.g. [2, 2, 2] for three qubits.
    normalize : bool, default True
        If True, divide by the Liouville dimension of traced-out subsystems.
        This makes partial_trace_superoperator(I, [q], [2]*n) return I_4.

    Returns
    -------
    np.ndarray or scalar
        Reduced superoperator.  For qubits, retaining one physical qubit returns
        a (4, 4) matrix in column-stacked Liouville order.
    """
    S_total = _validate_square_matrix(S_total, "S_total")
    dims = _validate_dims(dims)
    keep = _validate_keep_indices(keep_indices, len(dims))

    hilbert_dim = int(np.prod(dims))
    liouville_dim = hilbert_dim * hilbert_dim
    if S_total.shape != (liouville_dim, liouville_dim):
        raise ValueError(
            f"S_total shape must be ({liouville_dim}, {liouville_dim}) "
            f"for dims={dims}, got {S_total.shape}"
        )

    n_subsystems = len(dims)
    liouville_dims = [d * d for d in dims]

    # For a column-stacked Liouville operator, each physical qubit contributes
    # two Hilbert-space axes (ket row, ket col) and two mirrored Liouville axes
    # (bra row, bra col).  Group (ket_col, bra_col) → (ket_row, bra_row) into
    # one Liouville subsystem per physical qubit.
    # Start from (ket_col0,..., ket_col_{n-1}, ket_row0,...) shape:
    tensor = S_total.reshape(dims + dims + dims + dims)

    # Permute so each physical qubit's Liouville pair sits together:
    # (ket_col_q, ket_row_q, bra_col_q, bra_row_q) for each q.
    perm = []
    for q in range(n_subsystems):
        perm.extend([q, n_subsystems + q])          # ket col & ket row
    for q in range(n_subsystems):
        perm.extend([2 * n_subsystems + q, 3 * n_subsystems + q])  # bra col & bra row

    grouped = np.transpose(tensor, axes=perm).reshape(
        tuple(liouville_dims) + tuple(liouville_dims)
    )

    # cirq.partial_trace on the Liouville tensor
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


# ======================================================================
#  per-bit reduced-channel loss — scalable one-qubit reductions
# ======================================================================

def _embed_bits(env_val, q_val, lightcone_pos, num_lightcone):
    """Build a light-cone basis index from environment bits and one qubit bit."""
    bits = [0] * num_lightcone
    bits[lightcone_pos] = int(q_val)
    env_idx = 0
    for pos in range(num_lightcone):
        if pos != lightcone_pos:
            bits[pos] = (int(env_val) >> (num_lightcone - 2 - env_idx)) & 1
            env_idx += 1
    return sum(bit << (num_lightcone - 1 - i) for i, bit in enumerate(bits))


def _validate_qubit_labels(values, name):
    labels = [int(v) for v in values]
    if len(labels) != len(set(labels)):
        raise ValueError(f"{name} must not contain duplicates, got {labels}")
    return labels


def _selected_target_bits(block_qubits, lightcone_qubits, target_bits):
    block = _validate_qubit_labels(block_qubits, "block_qubits")
    lightcone = _validate_qubit_labels(lightcone_qubits, "lightcone_qubits")
    missing_block = [q for q in block if q not in lightcone]
    if missing_block:
        raise ValueError(
            f"block_qubits must be contained in lightcone_qubits; missing {missing_block}"
        )

    if target_bits is None:
        selected = list(block)
    else:
        selected = _validate_qubit_labels(target_bits, "target_bits")
        missing_selected = [q for q in selected if q not in block]
        if missing_selected:
            raise ValueError(
                f"target_bits must be contained in block_qubits; missing {missing_selected}"
            )
    return block, lightcone, selected


def reduced_bit_superoperator_from_V(V, target_bit, block_qubits, lightcone_qubits):
    """Return the one-qubit reduced superoperator for one global target bit.

    The returned 4x4 matrix uses this module's column-stacking convention:
    ``vec(|a><b|)`` sits at index ``a + 2*b``.
    """
    v_mat = _validate_square_matrix(V, "V")
    block, lightcone, selected = _selected_target_bits(
        block_qubits,
        lightcone_qubits,
        [target_bit],
    )
    target = selected[0]
    s = len(lightcone)
    dim = 1 << s
    if v_mat.shape != (dim, dim):
        raise ValueError(f"V shape must be ({dim}, {dim}) for lightcone_qubits={lightcone}")

    local_pos = lightcone.index(target)
    env_dim = 1 << (s - 1)
    dims = [2] * s
    reduced_channel = np.zeros((4, 4), dtype=complex)

    for c in range(2):
        for d in range(2):
            col_idx = c + 2 * d
            rho_in = np.zeros((dim, dim), dtype=complex)
            for env in range(env_dim):
                row_idx = _embed_bits(env, c, local_pos, s)
                col_idx_full = _embed_bits(env, d, local_pos, s)
                rho_in[row_idx, col_idx_full] = 1.0 / env_dim

            rho_out = v_mat @ rho_in @ v_mat.conj().T
            reduced = partial_trace(rho_out, [local_pos], dims)
            for a in range(2):
                for b in range(2):
                    row_idx = a + 2 * b
                    reduced_channel[row_idx, col_idx] = reduced[a, b]

    return reduced_channel


def reduced_bit_superoperator_from_V_zero_env(V, target_bit, block_qubits, lightcone_qubits):
    """Return one-qubit reduced superoperator with other cone bits fixed to |0...0>.

    This is the pure-|0> environment variant of
    ``reduced_bit_superoperator_from_V``.  The original function averages over
    all non-target light-cone bit strings, i.e. treats them as maximally mixed.
    Here the non-target light-cone bits are fixed to the all-zero basis state.
    """
    v_mat = _validate_square_matrix(V, "V")
    block, lightcone, selected = _selected_target_bits(
        block_qubits,
        lightcone_qubits,
        [target_bit],
    )
    target = selected[0]
    s = len(lightcone)
    dim = 1 << s
    if v_mat.shape != (dim, dim):
        raise ValueError(f"V shape must be ({dim}, {dim}) for lightcone_qubits={lightcone}")

    local_pos = lightcone.index(target)
    dims = [2] * s
    reduced_channel = np.zeros((4, 4), dtype=complex)

    zero_env = 0
    for c in range(2):
        for d in range(2):
            col_idx = c + 2 * d
            rho_in = np.zeros((dim, dim), dtype=complex)
            row_idx = _embed_bits(zero_env, c, local_pos, s)
            col_idx_full = _embed_bits(zero_env, d, local_pos, s)
            rho_in[row_idx, col_idx_full] = 1.0

            rho_out = v_mat @ rho_in @ v_mat.conj().T
            reduced = partial_trace(rho_out, [local_pos], dims)
            for a in range(2):
                for b in range(2):
                    row_idx = a + 2 * b
                    reduced_channel[row_idx, col_idx] = reduced[a, b]

    return reduced_channel


def reduced_bit_superoperator_from_V_one_env(V, target_bit, block_qubits, lightcone_qubits):
    """Return one-qubit reduced superoperator with other cone bits fixed to |1...1>."""
    v_mat = _validate_square_matrix(V, "V")
    block, lightcone, selected = _selected_target_bits(
        block_qubits,
        lightcone_qubits,
        [target_bit],
    )
    target = selected[0]
    s = len(lightcone)
    dim = 1 << s
    if v_mat.shape != (dim, dim):
        raise ValueError(f"V shape must be ({dim}, {dim}) for lightcone_qubits={lightcone}")

    local_pos = lightcone.index(target)
    dims = [2] * s
    reduced_channel = np.zeros((4, 4), dtype=complex)

    one_env = (1 << (s - 1)) - 1
    for c in range(2):
        for d in range(2):
            col_idx = c + 2 * d
            rho_in = np.zeros((dim, dim), dtype=complex)
            row_idx = _embed_bits(one_env, c, local_pos, s)
            col_idx_full = _embed_bits(one_env, d, local_pos, s)
            rho_in[row_idx, col_idx_full] = 1.0

            rho_out = v_mat @ rho_in @ v_mat.conj().T
            reduced = partial_trace(rho_out, [local_pos], dims)
            for a in range(2):
                for b in range(2):
                    row_idx = a + 2 * b
                    reduced_channel[row_idx, col_idx] = reduced[a, b]

    return reduced_channel


def per_bit_losses_from_V(V, block_qubits, lightcone_qubits, target_bits=None):
    """Compute deterministic one-qubit local losses from a residual operator.

    This explicit reduced-channel path avoids building the full
    ``4**s x 4**s`` Liouville superoperator for a light cone of size ``s``.
    Qubit labels are global labels; returned keys are global labels as well.

    Parameters
    ----------
    V : np.ndarray, shape (2**s, 2**s)
        Residual operator on ``lightcone_qubits``.
    block_qubits : iterable of int
        Global labels of the optimized block.
    lightcone_qubits : iterable of int
        Global labels defining the Hilbert-space order of ``V``.
    target_bits : iterable of int, optional
        Optional subset of global block labels to evaluate.

    Returns
    -------
    dict[int, float]
        Frobenius loss for each selected global target bit.
    """
    _, lightcone, selected = _selected_target_bits(
        block_qubits,
        lightcone_qubits,
        target_bits,
    )
    identity_super = np.eye(4, dtype=complex)
    losses = {}
    for q in selected:
        reduced = reduced_bit_superoperator_from_V(V, q, block_qubits, lightcone)
        losses[q] = float(np.linalg.norm(reduced - identity_super, ord="fro") ** 2)
    return losses


def per_bit_losses_from_V_zero_env(V, block_qubits, lightcone_qubits, target_bits=None):
    """Compute per-bit losses with non-target light-cone bits fixed to |0...0>."""
    _, lightcone, selected = _selected_target_bits(
        block_qubits,
        lightcone_qubits,
        target_bits,
    )
    identity_super = np.eye(4, dtype=complex)
    losses = {}
    for q in selected:
        reduced = reduced_bit_superoperator_from_V_zero_env(V, q, block_qubits, lightcone)
        losses[q] = float(np.linalg.norm(reduced - identity_super, ord="fro") ** 2)
    return losses


def per_bit_losses_from_V_one_env(V, block_qubits, lightcone_qubits, target_bits=None):
    """Compute per-bit losses with non-target light-cone bits fixed to |1...1>."""
    _, lightcone, selected = _selected_target_bits(
        block_qubits,
        lightcone_qubits,
        target_bits,
    )
    identity_super = np.eye(4, dtype=complex)
    losses = {}
    for q in selected:
        reduced = reduced_bit_superoperator_from_V_one_env(V, q, block_qubits, lightcone)
        losses[q] = float(np.linalg.norm(reduced - identity_super, ord="fro") ** 2)
    return losses


# ======================================================================
#  Sanity checks (run when module is executed directly)
# ======================================================================

def _run_sanity_checks():
    A = np.array([[1, 2], [3, 4]], dtype=complex)
    assert np.allclose(vec(A), np.array([1, 3, 2, 4], dtype=complex))
    assert np.allclose(unvec(vec(A), 2), A)

    I2 = np.eye(2, dtype=complex)
    assert np.allclose(superoperator(I2), np.eye(4, dtype=complex))

    # Global phase invariance
    phi = 0.77
    U = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S1 = superoperator(U)
    S2 = superoperator(np.exp(1j * phi) * U)
    assert np.allclose(S1, S2), "global phase invariance broken"

    # Single-qubit partial trace of Bell state should give I/2 on kept qubit
    bell = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]], dtype=complex) / 2
    rho_0 = partial_trace(bell, [0], [2, 2])
    assert np.allclose(rho_0, np.eye(2) / 2), f"bell partial trace [0] wrong: {rho_0}"
    rho_1 = partial_trace(bell, [1], [2, 2])
    assert np.allclose(rho_1, np.eye(2) / 2), f"bell partial trace [1] wrong: {rho_1}"

    # partial_trace_superoperator of identity should reduce to I_4
    nq = 3
    S_full = superoperator(np.eye(2 ** nq, dtype=complex))
    reduced = partial_trace_superoperator(S_full, [1], [2] * nq)
    assert np.allclose(reduced, np.eye(4, dtype=complex)), (
        f"partial_trace_superoperator(I, [1], [2]*3) != I_4"
    )

    # CZ on |++> creates an entangled state; each single-qubit reduced
    # state should be maximally mixed.
    U_cz = np.diag([1, 1, 1, -1]).astype(complex)
    plus_plus = np.ones(4, dtype=complex) / 2  # (|00>+|01>+|10>+|11>)/2
    rho_cz = U_cz @ np.outer(plus_plus, plus_plus.conj()) @ U_cz.conj().T
    rho_reduced = partial_trace(rho_cz, [0], [2, 2])
    assert np.allclose(rho_reduced, np.eye(2) / 2), (
        f"CZ reduced state should be I/2, got {rho_reduced}"
    )

    # Asymmetric state: |01> (q0=0,q1=1).  In [q0,q1] order the non-zero
    # element is at (1,1); in [q1,q0] order it is at (2,2).
    state_01 = np.zeros(4, dtype=complex); state_01[1] = 1   # q0=0,q1=1
    rho_01 = np.outer(state_01, state_01.conj())
    rho_natural = partial_trace(rho_01, [0, 1], [2, 2])
    rho_swapped = partial_trace(rho_01, [1, 0], [2, 2])
    assert np.allclose(rho_natural, rho_01), "keep natural order should preserve input"
    assert rho_swapped[2, 2] > 0.99, f"keep=[1,0] should put element at (2,2)"

    print("superoperator sanity checks PASSED (Cirq-native edition)")


if __name__ == "__main__":
    _run_sanity_checks()
