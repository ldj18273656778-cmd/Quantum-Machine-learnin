"""Module D: Mode 1 deterministic local-inversion loss."""

import numpy as np
from numpy.typing import NDArray
from collections.abc import Sequence
from typing import TypeAlias

from task2_code_auto import superoperator as superoperator_module
from task2_code_auto.superoperator_registry import get_active_superop


ComplexArray: TypeAlias = NDArray[np.complex128]


def _superoperator(matrix: ComplexArray) -> ComplexArray:
    return np.asarray(superoperator_module.superoperator(matrix), dtype=complex)


def _partial_trace_superoperator(
    matrix: ComplexArray,
    keep_indices: Sequence[int],
    dims: Sequence[int],
) -> ComplexArray:
    return np.asarray(
        superoperator_module.partial_trace_superoperator(matrix, keep_indices, dims),
        dtype=complex,
    )


def _validate_square_matrix(matrix: object, name: str) -> ComplexArray:
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix, got {matrix.shape}")
    return matrix


def _validate_qubit_list(values: Sequence[int], name: str) -> list[int]:
    items = [int(v) for v in values]
    if len(items) != len(set(items)):
        raise ValueError(f"{name} must not contain duplicates, got {items}")
    return items


def _bits_from_int(index: int, width: int) -> list[int]:
    return [(index >> (width - 1 - pos)) & 1 for pos in range(width)]


def _int_from_bits(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _axis_positions(
    lightcone_qubits: Sequence[int],
    block_qubits: Sequence[int],
) -> list[int]:
    lightcone = _validate_qubit_list(lightcone_qubits, "lightcone_qubits")
    block = _validate_qubit_list(block_qubits, "block_qubits")
    missing = [q for q in block if q not in lightcone]
    if missing:
        raise ValueError(
            f"block_qubits must be contained in lightcone_qubits; missing {missing}"
        )
    return [lightcone.index(q) for q in block]


def _project_bits(full_bits: Sequence[int], positions: Sequence[int]) -> list[int]:
    return [full_bits[pos] for pos in positions]


def _replace_bits(
    full_bits: Sequence[int],
    positions: Sequence[int],
    block_bits: Sequence[int],
) -> list[int]:
    out = list(full_bits)
    for pos, bit in zip(positions, block_bits):
        out[pos] = bit
    return out


def embed_block_unitary_in_lightcone(
    U_block: object,
    lightcone_qubits: Sequence[int],
    block_qubits: Sequence[int],
) -> ComplexArray:
    """Embed a block unitary into the Hilbert space ordered by lightcone_qubits.

    Matrix axes follow ascending/global ``lightcone_qubits`` order.  The block
    matrix axes follow the caller-provided ``block_qubits`` order, usually the
    same order used to build the 4-qubit ansatz unitary.
    """
    U_block = _validate_square_matrix(U_block, "U_block")
    lightcone = _validate_qubit_list(lightcone_qubits, "lightcone_qubits")
    block = _validate_qubit_list(block_qubits, "block_qubits")
    if not block:
        raise ValueError("block_qubits must contain at least one qubit")
    if not lightcone:
        raise ValueError("lightcone_qubits must contain at least one qubit")

    block_dim = 1 << len(block)
    if U_block.shape != (block_dim, block_dim):
        raise ValueError(
            f"U_block shape must be ({block_dim}, {block_dim}) for "
            + f"{len(block)} block qubits, got {U_block.shape}"
        )

    positions = _axis_positions(lightcone, block)
    cone_dim = 1 << len(lightcone)
    embedded = np.zeros((cone_dim, cone_dim), dtype=complex)
    for in_index in range(cone_dim):
        in_bits = _bits_from_int(in_index, len(lightcone))
        block_in = _int_from_bits(_project_bits(in_bits, positions))
        for block_out in range(block_dim):
            amplitude: complex = U_block[block_out, block_in].item()
            if amplitude == 0:
                continue
            out_bits = _replace_bits(
                in_bits,
                positions,
                _bits_from_int(block_out, len(block)),
            )
            out_index = _int_from_bits(out_bits)
            embedded[out_index, in_index] += amplitude
    return embedded


def compute_loss(
    U_target_Sj: object,
    U_trial_Bj: object,
    block_qubits: Sequence[int],
    lightcone_qubits: Sequence[int] | None = None,
    require_unitary: bool = True,
    atol: float = 1e-8,
) -> float:
    """Compute the Mode 1 deterministic local-inversion loss.

    The loss is the sum over block qubits of Frobenius distances between each
    reduced one-qubit error superoperator and ``I_4``.
    """
    U_target_Sj = _validate_square_matrix(U_target_Sj, "U_target_Sj")
    U_trial_Bj = _validate_square_matrix(U_trial_Bj, "U_trial_Bj")
    block = _validate_qubit_list(block_qubits, "block_qubits")
    if not block:
        raise ValueError("block_qubits must contain at least one qubit")

    if lightcone_qubits is None:
        lightcone = list(block)
    else:
        lightcone = _validate_qubit_list(lightcone_qubits, "lightcone_qubits")

    cone_dim = 1 << len(lightcone)
    block_dim = 1 << len(block)
    if U_target_Sj.shape != (cone_dim, cone_dim):
        raise ValueError(
            f"U_target_Sj shape must be ({cone_dim}, {cone_dim}) for "
            + f"{len(lightcone)} light-cone qubits, got {U_target_Sj.shape}"
        )
    if U_trial_Bj.shape != (block_dim, block_dim):
        raise ValueError(
            f"U_trial_Bj shape must be ({block_dim}, {block_dim}) for "
            + f"{len(block)} block qubits, got {U_trial_Bj.shape}"
        )

    if require_unitary:
        identity_cone = np.eye(cone_dim, dtype=complex)
        target_error = np.linalg.norm(U_target_Sj.conj().T @ U_target_Sj - identity_cone)
        if target_error > atol:
            raise ValueError(
                "U_target_Sj is not unitary within tolerance; "
                + f"error={target_error:.3e}"
            )
        identity_block = np.eye(block_dim, dtype=complex)
        trial_error = np.linalg.norm(U_trial_Bj.conj().T @ U_trial_Bj - identity_block)
        if trial_error > atol:
            raise ValueError(
                "U_trial_Bj is not unitary within tolerance; "
                + f"error={trial_error:.3e}"
            )

    trial_tilde = embed_block_unitary_in_lightcone(
        U_trial_Bj,
        lightcone,
        block,
    )
    residual = U_target_Sj @ trial_tilde.conj().T
    channel = _superoperator(residual)
    dims = [2] * len(lightcone)
    identity_super = np.eye(4, dtype=complex)

    total = 0.0
    for local_pos in _axis_positions(lightcone, block):
        reduced = _partial_trace_superoperator(channel, [local_pos], dims)
        total += float(np.linalg.norm(reduced - identity_super, ord="fro"))
    return total


def compute_lightcone_loss(
    U_target_C: object,
    U_trial_C: object,
    block_qubits: Sequence[int],
    lightcone_qubits: Sequence[int],
    target_bits: Sequence[int] | None = None,
    require_unitary: bool = True,
    atol: float = 1e-8,
) -> float:
    """Compute the squared Frobenius Mode 1 loss for a full-cone trial.

    Both operators act on the full light cone ordered by ``lightcone_qubits``.
    This is the dynamic ``15*n_C`` ansatz path; ``compute_loss`` keeps the
    legacy block-unitary embedding API.
    """
    target = _validate_square_matrix(U_target_C, "U_target_C")
    trial = _validate_square_matrix(U_trial_C, "U_trial_C")
    lightcone = _validate_qubit_list(lightcone_qubits, "lightcone_qubits")
    block = _validate_qubit_list(block_qubits, "block_qubits")
    missing = [q for q in block if q not in lightcone]
    if missing:
        raise ValueError(f"block_qubits must be contained in lightcone_qubits; missing {missing}")

    cone_dim = 1 << len(lightcone)
    expected = (cone_dim, cone_dim)
    if target.shape != expected:
        raise ValueError(f"U_target_C shape must be {expected} for lightcone_qubits={lightcone}")
    if trial.shape != expected:
        raise ValueError(f"U_trial_C shape must be {expected} for lightcone_qubits={lightcone}")

    if require_unitary:
        identity = np.eye(cone_dim, dtype=complex)
        target_error = np.linalg.norm(target.conj().T @ target - identity)
        trial_error = np.linalg.norm(trial.conj().T @ trial - identity)
        if target_error > atol:
            raise ValueError(f"U_target_C is not unitary within tolerance; error={target_error:.3e}")
        if trial_error > atol:
            raise ValueError(f"U_trial_C is not unitary within tolerance; error={trial_error:.3e}")

    residual = target @ trial.conj().T
    losses = get_active_superop()(
        residual,
        block,
        lightcone,
        target_bits=target_bits,
    )
    total = float(sum(losses.values()))
    if not np.isfinite(total):
        raise FloatingPointError(f"non-finite light-cone loss: {total}")
    return total
