"""Module C: light-cone extraction helpers for Task 2 Mode 1.

The dense extraction path implemented here is deliberately named as an
outside-|0> projection.  It is not a partial trace of a unitary and is not
guaranteed to be a true local unitary unless the selected cone is closed.
"""

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import TypeAlias

import cirq
import numpy as np
from numpy.typing import NDArray


ComplexArray: TypeAlias = NDArray[np.complex128]


@dataclass(frozen=True)
class LightConeDiagnostics:
    """Numerical diagnostics for a projected light-cone operator."""

    left_unitarity_error: float
    right_unitarity_error: float
    max_column_leakage: float


@dataclass(frozen=True)
class LightConeResult:
    """Projected target operator and qubit metadata for one light cone."""

    operator: ComplexArray
    lightcone_qubits: tuple[int, ...]
    block_positions: tuple[int, ...]
    outside_qubits: tuple[int, ...]
    semantics: str
    diagnostics: LightConeDiagnostics


def _as_unique_int_list(values: Sequence[int], name: str) -> list[int]:
    items = [int(v) for v in values]
    if len(items) != len(set(items)):
        raise ValueError(f"{name} must not contain duplicates, got {items}")
    return items


def _validate_qubits(qubits: Sequence[int], n_qubits: int, name: str) -> list[int]:
    items = _as_unique_int_list(qubits, name)
    bad = [q for q in items if q < 0 or q >= n_qubits]
    if bad:
        raise ValueError(f"{name} out of range for n_qubits={n_qubits}: {bad}")
    return items


def _validate_square_matrix(matrix: object, name: str) -> ComplexArray:
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix, got {matrix.shape}")
    return matrix


def _bits_from_int(index: int, width: int) -> list[int]:
    return [(index >> (width - 1 - pos)) & 1 for pos in range(width)]


def _int_from_bits(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _outside_assignment(
    outside_qubits: Sequence[int],
    outside_state: int | Sequence[int],
) -> dict[int, int]:
    if isinstance(outside_state, int):
        if outside_state < 0 or outside_state >= 2 ** len(outside_qubits):
            raise ValueError(
                "outside_state integer must fit the number of outside qubits"
            )
        bits = _bits_from_int(outside_state, len(outside_qubits))
    else:
        bits = [int(bit) for bit in outside_state]
        if len(bits) != len(outside_qubits):
            raise ValueError(
                "outside_state bit sequence must match the number of outside qubits"
            )
        if any(bit not in (0, 1) for bit in bits):
            raise ValueError("outside_state bits must be 0 or 1")
    return dict(zip(outside_qubits, bits))#把外部 qubit 的某个计算基状态配置转换成清晰的 qubit-to-bit 映射。


def _full_index(
    local_index: int,
    keep_qubits: Sequence[int],
    n_qubits: int,
    outside_values: Mapping[int, int],
) -> int:
    local_bits = _bits_from_int(local_index, len(keep_qubits))
    values = dict(outside_values)
    values.update(dict(zip(keep_qubits, local_bits)))
    return _int_from_bits([values[q] for q in range(n_qubits)])


def _line_qubit_index(qubit: cirq.Qid) -> int:
    if not isinstance(qubit, cirq.LineQubit):
        raise ValueError(f"expected cirq.LineQubit, got {qubit!r}")
    return int(qubit.x)


def lightcone_qubits_for_block(
    block_qubits: Sequence[int],
    n_qubits: int,
    radius: int = 1,
) -> list[int]:
    """Return sorted global qubits in a clipped 1D radius light cone.

    The current Task 2 construction uses `cirq.LineQubit` order and a 1D
    nearest-neighbor geometry.  This helper implements the plan's explicit
    radius policy; circuit-based cone discovery is provided separately for
    diagnostics.
    """
    n_qubits = int(n_qubits)
    radius = int(radius)
    if n_qubits <= 0:
        raise ValueError(f"n_qubits must be positive, got {n_qubits}")
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")

    block = _validate_qubits(block_qubits, n_qubits, "block_qubits")
    if not block:
        raise ValueError("block_qubits must contain at least one qubit")

    start = max(0, min(block) - radius)
    stop = min(n_qubits - 1, max(block) + radius)
    return list(range(start, stop + 1))


def backward_lightcone_from_circuit(
    circuit: cirq.Circuit,
    output_qubits: Sequence[int],
) -> list[int]:
    """Find a backward causal cone by walking Cirq operations in reverse."""
    affected = set(_as_unique_int_list(output_qubits, "output_qubits"))
    for op in reversed(list(circuit.all_operations())):
        op_qubits = {_line_qubit_index(q) for q in op.qubits}
        if op_qubits & affected:
            affected |= op_qubits
    return sorted(affected)


def backward_block_lightcone_from_circuit(
    circuit: cirq.Circuit,
    block_qubits: Sequence[int],
) -> list[int]:
    """Return the sorted backward light cone for a block of LineQubit labels."""
    return backward_lightcone_from_circuit(circuit, block_qubits)


def build_lightcone_target_unitary(
    circuit: cirq.Circuit,
    lightcone_qubits: Sequence[int],
) -> ComplexArray:
    """Build the target unitary restricted to a circuit backward light cone.

    Operations whose qubits are all contained in ``lightcone_qubits`` are kept
    and remapped from global ``LineQubit`` labels to local labels
    ``0..len(lightcone_qubits)-1``.  The returned matrix uses the sorted global
    light-cone order.
    """
    cone = sorted(_as_unique_int_list(lightcone_qubits, "lightcone_qubits"))
    if not cone:
        raise ValueError("lightcone_qubits must contain at least one qubit")

    local_qubits = cirq.LineQubit.range(len(cone))
    remap = {global_q: local_qubits[pos] for pos, global_q in enumerate(cone)}
    cone_set = set(remap)
    remapped_ops: list[cirq.Operation] = []
    for op in circuit.all_operations():
        op_qubits = {_line_qubit_index(q) for q in op.qubits}
        if not op_qubits <= cone_set:
            continue
        remapped_ops.append(op.with_qubits(*(remap[_line_qubit_index(q)] for q in op.qubits)))

    lightcone_circuit = cirq.Circuit(remapped_ops)
    return np.asarray(lightcone_circuit.unitary(qubit_order=list(local_qubits)), dtype=complex)


def projected_operator_on_lightcone(
    U_full: object,
    lightcone_qubits: Sequence[int],
    n_qubits: int,
    outside_state: int | Sequence[int] = 0,
    require_unitary: bool = False,
    atol: float = 1e-8,
) -> tuple[ComplexArray, LightConeDiagnostics]:
    """Project a full-system operator onto a light-cone subspace.

    Returns ``K_S = <outside_state| U_full |outside_state>`` in the basis whose
    axes follow sorted ``lightcone_qubits``.  This is a conditional projected
    operator, not a Hilbert-space partial trace.
    """
    n_qubits = int(n_qubits)
    if n_qubits <= 0:
        raise ValueError(f"n_qubits must be positive, got {n_qubits}")

    U_full = _validate_square_matrix(U_full, "U_full")
    full_dim = 1 << n_qubits
    if U_full.shape != (full_dim, full_dim):
        raise ValueError(
            f"U_full shape must be ({full_dim}, {full_dim}) for "
            + f"n_qubits={n_qubits}, got {U_full.shape}"
        )

    keep = sorted(_validate_qubits(lightcone_qubits, n_qubits, "lightcone_qubits"))
    outside = [q for q in range(n_qubits) if q not in keep]
    outside_values = _outside_assignment(outside, outside_state)

    keep_dim = 1 << len(keep)
    projected = np.empty((keep_dim, keep_dim), dtype=complex)
    for out_index in range(keep_dim):
        full_out = _full_index(out_index, keep, n_qubits, outside_values)
        for in_index in range(keep_dim):
            full_in = _full_index(in_index, keep, n_qubits, outside_values)
            projected[out_index, in_index] = U_full[full_out, full_in]

    diagnostics = projected_operator_diagnostics(projected)
    if require_unitary and (
        diagnostics.left_unitarity_error > atol
        or diagnostics.right_unitarity_error > atol
    ):
        raise ValueError(
            "projected light-cone operator is not unitary within tolerance; "
            + f"left_error={diagnostics.left_unitarity_error:.3e}, "
            + f"right_error={diagnostics.right_unitarity_error:.3e}"
        )

    return projected, diagnostics


def projected_operator_diagnostics(operator: object) -> LightConeDiagnostics:
    """Return unitarity and leakage diagnostics for a square operator."""
    operator = _validate_square_matrix(operator, "operator")
    dim = operator.shape[0]
    identity = np.eye(dim, dtype=complex)
    left_error = float(np.linalg.norm(operator.conj().T @ operator - identity))
    right_error = float(np.linalg.norm(operator @ operator.conj().T - identity))
    column_norms = np.asarray(np.sum(np.abs(operator) ** 2, axis=0), dtype=float)
    max_column_leakage = max(float(abs(1.0 - value)) for value in column_norms.tolist())
    return LightConeDiagnostics(left_error, right_error, max_column_leakage)


def extract_target_lightcone_operator(
    target: object,
    block_qubits: Sequence[int],
    n_qubits: int,
    radius: int = 1,
    lightcone_qubits: Sequence[int] | None = None,
    outside_state: int | Sequence[int] = 0,
    require_unitary: bool = False,
    max_n_qubits: int = 8,
    max_hilbert_dim: int = 256,
    atol: float = 1e-8,
) -> LightConeResult:
    """Return a projected target operator and metadata for Module D.

    ``target`` may be a dense matrix or a Cirq circuit.  Cirq circuits are
    converted with an explicit ``cirq.LineQubit.range(n_qubits)`` order.
    """
    n_qubits = int(n_qubits)
    max_n_qubits = int(max_n_qubits)
    max_hilbert_dim = int(max_hilbert_dim)
    hilbert_dim = 1 << n_qubits
    if n_qubits > max_n_qubits or hilbert_dim > max_hilbert_dim:
        raise ValueError(
            "dense light-cone projection is guarded for small systems; "
            + f"got n_qubits={n_qubits}, hilbert_dim={hilbert_dim}"
        )

    if lightcone_qubits is None:
        selected_lightcone = lightcone_qubits_for_block(block_qubits, n_qubits, radius)
    else:
        selected_lightcone = sorted(_validate_qubits(lightcone_qubits, n_qubits, "lightcone_qubits"))
    block = _validate_qubits(block_qubits, n_qubits, "block_qubits")
    missing = [q for q in block if q not in selected_lightcone]
    if missing:
        raise ValueError(f"block_qubits must be contained in lightcone_qubits; missing {missing}")
    block_positions = tuple(selected_lightcone.index(q) for q in block)
    outside_qubits = tuple(q for q in range(n_qubits) if q not in selected_lightcone)

    if isinstance(target, cirq.Circuit):
        qubit_order = [cirq.LineQubit(i) for i in range(n_qubits)]
        U_full = np.asarray(target.unitary(qubit_order=qubit_order), dtype=complex)
    else:
        U_full = target

    operator, diagnostics = projected_operator_on_lightcone(
        U_full,
        selected_lightcone,
        n_qubits,
        outside_state=outside_state,
        require_unitary=require_unitary,
        atol=atol,
    )
    return LightConeResult(
        operator=operator,
        lightcone_qubits=tuple(selected_lightcone),
        block_positions=block_positions,
        outside_qubits=outside_qubits,
        semantics="outside_zero_projection",
        diagnostics=diagnostics,
    )
