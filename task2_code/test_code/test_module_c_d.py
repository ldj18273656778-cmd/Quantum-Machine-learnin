"""Plain-Python tests for Module C and Module D.

Run from the repository root:

    python task2_code/test_code/test_module_c_d.py
"""

import os
import sys
from pathlib import Path

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import cirq
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from task2_code.lightcone import (  # noqa: E402
    backward_block_lightcone_from_circuit,
    build_lightcone_target_unitary,
    extract_target_lightcone_operator,
    lightcone_qubits_for_block,
    projected_operator_on_lightcone,
)
from task2_code.local_loss import (  # noqa: E402
    compute_lightcone_loss,
    compute_loss,
    embed_block_unitary_in_lightcone,
)


def _assert_allclose(actual: object, expected: object, atol: float = 1e-9) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if not np.allclose(actual_array, expected_array, atol=atol):
        raise AssertionError(f"arrays differ\nactual={actual}\nexpected={expected}")


def _random_unitary(dim: int, seed: int = 1234) -> NDArray[np.complex128]:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(matrix)
    phases = np.diag(r) / np.abs(np.diag(r))
    return np.asarray(q * phases.conj(), dtype=complex)


def test_lightcone_radius_zero_and_boundary() -> None:
    assert lightcone_qubits_for_block([2, 3, 4, 5], 8, 0) == [2, 3, 4, 5]
    assert lightcone_qubits_for_block([0, 1, 2, 3], 6, 3) == [0, 1, 2, 3, 4, 5]
    assert lightcone_qubits_for_block([2, 3], 5, 2) == [0, 1, 2, 3, 4]


def test_project_identity_on_lightcone() -> None:
    result = extract_target_lightcone_operator(
        np.eye(64, dtype=complex),
        block_qubits=[1, 2, 3, 4],
        n_qubits=6,
        radius=1,
        require_unitary=True,
    )
    _assert_allclose(result.operator, np.eye(64, dtype=complex))
    assert result.lightcone_qubits == (0, 1, 2, 3, 4, 5)
    assert result.block_positions == (1, 2, 3, 4)
    assert result.semantics == "outside_zero_projection"
    assert result.diagnostics.left_unitarity_error < 1e-12


def test_projection_detects_cross_boundary_nonunitarity() -> None:
    qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
    circuit = cirq.Circuit(cirq.CNOT(qubits[0], qubits[1]))
    unitary = np.asarray(circuit.unitary(qubit_order=qubits), dtype=complex)
    projected, diagnostics = projected_operator_on_lightcone(
        unitary,
        lightcone_qubits=[0],
        n_qubits=2,
        require_unitary=False,
    )
    _assert_allclose(projected, np.array([[1, 0], [0, 0]], dtype=complex))
    assert diagnostics.left_unitarity_error > 0.5
    try:
        _ = projected_operator_on_lightcone(
            unitary,
            lightcone_qubits=[0],
            n_qubits=2,
            require_unitary=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("nonunitary projected operator should raise")


def test_embed_nonleading_block_position() -> None:
    x_gate = np.array([[0, 1], [1, 0]], dtype=complex)
    embedded = embed_block_unitary_in_lightcone(
        x_gate,
        lightcone_qubits=[0, 1, 2],
        block_qubits=[1],
    )
    basis_000 = np.zeros(8); basis_000[0] = 1
    basis_101 = np.zeros(8); basis_101[5] = 1
    out_010 = np.zeros(8); out_010[2] = 1
    out_111 = np.zeros(8); out_111[7] = 1
    _assert_allclose(embedded @ basis_000, out_010)
    _assert_allclose(embedded @ basis_101, out_111)


def test_identity_and_global_phase_loss_zero() -> None:
    identity = np.eye(16, dtype=complex)
    block = [0, 1, 2, 3]
    phase_identity = np.asarray(np.exp(0.37j) * identity, dtype=complex)
    loss_identity = compute_loss(identity, identity, block, block)
    loss_phase = compute_loss(phase_identity, identity, block, block)
    assert loss_identity < 1e-9
    assert loss_phase < 1e-9


def test_exact_closed_support_loss_zero() -> None:
    target = _random_unitary(16, seed=5)
    block = [0, 1, 2, 3]
    loss = compute_loss(target, target, block, block)
    assert loss < 1e-8


def test_nonleading_exact_embedded_loss_zero() -> None:
    trial = _random_unitary(16, seed=6)
    lightcone = [0, 1, 2, 3, 4, 5]
    block = [1, 2, 3, 4]
    target = embed_block_unitary_in_lightcone(trial, lightcone, block)
    loss = compute_loss(target, trial, block, lightcone)
    assert loss < 1e-8


def test_full_lightcone_loss_zero_and_shape_checks() -> None:
    target = _random_unitary(32, seed=7)
    block = [1, 2, 3, 4]
    lightcone = [0, 1, 2, 3, 4]
    loss = compute_lightcone_loss(target, target, block, lightcone)
    assert loss < 1e-8

    try:
        _ = compute_lightcone_loss(target, np.eye(16, dtype=complex), block, lightcone)
    except ValueError:
        pass
    else:
        raise AssertionError("full-cone trial shape mismatch should raise")


def test_backward_block_lightcone_from_circuit() -> None:
    qubits = cirq.LineQubit.range(5)
    circuit = cirq.Circuit(
        cirq.CZ(qubits[0], qubits[1]),
        cirq.CZ(qubits[2], qubits[3]),
        cirq.CZ(qubits[1], qubits[2]),
        cirq.Z(qubits[4]),
    )
    assert backward_block_lightcone_from_circuit(circuit, [2]) == [0, 1, 2, 3]

    named = cirq.NamedQubit("q")
    try:
        _ = backward_block_lightcone_from_circuit(cirq.Circuit(cirq.X(named)), [0])
    except ValueError:
        pass
    else:
        raise AssertionError("non-LineQubit circuit should raise")


def test_build_lightcone_target_unitary_remaps_qubits() -> None:
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        cirq.X(qubits[1]),
        cirq.CZ(qubits[1], qubits[3]),
        cirq.Z(qubits[0]),
    )
    actual = build_lightcone_target_unitary(circuit, [1, 3])
    expected = cirq.Circuit(
        cirq.X(cirq.LineQubit(0)),
        cirq.CZ(cirq.LineQubit(0), cirq.LineQubit(1)),
    ).unitary(qubit_order=cirq.LineQubit.range(2))
    _assert_allclose(actual, expected)


def test_invalid_inputs_raise() -> None:
    try:
        _ = lightcone_qubits_for_block([1, 1], 4, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate block qubits should raise")

    try:
        _ = extract_target_lightcone_operator(
            np.eye(2**9, dtype=complex),
            block_qubits=[0, 1, 2, 3],
            n_qubits=9,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("dense guard should reject n_qubits=9 by default")


def run_all_tests() -> None:
    tests = [
        test_lightcone_radius_zero_and_boundary,
        test_project_identity_on_lightcone,
        test_projection_detects_cross_boundary_nonunitarity,
        test_embed_nonleading_block_position,
        test_identity_and_global_phase_loss_zero,
        test_exact_closed_support_loss_zero,
        test_nonleading_exact_embedded_loss_zero,
        test_full_lightcone_loss_zero_and_shape_checks,
        test_backward_block_lightcone_from_circuit,
        test_build_lightcone_target_unitary_remaps_qubits,
        test_invalid_inputs_raise,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    run_all_tests()
