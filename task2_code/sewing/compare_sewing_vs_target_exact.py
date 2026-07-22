"""Compare U_sewing and U_target via exact <Z_q> expectation from state vectors.

Unlike the statistical sampling script, this computes the **theoretical**
expectation value of Pauli-Z on a specified qubit directly from the final
state vector after unitary evolution.  No sampling noise.

Usage:
    python task2_code/sewing/compare_sewing_vs_target_exact.py
    python task2_code/sewing/compare_sewing_vs_target_exact.py --measure-qubit 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cirq
import numpy as np
from numpy.typing import NDArray

from task2_code.U_target import build_u_target_circuit
from task2_code.sewing.block_sewing import (
    DEFAULT_PARAMS,
    build_block_sew_circuit,
    load_block_specs,
    resolve_order,
)


MAX_EXACT_SYSTEM_QUBITS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare U_sewing and U_target via exact <Z_q> from state vectors.",
    )
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--order", default="odd-even")
    parser.add_argument("--dagger-convention", default="inverse")
    parser.add_argument("--measure-qubit", type=int, default=2)
    parser.add_argument("--system-basis", type=int, default=0)
    parser.add_argument("--ancilla-basis", type=int, default=0)
    parser.add_argument("--max-system-qubits", type=int, default=MAX_EXACT_SYSTEM_QUBITS,
                        help="Refuse legacy exact full-state simulation above this system size; use run_sew_and_compare.py --backend causal-cone for larger systems.")
    return parser.parse_args()


def _metadata_path(params_path: Path, metadata_path: Path | None) -> Path:
    return metadata_path if metadata_path is not None else params_path.with_suffix(".json")


def _z_expectation_from_state(state: NDArray[np.complex128], qubit_index: int, total_qubits: int) -> float:
    """Compute <Z_q> = P(0) - P(1) from a state vector."""
    probs = np.abs(state.reshape((2,) * total_qubits)) ** 2
    p0 = float(np.take(probs, 0, axis=qubit_index).sum())
    p1 = float(np.take(probs, 1, axis=qubit_index).sum())
    return p0 - p1


def _basis_state(basis_int: int, n_qubits: int) -> NDArray[np.complex128]:
    """Return the |basis_int⟩ state vector for n_qubits."""
    state = np.zeros(1 << n_qubits, dtype=np.complex128)
    state[basis_int] = 1.0
    return state


def _validate_inputs(args: argparse.Namespace, n_qubits: int) -> None:
    if n_qubits > args.max_system_qubits:
        raise ValueError(
            "legacy exact sewing comparison uses full-state simulation and is "
            f"guarded to n_qubits <= {args.max_system_qubits}; use "
            "task2_code/run_sew_and_compare.py --backend causal-cone for larger systems"
        )
    if args.measure_qubit < 0 or args.measure_qubit >= n_qubits:
        raise ValueError(f"--measure-qubit must be in 0..{n_qubits - 1}")
    if args.system_basis < 0 or args.system_basis >= (1 << n_qubits):
        raise ValueError(f"--system-basis must be in 0..{(1 << n_qubits) - 1}")
    if args.ancilla_basis < 0 or args.ancilla_basis >= (1 << n_qubits):
        raise ValueError(f"--ancilla-basis must be in 0..{(1 << n_qubits) - 1}")


def build_target_circuit(metadata_path: Path) -> cirq.Circuit:
    with open(metadata_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    n_val = int(meta["n_qubits"])
    target_seed = 42
    time_k = 5
    rng = np.random.default_rng(target_seed)
    h_arr = rng.uniform(-1.0, 1.0, size=n_val)
    t_val = 3.0 * math.pi / 40.0 * time_k + 0.001
    return build_u_target_circuit(n_val, rng, h_arr, t_val)


def main() -> None:
    args = parse_args()
    meta_path = _metadata_path(args.params, args.metadata)

    n_qubits, specs, _meta = load_block_specs(args.params, args.metadata)
    _validate_inputs(args, n_qubits)
    print(
        "Note: this legacy script uses exact full-state vectors; "
        "for larger local-observable checks use task2_code/run_sew_and_compare.py --backend causal-cone."
    )

    # ── build U_target ──
    print("Building U_target …")
    target_circuit = build_target_circuit(meta_path)
    target_qubits = list(cirq.LineQubit.range(n_qubits))

    # ── build U_sewing ──
    print("Building U_sewing …")
    ordered_specs = resolve_order(specs, args.order)
    sewing_circuit = build_block_sew_circuit(n_qubits, ordered_specs, args.dagger_convention)
    sewing_qubits = list(cirq.LineQubit.range(2 * n_qubits))

    # ── evolve (no measurement appended — pure unitary) ──
    sim = cirq.Simulator(dtype=np.complex128)

    print("Evolving U_target …")
    t0 = _basis_state(args.system_basis, n_qubits)
    target_result = sim.simulate(target_circuit, qubit_order=target_qubits,
                                  initial_state=t0)
    target_state = np.asarray(target_result.final_state_vector, dtype=np.complex128)

    print("Evolving U_sewing …")
    init = _basis_state((args.system_basis << n_qubits) | args.ancilla_basis, 2 * n_qubits)
    sewing_result = sim.simulate(sewing_circuit, qubit_order=sewing_qubits,
                                  initial_state=init)
    sewing_state = np.asarray(sewing_result.final_state_vector, dtype=np.complex128)

    # ── compute <Z_q> ──
    target_z = _z_expectation_from_state(target_state, args.measure_qubit, n_qubits)
    sewing_z = _z_expectation_from_state(sewing_state, args.measure_qubit, 2 * n_qubits)

    # ── report ──
    print()
    print("=" * 56)
    print(f"U_target    <Z_q{args.measure_qubit}> = {target_z:+.12g}")
    print(f"U_sewing    <Z_q{args.measure_qubit}> = {sewing_z:+.12g}")
    diff = abs(target_z - sewing_z)
    print(f"difference        = {diff:.12g}")
    print("=" * 56)
    if diff < 0.05:
        print("Note: difference is small; U_sewing may be faithful to U_target on this observable.")
    else:
        print("Note: difference is non-negligible. Possible causes:")
        print("  - local inversion training may not be perfect on the lightcone")
        print("  - the block-sew channel is an approximation, not an exact equivalence")
        print("  - the effective measured qubit semantics differ between the two circuits")


if __name__ == "__main__":
    main()
