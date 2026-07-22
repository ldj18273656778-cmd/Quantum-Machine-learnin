"""Compare U_sewing and U_target via sampled <Z_q> from Cirq simulation.

The script rebuilds the exact same U_target that was used during training
(using the h_arr and seed from the saved metadata JSON), constructs U_sewing
from the trained block parameters, and compares <Z_q> for both circuits.

Usage:
    python task2_code/sewing/compare_sewing_vs_target.py
    python task2_code/sewing/compare_sewing_vs_target.py --repetitions 2000
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
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

from task2_code.U_target import build_u_target_circuit
from task2_code.sewing.block_sewing import (
    DEFAULT_PARAMS,
    build_block_sew_circuit,
    load_block_specs,
    resolve_order,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare U_sewing vs U_target via sampled <Z_q> from repeated simulation.",
    )
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--order", default="odd-even")
    parser.add_argument("--dagger-convention", default="inverse")
    parser.add_argument("--measure-qubit", type=int, default=9,
                        help="Qubit index for Z measurement on U_target (system) and U_sewing (system).")
    parser.add_argument("--repetitions", type=int, default=1000,
                        help="Number of measurement repetitions per circuit.")
    parser.add_argument("--system-basis", type=int, default=0,
                        help="Initial system basis state (integer).")
    parser.add_argument("--ancilla-basis", type=int, default=0,
                        help="Initial ancilla basis state for U_sewing.")
    parser.add_argument("--max-system-qubits", type=int, default=12,
                        help="Refuse legacy full-state simulation above this system size; use run_sew_and_compare.py --backend causal-cone for larger systems.")
    return parser.parse_args()


def _metadata_path(params_path: Path, metadata_path: Path | None) -> Path:
    return metadata_path if metadata_path is not None else params_path.with_suffix(".json")


def _z_expectation(counts: dict[int, int], total: int) -> float:
    return (counts.get(0, 0) - counts.get(1, 0)) / total


def build_target_circuit(metadata_path: Path) -> cirq.Circuit:
    with open(metadata_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    n_val = int(meta["n_qubits"])
    # These constants match the training run that produced the saved params.
    target_seed = 42
    time_k = 5
    rng = np.random.default_rng(target_seed)
    h_arr = rng.uniform(-1.0, 1.0, size=n_val)
    t_val = 3.0 * math.pi / 40.0 * time_k + 0.001
    return build_u_target_circuit(n_val, rng, h_arr, t_val)


def _basis_circuit(basis_int: int, n_qubits: int) -> cirq.Circuit:
    circuit = cirq.Circuit()
    qubits = list(cirq.LineQubit.range(n_qubits))
    for idx, qubit in enumerate(qubits):
        bit = (basis_int >> (n_qubits - 1 - idx)) & 1
        if bit:
            circuit.append(cirq.X(qubit))
    return circuit


def simulate_z(circuit: cirq.Circuit, qubits: Sequence[cirq.Qid],
               measure_qubit: int, repetitions: int,
               initial_state: int | None = None,
               label: str = "") -> float:
    """Run measurement on measure_qubit in batches with a progress bar."""
    sim = cirq.Simulator()
    runnable = cirq.Circuit()
    if initial_state is not None:
        runnable += _basis_circuit(initial_state, len(qubits))
    runnable += circuit
    runnable.append(cirq.measure(qubits[measure_qubit], key="m"))

    from tqdm import tqdm as _tqdm
    batch_size = max(1, min(200, repetitions // 10))
    counts = {0: 0, 1: 0}
    pbar = _tqdm(total=repetitions, desc=label.strip(), unit="shot")
    for start in range(0, repetitions, batch_size):
        end = min(start + batch_size, repetitions)
        result = sim.run(runnable, repetitions=end - start)
        hist = result.histogram(key="m")
        counts[0] += int(hist.get(0, 0))
        counts[1] += int(hist.get(1, 0))
        pbar.update(end - start)
    pbar.close()
    return _z_expectation(counts, repetitions)


def _validate_inputs(args: argparse.Namespace, n_qubits: int) -> None:
    if n_qubits > args.max_system_qubits:
        raise ValueError(
            "legacy sampled sewing comparison uses full-state simulation and is "
            f"guarded to n_qubits <= {args.max_system_qubits}; use "
            "task2_code/run_sew_and_compare.py --backend causal-cone for larger systems"
        )
    if args.measure_qubit < 0 or args.measure_qubit >= n_qubits:
        raise ValueError(f"--measure-qubit must be in 0..{n_qubits - 1}")
    if args.system_basis < 0 or args.system_basis >= (1 << n_qubits):
        raise ValueError(f"--system-basis must be in 0..{(1 << n_qubits) - 1}")
    if args.ancilla_basis < 0 or args.ancilla_basis >= (1 << n_qubits):
        raise ValueError(f"--ancilla-basis must be in 0..{(1 << n_qubits) - 1}")


def main() -> None:
    args = parse_args()
    meta_path = _metadata_path(args.params, args.metadata)

    n_qubits, specs, _meta = load_block_specs(args.params, args.metadata)
    _validate_inputs(args, n_qubits)
    print(
        "Note: this legacy script still uses full-state Cirq simulation; "
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

    # ── simulate ──
    print(f"\nSimulating (repetitions = {args.repetitions}) …")
    target_z = simulate_z(target_circuit, target_qubits, args.measure_qubit,
                          args.repetitions, args.system_basis,
                          label="U_target  ")
    sewing_z = simulate_z(sewing_circuit, sewing_qubits, args.measure_qubit,
                          args.repetitions,
                          (args.system_basis << n_qubits) | args.ancilla_basis,
                          label="U_sewing  ")

    # ── report ──
    print("\n" + "=" * 56)
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
        print("  - the effective measured system-qubit semantics differ between the two circuits")


if __name__ == "__main__":
    main()
