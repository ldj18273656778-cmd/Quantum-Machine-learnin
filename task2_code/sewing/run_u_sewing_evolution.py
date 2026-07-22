"""Minimal Cirq evolution runner for the saved U_sewing circuit.

Examples:
    python task2_code/sewing/run_u_sewing_evolution.py
    python task2_code/sewing/run_u_sewing_evolution.py --save-state
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cirq
import numpy as np

from task2_code.sewing.block_sewing import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PARAMS,
    build_block_sew_circuit,
    load_block_specs,
    resolve_order,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run state-vector evolution under U_sewing.")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--order", choices=["odd-even", "metadata", "reverse"], default="odd-even")
    parser.add_argument("--dagger-convention", choices=["trained", "inverse"], default="inverse")
    parser.add_argument("--system-basis", type=int, default=0, help="Initial system basis integer.")
    parser.add_argument("--ancilla-basis", type=int, default=0, help="Initial ancilla basis integer.")
    parser.add_argument("--measure-qubit", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def basis_index(system_basis: int, ancilla_basis: int, n_qubits: int) -> int:
    if system_basis < 0 or system_basis >= (1 << n_qubits):
        raise ValueError(f"system_basis must be in [0, {1 << n_qubits}), got {system_basis}")
    if ancilla_basis < 0 or ancilla_basis >= (1 << n_qubits):
        raise ValueError(f"ancilla_basis must be in [0, {1 << n_qubits}), got {ancilla_basis}")
    return int((system_basis << n_qubits) | ancilla_basis)


def basis_preparation_circuit(initial_state: int, total_qubits: int) -> cirq.Circuit:
    qubits = list(cirq.LineQubit.range(total_qubits))
    circuit = cirq.Circuit()
    for q_index, qubit in enumerate(qubits):
        bit = (initial_state >> (total_qubits - 1 - q_index)) & 1
        if bit:
            circuit.append(cirq.X(qubit))
    return circuit


def main() -> None:
    args = parse_args()
    n_qubits, specs, _metadata = load_block_specs(args.params, args.metadata)
    ordered_specs = resolve_order(specs, args.order)
    circuit = build_block_sew_circuit(n_qubits, ordered_specs, args.dagger_convention)
    total_qubits = 2 * n_qubits
    qubit_order = list(cirq.LineQubit.range(total_qubits))
    initial_state = basis_index(args.system_basis, args.ancilla_basis, n_qubits)
    if args.measure_qubit < 0 or args.measure_qubit >= total_qubits:
        raise ValueError(f"measure_qubit must be in [0, {total_qubits}), got {args.measure_qubit}")
    if args.repetitions <= 0:
        raise ValueError(f"repetitions must be positive, got {args.repetitions}")

    print("mode = block-sew")
    print(f"order = {args.order}, dagger_convention = {args.dagger_convention}")
    print(f"total_qubits = {total_qubits}")
    print(f"operation_count = {sum(1 for _ in circuit.all_operations())}, moments = {len(circuit)}")
    print(f"initial_basis_index = {initial_state}")

    simulator = cirq.Simulator(dtype=np.complex128)
    measured_circuit = basis_preparation_circuit(initial_state, total_qubits)
    measured_circuit += circuit
    measured_circuit.append(cirq.measure(cirq.LineQubit(args.measure_qubit), key="m"))
    measurement_result = simulator.run(measured_circuit, repetitions=args.repetitions)
    histogram = measurement_result.histogram(key="m")
    zero_count = int(histogram.get(0, 0))
    one_count = int(histogram.get(1, 0))
    sampled_z = (zero_count - one_count) / args.repetitions
    print(f"measurement = cirq.measure(q{args.measure_qubit}, key='m')")
    print(f"repetitions = {args.repetitions}")
    print(f"counts = {{0: {zero_count}, 1: {one_count}}}")
    print(f"sampled_<Z_q{args.measure_qubit}> = {sampled_z:.12g}")

    if args.save_state:
        result = simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state)
        final_state = np.asarray(result.final_state_vector, dtype=np.complex128)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = args.output_dir / f"u_sewing_evolved_state_block-sew_{timestamp}.npy"
        np.save(path, final_state)
        print(f"Final state saved: {path}")


if __name__ == "__main__":
    main()
