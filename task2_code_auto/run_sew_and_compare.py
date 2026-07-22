"""Build U_sewing from saved params and compare XYZ observables to U_target.

Examples:
    python task2_code/run_sew_and_compare.py --run-dir task2_code/data/task2_training_n12_3blocks_zero_superoperator_from_zero_...
    python task2_code/run_sew_and_compare.py --params task2_code/data/task2_training_params_...npz --input-state ghz
    python task2_code/run_sew_and_compare.py --params task2_code/data/sum_block_123_best_params_20260520_193507.npz --input-state zero
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from collections.abc import Sequence
from typing import Any

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cirq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from task2_code_auto.sewing.block_sewing import (
    DEFAULT_PARAMS,
    build_block_sew_circuit,
    load_block_specs,
    resolve_order,
)
from task2_code_auto.target_factory import build_target_from_metadata

X_MAT = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Y_MAT = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
Z_MAT = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
PAULIS = {"X": X_MAT, "Y": Y_MAT, "Z": Z_MAT}
ComplexArray = NDArray[np.complex128]


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sew saved Task 2 blocks and compare XYZ expectations.")
    parser.add_argument("--run-dir", type=Path, default=None, help="training run directory containing params.npz and metadata.json; .npz files are also accepted")
    parser.add_argument("--params", type=Path, default=None, help="params .npz file, or a training run directory")
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--order", choices=["odd-even", "metadata", "reverse"], default="odd-even")
    parser.add_argument("--dagger-convention", choices=["trained", "inverse"], default="trained")
    parser.add_argument("--ansatz", default=None, help="ansatz registry key; defaults to metadata ansatz or default_5layer_cz")
    parser.add_argument("--input-state", default="zero", help="zero, ones, ghz, or basis:<int>")
    parser.add_argument("--observable-qubits", default=None, help="comma-separated system qubits to evaluate; defaults to all system qubits")
    parser.add_argument("--backend", choices=["auto", "exact", "causal-cone"], default="auto", help="observable simulation backend")
    parser.add_argument("--max-exact-n-qubits", type=int, default=12, help="largest system size allowed for full exact state-vector backend")
    parser.add_argument("--max-cone-qubits", type=int, default=24, help="largest causal cone to simulate exactly")
    parser.add_argument("--output-dir", type=Path, default=None, help="comparison output directory; defaults to the run directory when provided")
    return parser.parse_args()


def _parse_observable_qubits(value: str | None, n_qubits: int) -> list[int]:
    if value is None:
        return list(range(n_qubits))
    items = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError("--observable-qubits must contain at least one qubit")
    invalid = [q for q in items if q < 0 or q >= n_qubits]
    if invalid:
        raise ValueError(f"observable qubits out of range 0..{n_qubits - 1}: {invalid}")
    if len(items) != len(set(items)):
        raise ValueError(f"observable qubits must not contain duplicates: {items}")
    return items


def _single_existing_file(directory: Path, names: list[str], patterns: list[str]) -> Path:
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in directory.glob(pattern) if path.is_file())
    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    if not unique_matches:
        expected = ", ".join(names + patterns)
        raise FileNotFoundError(f"no matching file found in {directory}; expected one of: {expected}")
    choices = "\n".join(str(path) for path in unique_matches)
    raise ValueError(f"multiple matching files found in {directory}; pass the file explicitly:\n{choices}")


def _metadata_path(params_path: Path, metadata_path: Path | None) -> Path:
    if metadata_path is not None:
        return metadata_path
    if params_path.name == "params.npz":
        candidate = params_path.with_name("metadata.json")
        if candidate.exists():
            return candidate
    if params_path.name.startswith("task2_training_params_"):
        metadata_name = params_path.name.replace("task2_training_params_", "task2_training_metadata_", 1)
        candidate = params_path.with_name(metadata_name).with_suffix(".json")
        if candidate.exists():
            return candidate
    return params_path.with_suffix(".json")


def _resolve_input_paths(run_dir: Path | None, params_path: Path | None, metadata_path: Path | None) -> tuple[Path, Path, Path | None]:
    if run_dir is not None and run_dir.is_file():
        params_path = run_dir
        run_dir = None

    if run_dir is None and params_path is not None and params_path.is_dir():
        run_dir = params_path
        params_path = None

    if run_dir is not None:
        if params_path is None:
            params_path = _single_existing_file(run_dir, ["params.npz"], ["task2_training_params_*.npz", "*.npz"])
        elif not params_path.is_absolute():
            params_path = run_dir / params_path

        if metadata_path is None:
            metadata_path = _single_existing_file(run_dir, ["metadata.json"], ["task2_training_metadata_*.json"])
        elif not metadata_path.is_absolute():
            metadata_path = run_dir / metadata_path
        return params_path, metadata_path, run_dir

    params_path = params_path or DEFAULT_PARAMS
    return params_path, _metadata_path(params_path, metadata_path), None


def _basis_state(index: int, n_qubits: int) -> ComplexArray:
    dim = 1 << n_qubits
    if index < 0 or index >= dim:
        raise ValueError(f"basis index {index} out of range for {n_qubits} qubits")
    state = np.zeros(dim, dtype=np.complex128)
    state[index] = 1.0
    return state


def _ghz_state(n_qubits: int) -> ComplexArray:
    state = np.zeros(1 << n_qubits, dtype=np.complex128)
    state[0] = 1.0
    state[(1 << n_qubits) - 1] = 1.0
    return state / np.sqrt(2.0)


def prepare_system_state(spec: str, n_qubits: int) -> ComplexArray:
    value = spec.strip().lower()
    if value == "zero":
        return _basis_state(0, n_qubits)
    if value == "ones":
        return _basis_state((1 << n_qubits) - 1, n_qubits)
    if value == "ghz":
        return _ghz_state(n_qubits)
    if value.startswith("basis:"):
        return _basis_state(int(value.split(":", 1)[1]), n_qubits)
    raise ValueError("input-state must be 'zero', 'ones', 'ghz', or 'basis:<int>'")


def _reduced_density_matrix(state: ComplexArray, qubit_index: int, total_qubits: int) -> ComplexArray:
    q_pos = qubit_index
    before = 1 << q_pos
    after = 1 << (total_qubits - q_pos - 1)
    reshaped = state.reshape(before, 2, after)
    psi0 = reshaped[:, 0, :].ravel()
    psi1 = reshaped[:, 1, :].ravel()
    rho = np.zeros((2, 2), dtype=complex)
    rho[0, 0] = np.sum(np.abs(psi0) ** 2)
    rho[1, 1] = np.sum(np.abs(psi1) ** 2)
    rho[0, 1] = np.vdot(psi0, psi1)
    rho[1, 0] = np.conj(rho[0, 1])
    return rho


def _expectation(state: ComplexArray, qubit_index: int, total_qubits: int, pauli: ComplexArray) -> float:
    return float(np.trace(_reduced_density_matrix(state, qubit_index, total_qubits) @ pauli).real)


def compute_expectations(state: ComplexArray, qubits: list[int], total_qubits: int) -> dict[str, list[float]]:
    return {
        label: [_expectation(state, q, total_qubits, matrix) for q in qubits]
        for label, matrix in PAULIS.items()
    }


def _bits_from_int(index: int, width: int) -> list[int]:
    return [(index >> (width - 1 - pos)) & 1 for pos in range(width)]


def _basis_index_for_active(
    active_qubits: Sequence[cirq.LineQubit],
    n_system_qubits: int,
    input_state: str,
    ghz_component: int | None = None,
) -> int:
    value = input_state.strip().lower()
    basis_bits: list[int] | None = None
    if value.startswith("basis:"):
        basis_bits = _bits_from_int(int(value.split(":", 1)[1]), n_system_qubits)

    out = 0
    for qubit in active_qubits:
        q = int(qubit.x)
        if q >= n_system_qubits:
            bit = 0
        elif value == "zero":
            bit = 0
        elif value == "ones":
            bit = 1
        elif value == "ghz":
            if ghz_component is None:
                raise ValueError("ghz_component is required for GHZ initial states")
            bit = int(ghz_component)
        elif basis_bits is not None:
            bit = basis_bits[q]
        else:
            raise ValueError("input-state must be 'zero', 'ones', 'ghz', or 'basis:<int>'")
        out = (out << 1) | bit
    return out


def _active_basis_state(
    active_qubits: Sequence[cirq.LineQubit],
    n_system_qubits: int,
    input_state: str,
    ghz_component: int | None = None,
) -> ComplexArray:
    state = np.zeros(1 << len(active_qubits), dtype=np.complex128)
    state[_basis_index_for_active(active_qubits, n_system_qubits, input_state, ghz_component)] = 1.0
    return state


def _causal_cone_circuit(circuit: cirq.Circuit, measured_qubit: cirq.LineQubit) -> tuple[cirq.Circuit, list[cirq.LineQubit]]:
    support: set[cirq.Qid] = {measured_qubit}
    kept_reversed: list[cirq.Operation] = []
    for op in reversed(list(circuit.all_operations())):
        op_qubits = set(op.qubits)
        if support & op_qubits:
            support |= op_qubits
            kept_reversed.append(op)
    def line_index(qubit: cirq.Qid) -> int:
        if not isinstance(qubit, cirq.LineQubit):
            raise TypeError(f"causal-cone backend expects LineQubit labels, got {qubit!r}")
        return int(qubit.x)

    active_qubits = [cirq.LineQubit(line_index(qubit)) for qubit in sorted(support, key=line_index)]
    return cirq.Circuit(reversed(kept_reversed)), active_qubits


def _state_pauli_expectation(state: ComplexArray, pauli: str, qubit_position: int, total_qubits: int) -> float:
    return _expectation(state, qubit_position, total_qubits, PAULIS[pauli])


def _simulate_active_state(circuit: cirq.Circuit, active_qubits: Sequence[cirq.LineQubit], initial_state: ComplexArray) -> ComplexArray:
    result = cirq.Simulator(dtype=np.complex128).simulate(
        circuit,
        qubit_order=list(active_qubits),
        initial_state=initial_state,
    )
    return np.asarray(result.final_state_vector, dtype=np.complex128)


def _causal_cone_expectation_for_qubit(
    circuit: cirq.Circuit,
    measured_qubit: int,
    n_system_qubits: int,
    input_state: str,
    *,
    max_cone_qubits: int,
) -> tuple[dict[str, float], int]:
    measured = cirq.LineQubit(int(measured_qubit))
    cone_circuit, active_qubits = _causal_cone_circuit(circuit, measured)
    if len(active_qubits) > max_cone_qubits:
        raise ValueError(
            f"causal cone for qubit {measured_qubit} has {len(active_qubits)} qubits, "
            + f"exceeding --max-cone-qubits={max_cone_qubits}"
        )
    measured_pos = active_qubits.index(measured)
    total_active = len(active_qubits)
    value = input_state.strip().lower()

    if value == "ghz":
        active_system = {int(q.x) for q in active_qubits if int(q.x) < n_system_qubits}
        include_coherence = len(active_system) == n_system_qubits
        zero_initial = _active_basis_state(active_qubits, n_system_qubits, value, ghz_component=0)
        one_initial = _active_basis_state(active_qubits, n_system_qubits, value, ghz_component=1)
        if include_coherence:
            ghz_initial = (zero_initial + one_initial) / np.sqrt(2.0)
            ghz_final = _simulate_active_state(cone_circuit, active_qubits, ghz_initial)
            return {pauli: _state_pauli_expectation(ghz_final, pauli, measured_pos, total_active) for pauli in PAULIS}, total_active

        zero_final = _simulate_active_state(cone_circuit, active_qubits, zero_initial)
        one_final = _simulate_active_state(cone_circuit, active_qubits, one_initial)
        out: dict[str, float] = {}
        for pauli in PAULIS:
            exp_value = 0.5 * _state_pauli_expectation(zero_final, pauli, measured_pos, total_active)
            exp_value += 0.5 * _state_pauli_expectation(one_final, pauli, measured_pos, total_active)
            out[pauli] = exp_value
        return out, total_active

    final = _simulate_active_state(
        cone_circuit,
        active_qubits,
        _active_basis_state(active_qubits, n_system_qubits, value),
    )
    return {pauli: _state_pauli_expectation(final, pauli, measured_pos, total_active) for pauli in PAULIS}, total_active


def compute_expectations_causal_cone(
    circuit: cirq.Circuit,
    qubits: list[int],
    n_system_qubits: int,
    input_state: str,
    *,
    max_cone_qubits: int,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    out = {label: [] for label in PAULIS}
    cone_sizes: dict[str, int] = {}
    for q in qubits:
        per_pauli, cone_size = _causal_cone_expectation_for_qubit(
            circuit,
            q,
            n_system_qubits,
            input_state,
            max_cone_qubits=max_cone_qubits,
        )
        cone_sizes[str(q)] = cone_size
        for label in PAULIS:
            out[label].append(per_pauli[label])
    return out, {"cone_sizes": cone_sizes, "max_cone_size": max(cone_sizes.values()) if cone_sizes else 0}


def plot_comparison(qubits: list[int], target: dict[str, list[float]], sewing: dict[str, list[float]], path: Path, title: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, label in zip(axes, ["X", "Y", "Z"]):
        x = np.arange(len(qubits))
        width = 0.35
        ax.bar(x - width / 2, target[label], width, label="U_target", alpha=0.75)
        ax.bar(x + width / 2, sewing[label], width, label="U_sewing", alpha=0.75)
        ax.set_ylabel(f"<{label}_q>")
        ax.set_xticks(x)
        ax.set_xticklabels([f"q{q}" for q in qubits])
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title(title)
    axes[-1].set_xlabel("system qubit")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    args = parse_args()
    params_path, meta_path, run_dir = _resolve_input_paths(args.run_dir, args.params, args.metadata)
    n_qubits, specs, loaded_metadata = load_block_specs(params_path, meta_path)
    backend = args.backend
    if backend == "auto":
        backend = "exact" if n_qubits <= args.max_exact_n_qubits else "causal-cone"
    if backend == "exact" and n_qubits > args.max_exact_n_qubits:
        raise ValueError(
            f"exact state-vector sewing comparison is guarded to n_qubits <= {args.max_exact_n_qubits}; "
            + "use --backend causal-cone for larger systems"
        )

    ansatz_name = args.ansatz or str(loaded_metadata.get("ansatz", "default_5layer_cz"))
    target = build_target_from_metadata(loaded_metadata)
    ordered_specs = resolve_order(specs, args.order)
    sewing_circuit = build_block_sew_circuit(
        n_qubits,
        ordered_specs,
        args.dagger_convention,
        ansatz_name=ansatz_name,
    )

    qubits = _parse_observable_qubits(args.observable_qubits, n_qubits)
    backend_metadata: dict[str, Any] = {"backend": backend}
    if backend == "exact":
        system_state = prepare_system_state(args.input_state, n_qubits)
        ancilla_state = _basis_state(0, n_qubits)
        sewing_initial = np.kron(system_state, ancilla_state)

        sim = cirq.Simulator(dtype=np.complex128)
        target_result = sim.simulate(
            target.circuit,
            qubit_order=list(cirq.LineQubit.range(n_qubits)),
            initial_state=system_state,
        )
        sewing_result = sim.simulate(
            sewing_circuit,
            qubit_order=list(cirq.LineQubit.range(2 * n_qubits)),
            initial_state=sewing_initial,
        )
        target_state = np.asarray(target_result.final_state_vector, dtype=np.complex128)
        sewing_state = np.asarray(sewing_result.final_state_vector, dtype=np.complex128)
        target_exp = compute_expectations(target_state, qubits, n_qubits)
        sewing_exp = compute_expectations(sewing_state, qubits, 2 * n_qubits)
    else:
        target_exp, target_backend = compute_expectations_causal_cone(
            target.circuit,
            qubits,
            n_qubits,
            args.input_state,
            max_cone_qubits=args.max_cone_qubits,
        )
        sewing_exp, sewing_backend = compute_expectations_causal_cone(
            sewing_circuit,
            qubits,
            n_qubits,
            args.input_state,
            max_cone_qubits=args.max_cone_qubits,
        )
        backend_metadata["target"] = target_backend
        backend_metadata["sewing"] = sewing_backend

    summary: dict[str, Any] = {}
    for label in ["X", "Y", "Z"]:
        diffs = [abs(a - b) for a, b in zip(target_exp[label], sewing_exp[label])]
        summary[label] = {
            "max_diff": max(diffs),
            "mean_diff": float(sum(diffs) / len(diffs)),
        }
        print(f"<{label}> max_diff={summary[label]['max_diff']:.6g} mean_diff={summary[label]['mean_diff']:.6g}")

    output_dir = args.output_dir or run_dir or Path("task2_code/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    superoperator_name = str(loaded_metadata.get("superoperator", "unknown_superoperator"))
    stem = f"sew_compare_{_safe_stem(superoperator_name)}_{meta_path.stem}_{args.input_state.replace(':', '_')}"
    json_path = output_dir / f"{stem}.json"
    png_path = output_dir / f"{stem}.png"
    payload = {
        "run_dir": run_dir,
        "params": params_path,
        "metadata": meta_path,
        "input_state": args.input_state,
        "observable_qubits": qubits,
        "order": args.order,
        "dagger_convention": args.dagger_convention,
        "ansatz": ansatz_name,
        "n_qubits": n_qubits,
        "backend": backend_metadata,
        "target": target_exp,
        "sewing": sewing_exp,
        "summary": summary,
    }
    json_path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    plot_comparison(qubits, target_exp, sewing_exp, png_path, f"U_target vs U_sewing ({args.input_state})")
    print(f"Saved comparison JSON: {json_path}")
    print(f"Saved comparison plot: {png_path}")


if __name__ == "__main__":
    main()
