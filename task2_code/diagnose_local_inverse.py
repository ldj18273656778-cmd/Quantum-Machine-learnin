"""Check a saved single-block local inverse on a chosen input state.

The saved theta parameterizes U_trial(theta).  In the current training/sewing
convention, the learned local inverse applied before U_target is
U_trial(theta)^dagger, because the training residual is
U_target U_trial(theta)^dagger.

Examples:
    python task2_code/diagnose_local_inverse.py --run-dir task2_code/data/task2_training_n12_3blocks_zero_superoperator_from_zero_20260528_145828 --block 0,1,2,3
    python task2_code/diagnose_local_inverse.py --params task2_code/data/task2_training_params_superoperator_from_zero_20260527_215029.npz --block-index 1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cirq
import numpy as np

from task2_code.ansatz_registry import resolve_ansatz
from task2_code.run_sew_and_compare import compute_expectations, prepare_system_state
from task2_code.sewing.block_sewing import DEFAULT_PARAMS, BlockSpec, load_block_specs
from task2_code.target_factory import build_target_from_metadata


def _parse_block(value: str) -> tuple[int, ...]:
    items = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not items:
        raise argparse.ArgumentTypeError("block must contain at least one qubit label")
    if len(items) != len(set(items)):
        raise argparse.ArgumentTypeError(f"block labels must be unique, got {items}")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose one saved block as a local inverse before U_target.")
    parser.add_argument("--run-dir", type=Path, default=None, help="training run directory containing params.npz and metadata.json")
    parser.add_argument("--params", type=Path, default=None, help="params .npz file; defaults to run-dir/params.npz")
    parser.add_argument("--metadata", type=Path, default=None, help="metadata .json file; defaults to run-dir/metadata.json")
    parser.add_argument("--block", type=_parse_block, default=(0, 1, 2, 3), help="comma-separated block labels, e.g. 0,1,2,3")
    parser.add_argument("--block-index", type=int, default=None, help="1-based block index; overrides --block")
    parser.add_argument("--ansatz", default=None, help="ansatz registry key; defaults to metadata ansatz or default_5layer_cz")
    parser.add_argument("--input-state", default="zero", help="zero, ones, ghz, or basis:<int>")
    parser.add_argument("--output", type=Path, default=None, help="optional JSON output path")
    return parser.parse_args()


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


def _resolve_paths(run_dir: Path | None, params_path: Path | None, metadata_path: Path | None) -> tuple[Path, Path, Path | None]:
    if run_dir is not None:
        if run_dir.is_file():
            params_path = run_dir
            run_dir = None
        else:
            params_path = params_path or run_dir / "params.npz"
            metadata_path = metadata_path or run_dir / "metadata.json"
    params_path = params_path or DEFAULT_PARAMS
    return params_path, _metadata_path(params_path, metadata_path), run_dir


def _select_block(specs: list[BlockSpec], block: tuple[int, ...], block_index: int | None) -> BlockSpec:
    if block_index is not None:
        for spec in specs:
            if spec.block_index == block_index:
                return spec
        raise ValueError(f"block_index {block_index} not found; available: {[spec.block_index for spec in specs]}")
    for spec in specs:
        if spec.block_qubits == block:
            return spec
    raise ValueError(f"block {block} not found; available: {[spec.block_qubits for spec in specs]}")


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def build_local_inverse_circuit(spec: BlockSpec, ansatz_name: str) -> cirq.Circuit:
    builder = resolve_ansatz(ansatz_name)
    qubits = [cirq.LineQubit(q) for q in spec.lightcone_qubits]
    trial = builder(spec.theta, qubits=qubits, n_qubits=len(qubits))
    return cirq.Circuit(cirq.inverse(trial))


def main() -> None:
    args = parse_args()
    params_path, metadata_path, run_dir = _resolve_paths(args.run_dir, args.params, args.metadata)
    n_qubits, specs, metadata = load_block_specs(params_path, metadata_path)
    spec = _select_block(specs, args.block, args.block_index)
    ansatz_name = args.ansatz or str(metadata.get("ansatz", "default_5layer_cz"))

    target = build_target_from_metadata(metadata)
    local_inverse = build_local_inverse_circuit(spec, ansatz_name)
    diagnostic_circuit = cirq.Circuit()
    diagnostic_circuit += local_inverse
    diagnostic_circuit += target.circuit

    initial_state = prepare_system_state(args.input_state, n_qubits)
    sim = cirq.Simulator(dtype=np.complex128)
    result = sim.simulate(
        diagnostic_circuit,
        qubit_order=list(cirq.LineQubit.range(n_qubits)),
        initial_state=initial_state,
    )
    final_state = np.asarray(result.final_state_vector, dtype=np.complex128)
    expectations = compute_expectations(final_state, list(range(n_qubits)), n_qubits)
    block_expectations = {
        label: {str(q): float(expectations[label][q]) for q in spec.block_qubits}
        for label in ["X", "Y", "Z"]
    }
    block_z = {str(q): float(expectations["Z"][q]) for q in spec.block_qubits}

    payload = {
        "run_dir": run_dir,
        "params": params_path,
        "metadata": metadata_path,
        "trained_superoperator": metadata.get("superoperator"),
        "ansatz": ansatz_name,
        "input_state": args.input_state,
        "operation_order": "U_target after U_trial(theta)^dagger",
        "matrix_order": "U_target * U_trial(theta)^dagger",
        "z_convention": "Pauli Z expectation: |0> -> +1, |1> -> -1",
        "block_index": spec.block_index,
        "block_qubits": list(spec.block_qubits),
        "lightcone_qubits": list(spec.lightcone_qubits),
        "block_expectations": block_expectations,
        "block_z_expectations": block_z,
        "all_expectations": {
            label: {str(q): float(value) for q, value in enumerate(expectations[label])}
            for label in ["X", "Y", "Z"]
        },
        "all_z_expectations": {str(q): float(value) for q, value in enumerate(expectations["Z"])},
    }

    print(f"params = {params_path}")
    print(f"metadata = {metadata_path}")
    print(f"trained_superoperator = {metadata.get('superoperator')}")
    print(f"block = {spec.block_qubits}, lightcone = {spec.lightcone_qubits}")
    print("Convention: U_trial(theta) is the ansatz; local inverse is U_trial(theta)^dagger.")
    print("Circuit order: apply U_trial(theta)^dagger first, then U_target, i.e. U_target * U_trial(theta)^dagger.")
    print("Pauli-Z convention: |0> has <Z>=+1 and |1> has <Z>=-1.")
    print("Block X/Y/Z expectations:")
    for q in spec.block_qubits:
        print(
            f"  q{q}: "
            f"X={block_expectations['X'][str(q)]: .12g}  "
            f"Y={block_expectations['Y'][str(q)]: .12g}  "
            f"Z={block_expectations['Z'][str(q)]: .12g}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
        print(f"Saved diagnostic JSON: {args.output}")


if __name__ == "__main__":
    main()
