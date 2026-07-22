"""Construct U_sewing from saved block-local inversion parameters.

Default usage:
    python task2_code/sewing/block_sewing.py --params <bundle>/params.npz --metadata <bundle>/metadata.json

The artifact is a 2N-qubit Cirq circuit implementing the block-sewing
construction.  Dense full-system unitary export is intentionally unsupported.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cirq
import numpy as np
from numpy.typing import NDArray

from task2_code_auto.ansatz_registry import (
    ansatz_theta_count,
    build_registered_scoped_ansatz_circuit,
    registered_ansatz_unitary_on_lightcone,
)


DEFAULT_PARAMS = Path("task2_code/data/sum_block_123_best_params_20260520_193507.npz")
DEFAULT_OUTPUT_DIR = Path("task2_code/data")


@dataclass(frozen=True)
class BlockSpec:
    block_index: int
    block_qubits: tuple[int, ...]
    target_bit: int
    lightcone_qubits: tuple[int, ...]
    ansatz: str
    block_only_ansatz: bool
    params_key: str
    theta: NDArray[np.float64]
    best_loss: float
    per_bit_losses: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build U_sewing from saved block-local inversion params.")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--order", choices=["odd-even", "metadata", "reverse"], default="odd-even")
    parser.add_argument("--dagger-convention", choices=["trained", "inverse"], default="inverse")
    parser.add_argument("--ansatz", default=None, help="ansatz registry key; defaults to metadata ansatz")
    parser.add_argument("--save-local-unitaries", action="store_true")
    parser.add_argument("--print-circuit", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"metadata must contain a JSON object: {path}")
    return data


def _as_int_tuple(values: object, name: str) -> tuple[int, ...]:
    if not isinstance(values, list | tuple):
        raise ValueError(f"{name} must be a list or tuple, got {type(values).__name__}")
    items = tuple(int(value) for value in values)
    if len(items) != len(set(items)):
        raise ValueError(f"{name} must not contain duplicates, got {items}")
    return items


def _metadata_path(params_path: Path, metadata_path: Path | None) -> Path:
    return metadata_path if metadata_path is not None else params_path.with_suffix(".json")


def load_block_specs(params_path: Path, metadata_path: Path | None = None) -> tuple[int, list[BlockSpec], dict[str, Any]]:
    meta_path = _metadata_path(params_path, metadata_path)
    if not params_path.exists():
        raise FileNotFoundError(f"params file not found: {params_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata file not found: {meta_path}")

    metadata = _load_json(meta_path)
    npz = np.load(params_path, allow_pickle=False)
    n_qubits = int(metadata.get("n_qubits", 0))
    if n_qubits <= 0:
        raise ValueError(f"metadata n_qubits must be positive, got {n_qubits}")
    blocks_meta = metadata.get("blocks")
    if not isinstance(blocks_meta, list) or not blocks_meta:
        raise ValueError("metadata must contain a non-empty 'blocks' list")

    npz_blocks = np.asarray(npz["blocks"], dtype=int) if "blocks" in npz.files else None
    npz_target_bits = np.asarray(npz["target_bits"], dtype=int) if "target_bits" in npz.files else None
    npz_theta_sizes = np.asarray(npz["theta_sizes"], dtype=int) if "theta_sizes" in npz.files else None

    specs: list[BlockSpec] = []
    owned: list[int] = []
    for offset, block_meta in enumerate(blocks_meta):
        if not isinstance(block_meta, dict):
            raise ValueError(f"block metadata at index {offset} must be an object")
        block_index = int(block_meta["block_index"])
        block_qubits = _as_int_tuple(block_meta["block_qubits"], f"block {block_index} block_qubits")
        lightcone_qubits = _as_int_tuple(block_meta["lightcone_qubits"], f"block {block_index} lightcone_qubits")
        ansatz = str(block_meta.get("ansatz", metadata.get("ansatz", "default_5layer_cz")))
        block_only_ansatz = bool(block_meta.get("block_only_ansatz", metadata.get("block_only_ansatz", False)))
        target_bit = int(block_meta["target_bit"])
        params_key = str(block_meta.get("params_key", f"best_params_block_{block_index}"))
        if params_key not in npz.files:
            raise ValueError(f"params key {params_key!r} missing from {params_path}")
        theta = np.asarray(npz[params_key], dtype=float)
        ansatz_qubits = len(block_qubits) if block_only_ansatz else len(lightcone_qubits)
        expected_theta = ansatz_theta_count(ansatz, ansatz_qubits)
        if theta.shape != (expected_theta,):
            raise ValueError(f"{params_key} must have shape ({expected_theta},), got {theta.shape}")
        if int(block_meta["theta_size"]) != expected_theta:
            raise ValueError(f"metadata theta_size mismatch for block {block_index}")
        if int(block_meta["ansatz_qubits"]) != ansatz_qubits:
            raise ValueError(f"metadata ansatz_qubits mismatch for block {block_index}")
        if target_bit not in block_qubits:
            raise ValueError(f"target_bit {target_bit} is not in block {block_qubits}")
        missing = [q for q in block_qubits if q not in lightcone_qubits]
        if missing:
            raise ValueError(f"block {block_index} qubits missing from lightcone: {missing}")
        out_of_range = [q for q in lightcone_qubits if q < 0 or q >= n_qubits]
        if out_of_range:
            raise ValueError(f"block {block_index} lightcone qubits out of range: {out_of_range}")
        if npz_blocks is not None and tuple(npz_blocks[offset].tolist()) != block_qubits:
            raise ValueError(f"NPZ blocks mismatch for block {block_index}")
        if npz_target_bits is not None and int(npz_target_bits[offset]) != target_bit:
            raise ValueError(f"NPZ target_bits mismatch for block {block_index}")
        if npz_theta_sizes is not None and int(npz_theta_sizes[offset]) != expected_theta:
            raise ValueError(f"NPZ theta_sizes mismatch for block {block_index}")
        owned.extend(block_qubits)
        losses_obj = block_meta.get("per_bit_losses", {})
        per_bit_losses = {str(key): float(value) for key, value in dict(losses_obj).items()}
        specs.append(
            BlockSpec(
                block_index=block_index,
                block_qubits=block_qubits,
                target_bit=target_bit,
                lightcone_qubits=lightcone_qubits,
                ansatz=ansatz,
                block_only_ansatz=block_only_ansatz,
                params_key=params_key,
                theta=theta,
                best_loss=float(block_meta.get("best_loss", np.nan)),
                per_bit_losses=per_bit_losses,
            )
        )

    if sorted(owned) != list(range(n_qubits)):
        raise ValueError(f"blocks must own each data qubit exactly once; got {sorted(owned)}")
    return n_qubits, specs, metadata


def apply_ansatz_override(specs: list[BlockSpec], ansatz_name: str | None) -> list[BlockSpec]:
    if ansatz_name is None:
        return list(specs)
    overridden: list[BlockSpec] = []
    for spec in specs:
        expected_theta = ansatz_theta_count(ansatz_name, len(spec.block_qubits) if spec.block_only_ansatz else len(spec.lightcone_qubits))
        if spec.theta.shape != (expected_theta,):
            raise ValueError(
                f"{spec.params_key} has shape {spec.theta.shape}, but ansatz {ansatz_name!r} "
                + f"expects ({expected_theta},)"
            )
        overridden.append(
            BlockSpec(
                block_index=spec.block_index,
                block_qubits=spec.block_qubits,
                target_bit=spec.target_bit,
                lightcone_qubits=spec.lightcone_qubits,
                ansatz=ansatz_name,
                block_only_ansatz=spec.block_only_ansatz,
                params_key=spec.params_key,
                theta=spec.theta,
                best_loss=spec.best_loss,
                per_bit_losses=spec.per_bit_losses,
            )
        )
    return overridden


def resolve_order(specs: list[BlockSpec], order: str) -> list[BlockSpec]:
    if order == "odd-even":
        odd = [spec for spec in specs if spec.block_index % 2 == 1]
        even = [spec for spec in specs if spec.block_index % 2 == 0]
        return odd + even
    if order == "metadata":
        return list(specs)
    if order == "reverse":
        return list(reversed(specs))
    raise ValueError(f"unsupported order: {order}")


def build_local_ansatz_circuit(spec: BlockSpec, ansatz_name: str | None = None) -> cirq.Circuit:
    """Build the ansatz circuit from the registry for the given BlockSpec."""
    name = spec.ansatz if ansatz_name is None else ansatz_name
    return build_registered_scoped_ansatz_circuit(
        name,
        spec.theta,
        spec.lightcone_qubits,
        spec.block_qubits,
        block_only_ansatz=spec.block_only_ansatz,
    )


def _append_ansatz_pair(
    circuit: cirq.Circuit,
    ansatz: cirq.Circuit,
    inverse_first: bool,
) -> tuple[cirq.OP_TREE, cirq.OP_TREE]:
    # build_ansatz returns U_trial.  The learned local inverse used by sewing is
    # U_trial^dagger, so the default convention applies the inverse first.
    return (cirq.inverse(ansatz), ansatz) if inverse_first else (ansatz, cirq.inverse(ansatz))


def build_block_sew_circuit(
    n_qubits: int,
    ordered_specs: list[BlockSpec],
    dagger_convention: str,
    ansatz_name: str | None = None,
) -> cirq.Circuit:
    circuit = cirq.Circuit()
    system = list(cirq.LineQubit.range(n_qubits))
    ancilla = list(cirq.LineQubit.range(n_qubits, 2 * n_qubits))
    inverse_first = dagger_convention == "inverse"
    for spec in ordered_specs:
        ansatz = build_local_ansatz_circuit(spec, ansatz_name=ansatz_name)
        before, after = _append_ansatz_pair(circuit, ansatz, inverse_first)
        circuit += before
        circuit.append([cirq.SWAP(system[q], ancilla[q]) for q in spec.block_qubits])
        circuit += after
    circuit.append([cirq.SWAP(system[q], ancilla[q]) for q in range(n_qubits)])
    return circuit


def overlap_report(specs: list[BlockSpec]) -> dict[str, list[int]]:
    report: dict[str, list[int]] = {}
    for left_idx, left in enumerate(specs):
        for right in specs[left_idx + 1 :]:
            key = f"block_{left.block_index}_block_{right.block_index}"
            report[key] = sorted(set(left.lightcone_qubits) & set(right.lightcone_qubits))
    return report


def save_local_unitaries(path: Path, specs: list[BlockSpec]) -> dict[str, float]:
    payload: dict[str, NDArray[np.complex128]] = {}
    errors: dict[str, float] = {}
    for spec in specs:
        unitary = registered_ansatz_unitary_on_lightcone(
            spec.ansatz,
            spec.theta,
            spec.lightcone_qubits,
            spec.block_qubits,
            block_only_ansatz=spec.block_only_ansatz,
        )
        identity = np.eye(unitary.shape[0], dtype=complex)
        errors[f"block_{spec.block_index}"] = float(np.linalg.norm(unitary.conj().T @ unitary - identity))
        payload[f"U_block_{spec.block_index}"] = unitary
        payload[f"lightcone_block_{spec.block_index}"] = np.asarray(spec.lightcone_qubits, dtype=int)
    np.savez(path, **payload)
    return errors


def write_outputs(args: argparse.Namespace, n_qubits: int, specs: list[BlockSpec], metadata: dict[str, Any]) -> None:
    effective_specs = apply_ansatz_override(specs, args.ansatz)
    ordered_specs = resolve_order(effective_specs, args.order)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_timestamp = str(metadata.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S")))
    stem = f"U_sewing_block-sew_{source_timestamp}"
    circuit = build_block_sew_circuit(n_qubits, ordered_specs, args.dagger_convention)

    if args.validate_only:
        print("Validation passed.")
        return

    circuit_path = args.output_dir / f"{stem}.txt"
    report_path = args.output_dir / f"{stem}_report.json"
    circuit_text = str(circuit)
    circuit_path.write_text(circuit_text, encoding="utf-8")

    local_unitary_path = None
    local_unitary_errors = None
    if args.save_local_unitaries:
        local_unitary_path = args.output_dir / f"{stem}_local_unitaries.npz"
        local_unitary_errors = save_local_unitaries(local_unitary_path, effective_specs)

    report = {
        "params": str(args.params),
        "metadata": str(_metadata_path(args.params, args.metadata)),
        "mode": "block-sew",
        "order": args.order,
        "resolved_order": [spec.block_index for spec in ordered_specs],
        "dagger_convention": args.dagger_convention,
        "n_qubits": n_qubits,
        "circuit_qubits": 2 * n_qubits,
        "blocks": [
            {
                "block_index": spec.block_index,
                "block_qubits": list(spec.block_qubits),
                "target_bit": spec.target_bit,
                "lightcone_qubits": list(spec.lightcone_qubits),
                "ansatz": spec.ansatz,
                "block_only_ansatz": spec.block_only_ansatz,
                "theta_size": int(spec.theta.size),
                "best_loss": spec.best_loss,
                "per_bit_losses": spec.per_bit_losses,
            }
            for spec in effective_specs
        ],
        "overlaps": overlap_report(effective_specs),
        "operation_count": sum(1 for _ in circuit.all_operations()),
        "moments": len(circuit),
        "circuit_path": str(circuit_path),
        "report_path": str(report_path),
        "local_unitary_path": str(local_unitary_path) if local_unitary_path is not None else None,
        "local_unitary_errors": local_unitary_errors,
        "dense_full_unitary_supported": False,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Constructed block-sew circuit with {report['circuit_qubits']} qubits")
    print(f"resolved_order = {report['resolved_order']}")
    print(f"operation_count = {report['operation_count']}, moments = {report['moments']}")
    print(f"Circuit saved: {circuit_path}")
    print(f"Report saved: {report_path}")
    if local_unitary_path is not None:
        print(f"Local unitaries saved: {local_unitary_path}")
    if args.print_circuit:
        print(circuit_text)


def main() -> None:
    args = parse_args()
    n_qubits, specs, metadata = load_block_specs(args.params, args.metadata)
    write_outputs(args, n_qubits, specs, metadata)


if __name__ == "__main__":
    main()
