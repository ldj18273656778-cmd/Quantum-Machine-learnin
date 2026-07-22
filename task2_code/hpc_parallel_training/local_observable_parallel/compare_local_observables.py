"""Shard-friendly local observable comparison for target versus sewing.

This script compares evolved one-qubit observables ``<P_i>`` without building
dense N=32 channels.  Each run may handle all requested qubits or one shard of
them, making it suitable for SLURM array jobs.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
from typing import Any

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cirq
import numpy as np

from task2_code.hpc_parallel_training.hpc_block_flow import atomic_write_json
from task2_code.run_sew_and_compare import PAULIS, _causal_cone_expectation_for_qubit, _safe_stem
from task2_code.sewing.block_sewing import (
    apply_ansatz_override,
    build_block_sew_circuit,
    load_block_specs,
    resolve_order,
)
from task2_code.target_factory import build_target_from_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare target and sewing evolved one-qubit <P_i> by causal-cone simulation."
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="directory containing params.npz and metadata.json")
    parser.add_argument("--params", type=Path, default=None, help="assembled params.npz path")
    parser.add_argument("--metadata", type=Path, default=None, help="assembled metadata.json path")
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to run-dir or params parent")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--input-state", default="zero", help="zero, ones, ghz, or basis:<int>")
    parser.add_argument("--observable-qubits", default=None, help="comma-separated system qubits; defaults to all")
    parser.add_argument("--qubit-start", type=int, default=None, help="inclusive slice start after observable-qubits selection")
    parser.add_argument("--qubit-end", type=int, default=None, help="exclusive slice end after observable-qubits selection")
    parser.add_argument("--num-shards", type=int, default=None, help="total number of qubit shards")
    parser.add_argument("--shard-index", type=int, default=None, help="0-based shard index; use with --num-shards")
    parser.add_argument("--paulis", default="X,Y,Z", help="comma-separated Pauli labels from X,Y,Z")
    parser.add_argument("--order", choices=["odd-even", "metadata", "reverse"], default="odd-even")
    parser.add_argument("--dagger-convention", choices=["trained", "inverse"], default="trained")
    parser.add_argument("--ansatz", default=None, help="ansatz registry key; defaults to metadata/block metadata")
    parser.add_argument("--max-cone-qubits", type=int, default=24)
    return parser.parse_args()


def resolve_paths(run_dir: Path | None, params: Path | None, metadata: Path | None) -> tuple[Path, Path, Path | None]:
    if run_dir is not None:
        if params is None:
            params = run_dir / "params.npz"
        if metadata is None:
            metadata = run_dir / "metadata.json"
    if params is not None and params.is_dir():
        run_dir = params
        params = run_dir / "params.npz"
        if metadata is None:
            metadata = run_dir / "metadata.json"
    if params is None or metadata is None:
        raise ValueError("provide --run-dir, or both --params and --metadata")
    if not params.exists():
        raise FileNotFoundError(f"params not found: {params}")
    if not metadata.exists():
        raise FileNotFoundError(f"metadata not found: {metadata}")
    return params, metadata, run_dir


def parse_qubits(value: str | None, n_qubits: int) -> list[int]:
    if value is None:
        return list(range(n_qubits))
    qubits = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not qubits:
        raise ValueError("--observable-qubits must contain at least one qubit")
    invalid = [q for q in qubits if q < 0 or q >= n_qubits]
    if invalid:
        raise ValueError(f"observable qubits out of range 0..{n_qubits - 1}: {invalid}")
    if len(qubits) != len(set(qubits)):
        raise ValueError(f"observable qubits must not contain duplicates: {qubits}")
    return qubits


def select_shard(qubits: list[int], args: argparse.Namespace) -> tuple[list[int], dict[str, Any]]:
    if (args.num_shards is None) != (args.shard_index is None):
        raise ValueError("--num-shards and --shard-index must be provided together")
    if args.num_shards is not None:
        if args.num_shards <= 0:
            raise ValueError("--num-shards must be positive")
        if args.shard_index < 0 or args.shard_index >= args.num_shards:
            raise ValueError("--shard-index must be in 0..num-shards-1")
        selected = [q for offset, q in enumerate(qubits) if offset % args.num_shards == args.shard_index]
        if not selected:
            raise ValueError(f"shard {args.shard_index}/{args.num_shards} selected no qubits")
        return selected, {"mode": "modulo", "num_shards": args.num_shards, "shard_index": args.shard_index}

    start = 0 if args.qubit_start is None else int(args.qubit_start)
    end = len(qubits) if args.qubit_end is None else int(args.qubit_end)
    if start < 0 or end < start or end > len(qubits):
        raise ValueError(f"invalid qubit slice [{start}:{end}] for {len(qubits)} selected qubits")
    selected = qubits[start:end]
    if not selected:
        raise ValueError(f"qubit slice [{start}:{end}] selected no qubits")
    return selected, {"mode": "slice", "qubit_start": start, "qubit_end": end}


def parse_paulis(value: str) -> list[str]:
    labels = [part.strip().upper() for part in value.split(",") if part.strip()]
    if not labels:
        raise ValueError("--paulis must contain at least one label")
    invalid = [label for label in labels if label not in PAULIS]
    if invalid:
        raise ValueError(f"unsupported Pauli labels {invalid}; expected labels from {sorted(PAULIS)}")
    if len(labels) != len(set(labels)):
        raise ValueError(f"Pauli labels must not contain duplicates: {labels}")
    return labels


def expectations_for_circuit(
    circuit: cirq.Circuit,
    qubits: list[int],
    n_system_qubits: int,
    input_state: str,
    paulis: list[str],
    max_cone_qubits: int,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    values: dict[str, dict[str, float]] = {}
    cone_sizes: dict[str, int] = {}
    for qubit in qubits:
        per_pauli, cone_size = _causal_cone_expectation_for_qubit(
            circuit,
            qubit,
            n_system_qubits,
            input_state,
            max_cone_qubits=max_cone_qubits,
        )
        cone_sizes[str(qubit)] = cone_size
        values[str(qubit)] = {label: float(per_pauli[label]) for label in paulis}
    return values, cone_sizes


def summarize(rows: list[dict[str, Any]], paulis: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pauli in paulis:
        diffs = [float(row["abs_diff"]) for row in rows if row["pauli"] == pauli]
        out[pauli] = {
            "max_abs_diff": max(diffs) if diffs else 0.0,
            "mean_abs_diff": float(np.mean(diffs)) if diffs else 0.0,
            "rms_abs_diff": float(np.sqrt(np.mean(np.square(diffs)))) if diffs else 0.0,
        }
    all_diffs = [float(row["abs_diff"]) for row in rows]
    out["all"] = {
        "max_abs_diff": max(all_diffs) if all_diffs else 0.0,
        "mean_abs_diff": float(np.mean(all_diffs)) if all_diffs else 0.0,
        "rms_abs_diff": float(np.sqrt(np.mean(np.square(all_diffs)))) if all_diffs else 0.0,
    }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "qubit",
        "pauli",
        "target_expectation",
        "sewing_expectation",
        "signed_diff",
        "abs_diff",
        "target_cone_size",
        "sewing_cone_size",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def output_stem(metadata_path: Path, input_state: str, args: argparse.Namespace, shard_meta: dict[str, Any]) -> str:
    if args.output_prefix is not None:
        return str(args.output_prefix)
    base = f"local_observables_{_safe_stem(metadata_path.stem)}_{_safe_stem(input_state)}"
    if shard_meta["mode"] == "modulo":
        return f"{base}_shard_{shard_meta['shard_index']:03d}_of_{shard_meta['num_shards']:03d}"
    if args.qubit_start is not None or args.qubit_end is not None:
        return f"{base}_qubits_{shard_meta['qubit_start']:03d}_{shard_meta['qubit_end']:03d}"
    return base


def main() -> None:
    args = parse_args()
    params_path, metadata_path, run_dir = resolve_paths(args.run_dir, args.params, args.metadata)
    n_qubits, specs, metadata = load_block_specs(params_path, metadata_path)
    specs = apply_ansatz_override(specs, args.ansatz)
    ordered_specs = resolve_order(specs, args.order)
    target = build_target_from_metadata(metadata)
    sewing_circuit = build_block_sew_circuit(
        n_qubits,
        ordered_specs,
        args.dagger_convention,
        ansatz_name=args.ansatz,
    )

    all_qubits = parse_qubits(args.observable_qubits, n_qubits)
    selected_qubits, shard_meta = select_shard(all_qubits, args)
    paulis = parse_paulis(args.paulis)
    target_values, target_cones = expectations_for_circuit(
        target.circuit,
        selected_qubits,
        n_qubits,
        args.input_state,
        paulis,
        args.max_cone_qubits,
    )
    sewing_values, sewing_cones = expectations_for_circuit(
        sewing_circuit,
        selected_qubits,
        n_qubits,
        args.input_state,
        paulis,
        args.max_cone_qubits,
    )

    rows: list[dict[str, Any]] = []
    for qubit in selected_qubits:
        key = str(qubit)
        for pauli in paulis:
            target_exp = float(target_values[key][pauli])
            sewing_exp = float(sewing_values[key][pauli])
            signed_diff = sewing_exp - target_exp
            rows.append(
                {
                    "qubit": qubit,
                    "pauli": pauli,
                    "target_expectation": target_exp,
                    "sewing_expectation": sewing_exp,
                    "signed_diff": signed_diff,
                    "abs_diff": abs(signed_diff),
                    "target_cone_size": target_cones[key],
                    "sewing_cone_size": sewing_cones[key],
                }
            )

    output_dir = args.output_dir or run_dir or params_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(metadata_path, args.input_state, args, shard_meta)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    summary = summarize(rows, paulis)
    payload = {
        "params": params_path,
        "metadata": metadata_path,
        "n_qubits": n_qubits,
        "input_state": args.input_state,
        "observable_qubits": selected_qubits,
        "all_requested_observable_qubits": all_qubits,
        "shard": shard_meta,
        "paulis": paulis,
        "order": args.order,
        "dagger_convention": args.dagger_convention,
        "ansatz": args.ansatz or metadata.get("ansatz"),
        "max_cone_qubits": args.max_cone_qubits,
        "target_cone_sizes": target_cones,
        "sewing_cone_sizes": sewing_cones,
        "summary": summary,
        "rows": rows,
    }
    atomic_write_json(json_path, payload)
    write_csv(csv_path, rows)

    print(f"Saved local observable JSON: {json_path}")
    print(f"Saved local observable CSV: {csv_path}")
    print(f"selected_qubits = {selected_qubits}")
    for label, stats in summary.items():
        print(
            f"{label}: max_abs_diff={stats['max_abs_diff']:.6g} "
            f"mean_abs_diff={stats['mean_abs_diff']:.6g} rms_abs_diff={stats['rms_abs_diff']:.6g}"
        )


if __name__ == "__main__":
    main()
