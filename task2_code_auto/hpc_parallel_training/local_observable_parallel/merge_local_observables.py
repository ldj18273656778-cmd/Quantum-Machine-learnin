"""Merge local-observable shard outputs into one JSON and CSV report."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from task2_code_auto.hpc_parallel_training.hpc_block_flow import atomic_write_json
from task2_code_auto.hpc_parallel_training.local_observable_parallel.compare_local_observables import summarize, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge local observable comparison shard JSON files.")
    parser.add_argument("--input-dir", type=Path, required=True, help="directory containing shard JSON files")
    parser.add_argument("--pattern", default="local_observables_*_shard_*.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to input-dir")
    parser.add_argument("--output-prefix", default="local_observables_merged")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"shard must contain a JSON object: {path}")
    return data


def _metadata_key(payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        payload.get("params"),
        payload.get("metadata"),
        payload.get("n_qubits"),
        payload.get("input_state"),
        tuple(payload.get("paulis", [])),
        payload.get("order"),
        payload.get("dagger_convention"),
        payload.get("ansatz"),
        payload.get("max_cone_qubits"),
    )


def _write_coverage_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["qubit", "pauli", "count"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    files = sorted(path for path in args.input_dir.glob(args.pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"no shard JSON files match {args.pattern!r} under {args.input_dir}")

    payloads = [_load_json(path) for path in files]
    expected_key = _metadata_key(payloads[0])
    for path, payload in zip(files, payloads):
        if _metadata_key(payload) != expected_key:
            raise ValueError(f"shard metadata mismatch: {path}")

    rows: list[dict[str, Any]] = []
    for payload in payloads:
        shard_rows = payload.get("rows")
        if not isinstance(shard_rows, list):
            raise ValueError("each shard JSON must contain a rows list")
        rows.extend(dict(row) for row in shard_rows)

    rows.sort(key=lambda row: (int(row["qubit"]), str(row["pauli"])))
    paulis = [str(label) for label in payloads[0].get("paulis", [])]
    seen: dict[tuple[int, str], int] = {}
    for row in rows:
        key = (int(row["qubit"]), str(row["pauli"]))
        seen[key] = seen.get(key, 0) + 1
    duplicates = [key for key, count in seen.items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate qubit/pauli rows across shards: {duplicates[:10]}")

    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.output_prefix}.json"
    csv_path = output_dir / f"{args.output_prefix}.csv"
    coverage_path = output_dir / f"{args.output_prefix}_coverage.csv"
    summary = summarize(rows, paulis)
    qubits = sorted({int(row["qubit"]) for row in rows})
    coverage_rows = [
        {"qubit": qubit, "pauli": pauli, "count": seen.get((qubit, pauli), 0)}
        for qubit in qubits
        for pauli in paulis
    ]
    missing = [row for row in coverage_rows if int(row["count"]) == 0]
    payload = {
        "merged_from": files,
        "shard_count": len(files),
        "n_qubits": payloads[0].get("n_qubits"),
        "input_state": payloads[0].get("input_state"),
        "observable_qubits": qubits,
        "paulis": paulis,
        "row_count": len(rows),
        "missing_count": len(missing),
        "summary": summary,
        "rows": rows,
    }
    atomic_write_json(json_path, payload)
    write_csv(csv_path, rows)
    _write_coverage_csv(coverage_path, coverage_rows)

    print(f"Merged {len(files)} shard files")
    print(f"Saved merged JSON: {json_path}")
    print(f"Saved merged CSV: {csv_path}")
    print(f"Saved coverage CSV: {coverage_path}")
    if missing:
        print(f"missing rows: {len(missing)}")
    for label, stats in summary.items():
        print(
            f"{label}: max_abs_diff={stats['max_abs_diff']:.6g} "
            f"mean_abs_diff={stats['mean_abs_diff']:.6g} rms_abs_diff={stats['rms_abs_diff']:.6g}"
        )


if __name__ == "__main__":
    main()
