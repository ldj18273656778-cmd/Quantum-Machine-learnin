"""Re-evaluate saved Task 2 runs under mix/zero/one superoperator losses.

Examples:
    python task2_code/diagnose_superoperator_losses.py --run-dir task2_code/data/task2_training_n12_3blocks_zero_superoperator_from_zero_20260528_145828
    python task2_code/diagnose_superoperator_losses.py --run-dir task2_code/data/task2_training_n12_3blocks_* --output task2_code/data/superoperator_loss_diagnostic.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from task2_code.module_e_training import build_target_objective_context, residual_operator_for_context
from task2_code.sewing.block_sewing import load_block_specs
from task2_code.superoperator_registry import resolve_superop

LOSS_MODES = ("superoperator_from_mix", "superoperator_from_zero", "superoperator_from_one")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved params under mix/zero/one per-bit superoperator losses.")
    parser.add_argument("--run-dir", type=str, nargs="+", required=True, help="training run directory, params.npz file, or glob pattern")
    parser.add_argument("--output", type=Path, default=None, help="optional JSON output path")
    return parser.parse_args()


def _metadata_path(params_path: Path) -> Path:
    if params_path.name == "params.npz":
        return params_path.with_name("metadata.json")
    if params_path.name.startswith("task2_training_params_"):
        metadata_name = params_path.name.replace("task2_training_params_", "task2_training_metadata_", 1)
        candidate = params_path.with_name(metadata_name).with_suffix(".json")
        if candidate.exists():
            return candidate
    return params_path.with_suffix(".json")


def _resolve_artifact(value: str) -> list[tuple[Path, Path, Path | None]]:
    matches = [Path(path) for path in glob.glob(value)] or [Path(value)]
    artifacts: list[tuple[Path, Path, Path | None]] = []
    for path in matches:
        if path.is_dir():
            params_path = path / "params.npz"
            metadata_path = path / "metadata.json"
            run_dir = path
        else:
            params_path = path
            metadata_path = _metadata_path(path)
            run_dir = path.parent if path.name == "params.npz" else None
        artifacts.append((params_path, metadata_path, run_dir))
    return artifacts


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


def evaluate_artifact(params_path: Path, metadata_path: Path, run_dir: Path | None) -> dict[str, Any]:
    n_qubits, specs, metadata = load_block_specs(params_path, metadata_path)
    radius = int(metadata.get("radius", 2))
    target_seed = int(metadata["target_seed"])
    time_k = int(metadata["time_k"])
    lightcone_mode = str(metadata.get("lightcone_mode", "circuit"))
    loss_mode = str(metadata.get("loss_mode", "lightcone"))

    block_results: list[dict[str, Any]] = []
    totals = {name: 0.0 for name in LOSS_MODES}
    max_per_bit = {name: 0.0 for name in LOSS_MODES}

    for spec in specs:
        context, _ = build_target_objective_context(
            n_qubits,
            spec.block_qubits,
            spec.target_bit,
            radius,
            target_seed,
            time_k,
            lightcone_mode=lightcone_mode,
            loss_mode=loss_mode,
            require_unitary=False,
            max_n_qubits=n_qubits,
            max_hilbert_dim=4096,
            ansatz=spec.ansatz,
            block_only_ansatz=spec.block_only_ansatz,
        )
        residual, loss_qubits = residual_operator_for_context(spec.theta, context)
        mode_results: dict[str, Any] = {}
        for mode in LOSS_MODES:
            losses = resolve_superop(mode)(residual, context.block_qubits, loss_qubits, target_bits=None)
            loss_sum = float(sum(losses.values()))
            loss_max = float(max(losses.values()))
            totals[mode] += loss_sum
            max_per_bit[mode] = max(max_per_bit[mode], loss_max)
            mode_results[mode] = {
                "sum": loss_sum,
                "max": loss_max,
                "per_bit": {str(k): float(v) for k, v in losses.items()},
            }
        block_results.append({
            "block_index": spec.block_index,
            "block_qubits": list(spec.block_qubits),
            "ansatz": spec.ansatz,
            "block_only_ansatz": spec.block_only_ansatz,
            "trained_best_loss": spec.best_loss,
            "trained_loss_function": metadata.get("loss_function", "edge_quantum_channel"),
            "trained_superoperator": metadata.get("superoperator"),
            "losses": mode_results,
        })

    return {
        "run_dir": run_dir,
        "params": params_path,
        "metadata": metadata_path,
        "preset": metadata.get("preset"),
        "trained_loss_function": metadata.get("loss_function", "edge_quantum_channel"),
        "trained_superoperator": metadata.get("superoperator"),
        "n_qubits": n_qubits,
        "target_seed": target_seed,
        "time_k": time_k,
        "totals": totals,
        "max_per_bit": max_per_bit,
        "blocks": block_results,
    }


def print_summary(results: list[dict[str, Any]]) -> None:
    for item in results:
        label = item.get("preset") or item["params"]
        print(f"\n{label}")
        print(f"  trained_superoperator = {item['trained_superoperator']}")
        print(f"  target_seed = {item['target_seed']}  time_k = {item['time_k']}")
        for mode in LOSS_MODES:
            print(
                f"  {mode}: total={item['totals'][mode]:.12g}  "
                f"max_per_bit={item['max_per_bit'][mode]:.12g}"
            )


def main() -> None:
    args = parse_args()
    artifacts: list[tuple[Path, Path, Path | None]] = []
    for value in args.run_dir:
        artifacts.extend(_resolve_artifact(value))
    results = [evaluate_artifact(params_path, metadata_path, run_dir) for params_path, metadata_path, run_dir in artifacts]
    print_summary(results)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(_json_ready(results), indent=2), encoding="utf-8")
        print(f"\nSaved diagnostic JSON: {args.output}")


if __name__ == "__main__":
    main()
