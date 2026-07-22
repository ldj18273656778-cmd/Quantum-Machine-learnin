"""Run one restart group for a loss-mode success-probability experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from task2_code.ansatz_registry import random_ansatz_theta
from task2_code.hpc_parallel_training.hpc_block_flow import PRESETS, atomic_write_json, safe_stem
from task2_code.loss_registry import loss_function_uses_superoperator, set_active_loss_function
from task2_code.module_e_training import AdamConfig, adam_optimize, build_target_objective_context, sum_block_loss
from task2_code.superoperator_registry import set_active_superop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one loss-success restart group.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--group-index", type=int, required=True, help="1-based group index")
    parser.add_argument("--mode", default=None, help="optional mode label to run only one mode")
    parser.add_argument("--no-progress", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def _load_manifest(experiment_root: Path) -> dict[str, Any]:
    path = experiment_root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    return data


def _select_modes(manifest: dict[str, Any], mode_label: str | None) -> list[dict[str, Any]]:
    modes = [dict(item) for item in manifest.get("modes", [])]
    if not modes:
        raise ValueError("manifest must contain non-empty modes")
    if mode_label is None:
        return modes
    selected = [mode for mode in modes if str(mode["label"]) == mode_label]
    if not selected:
        raise ValueError(f"unknown mode label {mode_label!r}; available: {[mode['label'] for mode in modes]}")
    return selected


def _context_from_manifest(manifest: dict[str, Any]):
    preset = str(manifest["preset"])
    cfg = PRESETS[preset]
    context, _context_meta = build_target_objective_context(
        int(manifest["n_qubits"]),
        [int(q) for q in manifest["block_qubits"]],
        int(manifest["target_bit"]),
        int(manifest["radius"]),
        int(manifest["target_seed"]),
        int(manifest["time_k"]),
        lightcone_mode=str(manifest["lightcone_mode"]),
        loss_mode=str(manifest["loss_mode"]),
        require_unitary=False,
        max_n_qubits=int(manifest["n_qubits"]),
        max_hilbert_dim=4096,
        ansatz=str(manifest.get("ansatz", cfg.ansatz)),
        block_only_ansatz=bool(manifest.get("block_only_ansatz", cfg.block_only_ansatz)),
    )
    return context


def _run_one_mode(
    manifest: dict[str, Any],
    mode: dict[str, Any],
    group_index: int,
    *,
    show_progress: bool,
) -> dict[str, Any]:
    context = _context_from_manifest(manifest)
    loss_function = str(mode["loss_function"])
    set_active_loss_function(loss_function)
    if loss_function_uses_superoperator(loss_function):
        set_active_superop(str(mode["superoperator"]))

    restarts_per_group = int(manifest["restarts_per_group"])
    group_offset = (group_index - 1) * restarts_per_group
    mode_index = int(mode.get("mode_index", 0))
    base_seed = int(manifest["training_seed_start"]) + mode_index * 1_000_000 + group_offset
    threshold = float(mode.get("success_threshold", manifest.get("success_threshold", 0.01)))
    adam_cfg = AdamConfig(
        iterations=int(manifest["iterations"]),
        lr=float(manifest["lr"]),
        fd_eps=float(manifest["fd_eps"]),
    )
    loss_fn = lambda theta, ctx=context: sum_block_loss(theta, ctx)

    trials: list[dict[str, Any]] = []
    for local_index in range(restarts_per_group):
        restart_index = group_offset + local_index + 1
        seed = base_seed + local_index
        rng = np.random.default_rng(seed)
        initial_theta = random_ansatz_theta(context.ansatz, rng, n_qubits=context.ansatz_qubits)
        initial_loss = float(loss_fn(initial_theta))
        result = adam_optimize(loss_fn, initial_theta, adam_cfg, show_progress=show_progress)
        trials.append(
            {
                "restart_index": restart_index,
                "seed": seed,
                "initial_loss": initial_loss,
                "best_loss": float(result.best_loss),
                "best_iteration": int(result.best_iteration),
                "final_loss": float(result.loss_history[-1]),
                "success_threshold": threshold,
                "success": bool((not result.failed) and result.best_loss <= threshold),
                "failed": bool(result.failed),
                "failure_reason": str(result.failure_reason),
                "loss_history": np.asarray(result.loss_history, dtype=float).tolist(),
            }
        )
    success_count = sum(1 for trial in trials if trial["success"])
    return {
        "mode": mode,
        "trials": trials,
        "success_count": success_count,
        "total_count": len(trials),
        "success_rate": float(success_count / len(trials)),
    }


def main() -> None:
    args = parse_args()
    manifest = _load_manifest(args.experiment_root)
    group_index = int(args.group_index)
    group_count = int(manifest["group_count"])
    if group_index < 1 or group_index > group_count:
        raise ValueError(f"--group-index must be in 1..{group_count}")

    modes = _select_modes(manifest, args.mode)
    indexed_modes: list[dict[str, Any]] = []
    all_modes = [dict(item) for item in manifest["modes"]]
    for mode in modes:
        for idx, candidate in enumerate(all_modes):
            if str(candidate["label"]) == str(mode["label"]):
                mode["mode_index"] = idx
                break
        indexed_modes.append(mode)

    label = safe_stem(args.mode) if args.mode else "all_modes"
    output_path = args.experiment_root / "groups" / f"group_{group_index:06d}_{label}.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"group result already exists; pass --overwrite to replace it: {output_path}")

    results = [
        _run_one_mode(manifest, mode, group_index, show_progress=not args.no_progress)
        for mode in indexed_modes
    ]
    payload = {
        "schema_version": 1,
        "artifact_type": "loss_success_group_result",
        "experiment_root": args.experiment_root,
        "group_index": group_index,
        "results": results,
    }
    atomic_write_json(output_path, payload)
    print(f"Saved group result: {output_path}")


if __name__ == "__main__":
    main()
