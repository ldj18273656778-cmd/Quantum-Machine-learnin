"""Run one group of mixed-edge to Heisenberg warmstart trials."""

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
from task2_code.hpc_parallel_training.hpc_block_flow import PRESETS, atomic_write_json
from task2_code.loss_registry import loss_function_uses_superoperator, set_active_loss_function
from task2_code.module_e_training import AdamConfig, adam_optimize, build_target_objective_context, sum_block_loss
from task2_code.superoperator_registry import set_active_superop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one mixed-warmstart restart group.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--group-index", type=int, required=True, help="1-based group index")
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
    if data.get("artifact_type") != "mixed_warmstart_experiment":
        raise ValueError(f"unexpected artifact_type in manifest: {data.get('artifact_type')!r}")
    return data


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


def _set_loss(stage: dict[str, Any]) -> None:
    loss_function = str(stage["loss_function"])
    set_active_loss_function(loss_function)
    if loss_function_uses_superoperator(loss_function):
        set_active_superop(str(stage["superoperator"]))


def _stage_config(stage: dict[str, Any], fd_eps: float) -> AdamConfig:
    return AdamConfig(
        iterations=int(stage["iterations"]),
        lr=float(stage["lr"]),
        fd_eps=float(fd_eps),
    )


def _run_group(manifest: dict[str, Any], group_index: int, *, show_progress: bool) -> dict[str, Any]:
    context = _context_from_manifest(manifest)
    stage1 = dict(manifest["stage1"])
    stage2 = dict(manifest["stage2"])
    fd_eps = float(manifest["fd_eps"])
    mixed_cfg = _stage_config(stage1, fd_eps)
    heisenberg_cfg = _stage_config(stage2, fd_eps)
    threshold = float(stage2["success_threshold"])

    restarts_per_group = int(manifest["restarts_per_group"])
    group_offset = (group_index - 1) * restarts_per_group
    base_seed = int(manifest["training_seed_start"]) + group_offset

    trials: list[dict[str, Any]] = []
    for local_index in range(restarts_per_group):
        restart_index = group_offset + local_index + 1
        seed = base_seed + local_index
        rng = np.random.default_rng(seed)
        initial_theta = random_ansatz_theta(context.ansatz, rng, n_qubits=context.ansatz_qubits)

        _set_loss(stage1)
        mixed_loss_fn = lambda theta, ctx=context: sum_block_loss(theta, ctx)
        mixed_initial_loss = float(mixed_loss_fn(initial_theta))
        mixed_result = adam_optimize(mixed_loss_fn, initial_theta, mixed_cfg, show_progress=show_progress)
        mixed_best_params = np.asarray(mixed_result.best_params, dtype=float)
        mixed_best_params_mixed_edge_loss = float(mixed_loss_fn(mixed_best_params))

        _set_loss(stage2)
        heisenberg_loss_fn = lambda theta, ctx=context: sum_block_loss(theta, ctx)
        mixed_best_params_heisenberg_loss = float(heisenberg_loss_fn(mixed_best_params))
        heisenberg_result = adam_optimize(heisenberg_loss_fn, mixed_best_params, heisenberg_cfg, show_progress=show_progress)

        success = bool(
            (not mixed_result.failed)
            and (not heisenberg_result.failed)
            and heisenberg_result.best_loss <= threshold
        )
        trials.append(
            {
                "restart_index": restart_index,
                "seed": seed,
                "initial_params": np.asarray(initial_theta, dtype=float).tolist(),
                "mixed_initial_loss": mixed_initial_loss,
                "mixed_best_loss": float(mixed_result.best_loss),
                "mixed_best_iteration": int(mixed_result.best_iteration),
                "mixed_final_loss": float(mixed_result.loss_history[-1]),
                "mixed_failed": bool(mixed_result.failed),
                "mixed_failure_reason": str(mixed_result.failure_reason),
                "mixed_best_params": mixed_best_params.tolist(),
                "mixed_best_params_mixed_edge_loss": mixed_best_params_mixed_edge_loss,
                "mixed_best_params_heisenberg_loss": mixed_best_params_heisenberg_loss,
                "heisenberg_initial_loss": mixed_best_params_heisenberg_loss,
                "heisenberg_best_loss": float(heisenberg_result.best_loss),
                "heisenberg_best_iteration": int(heisenberg_result.best_iteration),
                "heisenberg_final_loss": float(heisenberg_result.loss_history[-1]),
                "heisenberg_failed": bool(heisenberg_result.failed),
                "heisenberg_failure_reason": str(heisenberg_result.failure_reason),
                "success_threshold": threshold,
                "success": success,
                "mixed_loss_history": np.asarray(mixed_result.loss_history, dtype=float).tolist(),
                "heisenberg_loss_history": np.asarray(heisenberg_result.loss_history, dtype=float).tolist(),
            }
        )

    success_count = sum(1 for trial in trials if trial["success"])
    return {
        "stage1": stage1,
        "stage2": stage2,
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

    output_path = args.experiment_root / "groups" / f"group_{group_index:06d}_mixed_warmstart.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"group result already exists; pass --overwrite to replace it: {output_path}")

    result = _run_group(manifest, group_index, show_progress=not args.no_progress)
    payload = {
        "schema_version": 1,
        "artifact_type": "mixed_warmstart_group_result",
        "experiment_root": args.experiment_root,
        "group_index": group_index,
        "result": result,
    }
    atomic_write_json(output_path, payload)
    print(f"Saved group result: {output_path}")


if __name__ == "__main__":
    main()
