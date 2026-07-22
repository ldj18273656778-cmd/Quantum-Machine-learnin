"""Run one group of paired Heisenberg-only and mixed-warmstart trials."""

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
from numpy.typing import NDArray

from task2_code.ansatz_registry import random_ansatz_theta
from task2_code.hpc_parallel_training.hpc_block_flow import PRESETS, atomic_savez, atomic_write_json
from task2_code.loss_registry import loss_function_uses_superoperator, set_active_loss_function
from task2_code.module_e_training import AdamConfig, build_target_objective_context, adam_optimize, sum_block_loss
from task2_code.superoperator_registry import set_active_superop


FloatArray = NDArray[np.float64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one paired warmstart group.")
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
    if data.get("artifact_type") != "paired_warmstart_experiment":
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


def _run_with_history(loss_fn, initial_theta: FloatArray, cfg: AdamConfig, *, show_progress: bool):
    params: list[FloatArray] = []
    losses: list[float] = []

    def capture(_step: int, theta: FloatArray, loss: float) -> None:
        params.append(np.asarray(theta, dtype=float).copy())
        losses.append(float(loss))

    result = adam_optimize(loss_fn, initial_theta, cfg, show_progress=show_progress, step_callback=capture)
    return result, np.asarray(params, dtype=float), np.asarray(losses, dtype=float)


def _array_key(prefix: str, pair_index: int) -> str:
    return f"{prefix}_{pair_index:06d}"


def _run_group(manifest: dict[str, Any], group_index: int, *, show_progress: bool) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    context = _context_from_manifest(manifest)
    heisenberg_stage = dict(manifest["heisenberg_only"])
    warmstart = dict(manifest["warmstart"])
    mixed_stage = dict(warmstart["stage1"])
    warm_heisenberg_stage = dict(warmstart["stage2"])
    fd_eps = float(manifest["fd_eps"])
    heisenberg_cfg = _stage_config(heisenberg_stage, fd_eps)
    mixed_cfg = _stage_config(mixed_stage, fd_eps)
    warm_heisenberg_cfg = _stage_config(warm_heisenberg_stage, fd_eps)
    threshold = float(heisenberg_stage["success_threshold"])

    pairs_per_group = int(manifest["pairs_per_group"])
    group_offset = (group_index - 1) * pairs_per_group
    base_seed = int(manifest["training_seed_start"]) + group_offset

    arrays: dict[str, FloatArray] = {}
    trials: list[dict[str, Any]] = []
    for local_index in range(pairs_per_group):
        pair_index = group_offset + local_index + 1
        seed = base_seed + local_index
        rng = np.random.default_rng(seed)
        initial_theta = np.asarray(random_ansatz_theta(context.ansatz, rng, n_qubits=context.ansatz_qubits), dtype=float)

        _set_loss(heisenberg_stage)
        heisenberg_loss_fn = lambda theta, ctx=context: sum_block_loss(theta, ctx)
        heis_result, heis_params, heis_losses = _run_with_history(
            heisenberg_loss_fn,
            initial_theta,
            heisenberg_cfg,
            show_progress=show_progress,
        )

        _set_loss(mixed_stage)
        mixed_loss_fn = lambda theta, ctx=context: sum_block_loss(theta, ctx)
        mixed_result, mixed_params, mixed_losses = _run_with_history(
            mixed_loss_fn,
            initial_theta,
            mixed_cfg,
            show_progress=show_progress,
        )
        mixed_best_params = np.asarray(mixed_result.best_params, dtype=float)
        mixed_best_params_mixed_loss = float(mixed_loss_fn(mixed_best_params))

        _set_loss(warm_heisenberg_stage)
        warm_heisenberg_loss_fn = lambda theta, ctx=context: sum_block_loss(theta, ctx)
        mixed_best_params_heisenberg_loss = float(warm_heisenberg_loss_fn(mixed_best_params))
        warm_result, warm_params, warm_losses = _run_with_history(
            warm_heisenberg_loss_fn,
            mixed_best_params,
            warm_heisenberg_cfg,
            show_progress=show_progress,
        )

        heis_success = bool((not heis_result.failed) and heis_result.best_loss <= threshold)
        warm_success = bool((not mixed_result.failed) and (not warm_result.failed) and warm_result.best_loss <= threshold)
        arrays[_array_key("initial_params", pair_index)] = initial_theta
        arrays[_array_key("heisenberg_param_history", pair_index)] = heis_params
        arrays[_array_key("heisenberg_loss_history", pair_index)] = heis_losses
        arrays[_array_key("heisenberg_best_params", pair_index)] = np.asarray(heis_result.best_params, dtype=float)
        arrays[_array_key("heisenberg_final_params", pair_index)] = np.asarray(heis_result.final_params, dtype=float)
        arrays[_array_key("mixed_param_history", pair_index)] = mixed_params
        arrays[_array_key("mixed_loss_history", pair_index)] = mixed_losses
        arrays[_array_key("mixed_best_params", pair_index)] = mixed_best_params
        arrays[_array_key("warm_heisenberg_param_history", pair_index)] = warm_params
        arrays[_array_key("warm_heisenberg_loss_history", pair_index)] = warm_losses
        arrays[_array_key("warm_heisenberg_best_params", pair_index)] = np.asarray(warm_result.best_params, dtype=float)
        arrays[_array_key("warm_heisenberg_final_params", pair_index)] = np.asarray(warm_result.final_params, dtype=float)

        trials.append(
            {
                "pair_index": pair_index,
                "seed": seed,
                "arrays_prefix": f"{pair_index:06d}",
                "success_threshold": threshold,
                "heisenberg_only": {
                    "initial_loss": float(heis_losses[0]),
                    "best_loss": float(heis_result.best_loss),
                    "best_iteration": int(heis_result.best_iteration),
                    "final_loss": float(heis_result.loss_history[-1]),
                    "failed": bool(heis_result.failed),
                    "failure_reason": str(heis_result.failure_reason),
                    "success": heis_success,
                    "history_length": int(heis_params.shape[0]),
                },
                "warmstart": {
                    "mixed_initial_loss": float(mixed_losses[0]),
                    "mixed_best_loss": float(mixed_result.best_loss),
                    "mixed_best_iteration": int(mixed_result.best_iteration),
                    "mixed_final_loss": float(mixed_result.loss_history[-1]),
                    "mixed_failed": bool(mixed_result.failed),
                    "mixed_failure_reason": str(mixed_result.failure_reason),
                    "mixed_best_params_mixed_loss": mixed_best_params_mixed_loss,
                    "mixed_best_params_heisenberg_loss": mixed_best_params_heisenberg_loss,
                    "heisenberg_initial_loss": mixed_best_params_heisenberg_loss,
                    "heisenberg_best_loss": float(warm_result.best_loss),
                    "heisenberg_best_iteration": int(warm_result.best_iteration),
                    "heisenberg_final_loss": float(warm_result.loss_history[-1]),
                    "heisenberg_failed": bool(warm_result.failed),
                    "heisenberg_failure_reason": str(warm_result.failure_reason),
                    "success": warm_success,
                    "mixed_history_length": int(mixed_params.shape[0]),
                    "heisenberg_history_length": int(warm_params.shape[0]),
                },
            }
        )

    heis_success_count = sum(1 for trial in trials if trial["heisenberg_only"]["success"])
    warm_success_count = sum(1 for trial in trials if trial["warmstart"]["success"])
    payload = {
        "trials": trials,
        "total_count": len(trials),
        "heisenberg_success_count": heis_success_count,
        "heisenberg_success_rate": float(heis_success_count / len(trials)),
        "warmstart_success_count": warm_success_count,
        "warmstart_success_rate": float(warm_success_count / len(trials)),
    }
    return payload, arrays


def main() -> None:
    args = parse_args()
    manifest = _load_manifest(args.experiment_root)
    group_index = int(args.group_index)
    group_count = int(manifest["group_count"])
    if group_index < 1 or group_index > group_count:
        raise ValueError(f"--group-index must be in 1..{group_count}")

    json_path = args.experiment_root / "groups" / f"group_{group_index:06d}_paired.json"
    npz_path = args.experiment_root / "groups" / f"group_{group_index:06d}_paired_params.npz"
    if (json_path.exists() or npz_path.exists()) and not args.overwrite:
        raise FileExistsError(f"group result already exists; pass --overwrite to replace it: {json_path} / {npz_path}")

    result, arrays = _run_group(manifest, group_index, show_progress=not args.no_progress)
    result["group_index"] = group_index
    result["arrays_file"] = npz_path.name
    atomic_savez(npz_path, **arrays)
    atomic_write_json(json_path, result)
    print(
        f"Wrote {json_path} and {npz_path} "
        + f"(heisenberg_success={result['heisenberg_success_count']}/{result['total_count']}, "
        + f"warmstart_success={result['warmstart_success_count']}/{result['total_count']})"
    )


if __name__ == "__main__":
    main()
