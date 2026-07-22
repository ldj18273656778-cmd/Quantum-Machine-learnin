"""Run one group for the warmstart-iteration success-rate sweep."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from numpy.typing import NDArray

from task2_code.ansatz_registry import random_ansatz_theta
from task2_code.hpc_parallel_training.hpc_block_flow import PRESETS, atomic_savez, atomic_write_json
from task2_code.loss_registry import loss_function_uses_superoperator, set_active_loss_function
from task2_code.module_e_training import AdamConfig, adam_optimize, build_target_objective_context, sum_block_loss
from task2_code.superoperator_registry import set_active_superop


FloatArray = NDArray[np.float64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one success-vs-warmstart-rate group.")
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
    if data.get("artifact_type") != "success_vs_warmstart_rate_experiment":
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
    set_active_loss_function(str(stage["loss_function"]))
    if loss_function_uses_superoperator(str(stage["loss_function"])):
        superoperator = stage.get("superoperator")
        if superoperator is None:
            raise ValueError(f"loss_function {stage['loss_function']} requires a superoperator")
        set_active_superop(str(superoperator))


def _run_stage_with_history(loss_fn, initial_theta: FloatArray, cfg: AdamConfig, *, show_progress: bool):
    params: list[FloatArray] = []
    losses: list[float] = []

    def capture(_step: int, theta: FloatArray, loss: float) -> None:
        params.append(np.asarray(theta, dtype=float).copy())
        losses.append(float(loss))

    result = adam_optimize(loss_fn, initial_theta, cfg, show_progress=show_progress, step_callback=capture)
    return result, np.asarray(params, dtype=float), np.asarray(losses, dtype=float)


def _trial_seed(manifest: dict[str, Any], warm_index: int, trial_index: int) -> int:
    return int(manifest["training_seed_start"]) + warm_index * int(manifest["trials_per_warmstart"]) + trial_index


def _array_key(prefix: str, warm_iterations: int, trial_index: int) -> str:
    return f"{prefix}_warm{warm_iterations:04d}_trial{trial_index:06d}"


def _run_group(manifest: dict[str, Any], group_index: int, *, show_progress: bool) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    context = _context_from_manifest(manifest)
    warmstart_stage = dict(manifest["warmstart_stage"])
    heisenberg_stage = dict(manifest["heisenberg_stage"])
    fd_eps = float(manifest["fd_eps"])
    heisenberg_cfg = AdamConfig(
        iterations=int(heisenberg_stage["iterations"]),
        lr=float(heisenberg_stage["lr"]),
        fd_eps=fd_eps,
    )
    threshold = float(manifest["success_threshold"])
    trials_per_group = int(manifest["trials_per_group_per_warmstart"])
    group_offset = (group_index - 1) * trials_per_group
    warmstart_iterations = [int(value) for value in manifest["warmstart_iterations"]]

    arrays: dict[str, FloatArray] = {}
    trials: list[dict[str, Any]] = []
    for warm_index, warm_iterations in enumerate(warmstart_iterations):
        warm_cfg = AdamConfig(iterations=warm_iterations, lr=float(warmstart_stage["lr"]), fd_eps=fd_eps)
        for local_index in range(trials_per_group):
            trial_index = group_offset + local_index + 1
            seed = _trial_seed(manifest, warm_index, trial_index - 1)
            rng = np.random.default_rng(seed)
            initial_theta = np.asarray(random_ansatz_theta(context.ansatz, rng, n_qubits=context.ansatz_qubits), dtype=float)

            _set_loss(warmstart_stage)
            warm_loss_fn = lambda theta, ctx=context: sum_block_loss(theta, ctx)
            initial_warmstart_loss = float(warm_loss_fn(initial_theta))
            warm_result, warm_params, warm_losses = _run_stage_with_history(
                warm_loss_fn,
                initial_theta,
                warm_cfg,
                show_progress=show_progress,
            )
            warm_best_params = np.asarray(warm_result.best_params, dtype=float)
            warm_final_params = np.asarray(warm_result.final_params, dtype=float)
            warm_final_loss = float(warm_loss_fn(warm_final_params))

            _set_loss(heisenberg_stage)
            heis_loss_fn = lambda theta, ctx=context: sum_block_loss(theta, ctx)
            heisenberg_initial_loss_after_warmstart = float(heis_loss_fn(warm_best_params))
            final_result, heis_params, heis_losses = _run_stage_with_history(
                heis_loss_fn,
                warm_best_params,
                heisenberg_cfg,
                show_progress=show_progress,
            )
            final_best_loss = float(final_result.best_loss)
            final_best_params = np.asarray(final_result.best_params, dtype=float)
            final_params = np.asarray(final_result.final_params, dtype=float)
            final_loss_from_best_params = float(heis_loss_fn(np.asarray(final_result.best_params, dtype=float)))
            key_prefix = f"warm{warm_iterations:04d}_trial{trial_index:06d}"
            trajectory_keys = {
                "initial_params": _array_key("initial_params", warm_iterations, trial_index),
                "warmstart_param_trajectory": _array_key("warmstart_param_trajectory", warm_iterations, trial_index),
                "warmstart_loss_history": _array_key("warmstart_loss_history", warm_iterations, trial_index),
                "warmstart_best_params": _array_key("warmstart_best_params", warm_iterations, trial_index),
                "warmstart_final_params": _array_key("warmstart_final_params", warm_iterations, trial_index),
                "heisenberg_param_trajectory": _array_key("heisenberg_param_trajectory", warm_iterations, trial_index),
                "heisenberg_loss_history": _array_key("heisenberg_loss_history", warm_iterations, trial_index),
                "heisenberg_best_params": _array_key("heisenberg_best_params", warm_iterations, trial_index),
                "heisenberg_final_params": _array_key("heisenberg_final_params", warm_iterations, trial_index),
            }
            arrays[trajectory_keys["initial_params"]] = initial_theta
            arrays[trajectory_keys["warmstart_param_trajectory"]] = warm_params
            arrays[trajectory_keys["warmstart_loss_history"]] = warm_losses
            arrays[trajectory_keys["warmstart_best_params"]] = warm_best_params
            arrays[trajectory_keys["warmstart_final_params"]] = warm_final_params
            arrays[trajectory_keys["heisenberg_param_trajectory"]] = heis_params
            arrays[trajectory_keys["heisenberg_loss_history"]] = heis_losses
            arrays[trajectory_keys["heisenberg_best_params"]] = final_best_params
            arrays[trajectory_keys["heisenberg_final_params"]] = final_params
            trials.append(
                {
                    "warmstart_iterations": warm_iterations,
                    "trial_index": trial_index,
                    "seed": seed,
                    "trajectory_key_prefix": key_prefix,
                    "trajectory_keys": trajectory_keys,
                    "initial_warmstart_loss": initial_warmstart_loss,
                    "warmstart_best_loss": float(warm_result.best_loss),
                    "warmstart_final_loss": warm_final_loss,
                    "warmstart_best_iteration": int(warm_result.best_iteration),
                    "heisenberg_initial_loss_after_warmstart": heisenberg_initial_loss_after_warmstart,
                    "final_best_loss": final_best_loss,
                    "heisenberg_best_iteration": int(final_result.best_iteration),
                    "final_loss_from_best_params": final_loss_from_best_params,
                    "success": bool(final_best_loss < threshold),
                }
            )
    return {"group_index": group_index, "trajectory_npz": f"group_{group_index:06d}_trajectories.npz", "trials": trials}, arrays


def main() -> None:
    args = parse_args()
    manifest = _load_manifest(args.experiment_root)
    group_count = int(manifest["group_count"])
    if args.group_index < 1 or args.group_index > group_count:
        raise ValueError(f"--group-index must be in 1..{group_count}")
    output = args.experiment_root / "groups" / f"group_{args.group_index:06d}.json"
    trajectory_output = args.experiment_root / "groups" / f"group_{args.group_index:06d}_trajectories.npz"
    if output.exists() and not args.overwrite:
        print(f"Skipping existing group result: {output}")
        return
    group, arrays = _run_group(manifest, args.group_index, show_progress=not args.no_progress)
    atomic_savez(trajectory_output, **arrays)
    atomic_write_json(output, group)
    successes = sum(1 for trial in group["trials"] if trial["success"])
    print(f"Wrote {trajectory_output}")
    print(f"Wrote {output}")
    print(f"successes={successes}/{len(group['trials'])}")


if __name__ == "__main__":
    main()
