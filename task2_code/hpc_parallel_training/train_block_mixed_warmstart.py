"""Train one block with repeated mixed-edge pretraining then Heisenberg warmstart.

Each warmstart trial starts from a random seed, optimizes the mixed edge-channel
loss, then uses the mixed stage best parameters to initialize Heisenberg Pauli
training.  The best Heisenberg result is saved using the same per-block artifact
contract as ``train_block.py`` so ``assemble_parallel_bundle.py`` can consume it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task2_code.ansatz_registry import random_ansatz_theta
from task2_code.hpc_parallel_training.hpc_block_flow import (
    atomic_savez,
    atomic_write_json,
    block_result_paths,
    config_from_manifest,
    load_manifest,
    parse_block_index,
)
from task2_code.loss_registry import active_loss_breakdown, loss_function_uses_superoperator, set_active_loss_function
from task2_code.module_e_training import AdamConfig, adam_optimize, build_target_objective_context, residual_operator_for_context, sum_block_loss
from task2_code.superoperator_registry import set_active_superop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one block with mixed-edge warmstart Heisenberg trials.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--block-index", type=int, required=True, help="1-based block index")
    parser.add_argument("--warmstart-restarts", type=int, default=5)
    parser.add_argument("--mixed-iterations", type=int, default=None)
    parser.add_argument("--heisenberg-iterations", type=int, default=None)
    parser.add_argument("--mixed-lr", type=float, default=None)
    parser.add_argument("--heisenberg-lr", type=float, default=None)
    parser.add_argument("--fd-eps", type=float, default=None)
    parser.add_argument("--success-threshold", type=float, default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true", default=False)
    parser.add_argument("--plot-loss", action="store_true", default=False, help="save selected Heisenberg loss trajectory plot")
    return parser.parse_args()


def _save_trajectory_plot(path: Path, mixed_history: list[float], heisenberg_history: list[float], block_index: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mixed_history, linewidth=1.2, label="selected mixed pretraining")
    offset = len(mixed_history) - 1
    x_vals = np.arange(len(heisenberg_history), dtype=int) + offset
    ax.plot(x_vals, heisenberg_history, linewidth=1.2, label="selected Heisenberg warmstart")
    ax.set_title(f"Block {block_index} mixed -> Heisenberg warmstart trajectory")
    ax.set_xlabel("combined iteration")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _set_loss(loss_function: str, superoperator: str | None) -> None:
    set_active_loss_function(loss_function)
    if loss_function_uses_superoperator(loss_function):
        if superoperator is None:
            raise ValueError(f"loss function {loss_function!r} requires a superoperator")
        set_active_superop(superoperator)


def _make_parameter_recorder() -> tuple[list[list[float]], Any]:
    trace: list[list[float]] = []

    def record_step(_step: int, theta: np.ndarray[Any, Any], _loss: float) -> None:
        trace.append(np.asarray(theta, dtype=float).tolist())

    return trace, record_step


def main() -> None:
    args = parse_args()
    if args.warmstart_restarts <= 0:
        raise ValueError("--warmstart-restarts must be positive")

    manifest = load_manifest(args.experiment_root)
    cfg = config_from_manifest(manifest)
    block_index = parse_block_index(args.block_index, cfg.block_count)
    block_qubits = cfg.blocks[block_index - 1]
    target_bit = cfg.target_bits[block_index - 1]
    context, context_meta = build_target_objective_context(
        cfg.n_qubits,
        block_qubits,
        target_bit,
        cfg.radius,
        cfg.target_seed,
        cfg.time_k,
        lightcone_mode=cfg.lightcone_mode,
        loss_mode=cfg.loss_mode,
        require_unitary=False,
        max_n_qubits=cfg.n_qubits,
        max_hilbert_dim=4096,
        ansatz=cfg.ansatz,
        block_only_ansatz=cfg.block_only_ansatz,
    )

    fd_eps = float(args.fd_eps if args.fd_eps is not None else manifest.get("fd_eps", 1e-5))
    mixed_cfg = AdamConfig(
        iterations=int(args.mixed_iterations if args.mixed_iterations is not None else cfg.iterations),
        lr=float(args.mixed_lr if args.mixed_lr is not None else cfg.lr),
        fd_eps=fd_eps,
    )
    heisenberg_cfg = AdamConfig(
        iterations=int(args.heisenberg_iterations if args.heisenberg_iterations is not None else cfg.iterations),
        lr=float(args.heisenberg_lr if args.heisenberg_lr is not None else cfg.lr),
        fd_eps=fd_eps,
    )
    success_threshold = float(args.success_threshold if args.success_threshold is not None else cfg.success_threshold)

    mixed_loss_function = "edge_quantum_channel"
    mixed_superoperator = "superoperator_from_mix"
    heisenberg_loss_function = "heisenberg_pauli"
    seed = cfg.training_seed_for_block(block_index - 1) + int(args.seed_offset)
    rng = np.random.default_rng(seed)

    print(f"experiment = {args.experiment_root}")
    print(f"block_index = {block_index}")
    print(f"block_qubits = {block_qubits}")
    print(f"target_bit = {target_bit}")
    print(f"lightcone = {context.lightcone_qubits}")
    print(f"theta_size = {context.theta_size}")
    print("training_strategy = mixed_edge_then_heisenberg_warmstart")
    print(f"warmstart_restarts = {args.warmstart_restarts}")

    trials: list[dict[str, Any]] = []
    best_trial_index = 0
    best_heisenberg_loss = float("inf")
    best_params: np.ndarray[Any, Any] | None = None
    selected_mixed_history: list[float] = []
    selected_heisenberg_history: list[float] = []
    selected_mixed_parameter_history: list[list[float]] = []
    selected_heisenberg_parameter_history: list[list[float]] = []
    all_mixed_parameter_histories: list[list[list[float]]] = []
    all_heisenberg_parameter_histories: list[list[list[float]]] = []

    for trial_index in range(int(args.warmstart_restarts)):
        initial_theta = random_ansatz_theta(context.ansatz, rng, n_qubits=context.ansatz_qubits)

        _set_loss(mixed_loss_function, mixed_superoperator)
        mixed_loss_fn = lambda theta, ctx=context: sum_block_loss(theta, ctx)
        mixed_initial_loss = float(mixed_loss_fn(initial_theta))
        mixed_parameter_history, record_mixed_parameter_step = _make_parameter_recorder()
        mixed_result = adam_optimize(
            mixed_loss_fn,
            initial_theta,
            mixed_cfg,
            show_progress=not args.no_progress,
            step_callback=record_mixed_parameter_step,
        )
        mixed_best_params = np.asarray(mixed_result.best_params, dtype=float)
        mixed_best_params_mixed_edge_loss = float(mixed_loss_fn(mixed_best_params))

        _set_loss(heisenberg_loss_function, None)
        heisenberg_loss_fn = lambda theta, ctx=context: sum_block_loss(theta, ctx)
        mixed_best_params_heisenberg_loss = float(heisenberg_loss_fn(mixed_best_params))
        heisenberg_parameter_history, record_heisenberg_parameter_step = _make_parameter_recorder()
        heisenberg_result = adam_optimize(
            heisenberg_loss_fn,
            mixed_best_params,
            heisenberg_cfg,
            show_progress=not args.no_progress,
            step_callback=record_heisenberg_parameter_step,
        )
        success = bool((not mixed_result.failed) and (not heisenberg_result.failed) and heisenberg_result.best_loss <= success_threshold)
        all_mixed_parameter_histories.append(mixed_parameter_history)
        all_heisenberg_parameter_histories.append(heisenberg_parameter_history)

        trial_payload: dict[str, Any] = {
            "trial_index": trial_index + 1,
            "mixed_initial_loss": mixed_initial_loss,
            "mixed_best_loss": float(mixed_result.best_loss),
            "mixed_best_iteration": int(mixed_result.best_iteration),
            "mixed_final_loss": float(mixed_result.loss_history[-1]),
            "mixed_failed": bool(mixed_result.failed),
            "mixed_failure_reason": str(mixed_result.failure_reason),
            "mixed_best_params_mixed_edge_loss": mixed_best_params_mixed_edge_loss,
            "mixed_best_params_heisenberg_loss": mixed_best_params_heisenberg_loss,
            "heisenberg_initial_loss": mixed_best_params_heisenberg_loss,
            "heisenberg_best_loss": float(heisenberg_result.best_loss),
            "heisenberg_best_iteration": int(heisenberg_result.best_iteration),
            "heisenberg_final_loss": float(heisenberg_result.loss_history[-1]),
            "heisenberg_failed": bool(heisenberg_result.failed),
            "heisenberg_failure_reason": str(heisenberg_result.failure_reason),
            "success": success,
            "mixed_loss_history": np.asarray(mixed_result.loss_history, dtype=float).tolist(),
            "heisenberg_loss_history": np.asarray(heisenberg_result.loss_history, dtype=float).tolist(),
        }
        trials.append(trial_payload)

        if float(heisenberg_result.best_loss) < best_heisenberg_loss:
            best_heisenberg_loss = float(heisenberg_result.best_loss)
            best_trial_index = trial_index + 1
            best_params = np.asarray(heisenberg_result.best_params, dtype=float)
            selected_mixed_history = trial_payload["mixed_loss_history"]
            selected_heisenberg_history = trial_payload["heisenberg_loss_history"]
            selected_mixed_parameter_history = mixed_parameter_history
            selected_heisenberg_parameter_history = heisenberg_parameter_history

        if success:
            if not args.no_progress and trial_index + 1 < int(args.warmstart_restarts):
                print(f"  reached success_threshold={success_threshold:.6g}; stopping warmstart trials")
            break

    if best_params is None:
        raise RuntimeError("no warmstart trials completed")

    _set_loss(heisenberg_loss_function, None)
    heisenberg_loss_fn = lambda theta, ctx=context: sum_block_loss(theta, ctx)
    loss_breakdown = active_loss_breakdown(best_params, context)
    _residual, loss_qubits = residual_operator_for_context(best_params, context)
    best_params_path, result_path = block_result_paths(args.experiment_root, block_index)
    block_dir = best_params_path.parent
    parameter_trajectory_path = block_dir / "parameter_trajectory.npz"
    params_key = f"best_params_block_{block_index}"
    atomic_savez(
        best_params_path,
        best_params=best_params,
        loss_history=np.asarray(selected_heisenberg_history, dtype=float),
        mixed_loss_history=np.asarray(selected_mixed_history, dtype=float),
        parameter_history=np.asarray(selected_heisenberg_parameter_history, dtype=float),
        mixed_parameter_history=np.asarray(selected_mixed_parameter_history, dtype=float),
    )
    atomic_savez(
        parameter_trajectory_path,
        selected_heisenberg_parameter_history=np.asarray(selected_heisenberg_parameter_history, dtype=float),
        selected_mixed_parameter_history=np.asarray(selected_mixed_parameter_history, dtype=float),
        **{
            f"trial_{trial_index + 1}_mixed_parameter_history": np.asarray(history, dtype=float)
            for trial_index, history in enumerate(all_mixed_parameter_histories)
        },
        **{
            f"trial_{trial_index + 1}_heisenberg_parameter_history": np.asarray(history, dtype=float)
            for trial_index, history in enumerate(all_heisenberg_parameter_histories)
        },
    )

    histories = [trial["heisenberg_loss_history"] for trial in trials]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "parallel_block_result",
        "experiment_root": args.experiment_root,
        "block_index": block_index,
        "block_qubits": list(block_qubits),
        "target_bit": int(target_bit),
        "lightcone_qubits": list(context.lightcone_qubits),
        "ansatz_qubits": context.ansatz_qubits,
        "ansatz": context.ansatz,
        "block_only_ansatz": context.block_only_ansatz,
        "theta_size": context.theta_size,
        "params_key": params_key,
        "selected_restart": best_trial_index,
        "completed_restarts": len(trials),
        "max_restarts": int(args.warmstart_restarts),
        "success_threshold": success_threshold,
        "best_iteration": int(trials[best_trial_index - 1]["heisenberg_best_iteration"]),
        "best_loss": float(heisenberg_loss_fn(best_params)),
        "loss_breakdown": {str(k): float(v) for k, v in loss_breakdown.items()},
        "per_bit_losses": {str(k): float(v) for k, v in loss_breakdown.items()},
        "loss_qubits": list(loss_qubits),
        "lightcone_semantics": context_meta.get("lightcone_semantics"),
        "loss_semantics": context_meta.get("loss_semantics"),
        "init": {
            "kind": "mixed_edge_then_heisenberg_warmstart",
            "seed": seed,
            "warmstart_restarts": int(args.warmstart_restarts),
            "mixed_loss_function": mixed_loss_function,
            "mixed_superoperator": mixed_superoperator,
            "heisenberg_loss_function": heisenberg_loss_function,
            "mixed_iterations": mixed_cfg.iterations,
            "heisenberg_iterations": heisenberg_cfg.iterations,
            "mixed_lr": mixed_cfg.lr,
            "heisenberg_lr": heisenberg_cfg.lr,
            "trials": trials,
        },
        "best_params_file": best_params_path.name,
        "parameter_trajectory_file": parameter_trajectory_path.name,
        "all_restart_loss_histories": histories,
    }
    atomic_write_json(result_path, payload)

    trajectory_path = block_dir / "loss_trajectory.json"
    atomic_write_json(
        trajectory_path,
        {
            "block_index": block_index,
            "block_qubits": list(block_qubits),
            "target_bit": int(target_bit),
            "best_loss": payload["best_loss"],
            "best_iteration": payload["best_iteration"],
            "selected_restart": best_trial_index,
            "selected_mixed_loss_history": selected_mixed_history,
            "selected_heisenberg_loss_history": selected_heisenberg_history,
            "all_restart_loss_histories": histories,
            "parameter_trajectory_file": parameter_trajectory_path.name,
            "mixed_parameter_history_keys": [
                f"trial_{trial_index + 1}_mixed_parameter_history"
                for trial_index in range(len(all_mixed_parameter_histories))
            ],
            "heisenberg_parameter_history_keys": [
                f"trial_{trial_index + 1}_heisenberg_parameter_history"
                for trial_index in range(len(all_heisenberg_parameter_histories))
            ],
        },
    )
    print(f"Saved loss trajectory: {trajectory_path}")
    if args.plot_loss:
        png_path = block_dir / "loss_trajectory.png"
        _save_trajectory_plot(png_path, selected_mixed_history, selected_heisenberg_history, block_index)
        print(f"Saved loss plot: {png_path}")

    print(f"Saved block params: {best_params_path}")
    print(f"Saved block result: {result_path}")
    print(f"best Heisenberg loss = {payload['best_loss']:.6e}")
    print(f"selected warmstart trial = {best_trial_index}")


if __name__ == "__main__":
    main()
