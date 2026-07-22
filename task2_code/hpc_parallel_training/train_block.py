"""Train one Task 2 block inside a parallel experiment directory.

This script is safe for HPC job arrays: each invocation writes only
``blocks/block_XX`` under the experiment root.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from task2_code.hpc_parallel_training.hpc_block_flow import (
    atomic_savez,
    atomic_write_json,
    block_result_paths,
    config_from_manifest,
    load_manifest,
    parse_block_index,
)
from task2_code.loss_registry import active_loss_breakdown, loss_function_uses_superoperator, set_active_loss_function
from task2_code.module_e_training import (
    AdamConfig,
    adam_optimize,
    build_target_objective_context,
    multi_restart_train,
    residual_operator_for_context,
    sum_block_loss,
)
from task2_code.superoperator_registry import set_active_superop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one block for a parallel Task 2 experiment.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--block-index", type=int, required=True, help="1-based block index")
    parser.add_argument("--init", choices=["random", "warmstart"], default="random")
    parser.add_argument("--warmstart-run-dir", type=Path, default=None, help="run directory containing params.npz for warmstart")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--fd-eps", type=float, default=None)
    parser.add_argument("--restarts", type=int, default=None, help="random-start restarts; ignored for warmstart")
    parser.add_argument("--success-threshold", type=float, default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true", default=False)
    parser.add_argument("--plot-loss", action="store_true", default=False, help="save loss trajectory plot to block directory")
    return parser.parse_args()


def _load_warmstart_theta(run_dir: Path, block_index: int, expected_size: int) -> NDArray[np.float64]:
    params_path = run_dir / "params.npz"
    if not params_path.exists():
        raise FileNotFoundError(f"warmstart params not found: {params_path}")
    key = f"best_params_block_{block_index}"
    with np.load(params_path, allow_pickle=False) as params:
        if key not in params.files:
            raise ValueError(f"warmstart params key {key!r} missing from {params_path}")
        theta = np.asarray(params[key], dtype=float)
    if theta.shape != (expected_size,):
        raise ValueError(f"warmstart {key} has shape {theta.shape}, expected ({expected_size},)")
    return theta


def _restart_histories(result) -> list[list[float]]:
    histories: list[list[float]] = []
    for restart in result.restart_results:
        histories.append(np.asarray(restart.loss_history, dtype=float).tolist())
    return histories


def _make_parameter_recorder() -> tuple[list[list[list[float]]], Any]:
    traces: list[list[list[float]]] = []
    current_trace: list[list[float]] = []

    def record_step(step: int, theta: NDArray[np.float64], _loss: float) -> None:
        nonlocal current_trace
        if step == 0:
            current_trace = []
            traces.append(current_trace)
        current_trace.append(np.asarray(theta, dtype=float).tolist())

    return traces, record_step


def _save_trajectory_plot(path: Path, histories: list[list[float]], selected_restart: int, block_index: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for ri, hist in enumerate(histories):
        label = f"restart {ri + 1}" + (" (best)" if ri + 1 == selected_restart else "")
        alpha = 0.9 if ri + 1 == selected_restart else 0.35
        ax.plot(hist, linewidth=1, alpha=alpha, label=label)
    ax.set_title(f"Block {block_index} loss trajectory")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.experiment_root)
    cfg = config_from_manifest(manifest)
    block_index = parse_block_index(args.block_index, cfg.block_count)

    set_active_loss_function(cfg.loss_function)
    if loss_function_uses_superoperator(cfg.loss_function):
        set_active_superop(cfg.superoperator)

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
    adam_cfg = AdamConfig(
        iterations=int(args.iterations if args.iterations is not None else cfg.iterations),
        lr=float(args.lr if args.lr is not None else cfg.lr),
        fd_eps=float(args.fd_eps if args.fd_eps is not None else manifest.get("fd_eps", 1e-5)),
    )
    success_threshold = float(args.success_threshold if args.success_threshold is not None else cfg.success_threshold)
    loss_fn = lambda theta, ctx=context: sum_block_loss(theta, ctx)

    print(f"experiment = {args.experiment_root}")
    print(f"block_index = {block_index}")
    print(f"block_qubits = {block_qubits}")
    print(f"target_bit = {target_bit}")
    print(f"lightcone = {context.lightcone_qubits}")
    print(f"theta_size = {context.theta_size}")
    print(f"init = {args.init}")

    parameter_histories: list[list[list[float]]]

    if args.init == "warmstart":
        if args.warmstart_run_dir is None:
            raise ValueError("--warmstart-run-dir is required when --init warmstart")
        initial_theta = _load_warmstart_theta(args.warmstart_run_dir, block_index, context.theta_size)
        init_loss = float(loss_fn(initial_theta))
        parameter_histories, record_parameter_step = _make_parameter_recorder()
        result = adam_optimize(
            loss_fn,
            initial_theta,
            adam_cfg,
            show_progress=not args.no_progress,
            step_callback=record_parameter_step,
        )
        histories = [np.asarray(result.loss_history, dtype=float).tolist()]
        selected_restart = 1
        completed_restarts = 1
        init_metadata: dict[str, Any] = {
            "kind": "warmstart",
            "warmstart_run_dir": args.warmstart_run_dir,
            "initial_loss": init_loss,
        }
    else:
        restarts = int(args.restarts if args.restarts is not None else cfg.max_restarts)
        seed = cfg.training_seed_for_block(block_index - 1) + int(args.seed_offset)
        rng = np.random.default_rng(seed)
        parameter_histories, record_parameter_step = _make_parameter_recorder()
        result = multi_restart_train(
            loss_fn,
            restarts,
            rng,
            adam_cfg,
            n_qubits=context.ansatz_qubits,
            ansatz=context.ansatz,
            show_progress=not args.no_progress,
            success_threshold=success_threshold,
            step_callback=record_parameter_step,
        )
        histories = _restart_histories(result)
        selected_restart = int(result.best_restart) + 1
        completed_restarts = len(result.restart_results)
        init_metadata = {"kind": "random", "seed": seed, "restarts": restarts}

    loss_breakdown = active_loss_breakdown(result.best_params, context)
    _residual, loss_qubits = residual_operator_for_context(result.best_params, context)
    best_params_path, result_path = block_result_paths(args.experiment_root, block_index)
    block_dir = best_params_path.parent
    parameter_trajectory_path = block_dir / "parameter_trajectory.npz"
    params_key = f"best_params_block_{block_index}"
    selected_parameter_history = np.asarray(parameter_histories[selected_restart - 1], dtype=float)
    atomic_savez(
        best_params_path,
        best_params=result.best_params,
        loss_history=np.asarray(histories[selected_restart - 1], dtype=float),
        parameter_history=selected_parameter_history,
    )
    atomic_savez(
        parameter_trajectory_path,
        selected_parameter_history=selected_parameter_history,
        **{
            f"restart_{restart_index + 1}_parameter_history": np.asarray(history, dtype=float)
            for restart_index, history in enumerate(parameter_histories)
        },
    )

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
        "selected_restart": selected_restart,
        "completed_restarts": completed_restarts,
        "max_restarts": int(args.restarts if args.restarts is not None else cfg.max_restarts),
        "success_threshold": success_threshold,
        "best_iteration": int(result.best_iteration),
        "best_loss": float(result.best_loss),
        "loss_breakdown": {str(k): float(v) for k, v in loss_breakdown.items()},
        "per_bit_losses": {str(k): float(v) for k, v in loss_breakdown.items()},
        "loss_qubits": list(loss_qubits),
        "lightcone_semantics": context_meta.get("lightcone_semantics"),
        "loss_semantics": context_meta.get("loss_semantics"),
        "init": init_metadata,
        "best_params_file": best_params_path.name,
        "parameter_trajectory_file": parameter_trajectory_path.name,
        "all_restart_loss_histories": histories,
    }
    atomic_write_json(result_path, payload)

    # ── loss trajectory (JSON + optional plot) ──
    trajectory_path = block_dir / "loss_trajectory.json"
    trajectory_payload: dict[str, Any] = {
        "block_index": block_index,
        "block_qubits": list(block_qubits),
        "target_bit": int(target_bit),
        "best_loss": float(result.best_loss),
        "best_iteration": int(result.best_iteration),
        "selected_restart": selected_restart,
        "all_restart_loss_histories": histories,
        "parameter_trajectory_file": parameter_trajectory_path.name,
        "parameter_history_keys": [
            f"restart_{restart_index + 1}_parameter_history"
            for restart_index in range(len(parameter_histories))
        ],
    }
    atomic_write_json(trajectory_path, trajectory_payload)
    print(f"Saved loss trajectory: {trajectory_path}")

    if args.plot_loss:
        png_path = block_dir / "loss_trajectory.png"
        _save_trajectory_plot(png_path, histories, selected_restart, block_index)
        print(f"Saved loss plot: {png_path}")

    print(f"best_loss = {result.best_loss:.12g}")
    print(f"Saved block result: {result_path}")


if __name__ == "__main__":
    main()
