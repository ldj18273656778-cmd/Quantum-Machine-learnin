"""Train Task 2 local-inversion blocks and save sewing-ready artifacts.

Examples:
    python task2_code/run_training_pipeline.py --preset n12_3blocks
    python task2_code/run_training_pipeline.py --preset n12_3blocks_zero
    python task2_code/run_sew_and_compare.py --run-dir task2_code/data/task2_training_n12_3blocks_zero_superoperator_from_zero_...
    python task2_code/run_training_pipeline.py --n-qubits 4 --blocks 0,1,2,3 --target-bits 1 --iterations 1 --max-restarts 1
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from task2_code.experiment_config import ExperimentConfig, N4_SINGLE_BLOCK, N4_SINGLE_BLOCK_HEISENBERG, N12_3BLOCKS, N12_3BLOCKS_CNOT_MIXED, N12_3BLOCKS_HEISENBERG, N12_3BLOCKS_ZERO, N20_5BLOCKS, N20_5BLOCKS_HEISENBERG, N24_6BLOCKS, N24_6BLOCKS_HEISENBERG, N28_7BLOCKS, N28_7BLOCKS_HEISENBERG, N32_8BLOCKS, N32_8BLOCKS_HEISENBERG, N12_3BLOCKS_ONE
from task2_code.module_e_training import (
    AdamConfig,
    build_target_objective_context,
    multi_restart_train,
    residual_operator_for_context,
    sum_block_loss,
)
from task2_code.loss_registry import active_loss_breakdown, get_active_loss_function, loss_function_uses_superoperator, set_active_loss_function
from task2_code.superoperator_registry import get_active_superop, set_active_superop
from task2_code.target_factory import build_target_from_seed, target_metadata

PRESETS = {
    "n4_single_block": N4_SINGLE_BLOCK,
    "n4_single_block_heisenberg": N4_SINGLE_BLOCK_HEISENBERG,
    "n12_3blocks": N12_3BLOCKS,
    "n12_3blocks_cnot_mixed": N12_3BLOCKS_CNOT_MIXED,
    "n12_3blocks_heisenberg": N12_3BLOCKS_HEISENBERG,
    "n12_3blocks_zero": N12_3BLOCKS_ZERO,
    "n12_3blocks_one": N12_3BLOCKS_ONE,
    "n20_5blocks": N20_5BLOCKS,
    "n20_5blocks_heisenberg": N20_5BLOCKS_HEISENBERG,
    "n24_6blocks": N24_6BLOCKS,
    "n24_6blocks_heisenberg": N24_6BLOCKS_HEISENBERG,
    "n28_7blocks": N28_7BLOCKS,
    "n28_7blocks_heisenberg": N28_7BLOCKS_HEISENBERG,
    "n32_8blocks": N32_8BLOCKS,
    "n32_8blocks_heisenberg": N32_8BLOCKS_HEISENBERG,
}


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def _run_dir_name(preset: str, ansatz: str, loss_function: str, superoperator: str | None, timestamp: str) -> str:
    parts = ["task2_training", _safe_stem(preset), _safe_stem(ansatz), _safe_stem(loss_function)]
    if superoperator is not None:
        parts.append(_safe_stem(superoperator))
    parts.append(timestamp)
    return "_".join(parts)


def _parse_blocks(value: str) -> list[list[int]]:
    blocks: list[list[int]] = []
    for chunk in value.split(";"):
        items = [int(part.strip()) for part in chunk.split(",") if part.strip()]
        if items:
            blocks.append(items)
    if not blocks:
        raise argparse.ArgumentTypeError("blocks must contain at least one block")
    return blocks


def _parse_int_list(value: str) -> list[int]:
    items = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not items:
        raise argparse.ArgumentTypeError("target-bits must contain at least one integer")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Task 2 blocks and save sewing-ready params/metadata.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="n12_3blocks_cnot_mixed")
    parser.add_argument("--n-qubits", type=int, default=None)
    parser.add_argument("--blocks", type=_parse_blocks, default=None, help="semicolon-separated blocks, e.g. 0,1,2,3;4,5,6,7")
    parser.add_argument("--target-bits", type=_parse_int_list, default=None, help="comma-separated target bits")
    parser.add_argument("--target-seed", type=int, default=None)
    parser.add_argument("--training-seed-start", type=int, default=None)
    parser.add_argument("--time-k", type=int, default=None)
    parser.add_argument("--radius", type=int, default=None)
    parser.add_argument("--lightcone-mode", choices=["circuit", "radius"], default=None)
    parser.add_argument("--loss-mode", choices=["lightcone", "full_system"], default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--fd-eps", type=float, default=1e-5)
    parser.add_argument("--max-restarts", type=int, default=None)
    parser.add_argument("--success-threshold", type=float, default=None, help="stop remaining restarts once best loss reaches this value")
    parser.add_argument("--ansatz", default=None)
    parser.add_argument("--block-only-ansatz", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--loss-function", default=None, help="scalar training objective registry key")
    parser.add_argument("--superoperator", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show ADAM iteration progress output (default: true)",
    )
    parser.add_argument(
        "--plot-loss",
        action="store_true",
        default=True,
        help="save per-block loss trajectory plot to the run directory",
    )
    return parser.parse_args()


def _resolved_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = PRESETS[args.preset]
    updates: dict[str, Any] = {}
    for arg_name, field_name in [
        ("n_qubits", "n_qubits"),
        ("blocks", "blocks"),
        ("target_bits", "target_bits"),
        ("target_seed", "target_seed"),
        ("training_seed_start", "training_seed_start"),
        ("time_k", "time_k"),
        ("radius", "radius"),
        ("lightcone_mode", "lightcone_mode"),
        ("loss_mode", "loss_mode"),
        ("iterations", "iterations"),
        ("lr", "lr"),
        ("max_restarts", "max_restarts"),
        ("success_threshold", "success_threshold"),
        ("ansatz", "ansatz"),
        ("block_only_ansatz", "block_only_ansatz"),
        ("loss_function", "loss_function"),
        ("superoperator", "superoperator"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            updates[field_name] = value
    if args.output_dir is not None:
        updates["data_dir"] = args.output_dir
    cfg = replace(cfg, **updates)
    if len(cfg.blocks) != len(cfg.target_bits):
        raise ValueError("number of blocks must match number of target bits")
    return cfg


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


def _save_loss_plot(
    run_dir: Path,
    loss_histories: list[NDArray[np.float64]],
    block_metadata: list[dict[str, Any]],
    cfg: ExperimentConfig,
    timestamp: str,
) -> None:
    mode_label = cfg.loss_function
    if loss_function_uses_superoperator(cfg.loss_function):
        mode_label = f"{mode_label} / {cfg.superoperator}"
    n_blocks = len(loss_histories)
    n_rows = (n_blocks + 2) // 3
    n_cols = min(n_blocks, 3)
    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    axes = np.asarray(axes_grid).ravel()
    for bi in range(n_blocks):
        ax = axes[bi]
        steps = list(range(len(loss_histories[bi])))
        ax.plot(steps, loss_histories[bi], linewidth=1)
        best_loss = block_metadata[bi]["best_loss"]
        ax.set_title(f"Block {bi + 1}  best loss={best_loss:.4f}")
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
    for bi in range(n_blocks, len(axes)):
        axes[bi].set_visible(False)
    fig.suptitle(
        f"Training loss  n={cfg.n_qubits}  {mode_label}  "
        f"restarts up to {cfg.max_restarts}  {timestamp}",
        fontsize=14,
    )
    fig.tight_layout()
    plot_path = run_dir / "loss_trajectory.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss plot: {plot_path}")


def main() -> None:
    args = parse_args()
    cfg = _resolved_config(args)
    set_active_loss_function(cfg.loss_function)
    uses_superoperator = loss_function_uses_superoperator(cfg.loss_function)
    if uses_superoperator:
        set_active_superop(cfg.superoperator)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.data_dir / _run_dir_name(
        args.preset,
        cfg.ansatz,
        cfg.loss_function,
        cfg.superoperator if uses_superoperator else None,
        timestamp,
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    params_path = run_dir / "params.npz"
    metadata_path = run_dir / "metadata.json"

    target = build_target_from_seed(cfg.n_qubits, cfg.target_seed, cfg.time_k)
    adam_cfg = AdamConfig(iterations=cfg.iterations, lr=cfg.lr, fd_eps=args.fd_eps)

    print(f"preset = {args.preset}")
    print(f"n_qubits = {cfg.n_qubits}")
    print(f"blocks = {cfg.blocks}")
    print(f"target_bits = {cfg.target_bits}")
    print(f"ansatz = {cfg.ansatz}")
    print(f"block_only_ansatz = {cfg.block_only_ansatz}")
    print(f"loss_function = {cfg.loss_function} -> {get_active_loss_function().__name__}")
    if uses_superoperator:
        print(f"superoperator = {cfg.superoperator} -> {get_active_superop().__name__}")
    else:
        print("superoperator = not used by this loss_function")

    arrays: dict[str, Any] = {}
    block_metadata: list[dict[str, Any]] = []
    theta_sizes: list[int] = []
    all_loss_histories: list[NDArray[np.float64]] = []
    _latest_block_results: list[Any] = []

    for block_idx, (block_qubits, target_bit) in enumerate(zip(cfg.blocks, cfg.target_bits), start=1):
        print(f"\nBlock {block_idx}: qubits {block_qubits}, target_bit {target_bit}")
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
        loss_fn = lambda theta, ctx=context: sum_block_loss(theta, ctx)
        rng = np.random.default_rng(cfg.training_seed_for_block(block_idx - 1))
        result = multi_restart_train(
            loss_fn,
            cfg.max_restarts,
            rng,
            adam_cfg,
            n_qubits=context.ansatz_qubits,
            ansatz=context.ansatz,
            show_progress=args.progress,
            success_threshold=cfg.success_threshold,
        )
        _latest_block_results.append(result)
        best_restart_losses = np.asarray(result.restart_results[result.best_restart].loss_history, dtype=float)
        all_loss_histories.append(best_restart_losses)
        _residual, loss_qubits = residual_operator_for_context(result.best_params, context)
        loss_breakdown = active_loss_breakdown(result.best_params, context)

        params_key = f"best_params_block_{block_idx}"
        arrays[params_key] = result.best_params
        theta_sizes.append(context.theta_size)
        block_metadata.append({
            "block_index": block_idx,
            "block_qubits": list(block_qubits),
            "target_bit": int(target_bit),
            "lightcone_qubits": list(context.lightcone_qubits),
            "ansatz_qubits": context.ansatz_qubits,
            "ansatz": context.ansatz,
            "block_only_ansatz": context.block_only_ansatz,
            "theta_size": context.theta_size,
            "params_key": params_key,
            "selected_restart": result.best_restart + 1,
            "completed_restarts": len(result.restart_results),
            "max_restarts": cfg.max_restarts,
            "success_threshold": cfg.success_threshold,
            "best_iteration": result.best_iteration,
            "best_loss": result.best_loss,
            "loss_breakdown": {str(k): float(v) for k, v in loss_breakdown.items()},
            "per_bit_losses": {str(k): float(v) for k, v in loss_breakdown.items()},
            "loss_qubits": list(loss_qubits),
            "lightcone_semantics": context_meta.get("lightcone_semantics"),
            "loss_semantics": context_meta.get("loss_semantics"),
        })
        print(f"  lightcone = {context.lightcone_qubits}")
        print(f"  best_loss = {result.best_loss:.12g}")
        print(f"  completed_restarts = {len(result.restart_results)}/{cfg.max_restarts}")
        print(f"  loss_breakdown = { {k: f'{v:.3e}' for k, v in loss_breakdown.items()} }")

    arrays["blocks"] = np.asarray(cfg.blocks, dtype=int)
    arrays["target_bits"] = np.asarray(cfg.target_bits, dtype=int)
    arrays["theta_sizes"] = np.asarray(theta_sizes, dtype=int)
    for bi, loss_hist in enumerate(all_loss_histories, start=1):
        arrays[f"loss_history_block_{bi}"] = loss_hist
    np.savez(params_path, **arrays)
    # also save per-block all-restart loss histories as human-readable JSON
    restart_records: list[dict[str, Any]] = []
    for bi, (block_qubits, target_bit) in enumerate(zip(cfg.blocks, cfg.target_bits)):
        block_result = _latest_block_results[bi] if bi < len(_latest_block_results) else None
        histories: list[list[float]] = []
        if block_result is not None:
            for rr in block_result.restart_results:
                histories.append(rr.loss_history.tolist() if hasattr(rr.loss_history, 'tolist') else list(rr.loss_history))
        restart_records.append({
            "block_index": bi + 1,
            "block_qubits": list(block_qubits),
            "target_bit": int(target_bit),
            "best_restart_index": block_result.best_restart if block_result is not None else None,
            "all_restart_loss_histories": histories,
        })
    trajectory_path = run_dir / "loss_trajectories.json"
    trajectory_path.write_text(json.dumps(restart_records, indent=2), encoding="utf-8")
    print(f"Saved loss trajectories: {trajectory_path}")

    metadata = {
        "schema_version": 2,
        "artifact_type": "multi_block_training",
        "timestamp": timestamp,
        "preset": args.preset,
        "run_dir": str(run_dir),
        **target_metadata(target),
        "blocks": block_metadata,
        "block_qubits": cfg.blocks,
        "target_bits": cfg.target_bits,
        "radius": cfg.radius,
        "lightcone_mode": cfg.lightcone_mode,
        "loss_mode": cfg.loss_mode,
        "ansatz": cfg.ansatz,
        "block_only_ansatz": cfg.block_only_ansatz,
        "loss_function": cfg.loss_function,
        "loss_function_uses_superoperator": uses_superoperator,
        "iterations": cfg.iterations,
        "lr": cfg.lr,
        "fd_eps": args.fd_eps,
        "max_restarts": cfg.max_restarts,
        "training_seed_start": cfg.training_seed_start,
        "params_file": params_path.name,
        "metadata_file": metadata_path.name,
    }
    if uses_superoperator:
        metadata["superoperator"] = cfg.superoperator
    metadata_path.write_text(json.dumps(_json_ready(metadata), indent=2), encoding="utf-8")

    if args.plot_loss:
        _save_loss_plot(run_dir, all_loss_histories, block_metadata, cfg, timestamp)

    print(f"\nSaved run directory: {run_dir}")
    print(f"\nSaved params: {params_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
