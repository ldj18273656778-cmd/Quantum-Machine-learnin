"""Plot landscapes by scanning two selected ansatz parameters."""

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

from task2_code.ansatz_registry import random_ansatz_theta
from task2_code.hpc_parallel_training.hpc_block_flow import PRESETS, atomic_savez, atomic_write_json, safe_stem, timestamp
from task2_code.loss_registry import loss_function_uses_superoperator, set_active_loss_function
from task2_code.module_e_training import build_target_objective_context, sum_block_loss
from task2_code.superoperator_registry import set_active_superop
from task2_code.target_factory import build_target_from_seed, target_metadata


FloatArray = NDArray[np.float64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan two selected parameters while fixing all other parameters.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="n32_8blocks")
    parser.add_argument("--block-index", type=int, default=5)
    parser.add_argument("--param-x", type=int, default=15, help="1-based parameter index for x axis")
    parser.add_argument("--param-y", type=int, default=35, help="1-based parameter index for y axis")
    parser.add_argument("--loss-function", default="heisenberg_pauli")
    parser.add_argument("--superoperator", default="superoperator_from_mix")
    parser.add_argument("--center-file", type=Path, default=Path("task2_code/loss_success_experiment/data/paired_warmstart_n32_8blocks_block05_400pairs/summary/paired_best_params.npz"), help=".npz center source; use 'none' for random fixed parameters")
    parser.add_argument("--center-key", default="warmstart_best_params")
    parser.add_argument("--center-loss-key", default="warmstart_best_losses")
    parser.add_argument("--center-row", type=int, default=None, help="0-based row; default chooses minimum center-loss-key")
    parser.add_argument("--center-seed", type=int, default=20260630, help="used when --center-file none")
    parser.add_argument("--grid-size", type=int, default=51)
    parser.add_argument("--theta-min", type=float, default=0.0)
    parser.add_argument("--theta-max", type=float, default=2.0 * np.pi)
    parser.add_argument("--output-dir", type=Path, default=Path("task2_code/landscape/data"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--contours", type=int, default=12)
    return parser.parse_args()


def _load_center(path: Path, key: str, loss_key: str, row: int | None) -> tuple[FloatArray, int, float | None]:
    if not path.exists():
        raise FileNotFoundError(f"center file not found: {path}")
    data = np.load(path)
    if key not in data:
        raise KeyError(f"center key {key!r} not found in {path}; available keys: {data.files}")
    params = np.asarray(data[key], dtype=float)
    if params.ndim == 1:
        return params.astype(float, copy=True), 0, None
    if params.ndim != 2:
        raise ValueError(f"center array must be 1D or 2D, got {params.shape}")
    if row is None:
        if loss_key not in data:
            raise KeyError(f"center loss key {loss_key!r} not found in {path}; available keys: {data.files}")
        losses = np.asarray(data[loss_key], dtype=float)
        if losses.shape != (params.shape[0],):
            raise ValueError(f"loss key {loss_key!r} must have shape ({params.shape[0]},), got {losses.shape}")
        row_value = int(np.argmin(losses))
        loss_value = float(losses[row_value])
    else:
        row_value = int(row)
        loss_value = None
    if row_value < 0 or row_value >= params.shape[0]:
        raise ValueError(f"center row must be in 0..{params.shape[0] - 1}, got {row_value}")
    center = np.asarray(params[row_value], dtype=float)
    if not np.all(np.isfinite(center)):
        raise ValueError("center contains non-finite values")
    return center.astype(float, copy=True), row_value, loss_value


def _is_random_center(path: Path) -> bool:
    return str(path).strip().lower() in {"none", "random"}


def _build_context(args: argparse.Namespace):
    cfg = PRESETS[str(args.preset)]
    block_index = int(args.block_index)
    if block_index < 1 or block_index > cfg.block_count:
        raise ValueError(f"--block-index must be in 1..{cfg.block_count}")
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
    return cfg, block_qubits, target_bit, context, context_meta


def _evaluate_pair_grid(center: FloatArray, x_index: int, y_index: int, values: FloatArray, context: Any) -> FloatArray:
    losses = np.empty((values.size, values.size), dtype=float)
    total = values.size * values.size
    done = 0
    for row, y_value in enumerate(values):
        for col, x_value in enumerate(values):
            theta = center.copy()
            theta[x_index] = float(x_value)
            theta[y_index] = float(y_value)
            losses[row, col] = float(sum_block_loss(theta, context))
            done += 1
        print(f"row {row + 1}/{values.size} complete ({done}/{total} evaluations)", flush=True)
    return losses


def _plot_heatmap(path: Path, values: FloatArray, losses: FloatArray, *, title: str, xlabel: str, ylabel: str, cmap: str, dpi: int, contours: int, center_x: float, center_y: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(
        losses,
        origin="lower",
        extent=(values[0], values[-1], values[0], values[-1]),
        aspect="auto",
        cmap=cmap,
    )
    if contours > 0:
        levels = np.linspace(float(losses.min()), float(losses.max()), int(contours))
        ax.contour(values, values, losses, levels=levels, colors="white", linewidths=0.45, alpha=0.55)
    ax.scatter([center_x], [center_y], c="white", s=34, edgecolors="black", linewidths=0.7, label="fixed center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("loss")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg, block_qubits, target_bit, context, context_meta = _build_context(args)
    if _is_random_center(args.center_file):
        rng = np.random.default_rng(int(args.center_seed))
        center = np.asarray(random_ansatz_theta(context.ansatz, rng, n_qubits=context.ansatz_qubits), dtype=float)
        center_row = -1
        center_loss = None
        center_source = "random"
    else:
        center, center_row, center_loss = _load_center(args.center_file, str(args.center_key), str(args.center_loss_key), args.center_row)
        center_source = str(args.center_file)
    if center.shape != (context.theta_size,):
        raise ValueError(f"center shape must be ({context.theta_size},), got {center.shape}")
    x_index = int(args.param_x) - 1
    y_index = int(args.param_y) - 1
    if x_index == y_index:
        raise ValueError("--param-x and --param-y must be different")
    if x_index < 0 or x_index >= context.theta_size or y_index < 0 or y_index >= context.theta_size:
        raise ValueError(f"parameter indices must be in 1..{context.theta_size}")
    if int(args.grid_size) < 2:
        raise ValueError("--grid-size must be at least 2")
    if float(args.theta_max) <= float(args.theta_min):
        raise ValueError("--theta-max must be greater than --theta-min")

    set_active_loss_function(str(args.loss_function))
    if loss_function_uses_superoperator(str(args.loss_function)):
        set_active_superop(str(args.superoperator))
    values = np.linspace(float(args.theta_min), float(args.theta_max), int(args.grid_size), dtype=float)
    center_loss_for_this_loss = float(sum_block_loss(center, context))
    losses = _evaluate_pair_grid(center, x_index, y_index, values, context)

    stamp = timestamp()
    run_name = args.run_name or "_".join(
        [
            "param_pair_landscape",
            safe_stem(str(args.preset)),
            f"block{int(args.block_index):02d}",
            safe_stem(str(args.loss_function)),
            f"theta{int(args.param_x):02d}_theta{int(args.param_y):02d}",
            f"grid{int(args.grid_size)}",
            stamp,
        ]
    )
    output_root = args.output_dir / run_name
    output_root.mkdir(parents=True, exist_ok=False)
    target = build_target_from_seed(cfg.n_qubits, cfg.target_seed, cfg.time_k)
    metadata = {
        "schema_version": 1,
        "artifact_type": "parameter_pair_loss_landscape",
        "timestamp": stamp,
        "preset": args.preset,
        "n_qubits": cfg.n_qubits,
        "block_index": int(args.block_index),
        "block_qubits": list(block_qubits),
        "target_bit": int(target_bit),
        "lightcone_qubits": list(context.lightcone_qubits),
        "lightcone_size": len(context.lightcone_qubits),
        "ansatz": context.ansatz,
        "theta_size": context.theta_size,
        "loss_function": str(args.loss_function),
        "superoperator": str(args.superoperator) if loss_function_uses_superoperator(str(args.loss_function)) else None,
        "param_x": int(args.param_x),
        "param_y": int(args.param_y),
        "param_x_zero_based": x_index,
        "param_y_zero_based": y_index,
        "fixed_parameter_source": center_source,
        "fixed_parameter_key": str(args.center_key),
        "fixed_parameter_row_zero_based": center_row,
        "fixed_parameter_seed": int(args.center_seed) if _is_random_center(args.center_file) else None,
        "fixed_parameter_recorded_loss": center_loss,
        "center_loss_for_this_loss": center_loss_for_this_loss,
        "center_param_x": float(center[x_index]),
        "center_param_y": float(center[y_index]),
        "grid_size": int(args.grid_size),
        "theta_min": float(values[0]),
        "theta_max": float(values[-1]),
        "loss_min": float(losses.min()),
        "loss_median": float(np.median(losses)),
        "loss_mean": float(losses.mean()),
        "loss_max": float(losses.max()),
        "lightcone_semantics": context_meta.get("lightcone_semantics"),
        "loss_semantics": context_meta.get("loss_semantics"),
        **target_metadata(target),
    }
    atomic_savez(output_root / "landscape.npz", theta_values=values, losses=losses, fixed_center=center)
    atomic_write_json(output_root / "metadata.json", metadata)
    title = args.title or f"{args.loss_function} landscape: theta_{args.param_x} vs theta_{args.param_y}"
    _plot_heatmap(
        output_root / "landscape_heatmap.png",
        values,
        losses,
        title=str(title),
        xlabel=f"theta_{int(args.param_x)}",
        ylabel=f"theta_{int(args.param_y)}",
        cmap=str(args.cmap),
        dpi=int(args.dpi),
        contours=int(args.contours),
        center_x=float(center[x_index]),
        center_y=float(center[y_index]),
    )
    print(f"Wrote {output_root / 'landscape.npz'}")
    print(f"Wrote {output_root / 'metadata.json'}")
    print(f"Wrote {output_root / 'landscape_heatmap.png'}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
