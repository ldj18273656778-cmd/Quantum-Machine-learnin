"""Plot a raw, non-log loss-landscape heatmap.

Two modes are supported:
1. pass ``--input landscape.npz`` to redraw an existing landscape;
2. omit ``--input`` to generate a landscape from preset/block/center parameters.
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
from numpy.typing import NDArray

from task2_code_auto.hpc_parallel_training.hpc_block_flow import PRESETS, atomic_savez, atomic_write_json, safe_stem, timestamp
from task2_code_auto.landscape.generate_loss_landscape import (
    _evaluate_grid,
    _load_center,
    _orthonormal_directions,
    _random_center,
)
from task2_code_auto.loss_registry import loss_function_uses_superoperator, set_active_loss_function
from task2_code_auto.module_e_training import build_target_objective_context, sum_block_loss
from task2_code_auto.superoperator_registry import set_active_superop
from task2_code_auto.target_factory import build_target_from_seed, target_metadata

FloatArray = NDArray[np.float64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot raw, non-log loss landscape.")
    parser.add_argument("--input", type=Path, default=None, help="optional existing landscape.npz to redraw")
    parser.add_argument("--output", type=Path, default=None, help="default: <input-dir>/landscape_heatmap_raw.png or <run-dir>/landscape_heatmap_raw.png")
    parser.add_argument("--title", default="Raw Heisenberg loss landscape")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--contours", type=int, default=12, help="set 0 to disable contour lines")

    parser.add_argument("--preset", choices=sorted(PRESETS), default="n20_5blocks")
    parser.add_argument("--block-index", type=int, default=2, help="1-based block index; default n20 block 2 has an 8-bit lightcone")
    parser.add_argument("--loss-function", default="heisenberg_pauli")
    parser.add_argument("--superoperator", default="superoperator_from_mix")
    parser.add_argument("--center-file", type=Path, default=None, help="optional .npy/.npz file containing the center theta")
    parser.add_argument("--center-key", default=None, help="key inside .npz; defaults to theta/best_params/final_params/params")
    parser.add_argument("--center-seed", type=int, default=1042, help="used only when --center-file is omitted")
    parser.add_argument("--direction-source", choices=["pca", "random"], default="pca")
    parser.add_argument("--direction-seed", type=int, default=20260616, help="used only when --direction-source=random")
    parser.add_argument("--trajectory-file", type=Path, default=None, help=".npz file containing parameter trajectory; defaults to --center-file")
    parser.add_argument("--trajectory-key", default="parameter_history")
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--span", type=float, default=1.0, help="fallback: scan alpha,beta in [-span, span]")
    parser.add_argument("--alpha-min", type=float, default=None)
    parser.add_argument("--alpha-max", type=float, default=None)
    parser.add_argument("--beta-min", type=float, default=None)
    parser.add_argument("--beta-max", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("task2_code/landscape/data"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-wrap", action="store_true", default=False, help="do not wrap scanned theta to [0, 2*pi)")
    return parser.parse_args()



def _axis_coordinates(min_value: float | None, max_value: float | None, span: float, grid_size: int, axis_name: str) -> FloatArray:
    if (min_value is None) != (max_value is None):
        raise ValueError(f"--{axis_name}-min and --{axis_name}-max must be provided together")
    lo = -float(span) if min_value is None else float(min_value)
    hi = float(span) if max_value is None else float(max_value)
    if hi <= lo:
        raise ValueError(f"{axis_name} max must be greater than min, got [{lo}, {hi}]")
    return np.linspace(lo, hi, int(grid_size), dtype=float)


def _load_parameter_history(path: Path, key: str, expected_size: int) -> FloatArray:
    if not path.exists():
        raise FileNotFoundError(f"trajectory file not found: {path}")
    if path.suffix != ".npz":
        raise ValueError(f"trajectory file must be .npz, got {path.suffix!r}")
    data = np.load(path)
    if key not in data:
        raise KeyError(f"trajectory key {key!r} not found in {path}; available keys: {list(data.keys())}")
    history = np.asarray(data[key], dtype=float)
    if history.ndim != 2 or history.shape[1] != expected_size:
        raise ValueError(f"trajectory must have shape (steps, {expected_size}), got {history.shape}")
    if history.shape[0] < 3:
        raise ValueError("trajectory must contain at least 3 parameter vectors for PCA")
    if not np.all(np.isfinite(history)):
        raise ValueError("trajectory contains non-finite values")
    return history.astype(float, copy=True)


def _pca_directions_from_history(history: FloatArray, center: FloatArray, *, wrap_angles: bool) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    deltas = history - center.reshape(1, -1)
    if wrap_angles:
        deltas = (deltas + np.pi) % (2.0 * np.pi) - np.pi
    centered = deltas - deltas.mean(axis=0, keepdims=True)
    _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if vh.shape[0] < 2 or singular_values[1] <= 0.0:
        raise ValueError("trajectory does not have two non-degenerate PCA directions")
    direction_x = vh[0].astype(float, copy=True)
    direction_y = vh[1].astype(float, copy=True)
    variances = singular_values * singular_values
    total_variance = float(variances.sum())
    explained = variances / total_variance if total_variance > 0.0 else np.zeros_like(variances)
    metadata = {
        "pca_singular_values": singular_values[:2].tolist(),
        "pca_explained_variance_ratio": explained[:2].tolist(),
    }
    return direction_x, direction_y, metadata


def _load_landscape(path: Path) -> tuple[FloatArray, FloatArray, FloatArray]:
    if not path.exists():
        raise FileNotFoundError(f"landscape file not found: {path}")
    data = np.load(path)
    if "losses" not in data:
        raise KeyError(f"{path} must contain 'losses'; got {list(data.keys())}")
    if "alpha_coordinates" in data and "beta_coordinates" in data:
        alpha_coordinates = np.asarray(data["alpha_coordinates"], dtype=float)
        beta_coordinates = np.asarray(data["beta_coordinates"], dtype=float)
    elif "coordinates" in data:
        alpha_coordinates = np.asarray(data["coordinates"], dtype=float)
        beta_coordinates = alpha_coordinates
    else:
        raise KeyError(f"{path} must contain coordinates or alpha/beta coordinates; got {list(data.keys())}")
    losses = np.asarray(data["losses"], dtype=float)
    if alpha_coordinates.ndim != 1 or beta_coordinates.ndim != 1:
        raise ValueError(f"coordinates must be 1D, got {alpha_coordinates.shape}, {beta_coordinates.shape}")
    if losses.shape != (beta_coordinates.size, alpha_coordinates.size):
        raise ValueError(f"losses must have shape ({beta_coordinates.size}, {alpha_coordinates.size}), got {losses.shape}")
    if not np.all(np.isfinite(losses)):
        raise ValueError("losses contain non-finite values")
    return alpha_coordinates, beta_coordinates, losses


def plot_raw_heatmap(
    alpha_coordinates: FloatArray,
    beta_coordinates: FloatArray,
    losses: FloatArray,
    output: Path,
    *,
    title: str,
    cmap: str,
    dpi: int,
    contours: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(
        losses,
        origin="lower",
        extent=(alpha_coordinates[0], alpha_coordinates[-1], beta_coordinates[0], beta_coordinates[-1]),
        aspect="auto",
        cmap=cmap,
    )
    if contours > 0:
        levels = np.linspace(float(losses.min()), float(losses.max()), contours)
        ax.contour(alpha_coordinates, beta_coordinates, losses, levels=levels, colors="white", linewidths=0.45, alpha=0.55)
    ax.scatter([0.0], [0.0], c="white", s=24, edgecolors="black", linewidths=0.6, label="center")
    ax.set_xlabel("alpha along random direction 1")
    ax.set_ylabel("beta along random direction 2")
    ax.set_title(title)
    ax.legend(loc="upper right")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("loss")
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _generate_landscape(args: argparse.Namespace) -> tuple[Path, FloatArray, FloatArray, FloatArray]:
    cfg = PRESETS[str(args.preset)]
    block_index = int(args.block_index)
    if block_index < 1 or block_index > cfg.block_count:
        raise ValueError(f"--block-index must be in 1..{cfg.block_count}")
    if int(args.grid_size) < 2:
        raise ValueError("--grid-size must be at least 2")
    if float(args.span) <= 0:
        raise ValueError("--span must be positive")

    block_qubits = cfg.blocks[block_index - 1]
    target_bit = cfg.target_bits[block_index - 1]
    loss_function = str(args.loss_function)
    set_active_loss_function(loss_function)
    if loss_function_uses_superoperator(loss_function):
        set_active_superop(str(args.superoperator))

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
    if len(context.lightcone_qubits) != 8:
        raise ValueError(f"this script is scoped to 8-bit lightcones; got {len(context.lightcone_qubits)}")

    center = (
        _load_center(args.center_file, args.center_key, context.theta_size)
        if args.center_file is not None
        else _random_center(int(args.center_seed), context.ansatz, context.ansatz_qubits, context.theta_size)
    )
    direction_metadata: dict[str, Any] = {"direction_source": str(args.direction_source)}
    if str(args.direction_source) == "pca":
        trajectory_file = args.trajectory_file or args.center_file
        if trajectory_file is None:
            raise ValueError("--direction-source=pca requires --trajectory-file or a .npz --center-file with parameter_history")
        history = _load_parameter_history(trajectory_file, str(args.trajectory_key), context.theta_size)
        direction_x, direction_y, pca_metadata = _pca_directions_from_history(
            history,
            center,
            wrap_angles=not bool(args.no_wrap),
        )
        direction_metadata.update(pca_metadata)
        direction_metadata["trajectory_file"] = trajectory_file
        direction_metadata["trajectory_key"] = str(args.trajectory_key)
    else:
        direction_x, direction_y = _orthonormal_directions(int(args.direction_seed), context.theta_size)
        direction_metadata["direction_seed"] = int(args.direction_seed)
    alpha_coordinates = _axis_coordinates(args.alpha_min, args.alpha_max, float(args.span), int(args.grid_size), "alpha")
    beta_coordinates = _axis_coordinates(args.beta_min, args.beta_max, float(args.span), int(args.grid_size), "beta")
    center_loss = float(sum_block_loss(center, context))
    losses = _evaluate_grid(
        center,
        direction_x,
        direction_y,
        alpha_coordinates,
        beta_coordinates,
        wrap_angles=not bool(args.no_wrap),
        context=context,
    )

    stamp = timestamp()
    run_name = args.run_name or "_".join(
        [
            "raw_landscape",
            safe_stem(str(args.preset)),
            f"block{block_index:02d}",
            safe_stem(loss_function),
            f"grid{int(args.grid_size)}",
            stamp,
        ]
    )
    output_root = Path(args.output_dir) / str(run_name)
    output_root.mkdir(parents=True, exist_ok=False)

    target = build_target_from_seed(cfg.n_qubits, cfg.target_seed, cfg.time_k)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "raw_loss_landscape_2d",
        "timestamp": stamp,
        "output_root": output_root,
        "preset": args.preset,
        "n_qubits": cfg.n_qubits,
        "block_index": block_index,
        "block_qubits": list(block_qubits),
        "target_bit": int(target_bit),
        "lightcone_qubits": list(context.lightcone_qubits),
        "lightcone_size": len(context.lightcone_qubits),
        "ansatz": context.ansatz,
        "ansatz_qubits": context.ansatz_qubits,
        "theta_size": context.theta_size,
        "block_only_ansatz": context.block_only_ansatz,
        "loss_function": loss_function,
        "superoperator": str(args.superoperator) if loss_function_uses_superoperator(loss_function) else None,
        "center_file": args.center_file,
        "center_key": args.center_key,
        "center_seed": int(args.center_seed) if args.center_file is None else None,
        "center_loss": center_loss,
        **direction_metadata,
        "grid_size": int(args.grid_size),
        "span": float(args.span),
        "alpha_min": float(alpha_coordinates[0]),
        "alpha_max": float(alpha_coordinates[-1]),
        "beta_min": float(beta_coordinates[0]),
        "beta_max": float(beta_coordinates[-1]),
        "wrap_angles": not bool(args.no_wrap),
        "loss_min": float(np.min(losses)),
        "loss_median": float(np.median(losses)),
        "loss_mean": float(np.mean(losses)),
        "loss_max": float(np.max(losses)),
        "lightcone_semantics": context_meta.get("lightcone_semantics"),
        "loss_semantics": context_meta.get("loss_semantics"),
        **target_metadata(target),
    }
    atomic_savez(
        output_root / "landscape.npz",
        coordinates=alpha_coordinates if np.array_equal(alpha_coordinates, beta_coordinates) else alpha_coordinates,
        alpha_coordinates=alpha_coordinates,
        beta_coordinates=beta_coordinates,
        losses=losses,
        center=center,
        direction_x=direction_x,
        direction_y=direction_y,
    )
    atomic_write_json(output_root / "metadata.json", metadata)
    return output_root, alpha_coordinates, beta_coordinates, losses


def main() -> None:
    args = parse_args()
    if args.input is not None:
        alpha_coordinates, beta_coordinates, losses = _load_landscape(args.input)
        output = args.output or args.input.with_name("landscape_heatmap_raw.png")
    else:
        output_root, alpha_coordinates, beta_coordinates, losses = _generate_landscape(args)
        output = args.output or output_root / "landscape_heatmap_raw.png"

    plot_raw_heatmap(
        alpha_coordinates,
        beta_coordinates,
        losses,
        output,
        title=str(args.title),
        cmap=str(args.cmap),
        dpi=int(args.dpi),
        contours=int(args.contours),
    )
    print(f"Saved raw landscape heatmap: {output}")
    print(f"loss_min = {float(losses.min()):.12g}")
    print(f"loss_median = {float(np.median(losses)):.12g}")
    print(f"loss_max = {float(losses.max()):.12g}")


if __name__ == "__main__":
    main()
