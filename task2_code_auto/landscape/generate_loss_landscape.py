"""Generate a 2D loss-landscape heatmap around a center parameter vector.

The script is intentionally scoped to small lightcones.  By default it uses
``n20_5blocks`` block 2, whose circuit lightcone has 8 qubits.
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

from task2_code_auto.ansatz_registry import random_ansatz_theta
from task2_code_auto.hpc_parallel_training.hpc_block_flow import PRESETS, atomic_savez, atomic_write_json, safe_stem, timestamp
from task2_code_auto.loss_registry import loss_function_uses_superoperator, set_active_loss_function
from task2_code_auto.module_e_training import build_target_objective_context, sum_block_loss
from task2_code_auto.superoperator_registry import set_active_superop
from task2_code_auto.target_factory import build_target_from_seed, target_metadata

FloatArray = NDArray[np.float64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an 8-lightcone loss-landscape heatmap.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="n20_5blocks")
    parser.add_argument("--block-index", type=int, default=2, help="1-based block index; default n20 block 2 has an 8-bit lightcone")
    parser.add_argument("--loss-function", default="heisenberg_pauli")
    parser.add_argument("--superoperator", default="superoperator_from_mix")
    parser.add_argument("--center-file", type=Path, default=None, help="optional .npy/.npz file containing the center theta")
    parser.add_argument("--center-key", default=None, help="key inside .npz; defaults to theta/best_params/final_params/params")
    parser.add_argument("--center-seed", type=int, default=1042, help="used only when --center-file is omitted")
    parser.add_argument("--direction-seed", type=int, default=20260616)
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


def _load_center(path: Path, key: str | None, expected_size: int) -> FloatArray:
    if not path.exists():
        raise FileNotFoundError(f"center file not found: {path}")
    if path.suffix == ".npy":
        theta = np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        if key is None:
            candidates = ["theta", "best_params", "final_params", "params"]
            found = [name for name in candidates if name in data]
            if not found:
                raise KeyError(f"no default center key found in {path}; available keys: {list(data.keys())}")
            key = found[0]
        theta = data[key]
    else:
        raise ValueError(f"center file must be .npy or .npz, got {path.suffix!r}")
    return _validate_theta(theta, expected_size, "center theta")


def _validate_theta(theta: object, expected_size: int, name: str) -> FloatArray:
    arr = np.asarray(theta, dtype=float).reshape(-1)
    if arr.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr.astype(float, copy=True)


def _random_center(seed: int, ansatz: str, n_qubits: int, expected_size: int) -> FloatArray:
    rng = np.random.default_rng(seed)
    theta = random_ansatz_theta(ansatz, rng, n_qubits=n_qubits)
    return _validate_theta(theta, expected_size, "random center theta")


def _orthonormal_directions(seed: int, size: int) -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    d1 = rng.normal(size=size)
    d1_norm = float(np.linalg.norm(d1))
    if d1_norm == 0.0:
        raise ValueError("first random direction has zero norm")
    d1 = d1 / d1_norm

    d2 = rng.normal(size=size)
    d2 = d2 - float(np.dot(d2, d1)) * d1
    d2_norm = float(np.linalg.norm(d2))
    if d2_norm == 0.0:
        raise ValueError("second random direction is degenerate")
    d2 = d2 / d2_norm
    return d1.astype(float), d2.astype(float)


def _axis_coordinates(min_value: float | None, max_value: float | None, span: float, grid_size: int, axis_name: str) -> FloatArray:
    if (min_value is None) != (max_value is None):
        raise ValueError(f"--{axis_name}-min and --{axis_name}-max must be provided together")
    lo = -float(span) if min_value is None else float(min_value)
    hi = float(span) if max_value is None else float(max_value)
    if hi <= lo:
        raise ValueError(f"{axis_name} max must be greater than min, got [{lo}, {hi}]")
    return np.linspace(lo, hi, int(grid_size), dtype=float)


def _evaluate_grid(
    center: FloatArray,
    direction_x: FloatArray,
    direction_y: FloatArray,
    alpha_coordinates: FloatArray,
    beta_coordinates: FloatArray,
    *,
    wrap_angles: bool,
    context: Any,
) -> FloatArray:
    losses = np.empty((beta_coordinates.size, alpha_coordinates.size), dtype=float)
    total = alpha_coordinates.size * beta_coordinates.size
    done = 0
    for row, beta in enumerate(beta_coordinates):
        for col, alpha in enumerate(alpha_coordinates):
            theta = center + float(alpha) * direction_x + float(beta) * direction_y
            if wrap_angles:
                theta = np.mod(theta, 2.0 * np.pi)
            losses[row, col] = float(sum_block_loss(theta, context))
            done += 1
        print(f"row {row + 1}/{beta_coordinates.size} complete ({done}/{total} evaluations)", flush=True)
    return losses


def _plot_heatmap(path: Path, alpha_coordinates: FloatArray, beta_coordinates: FloatArray, losses: FloatArray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    positive = losses[np.isfinite(losses) & (losses > 0)]
    if positive.size:
        image_values = np.log10(losses)
        label = "log10(loss)"
    else:
        image_values = losses
        label = "loss"
    im = ax.imshow(
        image_values,
        origin="lower",
        extent=(alpha_coordinates[0], alpha_coordinates[-1], beta_coordinates[0], beta_coordinates[-1]),
        aspect="auto",
        cmap="viridis",
    )
    ax.scatter([0.0], [0.0], c="white", s=24, edgecolors="black", linewidths=0.6, label="center")
    ax.set_xlabel("alpha along random direction 1")
    ax.set_ylabel("beta along random direction 2")
    ax.set_title(title)
    ax.legend(loc="upper right")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = PRESETS[args.preset]
    if args.block_index < 1 or args.block_index > cfg.block_count:
        raise ValueError(f"--block-index must be in 1..{cfg.block_count}")
    if args.grid_size < 2:
        raise ValueError("--grid-size must be at least 2")
    if args.span <= 0:
        raise ValueError("--span must be positive")

    block_qubits = cfg.blocks[args.block_index - 1]
    target_bit = cfg.target_bits[args.block_index - 1]
    set_active_loss_function(args.loss_function)
    if loss_function_uses_superoperator(args.loss_function):
        set_active_superop(args.superoperator)

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
        else _random_center(args.center_seed, context.ansatz, context.ansatz_qubits, context.theta_size)
    )
    direction_x, direction_y = _orthonormal_directions(args.direction_seed, context.theta_size)
    alpha_coordinates = _axis_coordinates(args.alpha_min, args.alpha_max, args.span, args.grid_size, "alpha")
    beta_coordinates = _axis_coordinates(args.beta_min, args.beta_max, args.span, args.grid_size, "beta")
    center_loss = float(sum_block_loss(center, context))
    losses = _evaluate_grid(
        center,
        direction_x,
        direction_y,
        alpha_coordinates,
        beta_coordinates,
        wrap_angles=not args.no_wrap,
        context=context,
    )

    stamp = timestamp()
    name = args.run_name or "_".join(
        [
            "landscape",
            safe_stem(args.preset),
            f"block{args.block_index:02d}",
            safe_stem(args.loss_function),
            f"grid{args.grid_size}",
            stamp,
        ]
    )
    output_root = args.output_dir / name
    output_root.mkdir(parents=True, exist_ok=False)

    target = build_target_from_seed(cfg.n_qubits, cfg.target_seed, cfg.time_k)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "loss_landscape_2d",
        "timestamp": stamp,
        "output_root": output_root,
        "preset": args.preset,
        "n_qubits": cfg.n_qubits,
        "block_index": args.block_index,
        "block_qubits": list(block_qubits),
        "target_bit": int(target_bit),
        "lightcone_qubits": list(context.lightcone_qubits),
        "lightcone_size": len(context.lightcone_qubits),
        "ansatz": context.ansatz,
        "ansatz_qubits": context.ansatz_qubits,
        "theta_size": context.theta_size,
        "block_only_ansatz": context.block_only_ansatz,
        "loss_function": args.loss_function,
        "superoperator": args.superoperator if loss_function_uses_superoperator(args.loss_function) else None,
        "center_file": args.center_file,
        "center_key": args.center_key,
        "center_seed": args.center_seed if args.center_file is None else None,
        "center_loss": center_loss,
        "direction_seed": args.direction_seed,
        "grid_size": args.grid_size,
        "span": args.span,
        "alpha_min": float(alpha_coordinates[0]),
        "alpha_max": float(alpha_coordinates[-1]),
        "beta_min": float(beta_coordinates[0]),
        "beta_max": float(beta_coordinates[-1]),
        "wrap_angles": not args.no_wrap,
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
    _plot_heatmap(
        output_root / "landscape_heatmap.png",
        alpha_coordinates,
        beta_coordinates,
        losses,
        title=f"{args.loss_function} landscape, {args.preset} block {args.block_index}",
    )

    print(f"Saved landscape: {output_root}")
    print(f"center_loss = {center_loss:.12g}")
    print(f"loss_min = {metadata['loss_min']:.12g}")
    print(f"loss_median = {metadata['loss_median']:.12g}")
    print(f"loss_max = {metadata['loss_max']:.12g}")


if __name__ == "__main__":
    main()
