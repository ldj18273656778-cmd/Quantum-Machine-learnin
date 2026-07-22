"""Plot Heisenberg-only and warm-start best parameters in a shared PCA plane."""

from __future__ import annotations

import argparse
import csv
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


FloatArray = NDArray[np.float64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paired best parameters projected onto a PCA plane.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "task2_code/loss_success_experiment/data/paired_warmstart_n32_8blocks_block05_400pairs/summary/paired_best_params.npz"
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="default: <input-dir>/best_params_pca_distribution.png")
    parser.add_argument("--projection-output", type=Path, default=None, help="default: <input-dir>/best_params_pca_projection.npz")
    parser.add_argument("--csv-output", type=Path, default=None, help="default: <input-dir>/best_params_pca_projection.csv")
    parser.add_argument("--title", default="Best parameter distribution in PCA plane")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--point-size", type=float, default=32.0)
    parser.add_argument("--alpha", type=float, default=0.78)
    parser.add_argument("--no-wrap", action="store_true", default=False, help="use raw parameter differences instead of angular wrap")
    return parser.parse_args()


def _require_array(data: Any, key: str, ndim: int) -> FloatArray:
    if key not in data:
        raise KeyError(f"missing array {key!r}; available keys: {data.files}")
    arr = np.asarray(data[key], dtype=float)
    if arr.ndim != ndim:
        raise ValueError(f"{key} must be {ndim}D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{key} contains non-finite values")
    return arr.astype(float, copy=True)


def _circular_mean(params: FloatArray) -> FloatArray:
    return np.angle(np.mean(np.exp(1j * params), axis=0)) % (2.0 * np.pi)


def _fit_pca(params: FloatArray, *, wrap_angles: bool) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    center = _circular_mean(params) if wrap_angles else params.mean(axis=0)
    deltas = params - center.reshape(1, -1)
    if wrap_angles:
        deltas = (deltas + np.pi) % (2.0 * np.pi) - np.pi
    centered = deltas - deltas.mean(axis=0, keepdims=True)
    _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if vh.shape[0] < 2 or singular_values[1] <= 0.0:
        raise ValueError("best-parameter cloud does not contain two non-degenerate PCA directions")
    directions = vh[:2].astype(float, copy=True)
    variances = singular_values * singular_values
    explained = variances / float(variances.sum())
    coordinates = centered @ directions.T
    return center.astype(float), directions, explained[:2].astype(float), coordinates.astype(float)


def _write_csv(
    path: Path,
    pair_indices: NDArray[np.int64],
    labels: list[str],
    coordinates: FloatArray,
    losses: FloatArray,
    success: NDArray[np.bool_],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_index", "training_type", "pc1", "pc2", "best_loss", "log10_best_loss", "success"])
        for idx, label, coord, loss, ok in zip(pair_indices, labels, coordinates, losses, success):
            writer.writerow([int(idx), label, float(coord[0]), float(coord[1]), float(loss), float(np.log10(loss)), bool(ok)])


def _plot(
    output: Path,
    heis_coords: FloatArray,
    warm_coords: FloatArray,
    heis_losses: FloatArray,
    warm_losses: FloatArray,
    explained: FloatArray,
    *,
    title: str,
    dpi: int,
    point_size: float,
    alpha: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    all_log_losses = np.log10(np.concatenate([heis_losses, warm_losses]))
    vmin = float(all_log_losses.min())
    vmax = float(all_log_losses.max())

    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    heis_scatter = ax.scatter(
        heis_coords[:, 0],
        heis_coords[:, 1],
        c=np.log10(heis_losses),
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        marker="o",
        s=point_size,
        alpha=alpha,
        edgecolors="black",
        linewidths=0.25,
        label="Heisenberg-only best params",
    )
    ax.scatter(
        warm_coords[:, 0],
        warm_coords[:, 1],
        c=np.log10(warm_losses),
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        marker="^",
        s=point_size,
        alpha=alpha,
        edgecolors="black",
        linewidths=0.25,
        label="Warm-start best params",
    )
    ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.set_xlabel(f"PC1 ({explained[0] * 100.0:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100.0:.1f}% variance)")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    cbar = fig.colorbar(heis_scatter, ax=ax)
    cbar.set_label("log10(best loss)")
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = np.load(args.input)
    heis_params = _require_array(data, "heisenberg_best_params", 2)
    warm_params = _require_array(data, "warmstart_best_params", 2)
    heis_losses = _require_array(data, "heisenberg_best_losses", 1)
    warm_losses = _require_array(data, "warmstart_best_losses", 1)
    heis_success = np.asarray(data["heisenberg_success"], dtype=bool)
    warm_success = np.asarray(data["warmstart_success"], dtype=bool)
    pair_indices = np.asarray(data["pair_indices"], dtype=np.int64)

    if heis_params.shape != warm_params.shape:
        raise ValueError(f"best parameter shape mismatch: {heis_params.shape} vs {warm_params.shape}")
    if heis_losses.shape != warm_losses.shape or heis_losses.shape != (heis_params.shape[0],):
        raise ValueError("loss arrays must have one value per best parameter")
    if np.any(heis_losses <= 0.0) or np.any(warm_losses <= 0.0):
        raise ValueError("best losses must be positive for log10 coloring")

    combined = np.vstack([heis_params, warm_params])
    center, directions, explained, combined_coords = _fit_pca(combined, wrap_angles=not bool(args.no_wrap))
    heis_coords = combined_coords[: heis_params.shape[0]]
    warm_coords = combined_coords[heis_params.shape[0] :]

    output = args.output or args.input.with_name("best_params_pca_distribution.png")
    projection_output = args.projection_output or args.input.with_name("best_params_pca_projection.npz")
    csv_output = args.csv_output or args.input.with_name("best_params_pca_projection.csv")

    labels = ["heisenberg_only"] * heis_params.shape[0] + ["warmstart"] * warm_params.shape[0]
    csv_pair_indices = np.concatenate([pair_indices, pair_indices])
    csv_coords = np.vstack([heis_coords, warm_coords])
    csv_losses = np.concatenate([heis_losses, warm_losses])
    csv_success = np.concatenate([heis_success, warm_success])

    np.savez_compressed(
        projection_output,
        pca_center=center,
        pca_directions=directions,
        pca_explained_variance_ratio=explained,
        heisenberg_coordinates=heis_coords,
        warmstart_coordinates=warm_coords,
        pair_indices=pair_indices,
        heisenberg_best_losses=heis_losses,
        warmstart_best_losses=warm_losses,
        heisenberg_success=heis_success,
        warmstart_success=warm_success,
        wrap_angles=bool(not args.no_wrap),
    )
    _write_csv(csv_output, csv_pair_indices, labels, csv_coords, csv_losses, csv_success)
    _plot(
        output,
        heis_coords,
        warm_coords,
        heis_losses,
        warm_losses,
        explained,
        title=str(args.title),
        dpi=int(args.dpi),
        point_size=float(args.point_size),
        alpha=float(args.alpha),
    )
    print(f"Wrote {output}")
    print(f"Wrote {projection_output}")
    print(f"Wrote {csv_output}")
    print(f"explained_variance_ratio = {explained.tolist()}")


if __name__ == "__main__":
    main()
