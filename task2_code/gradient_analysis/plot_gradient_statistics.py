"""Plot random-initialization gradient statistics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

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
    parser = argparse.ArgumentParser(description="Plot random-initialization gradient statistics.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _finite_range(values: FloatArray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot plot an array with no finite values")
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if lo == hi:
        pad = max(abs(lo) * 0.05, 1e-12)
        return lo - pad, hi + pad
    return lo, hi


def _log10_or_floor(values: FloatArray) -> FloatArray:
    arr = np.asarray(values, dtype=float)
    if np.any(arr < 0.0):
        raise ValueError("gradient squared norms must be non-negative")
    positive = arr[arr > 0.0]
    if positive.size == 0:
        return np.full(arr.shape, -300.0, dtype=float)
    floor = float(np.min(positive)) * 1e-3
    return np.log10(np.maximum(arr, floor))


def _bins_for(values: FloatArray, count: int) -> FloatArray:
    lo, hi = _finite_range(values)
    return np.linspace(lo, hi, max(int(count), 2))


def main() -> None:
    args = parse_args()
    data_path = args.experiment_root / "summary" / "gradient_statistics.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"aggregate output not found: {data_path}")
    data = np.load(data_path)
    mode_labels = [str(value) for value in data["mode_labels"].tolist()]
    selected_coordinates = [int(value) for value in data["selected_coordinates"]]
    losses = np.asarray(data["losses"], dtype=float)
    selected_gradients = np.asarray(data["selected_gradients"], dtype=float)
    grad_sq_norms = np.asarray(data["grad_sq_norms"], dtype=float)
    normalized_grad_sq_norms = np.asarray(data["normalized_grad_sq_norms"], dtype=float)
    mean_gradients = np.asarray(data["mean_gradients"], dtype=float)
    var_gradients = np.asarray(data["var_gradients"], dtype=float)
    figures = args.experiment_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(selected_coordinates), 1, figsize=(9, 3.2 * len(selected_coordinates)), constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    for coord_pos, coord in enumerate(selected_coordinates):
        ax = axes_arr[coord_pos]
        for mode_index, label in enumerate(mode_labels):
            ax.hist(selected_gradients[mode_index, :, coord_pos], bins=int(args.bins), alpha=0.55, density=True, label=label)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
        ax.set_title(f"Distribution of partial derivative coordinate i={coord}")
        ax.set_xlabel(f"partial_{coord} L(theta)")
        ax.set_ylabel("density")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
    out = figures / "selected_gradient_coordinate_histograms.png"
    fig.savefig(out, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    log_grad_sq_norms = _log10_or_floor(grad_sq_norms)
    log_normalized_grad_sq_norms = _log10_or_floor(normalized_grad_sq_norms)
    grad_bins = _bins_for(log_grad_sq_norms, int(args.bins))
    norm_bins = _bins_for(log_normalized_grad_sq_norms, int(args.bins))
    for mode_index, label in enumerate(mode_labels):
        axes[0].hist(log_grad_sq_norms[mode_index], bins=grad_bins, alpha=0.55, density=True, label=label)
        axes[1].hist(log_normalized_grad_sq_norms[mode_index], bins=norm_bins, alpha=0.55, density=True, label=label)
    axes[0].set_title("Distribution of gradient squared norm")
    axes[0].set_xlabel("log10(|grad L(theta)|^2)")
    axes[0].set_ylabel("density")
    axes[1].set_title("Distribution of normalized gradient squared norm")
    axes[1].set_xlabel("log10(|grad L(theta)|^2 / theta_size)")
    axes[1].set_ylabel("density")
    for ax in axes:
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
    out = figures / "gradient_signal_strength_histograms.png"
    fig.savefig(out, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    for mode_index, label in enumerate(mode_labels):
        ax.scatter(
            losses[mode_index],
            np.maximum(normalized_grad_sq_norms[mode_index], 1e-300),
            s=12,
            alpha=0.45,
            edgecolors="none",
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("loss at random initialization")
    ax.set_ylabel("|grad L(theta)|^2 / theta_size")
    ax.set_title("Loss vs normalized gradient signal at random initialization")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    out = figures / "loss_vs_normalized_gradient_squared.png"
    fig.savefig(out, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    coordinates = np.arange(1, mean_gradients.shape[1] + 1, dtype=int)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True, sharex=True)
    for mode_index, label in enumerate(mode_labels):
        axes[0].plot(coordinates, mean_gradients[mode_index], marker=".", linewidth=1.1, markersize=4, label=label)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("E[partial_i L(theta)]")
    axes[0].set_title("Per-parameter gradient mean over random initializations")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    for mode_index, label in enumerate(mode_labels):
        axes[1].plot(coordinates, var_gradients[mode_index], marker=".", linewidth=1.1, markersize=4, label=label)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Parameter coordinate i")
    axes[1].set_ylabel("Var[partial_i L(theta)]")
    axes[1].set_title("Per-parameter gradient variance over random initializations")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)
    out = figures / "gradient_coordinate_mean_variance.png"
    fig.savefig(out, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    width = 0.8 / max(len(mode_labels), 1)
    offsets = (np.arange(len(mode_labels)) - (len(mode_labels) - 1) / 2.0) * width
    for mode_index, label in enumerate(mode_labels):
        ax.bar(coordinates + offsets[mode_index], var_gradients[mode_index], width=width, alpha=0.78, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Parameter coordinate i")
    ax.set_ylabel("Var[partial_i L(theta)]")
    ax.set_title("Gradient variance by parameter coordinate")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    out = figures / "gradient_coordinate_variance_bar.png"
    fig.savefig(out, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {figures / 'selected_gradient_coordinate_histograms.png'}")
    print(f"Wrote {figures / 'gradient_signal_strength_histograms.png'}")
    print(f"Wrote {figures / 'loss_vs_normalized_gradient_squared.png'}")
    print(f"Wrote {figures / 'gradient_coordinate_mean_variance.png'}")
    print(f"Wrote {figures / 'gradient_coordinate_variance_bar.png'}")


if __name__ == "__main__":
    main()
