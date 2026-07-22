"""Plot random-parameter loss histograms."""

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
    parser = argparse.ArgumentParser(description="Plot random-parameter loss histograms.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _bins_for(values: FloatArray, count: int) -> FloatArray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("cannot plot non-finite loss array")
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if lo == hi:
        pad = max(abs(lo) * 0.05, 1e-12)
        lo -= pad
        hi += pad
    return np.linspace(lo, hi, max(int(count), 2))


def _log10_loss(values: FloatArray) -> FloatArray:
    if np.any(values < 0.0):
        raise ValueError("loss values must be non-negative")
    positive = values[values > 0.0]
    if positive.size == 0:
        return np.full(values.shape, -300.0, dtype=float)
    floor = float(np.min(positive)) * 1e-3
    return np.log10(np.maximum(values, floor))


def main() -> None:
    args = parse_args()
    data_path = args.experiment_root / "summary" / "loss_distribution.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"aggregate output not found: {data_path}")
    data = np.load(data_path)
    mode_labels = [str(value) for value in data["mode_labels"].tolist()]
    losses = np.asarray(data["losses"], dtype=float)
    figures = args.experiment_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(mode_labels), 1, figsize=(9, 3.8 * len(mode_labels)), constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    for mode_index, label in enumerate(mode_labels):
        values = losses[mode_index]
        axes_arr[mode_index].hist(values, bins=_bins_for(values, int(args.bins)), alpha=0.8, density=False)
        axes_arr[mode_index].set_title(f"Loss distribution: {label}")
        axes_arr[mode_index].set_xlabel("loss")
        axes_arr[mode_index].set_ylabel("count")
        axes_arr[mode_index].grid(alpha=0.25)
    path = figures / "loss_value_histograms.png"
    fig.savefig(path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(len(mode_labels), 1, figsize=(9, 3.8 * len(mode_labels)), constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    for mode_index, label in enumerate(mode_labels):
        values = _log10_loss(losses[mode_index])
        axes_arr[mode_index].hist(values, bins=_bins_for(values, int(args.bins)), alpha=0.8, density=False)
        axes_arr[mode_index].set_title(f"log10(loss) distribution: {label}")
        axes_arr[mode_index].set_xlabel("log10(loss)")
        axes_arr[mode_index].set_ylabel("count")
        axes_arr[mode_index].grid(alpha=0.25)
    log_path = figures / "log10_loss_value_histograms.png"
    fig.savefig(log_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for mode_index, label in enumerate(mode_labels):
        values = np.sort(losses[mode_index])
        y = np.arange(1, values.size + 1, dtype=float) / values.size
        ax.plot(values, y, linewidth=1.5, label=label)
    ax.set_xlabel("loss")
    ax.set_ylabel("empirical CDF")
    ax.set_title("Loss empirical CDF at random initial parameters")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    ecdf_path = figures / "loss_value_ecdf.png"
    fig.savefig(ecdf_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {path}")
    print(f"Wrote {log_path}")
    print(f"Wrote {ecdf_path}")


if __name__ == "__main__":
    main()
