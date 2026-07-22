"""Plot gradient coordinate statistics after removing causally invisible parameters."""

from __future__ import annotations

import argparse
import csv
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

from task2_code_auto.gradient_analysis.invisible_parameter_analysis import analyze_invisible_parameters


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot gradient coordinate statistics excluding invisible parameters.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _load_manifest(experiment_root: Path) -> dict[str, Any]:
    path = experiment_root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    return data


def _write_filtered_csv(path: Path, mode_labels: list[str], coordinates: IntArray, mean_gradients: FloatArray, var_gradients: FloatArray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "coordinate", "mean", "variance"])
        zero_based = coordinates - 1
        for mode_index, label in enumerate(mode_labels):
            for coord, idx in zip(coordinates, zero_based):
                writer.writerow([label, int(coord), float(mean_gradients[mode_index, idx]), float(var_gradients[mode_index, idx])])


def main() -> None:
    args = parse_args()
    manifest = _load_manifest(args.experiment_root)
    data_path = args.experiment_root / "summary" / "gradient_statistics.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"aggregate output not found: {data_path}")
    data = np.load(data_path)
    mode_labels = [str(value) for value in data["mode_labels"].tolist()]
    mean_gradients = np.asarray(data["mean_gradients"], dtype=float)
    var_gradients = np.asarray(data["var_gradients"], dtype=float)
    visible_coords, invisible_coords, metadata = analyze_invisible_parameters(manifest)
    visible_zero_based = visible_coords - 1
    figures = args.experiment_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = args.experiment_root / "summary"

    metadata_path = summary / "filtered_gradient_visible_parameters.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    csv_path = summary / "filtered_gradient_coordinate_statistics.csv"
    _write_filtered_csv(csv_path, mode_labels, visible_coords, mean_gradients, var_gradients)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True, sharex=True)
    for mode_index, label in enumerate(mode_labels):
        axes[0].plot(
            visible_coords,
            mean_gradients[mode_index, visible_zero_based],
            marker=".",
            linewidth=1.1,
            markersize=4,
            label=label,
        )
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("E[partial_i L(theta)]")
    axes[0].set_title("Gradient mean after removing causally invisible parameters")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    for mode_index, label in enumerate(mode_labels):
        axes[1].plot(
            visible_coords,
            var_gradients[mode_index, visible_zero_based],
            marker=".",
            linewidth=1.1,
            markersize=4,
            label=label,
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Original parameter coordinate i")
    axes[1].set_ylabel("Var[partial_i L(theta)]")
    axes[1].set_title("Gradient variance after removing causally invisible parameters")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)
    mean_var_path = figures / "filtered_gradient_coordinate_mean_variance.png"
    fig.savefig(mean_var_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    width = 0.8 / max(len(mode_labels), 1)
    offsets = (np.arange(len(mode_labels)) - (len(mode_labels) - 1) / 2.0) * width
    for mode_index, label in enumerate(mode_labels):
        ax.bar(
            visible_coords + offsets[mode_index],
            var_gradients[mode_index, visible_zero_based],
            width=width,
            alpha=0.78,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Original parameter coordinate i")
    ax.set_ylabel("Var[partial_i L(theta)]")
    ax.set_title("Gradient variance by visible parameter coordinate")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    variance_bar_path = figures / "filtered_gradient_coordinate_variance_bar.png"
    fig.savefig(variance_bar_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"Removed {invisible_coords.size} invisible parameters: {invisible_coords.tolist()}")
    print(f"Kept {visible_coords.size} visible parameters")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {mean_var_path}")
    print(f"Wrote {variance_bar_path}")


if __name__ == "__main__":
    main()
