"""Cluster paired best parameters with density-based clustering."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any
import warnings

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster best parameters from the paired warm-start experiment.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "task2_code/loss_success_experiment/data/paired_warmstart_n32_8blocks_block05_400pairs/summary/paired_best_params.npz"
        ),
    )
    parser.add_argument(
        "--projection",
        type=Path,
        default=Path(
            "task2_code/loss_success_experiment/data/paired_warmstart_n32_8blocks_block05_400pairs/summary/best_params_pca_projection.npz"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="default: <input-dir>")
    parser.add_argument("--min-cluster-size", type=int, default=8)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--title", default="Best-parameter clusters projected onto PCA plane")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--point-size", type=float, default=34.0)
    parser.add_argument("--no-wrap", action="store_true", default=False)
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


def _angular_features(params: FloatArray, *, wrap_angles: bool) -> FloatArray:
    center = _circular_mean(params) if wrap_angles else params.mean(axis=0)
    features = params - center.reshape(1, -1)
    if wrap_angles:
        features = (features + np.pi) % (2.0 * np.pi) - np.pi
    return features.astype(float, copy=True)


def _cluster(params: FloatArray, *, min_cluster_size: int, min_samples: int, wrap_angles: bool) -> IntArray:
    if params.shape[0] < min_cluster_size:
        return np.full(params.shape[0], -1, dtype=np.int64)
    cluster_module = importlib.import_module("sklearn.cluster")
    hdbscan_cls = getattr(cluster_module, "HDBSCAN")
    features = _angular_features(params, wrap_angles=wrap_angles)
    model = hdbscan_cls(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric="euclidean",
        cluster_selection_method="eom",
        allow_single_cluster=False,
        copy=True,
        n_jobs=-1,
    )
    labels = model.fit_predict(features)
    return np.asarray(labels, dtype=np.int64)


def _cluster_summary(labels: IntArray, losses: FloatArray, success: NDArray[np.bool_]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "total": int(labels.size),
        "noise_count": int(np.sum(labels == -1)),
        "cluster_count": int(len([label for label in sorted(set(labels.tolist())) if label != -1])),
        "clusters": [],
    }
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        out["clusters"].append(
            {
                "label": int(label),
                "kind": "noise" if label == -1 else "cluster",
                "count": int(mask.sum()),
                "best_loss_min": float(losses[mask].min()),
                "best_loss_median": float(np.median(losses[mask])),
                "best_loss_max": float(losses[mask].max()),
                "success_count": int(success[mask].sum()),
                "success_rate": float(success[mask].mean()),
            }
        )
    return out


def _scatter_clusters(
    ax: plt.Axes,
    coords: FloatArray,
    labels: IntArray,
    losses: FloatArray,
    *,
    marker: str,
    title: str,
    point_size: float,
) -> None:
    non_noise = sorted(label for label in set(labels.tolist()) if label != -1)
    cmap = plt.get_cmap("tab20")
    for pos, label in enumerate(non_noise):
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            marker=marker,
            s=point_size,
            color=cmap(pos % 20),
            edgecolors="black",
            linewidths=0.25,
            alpha=0.82,
            label=f"cluster {label} (n={int(mask.sum())}, med={np.median(losses[mask]):.3g})",
        )
    noise = labels == -1
    if np.any(noise):
        ax.scatter(
            coords[noise, 0],
            coords[noise, 1],
            marker=marker,
            s=point_size * 0.75,
            color="lightgray",
            edgecolors="black",
            linewidths=0.2,
            alpha=0.55,
            label=f"noise (n={int(noise.sum())})",
        )
    ax.axhline(0.0, color="0.82", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.82", linewidth=0.8, zorder=0)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=7, frameon=True)


def _write_labels_csv(
    path: Path,
    pair_indices: IntArray,
    labels_by_type: dict[str, IntArray],
    losses_by_type: dict[str, FloatArray],
    success_by_type: dict[str, NDArray[np.bool_]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_index", "training_type", "cluster_label", "best_loss", "log10_best_loss", "success"])
        for training_type, labels in labels_by_type.items():
            losses = losses_by_type[training_type]
            success = success_by_type[training_type]
            for idx, label, loss, ok in zip(pair_indices, labels, losses, success):
                writer.writerow([int(idx), training_type, int(label), float(loss), float(np.log10(loss)), bool(ok)])


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._hdbscan.hdbscan")
    args = parse_args()
    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.input)
    projection = np.load(args.projection)
    pair_indices = np.asarray(data["pair_indices"], dtype=np.int64)
    heis_params = _require_array(data, "heisenberg_best_params", 2)
    warm_params = _require_array(data, "warmstart_best_params", 2)
    heis_losses = _require_array(data, "heisenberg_best_losses", 1)
    warm_losses = _require_array(data, "warmstart_best_losses", 1)
    heis_success = np.asarray(data["heisenberg_success"], dtype=bool)
    warm_success = np.asarray(data["warmstart_success"], dtype=bool)
    heis_coords = _require_array(projection, "heisenberg_coordinates", 2)
    warm_coords = _require_array(projection, "warmstart_coordinates", 2)

    heis_labels = _cluster(
        heis_params,
        min_cluster_size=int(args.min_cluster_size),
        min_samples=int(args.min_samples),
        wrap_angles=not bool(args.no_wrap),
    )
    warm_labels = _cluster(
        warm_params,
        min_cluster_size=int(args.min_cluster_size),
        min_samples=int(args.min_samples),
        wrap_angles=not bool(args.no_wrap),
    )
    combined_labels = _cluster(
        np.vstack([heis_params, warm_params]),
        min_cluster_size=int(args.min_cluster_size),
        min_samples=int(args.min_samples),
        wrap_angles=not bool(args.no_wrap),
    )
    heis_failed_mask = ~heis_success
    heis_failed_subset_labels = _cluster(
        heis_params[heis_failed_mask],
        min_cluster_size=int(args.min_cluster_size),
        min_samples=int(args.min_samples),
        wrap_angles=not bool(args.no_wrap),
    )
    heis_failed_labels = np.full(heis_params.shape[0], -2, dtype=np.int64)
    heis_failed_labels[heis_failed_mask] = heis_failed_subset_labels

    labels_npz = output_dir / "best_params_hdbscan_clusters.npz"
    np.savez_compressed(
        labels_npz,
        pair_indices=pair_indices,
        heisenberg_cluster_labels=heis_labels,
        warmstart_cluster_labels=warm_labels,
        heisenberg_failed_cluster_labels=heis_failed_labels,
        combined_heisenberg_cluster_labels=combined_labels[: heis_params.shape[0]],
        combined_warmstart_cluster_labels=combined_labels[heis_params.shape[0] :],
        min_cluster_size=int(args.min_cluster_size),
        min_samples=int(args.min_samples),
        wrap_angles=bool(not args.no_wrap),
    )

    labels_csv = output_dir / "best_params_hdbscan_cluster_labels.csv"
    _write_labels_csv(
        labels_csv,
        pair_indices,
        {"heisenberg_only": heis_labels, "warmstart": warm_labels},
        {"heisenberg_only": heis_losses, "warmstart": warm_losses},
        {"heisenberg_only": heis_success, "warmstart": warm_success},
    )

    summary = {
        "artifact_type": "best_params_hdbscan_clusters",
        "input": str(args.input),
        "projection": str(args.projection),
        "min_cluster_size": int(args.min_cluster_size),
        "min_samples": int(args.min_samples),
        "wrap_angles": bool(not args.no_wrap),
        "labels_npz": labels_npz.name,
        "labels_csv": labels_csv.name,
        "heisenberg_only": _cluster_summary(heis_labels, heis_losses, heis_success),
        "heisenberg_failed_only": _cluster_summary(
            heis_failed_subset_labels,
            heis_losses[heis_failed_mask],
            heis_success[heis_failed_mask],
        ),
        "warmstart": _cluster_summary(warm_labels, warm_losses, warm_success),
        "combined": _cluster_summary(
            combined_labels,
            np.concatenate([heis_losses, warm_losses]),
            np.concatenate([heis_success, warm_success]),
        ),
    }
    summary_json = output_dir / "best_params_hdbscan_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)
    _scatter_clusters(
        axes[0],
        heis_coords,
        heis_labels,
        heis_losses,
        marker="o",
        title="Heisenberg-only best params",
        point_size=float(args.point_size),
    )
    _scatter_clusters(
        axes[1],
        warm_coords,
        warm_labels,
        warm_losses,
        marker="^",
        title="Warm-start best params",
        point_size=float(args.point_size),
    )
    fig.suptitle(str(args.title))
    cluster_png = output_dir / "best_params_hdbscan_clusters.png"
    fig.savefig(cluster_png, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {labels_npz}")
    print(f"Wrote {labels_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {cluster_png}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
