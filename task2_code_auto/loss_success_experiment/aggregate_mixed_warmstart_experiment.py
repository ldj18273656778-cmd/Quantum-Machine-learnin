"""Aggregate mixed-edge to Heisenberg warmstart experiment results."""

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

from task2_code_auto.hpc_parallel_training.hpc_block_flow import atomic_write_json


LOG_LOSS_RANGE = (-4.5, 1.8)
LOG_LOSS_BINS = np.linspace(LOG_LOSS_RANGE[0], LOG_LOSS_RANGE[1], 43)
SUCCESS_LOG_THRESHOLD = -2.0


SERIES_COLORS = {
    "Mixed-channel loss": "#2A9D8F",
    "Heisenberg loss after mixed pretraining": "#457B9D",
    "Warm-start Heisenberg best loss": "#E76F51",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate mixed-warmstart success results.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _group_files(experiment_root: Path) -> list[Path]:
    files = sorted((experiment_root / "groups").glob("group_*_mixed_warmstart.json"))
    if not files:
        raise FileNotFoundError(f"no group_*_mixed_warmstart.json files found in {experiment_root / 'groups'}")
    return files


def _collect_trials(manifest: dict[str, Any], files: list[Path]) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    expected_groups = set(range(1, int(manifest["group_count"]) + 1))
    seen_groups: set[int] = set()

    for path in files:
        payload = _load_json(path)
        if payload.get("artifact_type") != "mixed_warmstart_group_result":
            raise ValueError(f"unexpected artifact_type in {path}: {payload.get('artifact_type')!r}")
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError(f"unsupported schema_version in {path}: {payload.get('schema_version')!r}")
        group_index = int(payload["group_index"])
        if group_index not in expected_groups:
            raise ValueError(f"unexpected group_index {group_index} in {path}")
        if group_index in seen_groups:
            raise ValueError(f"duplicate group result for group {group_index}")
        seen_groups.add(group_index)
        for trial in payload["result"].get("trials", []):
            item = dict(trial)
            item["group_index"] = group_index
            trials.append(item)

    missing_groups = expected_groups - seen_groups
    if missing_groups:
        raise ValueError(f"missing group results: {sorted(missing_groups)}")
    expected_total = int(manifest["total_restarts"])
    if len(trials) != expected_total:
        raise ValueError(f"expected {expected_total} trials, collected {len(trials)}")
    return trials


def _stats(values: np.ndarray[Any, Any], prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_min": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_min": float(values.min()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_max": float(values.max()),
    }


def _summary_row(trials: list[dict[str, Any]]) -> dict[str, Any]:
    success = np.asarray([bool(trial["success"]) for trial in trials], dtype=bool)
    mixed_failed = np.asarray([bool(trial["mixed_failed"]) for trial in trials], dtype=bool)
    heisenberg_failed = np.asarray([bool(trial["heisenberg_failed"]) for trial in trials], dtype=bool)
    thresholds = {float(trial["success_threshold"]) for trial in trials}
    row: dict[str, Any] = {
        "total_count": int(len(trials)),
        "success_count": int(success.sum()),
        "success_rate": float(success.mean()) if len(success) else 0.0,
        "mixed_failed_count": int(mixed_failed.sum()),
        "heisenberg_failed_count": int(heisenberg_failed.sum()),
        "success_threshold": float(next(iter(thresholds))) if len(thresholds) == 1 else np.nan,
    }
    row.update(_stats(np.asarray([float(trial["mixed_best_loss"]) for trial in trials], dtype=float), "mixed_best_loss"))
    row.update(
        _stats(
            np.asarray([float(trial["mixed_best_params_mixed_edge_loss"]) for trial in trials], dtype=float),
            "mixed_best_params_mixed_edge_loss",
        )
    )
    row.update(
        _stats(
            np.asarray([float(trial["mixed_best_params_heisenberg_loss"]) for trial in trials], dtype=float),
            "mixed_best_params_heisenberg_loss",
        )
    )
    row.update(_stats(np.asarray([float(trial["heisenberg_best_loss"]) for trial in trials], dtype=float), "heisenberg_best_loss"))
    return row


def _write_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _plot_success(path: Path, row: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["mixed -> heisenberg"], [float(row["success_rate"])])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("success rate")
    ax.set_title("Warmstart Heisenberg success rate")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(path: Path, trials: list[dict[str, Any]]) -> None:
    x = np.asarray([float(trial["mixed_best_params_mixed_edge_loss"]) for trial in trials], dtype=float)
    y = np.asarray([float(trial["mixed_best_params_heisenberg_loss"]) for trial in trials], dtype=float)
    c = np.asarray([bool(trial["success"]) for trial in trials], dtype=bool)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x[~c], y[~c], s=10, alpha=0.35, label="failed")
    if c.any():
        ax.scatter(x[c], y[c], s=16, alpha=0.8, label="success")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mixed best params: mixed edge loss")
    ax.set_ylabel("mixed best params: Heisenberg loss")
    ax.set_title("Does mixed pretraining help Heisenberg?")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_histograms(path: Path, trials: list[dict[str, Any]]) -> None:
    series = [
        ("Mixed-channel loss", np.asarray([float(trial["mixed_best_loss"]) for trial in trials], dtype=float)),
        (
            "Heisenberg loss after mixed pretraining",
            np.asarray([float(trial["heisenberg_initial_loss"]) for trial in trials], dtype=float),
        ),
        ("Warm-start Heisenberg best loss", np.asarray([float(trial["heisenberg_best_loss"]) for trial in trials], dtype=float)),
    ]
    fig, axes = plt.subplots(len(series), 1, figsize=(10.5, 8.6), squeeze=False, sharex=True)
    for ax, (title, values) in zip(axes.ravel(), series):
        positive = values[values > 0]
        log_values = np.log10(positive) if positive.size else np.asarray([], dtype=float)
        weights = np.full(log_values.shape, 100.0 / values.size, dtype=float) if values.size else None
        ax.hist(
            log_values,
            bins=LOG_LOSS_BINS,
            weights=weights,
            color=SERIES_COLORS.get(title, "#457B9D"),
            alpha=0.82,
            edgecolor="white",
            linewidth=0.7,
        )
        ax.axvline(SUCCESS_LOG_THRESHOLD, color="#222222", linestyle="--", linewidth=1.4)
        ax.text(
            SUCCESS_LOG_THRESHOLD + 0.05,
            0.92,
            r"success threshold $10^{-2}$",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=9,
            color="#222222",
        )
        ax.set_xlim(LOG_LOSS_RANGE)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_ylabel("percentage (%)")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes.ravel()[-1].set_xlabel(r"$\log_{10}$(loss)")
    fig.suptitle("Warm-start loss distributions with shared success threshold", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest = _load_json(args.experiment_root / "manifest.json")
    if manifest.get("artifact_type") != "mixed_warmstart_experiment":
        raise ValueError(f"unexpected manifest artifact_type: {manifest.get('artifact_type')!r}")
    trials = _collect_trials(manifest, _group_files(args.experiment_root))
    row = _summary_row(trials)
    summary_dir = args.experiment_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        summary_dir / "summary.json",
        summary_dir / "summary.csv",
        summary_dir / "success_summary.png",
        summary_dir / "mixed_vs_heisenberg_scatter.png",
        summary_dir / "loss_histograms.png",
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"summary outputs already exist; pass --overwrite to replace them: {existing}")

    payload = {
        "schema_version": 1,
        "artifact_type": "mixed_warmstart_summary",
        "experiment_root": args.experiment_root,
        "preset": manifest.get("preset"),
        "block_index": manifest.get("block_index"),
        "block_qubits": manifest.get("block_qubits"),
        "target_bit": manifest.get("target_bit"),
        "lightcone_qubits": manifest.get("lightcone_qubits"),
        "stage1": manifest.get("stage1"),
        "stage2": manifest.get("stage2"),
        "row": row,
    }
    atomic_write_json(summary_dir / "summary.json", payload)
    _write_csv(summary_dir / "summary.csv", row)
    _plot_success(summary_dir / "success_summary.png", row)
    _plot_scatter(summary_dir / "mixed_vs_heisenberg_scatter.png", trials)
    _plot_histograms(summary_dir / "loss_histograms.png", trials)

    print(f"Saved summary: {summary_dir / 'summary.json'}")
    print(f"mixed->heisenberg: {row['success_count']}/{row['total_count']} = {row['success_rate']:.4f}")


if __name__ == "__main__":
    main()
