"""Aggregate and plot loss-mode success-probability experiment results."""

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

from task2_code.hpc_parallel_training.hpc_block_flow import atomic_write_json


LOG_LOSS_RANGE = (-4.5, 1.8)
LOG_LOSS_BINS = np.linspace(LOG_LOSS_RANGE[0], LOG_LOSS_RANGE[1], 43)
SUCCESS_LOG_THRESHOLD = -2.0


DISPLAY_LABELS = {
    "edge_quantum_channel_superoperator_from_mix": "Mixed-channel loss",
    "edge_quantum_channel_superoperator_from_zero": "Zero-channel loss",
    "edge_quantum_channel_superoperator_from_one": "One-channel loss",
    "heisenberg_pauli": "Heisenberg Pauli loss",
}


DISPLAY_COLORS = {
    "edge_quantum_channel_superoperator_from_mix": "#2A9D8F",
    "edge_quantum_channel_superoperator_from_zero": "#457B9D",
    "edge_quantum_channel_superoperator_from_one": "#E9C46A",
    "heisenberg_pauli": "#E76F51",
}


def _display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label.replace("_", " ").title())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate loss-mode success-probability results.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _group_files(experiment_root: Path) -> list[Path]:
    files = sorted((experiment_root / "groups").glob("group_*.json"))
    if not files:
        raise FileNotFoundError(f"no group_*.json files found in {experiment_root / 'groups'}")
    return files


def _collect_trials(manifest: dict[str, Any], files: list[Path]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[int, str]] = set()
    expected_modes = {str(mode["label"]) for mode in manifest["modes"]}
    expected_groups = set(range(1, int(manifest["group_count"]) + 1))
    seen_groups: set[int] = set()

    for path in files:
        payload = _load_json(path)
        if payload.get("artifact_type") != "loss_success_group_result":
            raise ValueError(f"unexpected artifact_type in {path}: {payload.get('artifact_type')!r}")
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError(f"unsupported schema_version in {path}: {payload.get('schema_version')!r}")
        group_index = int(payload["group_index"])
        if group_index not in expected_groups:
            raise ValueError(f"unexpected group_index {group_index} in {path}")
        seen_groups.add(group_index)

        for result in payload.get("results", []):
            mode = dict(result["mode"])
            label = str(mode["label"])
            if label not in expected_modes:
                raise ValueError(f"unexpected mode {label!r} in {path}")
            key = (group_index, label)
            if key in seen:
                raise ValueError(f"duplicate result for group {group_index}, mode {label!r}")
            seen.add(key)
            for trial in result.get("trials", []):
                item = dict(trial)
                item["group_index"] = group_index
                item["mode"] = mode
                out.setdefault(label, []).append(item)

    missing_groups = expected_groups - seen_groups
    if missing_groups:
        raise ValueError(f"missing group results: {sorted(missing_groups)}")
    missing_pairs = [(group, mode) for group in expected_groups for mode in expected_modes if (group, mode) not in seen]
    if missing_pairs:
        raise ValueError(f"missing group/mode results: {missing_pairs[:10]}")
    return out


def _summary_for_trials(trials_by_mode: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, trials in sorted(trials_by_mode.items()):
        best_losses = np.asarray([float(trial["best_loss"]) for trial in trials], dtype=float)
        success = np.asarray([bool(trial["success"]) for trial in trials], dtype=bool)
        failed = np.asarray([bool(trial.get("failed", False)) for trial in trials], dtype=bool)
        thresholds = {float(trial.get("success_threshold", np.nan)) for trial in trials}
        rows.append(
            {
                "mode": label,
                "total_count": int(len(trials)),
                "success_count": int(success.sum()),
                "success_rate": float(success.mean()) if len(success) else 0.0,
                "failed_count": int(failed.sum()),
                "success_threshold": float(next(iter(thresholds))) if len(thresholds) == 1 else np.nan,
                "best_loss_min": float(best_losses.min()) if best_losses.size else np.nan,
                "best_loss_median": float(np.median(best_losses)) if best_losses.size else np.nan,
                "best_loss_mean": float(best_losses.mean()) if best_losses.size else np.nan,
                "best_loss_max": float(best_losses.max()) if best_losses.size else np.nan,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [str(row["mode"]) for row in rows]
    success_rates = [float(row["success_rate"]) for row in rows]
    medians = [float(row["best_loss_median"]) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(max(10, 2.5 * len(rows)), 9), constrained_layout=True)
    axes[0].bar(labels, success_rates)
    axes[0].set_ylabel("success rate")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, medians)
    axes[1].set_ylabel("median best loss")
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", alpha=0.3)

    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_loss_histograms(path: Path, trials_by_mode: dict[str, list[dict[str, Any]]]) -> None:
    modes = sorted(trials_by_mode)
    fig, axes = plt.subplots(len(modes), 1, figsize=(10.5, max(3.2, 2.9 * len(modes))), squeeze=False, sharex=True)
    for ax, label in zip(axes.ravel(), modes):
        best_losses = np.asarray([float(trial["best_loss"]) for trial in trials_by_mode[label]], dtype=float)
        positive = best_losses[best_losses > 0]
        log_values = np.log10(positive) if positive.size else np.asarray([], dtype=float)
        weights = np.full(log_values.shape, 100.0 / best_losses.size, dtype=float) if best_losses.size else None
        ax.hist(
            log_values,
            bins=LOG_LOSS_BINS,
            weights=weights,
            color=DISPLAY_COLORS.get(label, "#457B9D"),
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
        ax.set_title(_display_label(label), fontsize=12, weight="bold")
        ax.set_ylabel("percentage (%)")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes.ravel()[-1].set_xlabel(r"$\log_{10}$(best loss)")
    fig.suptitle("Best-loss distribution with shared success threshold", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest = _load_json(args.experiment_root / "manifest.json")
    trials_by_mode = _collect_trials(manifest, _group_files(args.experiment_root))
    rows = _summary_for_trials(trials_by_mode)
    summary_dir = args.experiment_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [
        summary_dir / "summary.json",
        summary_dir / "summary.csv",
        summary_dir / "success_summary.png",
        summary_dir / "best_loss_histograms.png",
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"summary outputs already exist; pass --overwrite to replace them: {existing}")

    payload = {
        "schema_version": 1,
        "artifact_type": "loss_success_summary",
        "experiment_root": args.experiment_root,
        "preset": manifest.get("preset"),
        "block_index": manifest.get("block_index"),
        "block_qubits": manifest.get("block_qubits"),
        "target_bit": manifest.get("target_bit"),
        "lightcone_qubits": manifest.get("lightcone_qubits"),
        "success_threshold": manifest.get("success_threshold"),
        "rows": rows,
    }
    atomic_write_json(summary_dir / "summary.json", payload)
    _write_csv(summary_dir / "summary.csv", rows)
    _plot_summary(summary_dir / "success_summary.png", rows)
    _plot_loss_histograms(summary_dir / "best_loss_histograms.png", trials_by_mode)

    print(f"Saved summary: {summary_dir / 'summary.json'}")
    for row in rows:
        print(f"{row['mode']}: {row['success_count']}/{row['total_count']} = {row['success_rate']:.4f}")


if __name__ == "__main__":
    main()
