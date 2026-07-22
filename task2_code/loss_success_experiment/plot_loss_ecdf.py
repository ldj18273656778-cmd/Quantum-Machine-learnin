"""Plot ECDF figures for loss-success and mixed-warmstart experiments."""

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
from matplotlib.ticker import MultipleLocator
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


LOG_LOSS_RANGE = (-4.5, 1.8)
SUCCESS_LOG_THRESHOLD = -2.0

DISPLAY_LABELS = {
    "edge_quantum_channel_superoperator_from_mix": "Mixed-channel loss",
    "edge_quantum_channel_superoperator_from_zero": "Zero-channel loss",
    "edge_quantum_channel_superoperator_from_one": "One-channel loss",
    "heisenberg_pauli": "Heisenberg Pauli loss",
    "mixed_best_loss": "Mixed-channel loss",
    "heisenberg_initial_loss": "Heisenberg loss after mixed pretraining",
    "heisenberg_best_loss": "Warm-start Heisenberg best loss",
}

DISPLAY_COLORS = {
    "edge_quantum_channel_superoperator_from_mix": "#2A9D8F",
    "edge_quantum_channel_superoperator_from_zero": "#457B9D",
    "edge_quantum_channel_superoperator_from_one": "#E9C46A",
    "heisenberg_pauli": "#E76F51",
    "mixed_best_loss": "#2A9D8F",
    "heisenberg_initial_loss": "#457B9D",
    "heisenberg_best_loss": "#E76F51",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ECDF of best losses for a success experiment.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="default: <experiment-root>/summary/loss_ecdf.png")
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label.replace("_", " ").title())


def _collect_loss_success(experiment_root: Path, manifest: dict[str, Any]) -> list[tuple[str, FloatArray]]:
    expected_modes = {str(mode["label"]) for mode in manifest["modes"]}
    series: dict[str, list[float]] = {mode: [] for mode in sorted(expected_modes)}
    for path in sorted((experiment_root / "groups").glob("group_*.json")):
        payload = _load_json(path)
        if payload.get("artifact_type") != "loss_success_group_result":
            continue
        for result in payload.get("results", []):
            label = str(result["mode"]["label"])
            if label not in expected_modes:
                raise ValueError(f"unexpected mode {label!r} in {path}")
            for trial in result.get("trials", []):
                series[label].append(float(trial["best_loss"]))
    return [(label, np.asarray(values, dtype=float)) for label, values in series.items() if values]


def _collect_mixed_warmstart(experiment_root: Path) -> list[tuple[str, FloatArray]]:
    mixed: list[float] = []
    heis_initial: list[float] = []
    heis_best: list[float] = []
    for path in sorted((experiment_root / "groups").glob("group_*_mixed_warmstart.json")):
        payload = _load_json(path)
        if payload.get("artifact_type") != "mixed_warmstart_group_result":
            continue
        for trial in payload["result"].get("trials", []):
            mixed.append(float(trial["mixed_best_loss"]))
            heis_initial.append(float(trial["heisenberg_initial_loss"]))
            heis_best.append(float(trial["heisenberg_best_loss"]))
    return [
        ("mixed_best_loss", np.asarray(mixed, dtype=float)),
        ("heisenberg_initial_loss", np.asarray(heis_initial, dtype=float)),
        ("heisenberg_best_loss", np.asarray(heis_best, dtype=float)),
    ]


def _plot_ecdf(path: Path, series: list[tuple[str, FloatArray]], *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    for label, values in series:
        positive = values[values > 0]
        if positive.size == 0:
            continue
        log_values = np.sort(np.log10(positive))
        y = np.arange(1, log_values.size + 1, dtype=float) / float(log_values.size) * 100.0
        ax.step(
            log_values,
            y,
            where="post",
            linewidth=2.2,
            color=DISPLAY_COLORS.get(label, "#457B9D"),
            label=f"{_display_label(label)} (n={log_values.size})",
        )
    ax.axvline(SUCCESS_LOG_THRESHOLD, color="#222222", linestyle="--", linewidth=1.4)
    ax.text(
        SUCCESS_LOG_THRESHOLD + 0.05,
        8.0,
        r"success threshold $10^{-2}$",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#222222",
    )
    ax.set_xlim(LOG_LOSS_RANGE)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel(r"$\log_{10}$(loss)")
    ax.set_ylabel("cumulative percentage (%)")
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_facecolor("#F7F9FA")
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(20.0))
    ax.yaxis.set_minor_locator(MultipleLocator(10.0))
    ax.grid(which="major", color="#C8D2DC", linewidth=0.85, alpha=0.75)
    ax.grid(which="minor", color="#E1E7EC", linewidth=0.55, alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest = _load_json(args.experiment_root / "manifest.json")
    artifact_type = str(manifest.get("artifact_type", ""))
    if artifact_type == "mixed_warmstart_experiment":
        series = _collect_mixed_warmstart(args.experiment_root)
        default_title = f"Block {manifest.get('block_index')} warm-start loss ECDF"
        default_name = "loss_ecdf.png"
    else:
        series = _collect_loss_success(args.experiment_root, manifest)
        default_title = f"Block {manifest.get('block_index')} random-restart best-loss ECDF"
        default_name = "best_loss_ecdf.png"
    if not series or any(values.size == 0 for _label, values in series):
        raise ValueError(f"no ECDF data found under {args.experiment_root}")
    output = args.output or args.experiment_root / "summary" / default_name
    _plot_ecdf(output, series, title=str(args.title or default_title))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
