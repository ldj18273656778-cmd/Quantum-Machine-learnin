"""Add existing reduced-steps 0 and 150 results to a warmstart-rate sweep."""

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
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SWEEP_ROOT = Path(
    "/home/um202470160/task2_code/loss_success_experiment/success_vs_warmstart_rate/data/"
    "success_vs_warmstart_rate_n32_8blocks_block05_heis150_1000"
)
DEFAULT_HEIS_ROOT = Path(
    "/home/um202470160/task2_code/loss_success_experiment/data/"
    "loss_success_n32_8blocks_block05_20260615_144958"
)
DEFAULT_PAIRED_ROOT = Path(
    "/home/um202470160/task2_code/loss_success_experiment/data/"
    "paired_warmstart_n32_8blocks_block05_400pairs"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extend warmstart-rate plots with existing 0/150-step results.")
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--heisenberg-only-root", type=Path, default=DEFAULT_HEIS_ROOT)
    parser.add_argument("--paired-warmstart-root", type=Path, default=DEFAULT_PAIRED_ROOT)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _heisenberg_only_row(root: Path) -> dict[str, Any]:
    summary = _load_json(root / "summary" / "summary.json")
    source_row = next(row for row in summary["rows"] if row["mode"] == "heisenberg_pauli")
    losses: list[float] = []
    for path in sorted((root / "groups").glob("group_*_all_modes.json")):
        group = _load_json(path)
        result = next(item for item in group["results"] if item["mode"]["label"] == "heisenberg_pauli")
        losses.extend(float(trial["best_loss"]) for trial in result["trials"])
    values = np.asarray(losses, dtype=float)
    return {
        "warmstart_iterations": 0,
        "trials": int(source_row["total_count"]),
        "success_count": int(source_row["success_count"]),
        "success_rate": float(source_row["success_rate"]),
        "final_best_loss_mean": float(np.mean(values)),
        "final_best_loss_median": float(np.median(values)),
        "final_best_loss_q25": float(np.quantile(values, 0.25)),
        "final_best_loss_q75": float(np.quantile(values, 0.75)),
        "warmstart_best_loss_mean": "",
        "warmstart_best_loss_median": "",
        "source": str(root),
        "source_label": "heisenberg_only_150_steps",
    }


def _paired_warmstart_150_row(root: Path) -> dict[str, Any]:
    summary = _load_json(root / "summary" / "paired_summary.json")
    heis_losses: list[float] = []
    warm_losses: list[float] = []
    for path in sorted((root / "groups").glob("group_*_paired.json")):
        group = _load_json(path)
        for trial in group["trials"]:
            heis_losses.append(float(trial["warmstart"]["heisenberg_best_loss"]))
            warm_losses.append(float(trial["warmstart"]["mixed_best_loss"]))
    heis_values = np.asarray(heis_losses, dtype=float)
    warm_values = np.asarray(warm_losses, dtype=float)
    return {
        "warmstart_iterations": 150,
        "trials": int(summary["total_pairs"]),
        "success_count": int(summary["warmstart_success_count"]),
        "success_rate": float(summary["warmstart_success_rate"]),
        "final_best_loss_mean": float(np.mean(heis_values)),
        "final_best_loss_median": float(np.median(heis_values)),
        "final_best_loss_q25": float(np.quantile(heis_values, 0.25)),
        "final_best_loss_q75": float(np.quantile(heis_values, 0.75)),
        "warmstart_best_loss_mean": float(np.mean(warm_values)),
        "warmstart_best_loss_median": float(np.median(warm_values)),
        "source": str(root),
        "source_label": "paired_warmstart_150_reduced_150_heisenberg",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "warmstart_iterations",
        "trials",
        "success_count",
        "success_rate",
        "final_best_loss_mean",
        "final_best_loss_median",
        "final_best_loss_q25",
        "final_best_loss_q75",
        "warmstart_best_loss_mean",
        "warmstart_best_loss_median",
        "source_label",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _plot_success(path: Path, rows: list[dict[str, Any]], dpi: int) -> None:
    x = np.asarray([int(row["warmstart_iterations"]) for row in rows], dtype=float)
    y = np.asarray([float(row["success_rate"]) for row in rows], dtype=float)
    n = np.asarray([int(row["trials"]) for row in rows], dtype=float)
    stderr = np.sqrt(np.maximum(y * (1.0 - y) / n, 0.0))
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    ax.errorbar(x, y, yerr=1.96 * stderr, marker="o", capsize=4, linewidth=1.5)
    ax.set_xlabel("Reduced-loss warmstart iterations")
    ax.set_ylabel("Final Heisenberg success probability")
    ax.set_title("Success probability vs reduced-loss warmstart length")
    ax.set_ylim(bottom=0.0, top=min(1.0, max(0.05, float(np.max(y + 1.96 * stderr)) * 1.15)))
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_final_loss(path: Path, rows: list[dict[str, Any]], threshold: float, dpi: int) -> None:
    x = np.asarray([int(row["warmstart_iterations"]) for row in rows], dtype=float)
    median = np.asarray([float(row["final_best_loss_median"]) for row in rows], dtype=float)
    q25 = np.asarray([float(row["final_best_loss_q25"]) for row in rows], dtype=float)
    q75 = np.asarray([float(row["final_best_loss_q75"]) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    ax.plot(x, median, marker="o", linewidth=1.5, label="median")
    ax.fill_between(x, q25, q75, alpha=0.2, label="IQR")
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.0, label="success threshold")
    ax.set_xlabel("Reduced-loss warmstart iterations")
    ax.set_ylabel("Final best Heisenberg loss")
    ax.set_yscale("log")
    ax.set_title("Final loss vs reduced-loss warmstart length")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary = _load_json(args.sweep_root / "summary" / "success_vs_warmstart_rate_summary.json")
    rows = [_heisenberg_only_row(args.heisenberg_only_root)]
    for row in summary["rows"]:
        item = dict(row)
        item["source"] = str(args.sweep_root)
        item["source_label"] = "new_sweep_1000_trials"
        rows.append(item)
    rows.append(_paired_warmstart_150_row(args.paired_warmstart_root))
    rows = sorted(rows, key=lambda item: int(item["warmstart_iterations"]))

    extended = {
        "artifact_type": "success_vs_warmstart_rate_summary_extended",
        "experiment_root": str(args.sweep_root),
        "note": "Adds historical reduced-steps 0 and 150 points from loss_success_experiment/data. Step 0 uses Heisenberg-only 150-step block5 run; step 150 uses paired warmstart reduced-150 + Heisenberg-150 run with lr=0.1.",
        "rows": rows,
        "base_summary": summary,
        "external_sources": {
            "reduced_steps_0": str(args.heisenberg_only_root),
            "reduced_steps_150": str(args.paired_warmstart_root),
        },
    }
    summary_dir = args.sweep_root / "summary"
    figures = args.sweep_root / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / "success_vs_warmstart_rate_summary_extended.json"
    csv_path = summary_dir / "success_vs_warmstart_rate_summary_extended.csv"
    json_path.write_text(json.dumps(extended, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows)

    success_path = figures / "success_rate_vs_warmstart_iterations_extended.png"
    loss_path = figures / "final_loss_vs_warmstart_iterations_extended.png"
    _plot_success(success_path, rows, int(args.dpi))
    _plot_final_loss(loss_path, rows, float(summary["manifest"]["success_threshold"]), int(args.dpi))

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {success_path}")
    print(f"Wrote {loss_path}")
    for row in rows:
        print(
            f"warmstart_iterations={row['warmstart_iterations']}: "
            f"{row['success_count']}/{row['trials']} = {float(row['success_rate']):.4f} "
            f"({row['source_label']})"
        )


if __name__ == "__main__":
    main()
