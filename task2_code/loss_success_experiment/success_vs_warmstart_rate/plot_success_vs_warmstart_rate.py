"""Plot success probability versus reduced-loss warmstart iterations."""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a warmstart-iteration success-rate sweep.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def main() -> None:
    args = parse_args()
    summary_path = args.experiment_root / "summary" / "success_vs_warmstart_rate_summary.json"
    summary = _load_json(summary_path)
    rows = list(summary["rows"])
    iterations = np.asarray([int(row["warmstart_iterations"]) for row in rows], dtype=float)
    success_rates = np.asarray([float(row["success_rate"]) for row in rows], dtype=float)
    trials = np.asarray([int(row["trials"]) for row in rows], dtype=float)
    final_medians = np.asarray([float(row["final_best_loss_median"]) for row in rows], dtype=float)
    final_q25 = np.asarray([float(row["final_best_loss_q25"]) for row in rows], dtype=float)
    final_q75 = np.asarray([float(row["final_best_loss_q75"]) for row in rows], dtype=float)
    stderr = np.sqrt(np.maximum(success_rates * (1.0 - success_rates) / trials, 0.0))

    figures = args.experiment_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.errorbar(iterations, success_rates, yerr=1.96 * stderr, marker="o", capsize=4, linewidth=1.5)
    ax.set_xlabel("Reduced-loss warmstart iterations")
    ax.set_ylabel("Final Heisenberg success probability")
    ax.set_title("Success probability vs reduced-loss warmstart length")
    ax.set_ylim(bottom=0.0, top=min(1.0, max(0.05, float(np.max(success_rates + 1.96 * stderr)) * 1.15)))
    ax.grid(alpha=0.3)
    path = figures / "success_rate_vs_warmstart_iterations.png"
    fig.savefig(path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(iterations, final_medians, marker="o", linewidth=1.5, label="median")
    ax.fill_between(iterations, final_q25, final_q75, alpha=0.2, label="IQR")
    ax.axhline(float(summary["manifest"]["success_threshold"]), color="black", linestyle="--", linewidth=1.0, label="success threshold")
    ax.set_xlabel("Reduced-loss warmstart iterations")
    ax.set_ylabel("Final best Heisenberg loss")
    ax.set_yscale("log")
    ax.set_title("Final loss vs reduced-loss warmstart length")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    path_loss = figures / "final_loss_vs_warmstart_iterations.png"
    fig.savefig(path_loss, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {path}")
    print(f"Wrote {path_loss}")


if __name__ == "__main__":
    main()
