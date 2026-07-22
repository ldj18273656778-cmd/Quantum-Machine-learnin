"""Aggregate a success-vs-warmstart-rate experiment."""

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

import numpy as np

from task2_code.hpc_parallel_training.hpc_block_flow import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate a warmstart-iteration success-rate sweep.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


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
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    args = parse_args()
    manifest = _load_json(args.experiment_root / "manifest.json")
    if manifest.get("artifact_type") != "success_vs_warmstart_rate_experiment":
        raise ValueError(f"unexpected artifact_type: {manifest.get('artifact_type')!r}")

    group_count = int(manifest["group_count"])
    missing = []
    groups = []
    for group_index in range(1, group_count + 1):
        path = args.experiment_root / "groups" / f"group_{group_index:06d}.json"
        if not path.exists():
            missing.append(group_index)
            continue
        groups.append(_load_json(path))
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} group result files; first missing: {missing[:10]}")

    trials = [trial for group in groups for trial in group.get("trials", [])]
    expected_total = int(manifest["total_trials"])
    if len(trials) != expected_total:
        raise ValueError(f"expected {expected_total} trials, found {len(trials)}")

    rows: list[dict[str, Any]] = []
    by_iterations: dict[int, list[dict[str, Any]]] = {int(value): [] for value in manifest["warmstart_iterations"]}
    for trial in trials:
        by_iterations[int(trial["warmstart_iterations"])].append(trial)

    for warm_iterations in [int(value) for value in manifest["warmstart_iterations"]]:
        items = by_iterations[warm_iterations]
        expected = int(manifest["trials_per_warmstart"])
        if len(items) != expected:
            raise ValueError(f"warmstart_iterations={warm_iterations}: expected {expected} trials, found {len(items)}")
        final_losses = np.asarray([float(item["final_best_loss"]) for item in items], dtype=float)
        warm_losses = np.asarray([float(item["warmstart_best_loss"]) for item in items], dtype=float)
        success_count = sum(1 for item in items if bool(item["success"]))
        rows.append(
            {
                "warmstart_iterations": warm_iterations,
                "trials": len(items),
                "success_count": success_count,
                "success_rate": float(success_count / len(items)),
                "final_best_loss_mean": float(np.mean(final_losses)),
                "final_best_loss_median": float(np.median(final_losses)),
                "final_best_loss_q25": float(np.quantile(final_losses, 0.25)),
                "final_best_loss_q75": float(np.quantile(final_losses, 0.75)),
                "warmstart_best_loss_mean": float(np.mean(warm_losses)),
                "warmstart_best_loss_median": float(np.median(warm_losses)),
            }
        )

    summary = {
        "artifact_type": "success_vs_warmstart_rate_summary",
        "experiment_root": str(args.experiment_root),
        "total_trials": len(trials),
        "rows": rows,
        "manifest": manifest,
    }
    summary_dir = args.experiment_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / "success_vs_warmstart_rate_summary.json"
    csv_path = summary_dir / "success_vs_warmstart_rate_summary.csv"
    atomic_write_json(json_path, summary)
    _write_csv(csv_path, rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    for row in rows:
        print(
            f"warmstart_iterations={row['warmstart_iterations']}: "
            f"{row['success_count']}/{row['trials']} = {row['success_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
