"""Aggregate random-parameter loss-distribution batches."""

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

import numpy as np
from numpy.typing import NDArray

from task2_code.hpc_parallel_training.hpc_block_flow import atomic_savez, atomic_write_json


FloatArray = NDArray[np.float64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate a random-parameter loss-distribution experiment.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _summary(values: FloatArray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "var": float(np.var(values)),
        "std": float(np.std(values)),
    }


def main() -> None:
    args = parse_args()
    manifest = _load_json(args.experiment_root / "manifest.json")
    if manifest.get("artifact_type") != "random_parameter_loss_distribution_experiment":
        raise ValueError(f"unexpected artifact_type: {manifest.get('artifact_type')!r}")
    task_count = int(manifest["task_count"])
    sample_count = int(manifest["sample_count"])
    mode_labels = [str(mode["label"]) for mode in manifest["modes"]]

    missing = []
    sample_index_parts = []
    seed_parts = []
    loss_parts = []
    for task_index in range(1, task_count + 1):
        path = args.experiment_root / "batches" / f"batch_{task_index:06d}.npz"
        if not path.exists():
            missing.append(task_index)
            continue
        with np.load(path) as batch:
            sample_index_parts.append(np.asarray(batch["sample_indices"], dtype=np.int64))
            seed_parts.append(np.asarray(batch["seeds"], dtype=np.int64))
            loss_parts.append(np.asarray(batch["losses"], dtype=float))
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} batch outputs; first missing: {missing[:10]}")

    sample_indices = np.concatenate(sample_index_parts)
    seeds = np.concatenate(seed_parts)
    losses = np.concatenate(loss_parts, axis=1)
    order = np.argsort(sample_indices)
    sample_indices = sample_indices[order]
    seeds = seeds[order]
    losses = losses[:, order]
    expected = np.arange(1, sample_count + 1, dtype=np.int64)
    if not np.array_equal(sample_indices, expected):
        raise ValueError("sample indices are not complete and ordered after aggregation")

    summary: dict[str, Any] = {
        "artifact_type": "random_parameter_loss_distribution_summary",
        "experiment_root": str(args.experiment_root),
        "sample_count": sample_count,
        "theta_size": int(manifest["theta_size"]),
        "mode_labels": mode_labels,
        "modes": {},
    }
    rows = []
    for mode_index, label in enumerate(mode_labels):
        stats = _summary(losses[mode_index])
        summary["modes"][label] = {"loss": stats}
        row = {"mode": label, **stats}
        rows.append(row)

    summary_dir = args.experiment_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    atomic_savez(summary_dir / "loss_distribution.npz", sample_indices=sample_indices, seeds=seeds, mode_labels=np.asarray(mode_labels), losses=losses)
    atomic_write_json(summary_dir / "loss_distribution_summary.json", summary)
    csv_path = summary_dir / "loss_distribution_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["mode", "min", "q05", "q25", "median", "q75", "q95", "max", "mean", "var", "std"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary_dir / 'loss_distribution.npz'}")
    print(f"Wrote {summary_dir / 'loss_distribution_summary.json'}")
    print(f"Wrote {csv_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
