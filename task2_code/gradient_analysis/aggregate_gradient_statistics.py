"""Aggregate random-initialization gradient statistics."""

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

from task2_code.hpc_parallel_training.hpc_block_flow import atomic_write_json


FloatArray = NDArray[np.float64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate random-initialization gradient statistics.")
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
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "var": float(np.var(values)),
    }


def main() -> None:
    args = parse_args()
    manifest = _load_json(args.experiment_root / "manifest.json")
    if manifest.get("artifact_type") != "random_initial_gradient_statistics_experiment":
        raise ValueError(f"unexpected artifact_type: {manifest.get('artifact_type')!r}")
    task_count = int(manifest["task_count"])
    sample_count = int(manifest["sample_count"])
    mode_labels = [str(mode["label"]) for mode in manifest["modes"]]
    missing = []
    sample_index_parts = []
    seed_parts = []
    loss_parts = []
    gradient_parts = []
    grad_sq_norm_parts = []
    normalized_grad_sq_norm_parts = []
    for task_index in range(1, task_count + 1):
        path = args.experiment_root / "batches" / f"batch_{task_index:06d}.npz"
        if not path.exists():
            missing.append(task_index)
            continue
        with np.load(path) as batch:
            sample_index_parts.append(np.asarray(batch["sample_indices"], dtype=np.int64))
            seed_parts.append(np.asarray(batch["seeds"], dtype=np.int64))
            loss_parts.append(np.asarray(batch["losses"], dtype=float))
            gradient_parts.append(np.asarray(batch["gradients"], dtype=float))
            grad_sq_norm_parts.append(np.asarray(batch["grad_sq_norms"], dtype=float))
            normalized_grad_sq_norm_parts.append(np.asarray(batch["normalized_grad_sq_norms"], dtype=float))
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} batch files; first missing: {missing[:10]}")

    sample_indices = np.concatenate(sample_index_parts)
    seeds = np.concatenate(seed_parts)
    losses = np.concatenate(loss_parts, axis=1)
    gradients = np.concatenate(gradient_parts, axis=1)
    grad_sq_norms = np.concatenate(grad_sq_norm_parts, axis=1)
    normalized_grad_sq_norms = np.concatenate(normalized_grad_sq_norm_parts, axis=1)

    order = np.argsort(sample_indices)
    sample_indices = sample_indices[order]
    seeds = seeds[order]
    losses = losses[:, order]
    gradients = gradients[:, order, :]
    grad_sq_norms = grad_sq_norms[:, order]
    normalized_grad_sq_norms = normalized_grad_sq_norms[:, order]
    expected = np.arange(1, sample_count + 1, dtype=np.int64)
    if not np.array_equal(sample_indices, expected):
        raise ValueError("sample indices are not complete and ordered after aggregation")

    mean_gradients = np.mean(gradients, axis=1)
    var_gradients = np.var(gradients, axis=1)
    selected_coordinates = [int(value) for value in manifest["selected_coordinates"]]
    selected_zero_based = [coord - 1 for coord in selected_coordinates]
    selected_gradients = gradients[:, :, selected_zero_based]

    summary: dict[str, Any] = {
        "artifact_type": "random_initial_gradient_statistics_summary",
        "experiment_root": str(args.experiment_root),
        "sample_count": sample_count,
        "theta_size": int(manifest["theta_size"]),
        "mode_labels": mode_labels,
        "selected_coordinates": selected_coordinates,
        "outputs": {
            "npz": "gradient_statistics.npz",
            "coordinate_csv": "gradient_coordinate_statistics.csv",
        },
        "modes": {},
    }
    coordinate_rows = []
    for mode_index, label in enumerate(mode_labels):
        loss_values = losses[mode_index]
        grad_sq_values = grad_sq_norms[mode_index]
        norm_grad_sq_values = normalized_grad_sq_norms[mode_index]
        high_loss = loss_values >= np.quantile(loss_values, 0.75)
        small_grad = norm_grad_sq_values <= np.quantile(norm_grad_sq_values, 0.25)
        summary["modes"][label] = {
            "loss": _summary(loss_values),
            "grad_sq_norm": _summary(grad_sq_values),
            "normalized_grad_sq_norm": _summary(norm_grad_sq_values),
            "high_loss_small_gradient_count": int(np.sum(high_loss & small_grad)),
            "high_loss_small_gradient_fraction": float(np.mean(high_loss & small_grad)),
            "selected_coordinates": {},
        }
        for coord in selected_coordinates:
            values = gradients[mode_index, :, coord - 1]
            stats = _summary(values)
            summary["modes"][label]["selected_coordinates"][str(coord)] = stats
        for coord in range(1, int(manifest["theta_size"]) + 1):
            coordinate_rows.append(
                {
                    "mode": label,
                    "coordinate": coord,
                    "mean": float(mean_gradients[mode_index, coord - 1]),
                    "variance": float(var_gradients[mode_index, coord - 1]),
                }
            )

    output_npz = args.experiment_root / "summary" / "gradient_statistics.npz"
    np.savez_compressed(
        output_npz,
        sample_indices=sample_indices,
        seeds=seeds,
        losses=losses,
        gradients=gradients,
        grad_sq_norms=grad_sq_norms,
        normalized_grad_sq_norms=normalized_grad_sq_norms,
        mean_gradients=mean_gradients,
        var_gradients=var_gradients,
        selected_coordinates=np.asarray(selected_coordinates, dtype=np.int64),
        selected_gradients=selected_gradients,
        mode_labels=np.asarray(mode_labels),
    )
    coordinate_csv = args.experiment_root / "summary" / "gradient_coordinate_statistics.csv"
    with coordinate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "coordinate", "mean", "variance"])
        writer.writeheader()
        writer.writerows(coordinate_rows)
    output_json = args.experiment_root / "summary" / "gradient_statistics_summary.json"
    atomic_write_json(output_json, summary)
    print(f"Wrote {output_npz}")
    print(f"Wrote {coordinate_csv}")
    print(f"Wrote {output_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
