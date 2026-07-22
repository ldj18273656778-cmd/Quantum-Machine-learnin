"""Aggregate Heisenberg-loss gradient norm checks for paired best parameters."""

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

import numpy as np
from numpy.typing import NDArray

from task2_code.hpc_parallel_training.hpc_block_flow import atomic_write_json


FloatArray = NDArray[np.float64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate paired best-parameter gradient norm checks.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _summarize(values: FloatArray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def main() -> None:
    args = parse_args()
    manifest = _load_json(args.experiment_root / "manifest.json")
    pair_count = int(manifest["total_pairs"])
    total_tasks = 2 * pair_count
    result_dir = args.experiment_root / "summary" / "gradient_norms"
    missing = []
    records: list[dict[str, Any]] = []
    for task_index in range(1, total_tasks + 1):
        path = result_dir / f"gradient_norm_{task_index:06d}.json"
        if not path.exists():
            missing.append(task_index)
            continue
        records.append(_load_json(path))
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} gradient norm files; first missing: {missing[:10]}")

    records.sort(key=lambda item: int(item["task_index"]))
    task_indices = np.asarray([int(item["task_index"]) for item in records], dtype=np.int64)
    pair_indices = np.asarray([int(item["pair_index"]) for item in records], dtype=np.int64)
    type_codes = np.asarray([0 if item["training_type"] == "heisenberg_only" else 1 for item in records], dtype=np.int64)
    grad_norms = np.asarray([float(item["grad_norm"]) for item in records], dtype=float)
    grad_inf_norms = np.asarray([float(item["grad_inf_norm"]) for item in records], dtype=float)
    losses = np.asarray([float(item["loss_at_theta"]) for item in records], dtype=float)
    recorded_losses = np.asarray([float(item["best_loss_recorded"]) for item in records], dtype=float)
    successes = np.asarray([bool(item["success"]) for item in records], dtype=bool)
    elapsed_seconds = np.asarray([float(item["elapsed_seconds"]) for item in records], dtype=float)

    heis_mask = type_codes == 0
    warm_mask = type_codes == 1
    summary = {
        "artifact_type": "paired_best_params_gradient_norms",
        "experiment_root": str(args.experiment_root),
        "total_records": int(len(records)),
        "fd_eps": float(records[0]["fd_eps"]),
        "heisenberg_only": {
            "count": int(heis_mask.sum()),
            "grad_norm": _summarize(grad_norms[heis_mask]),
            "grad_inf_norm": _summarize(grad_inf_norms[heis_mask]),
            "loss": _summarize(losses[heis_mask]),
            "success_count": int(successes[heis_mask].sum()),
        },
        "warmstart": {
            "count": int(warm_mask.sum()),
            "grad_norm": _summarize(grad_norms[warm_mask]),
            "grad_inf_norm": _summarize(grad_inf_norms[warm_mask]),
            "loss": _summarize(losses[warm_mask]),
            "success_count": int(successes[warm_mask].sum()),
        },
        "elapsed_seconds_per_point": _summarize(elapsed_seconds),
        "max_abs_loss_difference_from_recorded": float(np.max(np.abs(losses - recorded_losses))),
    }

    output_npz = args.experiment_root / "summary" / "paired_best_params_gradient_norms.npz"
    np.savez_compressed(
        output_npz,
        task_indices=task_indices,
        pair_indices=pair_indices,
        type_codes=type_codes,
        grad_norms=grad_norms,
        grad_inf_norms=grad_inf_norms,
        losses=losses,
        recorded_losses=recorded_losses,
        successes=successes,
        elapsed_seconds=elapsed_seconds,
    )
    output_json = args.experiment_root / "summary" / "paired_best_params_gradient_norms_summary.json"
    atomic_write_json(output_json, summary)
    print(f"Wrote {output_npz}")
    print(f"Wrote {output_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
