"""Run one batch of random-parameter loss evaluations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from task2_code.ansatz_registry import random_ansatz_theta
from task2_code.hpc_parallel_training.hpc_block_flow import PRESETS, atomic_savez, atomic_write_json
from task2_code.loss_registry import loss_function_uses_superoperator, resolve_loss_function_spec, set_active_loss_function
from task2_code.module_e_training import build_target_objective_context, sum_block_loss
from task2_code.superoperator_registry import set_active_superop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one random-parameter loss-distribution batch.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True, help="1-based task index")
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def _load_manifest(experiment_root: Path) -> dict[str, Any]:
    path = experiment_root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    if data.get("artifact_type") != "random_parameter_loss_distribution_experiment":
        raise ValueError(f"unexpected artifact_type: {data.get('artifact_type')!r}")
    return data


def _context_from_manifest(manifest: dict[str, Any]):
    preset = str(manifest["preset"])
    cfg = PRESETS[preset]
    context, _context_meta = build_target_objective_context(
        int(manifest["n_qubits"]),
        [int(q) for q in manifest["block_qubits"]],
        int(manifest["target_bit"]),
        int(manifest["radius"]),
        int(manifest["target_seed"]),
        int(manifest["time_k"]),
        lightcone_mode=str(manifest["lightcone_mode"]),
        loss_mode=str(manifest["loss_mode"]),
        require_unitary=False,
        max_n_qubits=int(manifest["n_qubits"]),
        max_hilbert_dim=4096,
        ansatz=str(manifest.get("ansatz", cfg.ansatz)),
        block_only_ansatz=bool(manifest.get("block_only_ansatz", cfg.block_only_ansatz)),
    )
    return context


def _set_loss(mode: dict[str, Any]) -> None:
    loss_name = str(mode["loss_function"])
    spec = resolve_loss_function_spec(loss_name)
    set_active_loss_function(loss_name)
    if spec.uses_superoperator or loss_function_uses_superoperator(loss_name):
        superoperator = mode.get("superoperator")
        if superoperator is None:
            raise ValueError(f"loss_function {loss_name!r} requires a superoperator")
        set_active_superop(str(superoperator))


def main() -> None:
    args = parse_args()
    manifest = _load_manifest(args.experiment_root)
    task_index = int(args.task_index)
    task_count = int(manifest["task_count"])
    if task_index < 1 or task_index > task_count:
        raise ValueError(f"--task-index must be in 1..{task_count}")
    batch_npz = args.experiment_root / "batches" / f"batch_{task_index:06d}.npz"
    batch_json = args.experiment_root / "batches" / f"batch_{task_index:06d}.json"
    if (batch_npz.exists() or batch_json.exists()) and not args.overwrite:
        raise FileExistsError(f"batch output exists; pass --overwrite to replace: {batch_npz}")

    context = _context_from_manifest(manifest)
    modes = [dict(mode) for mode in manifest["modes"]]
    samples_per_task = int(manifest["samples_per_task"])
    theta_size = int(manifest["theta_size"])
    sample_start = (task_index - 1) * samples_per_task + 1
    sample_indices = np.arange(sample_start, sample_start + samples_per_task, dtype=np.int64)
    seeds = int(manifest["training_seed_start"]) + sample_indices - 1
    initial_params = np.empty((samples_per_task, theta_size), dtype=float)
    losses = np.empty((len(modes), samples_per_task), dtype=float)
    elapsed_by_mode = np.zeros(len(modes), dtype=float)
    started = perf_counter()

    for sample_pos, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        initial_params[sample_pos] = random_ansatz_theta(context.ansatz, rng, n_qubits=context.ansatz_qubits)

    for mode_index, mode in enumerate(modes):
        mode_start = perf_counter()
        _set_loss(mode)
        for sample_pos in range(samples_per_task):
            losses[mode_index, sample_pos] = float(sum_block_loss(initial_params[sample_pos], context))
        elapsed_by_mode[mode_index] = perf_counter() - mode_start
    elapsed_total = perf_counter() - started

    atomic_savez(batch_npz, sample_indices=sample_indices, seeds=seeds, initial_params=initial_params, losses=losses)
    payload = {
        "task_index": task_index,
        "sample_start": int(sample_indices[0]),
        "sample_stop": int(sample_indices[-1]),
        "sample_count": int(samples_per_task),
        "mode_labels": [mode["label"] for mode in modes],
        "elapsed_seconds": float(elapsed_total),
        "elapsed_seconds_by_mode": {str(mode["label"]): float(elapsed_by_mode[idx]) for idx, mode in enumerate(modes)},
    }
    atomic_write_json(batch_json, payload)
    print(f"Wrote {batch_npz}")
    print(f"Wrote {batch_json}")
    print(f"samples {sample_indices[0]}..{sample_indices[-1]} elapsed={elapsed_total:.2f}s")


if __name__ == "__main__":
    main()
