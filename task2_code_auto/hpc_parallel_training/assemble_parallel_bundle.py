"""Assemble per-block Task 2 training outputs into a sewing-compatible bundle."""

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

from task2_code_auto.hpc_parallel_training.hpc_block_flow import (
    atomic_savez,
    atomic_write_json,
    block_result_paths,
    bundle_name,
    config_from_manifest,
    load_manifest,
    timestamp,
)
from task2_code_auto.loss_registry import loss_function_uses_superoperator
from task2_code_auto.target_factory import build_target_from_seed, target_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble parallel block results into params.npz/metadata.json.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to <experiment-root>/bundle")
    return parser.parse_args()


def _load_block_result(experiment_root: Path, block_index: int) -> tuple[NDArray[np.float64], dict[str, Any]]:
    params_path, result_path = block_result_paths(experiment_root, block_index)
    if not params_path.exists():
        raise FileNotFoundError(f"block params not found: {params_path}")
    if not result_path.exists():
        raise FileNotFoundError(f"block result metadata not found: {result_path}")
    metadata = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"block result must contain a JSON object: {result_path}")
    with np.load(params_path, allow_pickle=False) as params:
        theta = np.asarray(params["best_params"], dtype=float)
        loss_history = np.asarray(params["loss_history"], dtype=float) if "loss_history" in params.files else np.asarray([], dtype=float)
    metadata["loss_history"] = loss_history
    return theta, metadata


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.experiment_root)
    cfg = config_from_manifest(manifest)
    stamp = timestamp()
    output_dir = args.output_dir or (args.experiment_root / "bundle" / bundle_name(str(manifest["preset"]), cfg, stamp))
    output_dir.mkdir(parents=True, exist_ok=False)

    arrays: dict[str, Any] = {}
    block_metadata: list[dict[str, Any]] = []
    theta_sizes: list[int] = []
    for block_index, (block_qubits, target_bit) in enumerate(zip(cfg.blocks, cfg.target_bits), start=1):
        theta, result_meta = _load_block_result(args.experiment_root, block_index)
        expected_size = int(result_meta["theta_size"])
        if theta.shape != (expected_size,):
            raise ValueError(f"block {block_index} theta has shape {theta.shape}, expected ({expected_size},)")
        params_key = f"best_params_block_{block_index}"
        arrays[params_key] = theta
        arrays[f"loss_history_block_{block_index}"] = np.asarray(result_meta.pop("loss_history"), dtype=float)
        theta_sizes.append(expected_size)
        block_entry = dict(result_meta)
        block_entry["block_index"] = block_index
        block_entry["block_qubits"] = list(block_qubits)
        block_entry["target_bit"] = int(target_bit)
        block_entry["params_key"] = params_key
        block_metadata.append(block_entry)

    arrays["blocks"] = np.asarray(cfg.blocks, dtype=int)
    arrays["target_bits"] = np.asarray(cfg.target_bits, dtype=int)
    arrays["theta_sizes"] = np.asarray(theta_sizes, dtype=int)
    params_path = output_dir / "params.npz"
    metadata_path = output_dir / "metadata.json"
    atomic_savez(params_path, **arrays)

    target = build_target_from_seed(cfg.n_qubits, cfg.target_seed, cfg.time_k)
    metadata: dict[str, Any] = {
        "schema_version": 3,
        "artifact_type": "parallel_block_training_bundle",
        "timestamp": stamp,
        "preset": manifest["preset"],
        "experiment_root": args.experiment_root,
        "run_dir": output_dir,
        **target_metadata(target),
        "blocks": block_metadata,
        "block_qubits": cfg.blocks,
        "target_bits": cfg.target_bits,
        "radius": cfg.radius,
        "lightcone_mode": cfg.lightcone_mode,
        "loss_mode": cfg.loss_mode,
        "ansatz": cfg.ansatz,
        "block_only_ansatz": cfg.block_only_ansatz,
        "loss_function": cfg.loss_function,
        "loss_function_uses_superoperator": loss_function_uses_superoperator(cfg.loss_function),
        "iterations": cfg.iterations,
        "lr": cfg.lr,
        "fd_eps": manifest.get("fd_eps", 1e-5),
        "max_restarts": cfg.max_restarts,
        "training_seed_start": cfg.training_seed_start,
        "params_file": params_path.name,
        "metadata_file": metadata_path.name,
    }
    if loss_function_uses_superoperator(cfg.loss_function):
        metadata["superoperator"] = cfg.superoperator
    atomic_write_json(metadata_path, metadata)
    print(f"Assembled sewing bundle: {output_dir}")
    print(f"params = {params_path}")
    print(f"metadata = {metadata_path}")


if __name__ == "__main__":
    main()
