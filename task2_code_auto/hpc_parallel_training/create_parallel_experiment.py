"""Create a manifest directory for parallel per-block Task 2 training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task2_code_auto.hpc_parallel_training.hpc_block_flow import (
    PRESETS,
    atomic_write_json,
    config_to_manifest,
    parallel_experiment_name,
    timestamp,
)
from task2_code_auto.loss_registry import loss_function_uses_superoperator, set_active_loss_function
from task2_code_auto.module_e_training import build_target_objective_context
from task2_code_auto.superoperator_registry import set_active_superop
from task2_code_auto.target_factory import build_target_from_seed, target_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Task 2 parallel block-training experiment manifest.")
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("task2_code_auto/hpc_parallel_training/data"))
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--fd-eps", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PRESETS[args.preset]
    set_active_loss_function(cfg.loss_function)
    if loss_function_uses_superoperator(cfg.loss_function):
        set_active_superop(cfg.superoperator)

    stamp = timestamp()
    name = args.experiment_name or parallel_experiment_name(args.preset, cfg, stamp)
    experiment_root = args.output_dir / name
    experiment_root.mkdir(parents=True, exist_ok=False)
    (experiment_root / "blocks").mkdir()

    target = build_target_from_seed(cfg.n_qubits, cfg.target_seed, cfg.time_k)
    block_specs: list[dict[str, Any]] = []
    for block_index, (block_qubits, target_bit) in enumerate(zip(cfg.blocks, cfg.target_bits), start=1):
        context, context_meta = build_target_objective_context(
            cfg.n_qubits,
            block_qubits,
            target_bit,
            cfg.radius,
            cfg.target_seed,
            cfg.time_k,
            lightcone_mode=cfg.lightcone_mode,
            loss_mode=cfg.loss_mode,
            require_unitary=False,
            max_n_qubits=cfg.n_qubits,
            max_hilbert_dim=4096,
            ansatz=cfg.ansatz,
            block_only_ansatz=cfg.block_only_ansatz,
        )
        block_specs.append(
            {
                "block_index": block_index,
                "block_qubits": list(block_qubits),
                "target_bit": int(target_bit),
                "lightcone_qubits": list(context.lightcone_qubits),
                "ansatz_qubits": context.ansatz_qubits,
                "ansatz": context.ansatz,
                "block_only_ansatz": context.block_only_ansatz,
                "theta_size": context.theta_size,
                "params_key": f"best_params_block_{block_index}",
                "lightcone_semantics": context_meta.get("lightcone_semantics"),
                "loss_semantics": context_meta.get("loss_semantics"),
            }
        )

    manifest = {
        "schema_version": 1,
        "artifact_type": "parallel_block_experiment",
        "timestamp": stamp,
        "experiment_root": experiment_root,
        **config_to_manifest(args.preset, cfg),
        **target_metadata(target),
        "fd_eps": args.fd_eps,
        "block_specs": block_specs,
    }
    atomic_write_json(experiment_root / "manifest.json", manifest)
    print(f"Created parallel experiment: {experiment_root}")
    print(f"Blocks: {cfg.block_count}")


if __name__ == "__main__":
    main()
