"""Create a random-parameter loss-distribution experiment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task2_code_auto.hpc_parallel_training.hpc_block_flow import PRESETS, atomic_write_json, safe_stem, timestamp
from task2_code_auto.loss_registry import loss_function_uses_superoperator, resolve_loss_function_spec, set_active_loss_function
from task2_code_auto.module_e_training import build_target_objective_context
from task2_code_auto.superoperator_registry import resolve_superop, set_active_superop
from task2_code_auto.target_factory import build_target_from_seed, target_metadata


DEFAULT_MODES = "edge_quantum_channel:superoperator_from_mix,heisenberg_pauli"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a random-parameter loss-distribution experiment.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="n32_8blocks")
    parser.add_argument("--block-index", type=int, default=5, help="1-based block index")
    parser.add_argument("--modes", default=DEFAULT_MODES, help="comma-separated loss[:superoperator] modes")
    parser.add_argument("--sample-count", type=int, default=10000)
    parser.add_argument("--samples-per-task", type=int, default=20)
    parser.add_argument("--training-seed-start", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("task2_code/barren_plateau_analysis/data"))
    parser.add_argument("--experiment-name", default=None)
    return parser.parse_args()


def _safe_experiment_name(value: str) -> str:
    name = safe_stem(value)
    if name != value or Path(value).name != value:
        raise ValueError("--experiment-name must be a plain safe name without path separators")
    return name


def _parse_modes(spec: str) -> list[dict[str, Any]]:
    modes = []
    seen = set()
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":") if part.strip()]
        if not parts or len(parts) > 2:
            raise ValueError(f"invalid mode spec {item!r}; expected loss[:superoperator]")
        loss_name = parts[0]
        superop_name = parts[1] if len(parts) == 2 else None
        loss_spec = resolve_loss_function_spec(loss_name)
        if loss_spec.uses_superoperator:
            if superop_name is None:
                raise ValueError(f"loss {loss_name!r} requires a superoperator")
            resolve_superop(superop_name)
        elif superop_name is not None:
            raise ValueError(f"loss {loss_name!r} does not use a superoperator")
        label = loss_name if superop_name is None else f"{loss_name}:{superop_name}"
        if label in seen:
            raise ValueError(f"duplicate mode {label!r}")
        seen.add(label)
        modes.append(
            {
                "label": label,
                "loss_function": loss_name,
                "superoperator": superop_name,
                "uses_superoperator": bool(loss_spec.uses_superoperator),
            }
        )
    if not modes:
        raise ValueError("--modes must contain at least one mode")
    return modes


def main() -> None:
    args = parse_args()
    cfg = PRESETS[args.preset]
    if args.block_index < 1 or args.block_index > cfg.block_count:
        raise ValueError(f"--block-index must be in 1..{cfg.block_count}")
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    if args.samples_per_task <= 0:
        raise ValueError("--samples-per-task must be positive")
    if args.sample_count % args.samples_per_task != 0:
        raise ValueError("--sample-count must be divisible by --samples-per-task")
    modes = _parse_modes(str(args.modes))

    block_qubits = cfg.blocks[args.block_index - 1]
    target_bit = cfg.target_bits[args.block_index - 1]
    set_active_loss_function(cfg.loss_function)
    if loss_function_uses_superoperator(cfg.loss_function):
        set_active_superop(cfg.superoperator)
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

    name = _safe_experiment_name(str(args.experiment_name)) if args.experiment_name else "_".join(
        [
            "loss_distribution",
            safe_stem(args.preset),
            f"block{args.block_index:02d}",
            f"samples{args.sample_count}",
            timestamp(),
        ]
    )
    experiment_root = args.output_dir / name
    experiment_root.mkdir(parents=True, exist_ok=False)
    (experiment_root / "batches").mkdir()
    (experiment_root / "summary").mkdir()
    (experiment_root / "figures").mkdir()

    target = build_target_from_seed(cfg.n_qubits, cfg.target_seed, cfg.time_k)
    training_seed_start = int(args.training_seed_start if args.training_seed_start is not None else cfg.training_seed_for_block(args.block_index - 1))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "random_parameter_loss_distribution_experiment",
        "preset": args.preset,
        "n_qubits": cfg.n_qubits,
        "block_index": args.block_index,
        "block_qubits": list(block_qubits),
        "target_bit": int(target_bit),
        "radius": cfg.radius,
        "time_k": cfg.time_k,
        "target_seed": cfg.target_seed,
        "training_seed_start": training_seed_start,
        "lightcone_mode": cfg.lightcone_mode,
        "loss_mode": cfg.loss_mode,
        "ansatz": cfg.ansatz,
        "block_only_ansatz": cfg.block_only_ansatz,
        "ansatz_qubits": context.ansatz_qubits,
        "theta_size": context.theta_size,
        "lightcone_qubits": list(context.lightcone_qubits),
        "sample_count": int(args.sample_count),
        "samples_per_task": int(args.samples_per_task),
        "task_count": int(args.sample_count // args.samples_per_task),
        "modes": modes,
        "target_metadata": target_metadata(target),
        "context_metadata": context_meta,
    }
    atomic_write_json(experiment_root / "manifest.json", manifest)
    print(f"Created loss-distribution experiment: {experiment_root}")
    print(f"block_index = {args.block_index}")
    print(f"lightcone_size = {len(context.lightcone_qubits)}")
    print(f"theta_size = {context.theta_size}")
    print(f"sample_count = {args.sample_count}")
    print(f"samples_per_task = {args.samples_per_task}")
    print(f"task_count = {manifest['task_count']}")
    print(f"modes = {[mode['label'] for mode in modes]}")


if __name__ == "__main__":
    main()
