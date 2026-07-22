"""Create a paired Heisenberg vs mixed-warmstart block experiment."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a paired Heisenberg vs mixed-warmstart experiment.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="n32_8blocks")
    parser.add_argument("--block-index", type=int, default=5, help="1-based block index")
    parser.add_argument("--total-pairs", type=int, default=400)
    parser.add_argument("--pairs-per-group", type=int, default=1)
    parser.add_argument("--mixed-iterations", type=int, default=None)
    parser.add_argument("--heisenberg-iterations", type=int, default=None)
    parser.add_argument("--mixed-lr", type=float, default=None)
    parser.add_argument("--heisenberg-lr", type=float, default=None)
    parser.add_argument("--fd-eps", type=float, default=1e-5)
    parser.add_argument("--training-seed-start", type=int, default=None)
    parser.add_argument("--success-threshold", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=Path("task2_code/loss_success_experiment/data"))
    parser.add_argument("--experiment-name", default=None)
    return parser.parse_args()


def _safe_experiment_name(value: str) -> str:
    name = safe_stem(value)
    if name != value or Path(value).name != value:
        raise ValueError("--experiment-name must be a plain safe name without path separators")
    return name


def _positive_or_none(value: int | float | None, name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive")


def main() -> None:
    args = parse_args()
    cfg = PRESETS[args.preset]
    if args.block_index < 1 or args.block_index > cfg.block_count:
        raise ValueError(f"--block-index must be in 1..{cfg.block_count}")
    if args.total_pairs <= 0:
        raise ValueError("--total-pairs must be positive")
    if args.pairs_per_group <= 0:
        raise ValueError("--pairs-per-group must be positive")
    if args.total_pairs % args.pairs_per_group != 0:
        raise ValueError("--total-pairs must be divisible by --pairs-per-group")
    _positive_or_none(args.mixed_iterations, "--mixed-iterations")
    _positive_or_none(args.heisenberg_iterations, "--heisenberg-iterations")
    _positive_or_none(args.mixed_lr, "--mixed-lr")
    _positive_or_none(args.heisenberg_lr, "--heisenberg-lr")
    _positive_or_none(args.fd_eps, "--fd-eps")
    _positive_or_none(args.success_threshold, "--success-threshold")

    mixed_loss = "edge_quantum_channel"
    mixed_superoperator = "superoperator_from_mix"
    heisenberg_loss = "heisenberg_pauli"
    resolve_loss_function_spec(mixed_loss)
    resolve_loss_function_spec(heisenberg_loss)
    resolve_superop(mixed_superoperator)

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

    stamp = timestamp()
    name = _safe_experiment_name(str(args.experiment_name)) if args.experiment_name else "_".join(
        [
            "paired_warmstart",
            safe_stem(args.preset),
            f"block{args.block_index:02d}",
            stamp,
        ]
    )
    experiment_root = args.output_dir / name
    experiment_root.mkdir(parents=True, exist_ok=False)
    (experiment_root / "groups").mkdir()
    (experiment_root / "summary").mkdir()

    target = build_target_from_seed(cfg.n_qubits, cfg.target_seed, cfg.time_k)
    mixed_iterations = int(args.mixed_iterations if args.mixed_iterations is not None else cfg.iterations)
    heisenberg_iterations = int(args.heisenberg_iterations if args.heisenberg_iterations is not None else cfg.iterations)
    mixed_lr = float(args.mixed_lr if args.mixed_lr is not None else cfg.lr)
    heisenberg_lr = float(args.heisenberg_lr if args.heisenberg_lr is not None else cfg.lr)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "paired_warmstart_experiment",
        "timestamp": stamp,
        "experiment_root": experiment_root,
        "preset": args.preset,
        "n_qubits": cfg.n_qubits,
        "block_index": args.block_index,
        "block_qubits": list(block_qubits),
        "target_bit": int(target_bit),
        "lightcone_qubits": list(context.lightcone_qubits),
        "lightcone_size": len(context.lightcone_qubits),
        "ansatz": context.ansatz,
        "ansatz_qubits": context.ansatz_qubits,
        "theta_size": context.theta_size,
        "block_only_ansatz": context.block_only_ansatz,
        "radius": cfg.radius,
        "lightcone_mode": cfg.lightcone_mode,
        "loss_mode": cfg.loss_mode,
        "fd_eps": args.fd_eps,
        "training_seed_start": int(args.training_seed_start if args.training_seed_start is not None else cfg.training_seed_start),
        "total_pairs": args.total_pairs,
        "pairs_per_group": args.pairs_per_group,
        "group_count": args.total_pairs // args.pairs_per_group,
        "heisenberg_only": {
            "name": "heisenberg_only",
            "loss_function": heisenberg_loss,
            "superoperator": None,
            "iterations": heisenberg_iterations,
            "lr": heisenberg_lr,
            "success_threshold": float(args.success_threshold),
        },
        "warmstart": {
            "stage1": {
                "name": "mixed_edge",
                "loss_function": mixed_loss,
                "superoperator": mixed_superoperator,
                "iterations": mixed_iterations,
                "lr": mixed_lr,
            },
            "stage2": {
                "name": "heisenberg_warmstart",
                "loss_function": heisenberg_loss,
                "superoperator": None,
                "iterations": heisenberg_iterations,
                "lr": heisenberg_lr,
                "success_threshold": float(args.success_threshold),
            },
        },
        "lightcone_semantics": context_meta.get("lightcone_semantics"),
        "loss_semantics": context_meta.get("loss_semantics"),
        **target_metadata(target),
    }
    atomic_write_json(experiment_root / "manifest.json", manifest)
    print(f"Created paired warmstart experiment: {experiment_root}")
    print(f"block_index = {args.block_index}")
    print(f"lightcone_size = {len(context.lightcone_qubits)}")
    print(f"total_pairs = {args.total_pairs}")
    print(f"pairs_per_group = {args.pairs_per_group}")
    print(f"group_count = {manifest['group_count']}")
    print(f"training_seed_start = {manifest['training_seed_start']}")


if __name__ == "__main__":
    main()
