"""Create a success-rate sweep over reduced-loss warmstart iterations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from task2_code.hpc_parallel_training.hpc_block_flow import PRESETS, atomic_write_json, safe_stem, timestamp
from task2_code.loss_registry import loss_function_uses_superoperator, resolve_loss_function_spec, set_active_loss_function
from task2_code.module_e_training import build_target_objective_context
from task2_code.superoperator_registry import resolve_superop, set_active_superop
from task2_code.target_factory import build_target_from_seed, target_metadata


DEFAULT_WARMSTART_ITERATIONS = "30,60,75,90,120,180"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a warmstart-iteration success-rate sweep.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="n32_8blocks")
    parser.add_argument("--block-index", type=int, default=5, help="1-based block index")
    parser.add_argument("--warmstart-iterations", default=DEFAULT_WARMSTART_ITERATIONS)
    parser.add_argument("--heisenberg-iterations", type=int, default=150)
    parser.add_argument("--trials-per-warmstart", type=int, default=1000)
    parser.add_argument("--groups", type=int, default=500)
    parser.add_argument("--warmstart-lr", type=float, default=None)
    parser.add_argument("--heisenberg-lr", type=float, default=None)
    parser.add_argument("--fd-eps", type=float, default=1e-5)
    parser.add_argument("--training-seed-start", type=int, default=None)
    parser.add_argument("--success-threshold", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=Path("task2_code/loss_success_experiment/success_vs_warmstart_rate/data"))
    parser.add_argument("--experiment-name", default=None)
    return parser.parse_args()


def _safe_experiment_name(value: str) -> str:
    name = safe_stem(value)
    if name != value or Path(value).name != value:
        raise ValueError("--experiment-name must be a plain safe name without path separators")
    return name


def _parse_positive_ints(value: str, name: str) -> list[int]:
    items = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one integer")
    if any(item <= 0 for item in items):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(items)) != len(items):
        raise ValueError(f"{name} must not contain duplicates")
    return items


def _positive_or_none(value: int | float | None, name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive")


def main() -> None:
    args = parse_args()
    cfg = PRESETS[args.preset]
    if args.block_index < 1 or args.block_index > cfg.block_count:
        raise ValueError(f"--block-index must be in 1..{cfg.block_count}")
    warmstart_iterations = _parse_positive_ints(str(args.warmstart_iterations), "--warmstart-iterations")
    if args.heisenberg_iterations <= 0:
        raise ValueError("--heisenberg-iterations must be positive")
    if args.trials_per_warmstart <= 0:
        raise ValueError("--trials-per-warmstart must be positive")
    if args.groups <= 0:
        raise ValueError("--groups must be positive")
    if args.trials_per_warmstart % args.groups != 0:
        raise ValueError("--trials-per-warmstart must be divisible by --groups")
    _positive_or_none(args.warmstart_lr, "--warmstart-lr")
    _positive_or_none(args.heisenberg_lr, "--heisenberg-lr")
    _positive_or_none(args.fd_eps, "--fd-eps")
    _positive_or_none(args.success_threshold, "--success-threshold")

    reduced_loss = "edge_quantum_channel"
    reduced_superoperator = "superoperator_from_mix"
    heisenberg_loss = "heisenberg_pauli"
    resolve_loss_function_spec(reduced_loss)
    resolve_loss_function_spec(heisenberg_loss)
    resolve_superop(reduced_superoperator)

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
            "success_vs_warmstart_rate",
            safe_stem(args.preset),
            f"block{args.block_index:02d}",
            f"heis{args.heisenberg_iterations}",
            f"trials{args.trials_per_warmstart}",
            timestamp(),
        ]
    )
    experiment_root = args.output_dir / name
    experiment_root.mkdir(parents=True, exist_ok=False)
    (experiment_root / "groups").mkdir()
    (experiment_root / "summary").mkdir()
    (experiment_root / "figures").mkdir()

    target = build_target_from_seed(cfg.n_qubits, cfg.target_seed, cfg.time_k)
    trials_per_group_per_warmstart = args.trials_per_warmstart // args.groups
    default_training_seed = cfg.training_seed_for_block(args.block_index - 1)
    training_seed_start = int(args.training_seed_start if args.training_seed_start is not None else default_training_seed)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "success_vs_warmstart_rate_experiment",
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
        "fd_eps": float(args.fd_eps),
        "success_threshold": float(args.success_threshold),
        "warmstart_iterations": warmstart_iterations,
        "heisenberg_iterations": int(args.heisenberg_iterations),
        "trials_per_warmstart": int(args.trials_per_warmstart),
        "group_count": int(args.groups),
        "trials_per_group_per_warmstart": int(trials_per_group_per_warmstart),
        "total_trials": int(args.trials_per_warmstart * len(warmstart_iterations)),
        "warmstart_stage": {
            "loss_function": reduced_loss,
            "superoperator": reduced_superoperator,
            "iterations_by_sweep": warmstart_iterations,
            "lr": float(args.warmstart_lr if args.warmstart_lr is not None else cfg.lr),
        },
        "heisenberg_stage": {
            "loss_function": heisenberg_loss,
            "superoperator": None,
            "iterations": int(args.heisenberg_iterations),
            "lr": float(args.heisenberg_lr if args.heisenberg_lr is not None else cfg.lr),
        },
        "target_metadata": target_metadata(target),
        "context_metadata": context_meta,
    }
    atomic_write_json(experiment_root / "manifest.json", manifest)
    print(f"Created success-vs-warmstart-rate experiment: {experiment_root}")
    print(f"block_index = {args.block_index}")
    print(f"warmstart_iterations = {warmstart_iterations}")
    print(f"heisenberg_iterations = {args.heisenberg_iterations}")
    print(f"trials_per_warmstart = {args.trials_per_warmstart}")
    print(f"group_count = {args.groups}")
    print(f"trials_per_group_per_warmstart = {trials_per_group_per_warmstart}")


if __name__ == "__main__":
    main()
