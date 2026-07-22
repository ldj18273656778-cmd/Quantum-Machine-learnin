"""Create a single-block loss-mode success-probability experiment."""

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


DEFAULT_MODES = [
    "edge_quantum_channel:superoperator_from_mix",
    "edge_quantum_channel:superoperator_from_zero",
    "edge_quantum_channel:superoperator_from_one",
    "heisenberg_pauli",
]

DEFAULT_SUCCESS_THRESHOLDS_BY_LOSS = {
    "edge_quantum_channel": 0.01,
    "heisenberg_pauli": 0.01,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a loss-mode success-probability experiment.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="n32_8blocks")
    parser.add_argument("--block-index", type=int, default=5, help="1-based block index")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help="comma-separated loss[:superoperator] modes")
    parser.add_argument("--total-restarts", type=int, default=500)
    parser.add_argument("--restarts-per-group", type=int, default=2)
    parser.add_argument("--success-threshold", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--fd-eps", type=float, default=1e-5)
    parser.add_argument("--training-seed-start", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("task2_code/loss_success_experiment/data"))
    parser.add_argument("--experiment-name", default=None)
    return parser.parse_args()


def _safe_experiment_name(value: str) -> str:
    name = safe_stem(value)
    if name != value or Path(value).name != value:
        raise ValueError("--experiment-name must be a plain safe name without path separators")
    return name


def _mode_threshold(loss_function: str, global_threshold: float | None, fallback: float) -> float:
    if global_threshold is not None:
        return float(global_threshold)
    return float(DEFAULT_SUCCESS_THRESHOLDS_BY_LOSS.get(loss_function, fallback))


def _parse_mode(
    spec: str,
    default_superoperator: str,
    success_threshold: float | None,
    fallback_threshold: float,
) -> dict[str, Any]:
    parts = [part.strip() for part in spec.split(":") if part.strip()]
    if not parts:
        raise ValueError("loss mode spec must not be empty")
    loss_function = parts[0]
    resolve_loss_function_spec(loss_function)
    uses_superoperator = loss_function_uses_superoperator(loss_function)
    superoperator = parts[1] if len(parts) > 1 else (default_superoperator if uses_superoperator else None)
    if len(parts) > 2:
        raise ValueError(f"loss mode spec has too many ':' parts: {spec!r}")
    if uses_superoperator:
        if superoperator is None:
            raise ValueError(f"loss mode {loss_function!r} requires a superoperator")
        resolve_superop(str(superoperator))
    elif superoperator is not None:
        raise ValueError(f"loss mode {loss_function!r} does not use a superoperator")
    label = loss_function if superoperator is None else f"{loss_function}_{superoperator}"
    return {
        "label": safe_stem(label),
        "loss_function": loss_function,
        "superoperator": superoperator,
        "uses_superoperator": uses_superoperator,
        "success_threshold": _mode_threshold(loss_function, success_threshold, fallback_threshold),
    }


def main() -> None:
    args = parse_args()
    cfg = PRESETS[args.preset]
    if args.block_index < 1 or args.block_index > cfg.block_count:
        raise ValueError(f"--block-index must be in 1..{cfg.block_count}")
    if args.total_restarts <= 0:
        raise ValueError("--total-restarts must be positive")
    if args.restarts_per_group <= 0:
        raise ValueError("--restarts-per-group must be positive")
    if args.total_restarts % args.restarts_per_group != 0:
        raise ValueError("--total-restarts must be divisible by --restarts-per-group")
    if args.iterations is not None and args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.lr is not None and args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.fd_eps <= 0:
        raise ValueError("--fd-eps must be positive")
    if args.success_threshold is not None and args.success_threshold <= 0:
        raise ValueError("--success-threshold must be positive")

    modes = [
        _parse_mode(item, cfg.superoperator, args.success_threshold, cfg.success_threshold)
        for item in args.modes.split(",")
        if item.strip()
    ]
    if not modes:
        raise ValueError("--modes must contain at least one mode")

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
            "loss_success",
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
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "loss_success_experiment",
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
        "iterations": int(args.iterations if args.iterations is not None else cfg.iterations),
        "lr": float(args.lr if args.lr is not None else cfg.lr),
        "fd_eps": args.fd_eps,
        "training_seed_start": int(args.training_seed_start if args.training_seed_start is not None else cfg.training_seed_start),
        "success_threshold": float(args.success_threshold) if args.success_threshold is not None else None,
        "total_restarts": args.total_restarts,
        "restarts_per_group": args.restarts_per_group,
        "group_count": args.total_restarts // args.restarts_per_group,
        "modes": modes,
        "lightcone_semantics": context_meta.get("lightcone_semantics"),
        "loss_semantics": context_meta.get("loss_semantics"),
        **target_metadata(target),
    }
    atomic_write_json(experiment_root / "manifest.json", manifest)
    print(f"Created loss success experiment: {experiment_root}")
    print(f"block_index = {args.block_index}")
    print(f"lightcone_size = {len(context.lightcone_qubits)}")
    thresholds = {mode["label"]: mode["success_threshold"] for mode in modes}
    print(f"modes = {[mode['label'] for mode in modes]}")
    print(f"thresholds = {thresholds}")
    print(f"groups = {manifest['group_count']} x {args.restarts_per_group} restarts")


if __name__ == "__main__":
    main()
