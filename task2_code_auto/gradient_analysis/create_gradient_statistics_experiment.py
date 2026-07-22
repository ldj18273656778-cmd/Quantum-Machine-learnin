"""Create a random-initialization gradient statistics experiment."""

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


DEFAULT_MODES = "heisenberg_pauli,edge_quantum_channel:superoperator_from_mix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a random-initialization gradient statistics experiment.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="n32_8blocks")
    parser.add_argument("--block-index", type=int, default=5, help="1-based block index")
    parser.add_argument("--modes", default=DEFAULT_MODES, help="comma-separated loss[:superoperator] modes")
    parser.add_argument("--sample-count", type=int, default=10000, help="number of random initial parameters")
    parser.add_argument("--samples-per-task", type=int, default=20, help="number of random initial parameters per batch task")
    parser.add_argument("--fd-eps", type=float, default=1e-5)
    parser.add_argument("--training-seed-start", type=int, default=None)
    parser.add_argument("--selected-coordinates", default="1,61,120", help="1-based gradient coordinates to emphasize in plots")
    parser.add_argument("--output-dir", type=Path, default=Path("task2_code/gradient_analysis/data"))
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
        loss_function = parts[0]
        resolve_loss_function_spec(loss_function)
        uses_superoperator = loss_function_uses_superoperator(loss_function)
        superoperator = parts[1] if len(parts) == 2 else ("superoperator_from_mix" if uses_superoperator else None)
        if uses_superoperator:
            resolve_superop(str(superoperator))
        elif superoperator is not None:
            raise ValueError(f"loss {loss_function!r} does not use a superoperator, but {superoperator!r} was provided")
        label = loss_function if superoperator is None else f"{loss_function}:{superoperator}"
        if label in seen:
            raise ValueError(f"duplicate mode {label!r}")
        seen.add(label)
        modes.append(
            {
                "mode_index": len(modes),
                "label": label,
                "loss_function": loss_function,
                "superoperator": superoperator,
            }
        )
    if not modes:
        raise ValueError("--modes must contain at least one mode")
    return modes


def _parse_selected_coordinates(value: str, theta_size: int) -> list[int]:
    coords = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        coord = int(item)
        if coord < 1 or coord > theta_size:
            raise ValueError(f"selected coordinate {coord} is outside 1..{theta_size}")
        if coord not in coords:
            coords.append(coord)
    if not coords:
        raise ValueError("--selected-coordinates must contain at least one coordinate")
    return coords


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
    if args.fd_eps <= 0:
        raise ValueError("--fd-eps must be positive")

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
    selected_coordinates = _parse_selected_coordinates(str(args.selected_coordinates), context.theta_size)

    stamp = timestamp()
    name = _safe_experiment_name(str(args.experiment_name)) if args.experiment_name else "_".join(
        [
            "gradient_stats",
            safe_stem(args.preset),
            f"block{args.block_index:02d}",
            f"samples{args.sample_count}",
            stamp,
        ]
    )
    experiment_root = args.output_dir / name
    experiment_root.mkdir(parents=True, exist_ok=False)
    (experiment_root / "batches").mkdir()
    (experiment_root / "summary").mkdir()
    (experiment_root / "figures").mkdir()

    target = build_target_from_seed(cfg.n_qubits, cfg.target_seed, cfg.time_k)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "random_initial_gradient_statistics_experiment",
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
        "fd_eps": float(args.fd_eps),
        "training_seed_start": int(args.training_seed_start if args.training_seed_start is not None else cfg.training_seed_start),
        "sample_count": int(args.sample_count),
        "samples_per_task": int(args.samples_per_task),
        "task_count": int(args.sample_count // args.samples_per_task),
        "selected_coordinates": selected_coordinates,
        "modes": modes,
        "lightcone_semantics": context_meta.get("lightcone_semantics"),
        "loss_semantics": context_meta.get("loss_semantics"),
        **target_metadata(target),
    }
    atomic_write_json(experiment_root / "manifest.json", manifest)
    print(f"Created gradient statistics experiment: {experiment_root}")
    print(f"block_index = {args.block_index}")
    print(f"lightcone_size = {len(context.lightcone_qubits)}")
    print(f"theta_size = {context.theta_size}")
    print(f"sample_count = {args.sample_count}")
    print(f"samples_per_task = {args.samples_per_task}")
    print(f"task_count = {manifest['task_count']}")
    print(f"modes = {[mode['label'] for mode in modes]}")


if __name__ == "__main__":
    main()
