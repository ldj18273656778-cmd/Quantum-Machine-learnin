"""Warm-start selected Heisenberg blocks from another training run.

Examples:
    python task2_code/warmstart_blocks_heisenberg.py \
        --preset n12_3blocks_heisenberg \
        --warmstart-run-dir task2_code/data/task2_training_n12_3blocks_edge_quantum_channel_superoperator_from_mix_... \
        --base-run-dir task2_code/data/task2_training_n12_3blocks_heisenberg_heisenberg_pauli_... \
        --block-indices 2,3

    python task2_code/warmstart_blocks_heisenberg.py \
        --preset n20_5blocks_heisenberg \
        --warmstart-run-dir task2_code/data/<n20_mixed_run> \
        --base-run-dir task2_code/data/<n20_heisenberg_run> \
        --block-indices 2,4,5
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from task2_code.experiment_config import (
    ExperimentConfig,
    N12_3BLOCKS_HEISENBERG,
    N20_5BLOCKS_HEISENBERG,
    N24_6BLOCKS_HEISENBERG,
    N28_7BLOCKS_HEISENBERG,
    N32_8BLOCKS_HEISENBERG,
)
from task2_code.loss_registry import active_loss_breakdown, set_active_loss_function
from task2_code.module_e_training import (
    AdamConfig,
    adam_optimize,
    build_target_objective_context,
    sum_block_loss,
)


PRESETS: dict[str, ExperimentConfig] = {
    "n12_3blocks_heisenberg": N12_3BLOCKS_HEISENBERG,
    "n20_5blocks_heisenberg": N20_5BLOCKS_HEISENBERG,
    "n24_6blocks_heisenberg": N24_6BLOCKS_HEISENBERG,
    "n28_7blocks_heisenberg": N28_7BLOCKS_HEISENBERG,
    "n32_8blocks_heisenberg": N32_8BLOCKS_HEISENBERG,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm-start selected Heisenberg blocks from saved params.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="n12_3blocks_heisenberg")
    parser.add_argument("--warmstart-run-dir", type=Path, required=True, help="run containing initial params for selected blocks")
    parser.add_argument("--base-run-dir", type=Path, required=True, help="run supplying params for non-selected blocks and base metadata")
    parser.add_argument("--block-indices", type=str, required=True, help="comma-separated 1-based block indices, e.g. 2,4,5")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--fd-eps", type=float, default=1e-5)
    parser.add_argument("--ansatz", default=None, help="override preset ansatz registry key")
    parser.add_argument("--block-only-ansatz", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-progress", action="store_true", default=False)
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _parse_block_indices(value: str, block_count: int) -> list[int]:
    items = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError("--block-indices must contain at least one block index")
    invalid = [idx for idx in items if idx < 1 or idx > block_count]
    if invalid:
        raise ValueError(f"block indices out of range 1..{block_count}: {invalid}")
    if len(items) != len(set(items)):
        raise ValueError(f"block indices must not contain duplicates: {items}")
    return items


def _load_params(run_dir: Path, label: str) -> dict[str, NDArray[np.float64]]:
    params_path = run_dir / "params.npz"
    if not params_path.exists():
        raise FileNotFoundError(f"{label} params.npz not found: {params_path}")
    return {name: np.asarray(value, dtype=float) for name, value in np.load(params_path, allow_pickle=False).items()}


def _load_metadata(run_dir: Path, label: str) -> dict[str, Any]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"{label} metadata.json not found: {metadata_path}")
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{label} metadata must be a JSON object: {metadata_path}")
    return data


def _params_key(block_index: int) -> str:
    return f"best_params_block_{block_index}"


def _require_param(params: dict[str, NDArray[np.float64]], key: str, source: Path) -> NDArray[np.float64]:
    if key not in params:
        raise KeyError(f"{key!r} missing from {source / 'params.npz'}")
    return np.asarray(params[key], dtype=float)


def _resolved_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = PRESETS[args.preset]
    updates: dict[str, Any] = {}
    if args.iterations is not None:
        updates["iterations"] = args.iterations
    if args.lr is not None:
        updates["lr"] = args.lr
    if args.ansatz is not None:
        updates["ansatz"] = args.ansatz
    if args.block_only_ansatz is not None:
        updates["block_only_ansatz"] = args.block_only_ansatz
    return replace(cfg, **updates)


def _build_context(cfg: ExperimentConfig, block_index: int):
    block_qubits = cfg.blocks[block_index - 1]
    target_bit = cfg.target_bits[block_index - 1]
    return build_target_objective_context(
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


def _update_block_metadata(
    metadata: dict[str, Any],
    block_index: int,
    result,
    init_loss: float,
    init_breakdown: dict[int, float],
    loss_breakdown: dict[int, float],
    warmstart_run_dir: Path,
) -> None:
    blocks = metadata.setdefault("blocks", [])
    if not isinstance(blocks, list):
        raise ValueError("metadata['blocks'] must be a list")
    target: dict[str, Any] | None = None
    for item in blocks:
        if isinstance(item, dict) and int(item.get("block_index", -1)) == block_index:
            target = item
            break
    if target is None:
        target = {"block_index": block_index}
        blocks.append(target)
    target["best_loss"] = float(result.best_loss)
    target["best_iteration"] = int(result.best_iteration)
    target["loss_breakdown"] = {str(k): float(v) for k, v in loss_breakdown.items()}
    target["per_bit_losses"] = {str(k): float(v) for k, v in loss_breakdown.items()}
    target["warmstart_initial_loss"] = float(init_loss)
    target["warmstart_initial_breakdown"] = {str(k): float(v) for k, v in init_breakdown.items()}
    target["warmstart_from"] = str(warmstart_run_dir)


def _save_loss_plot(out_dir: Path, records: list[dict[str, Any]], timestamp: str) -> None:
    if not records:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for record in records:
        ax.plot(record["loss_history"], linewidth=1, label=f"block {record['block_index']}")
    ax.set_title(f"Warm-start Heisenberg blocks  {timestamp}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("Heisenberg Pauli loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path = out_dir / "loss_trajectory.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss plot: {plot_path}")


def main() -> None:
    args = parse_args()
    cfg = _resolved_config(args)
    if cfg.loss_function != "heisenberg_pauli":
        raise ValueError(f"warm-start script expects heisenberg_pauli loss, got {cfg.loss_function!r}")
    set_active_loss_function(cfg.loss_function)

    block_indices = _parse_block_indices(args.block_indices, cfg.block_count)
    warm_params = _load_params(args.warmstart_run_dir, "warmstart")
    base_params = _load_params(args.base_run_dir, "base")
    metadata = _load_metadata(args.base_run_dir, "base")

    print(f"preset = {args.preset}")
    print(f"n_qubits = {cfg.n_qubits}")
    print(f"blocks to warm-start = {block_indices}")
    print(f"ansatz = {cfg.ansatz}")
    print(f"block_only_ansatz = {cfg.block_only_ansatz}")
    print(f"warmstart params = {args.warmstart_run_dir}")
    print(f"base params = {args.base_run_dir}")

    combined: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    adam_cfg = AdamConfig(iterations=cfg.iterations, lr=cfg.lr, fd_eps=args.fd_eps)

    selected = set(block_indices)
    for block_index in range(1, cfg.block_count + 1):
        key = _params_key(block_index)
        if block_index not in selected:
            theta = _require_param(base_params, key, args.base_run_dir)
            context, _ = _build_context(cfg, block_index)
            if theta.shape != (context.theta_size,):
                raise ValueError(f"base {key} has shape {theta.shape}, expected ({context.theta_size},)")
            combined[key] = theta
            continue

        context, _ = _build_context(cfg, block_index)
        initial_theta = _require_param(warm_params, key, args.warmstart_run_dir)
        if initial_theta.shape != (context.theta_size,):
            raise ValueError(f"warmstart {key} has shape {initial_theta.shape}, expected ({context.theta_size},)")

        print(f"\nBlock {block_index}: qubits {cfg.blocks[block_index - 1]}, target_bit {cfg.target_bits[block_index - 1]}")
        print(f"  lightcone = {context.lightcone_qubits}")
        print(f"  theta_size = {context.theta_size}")
        loss_fn = lambda theta, ctx=context: sum_block_loss(theta, ctx)
        init_loss = float(loss_fn(initial_theta))
        init_breakdown = active_loss_breakdown(initial_theta, context)
        print(f"  init_loss = {init_loss:.6e}")
        print(f"  init_breakdown = { {k: f'{v:.3e}' for k, v in init_breakdown.items()} }")

        result = adam_optimize(loss_fn, initial_theta, adam_cfg, show_progress=not args.no_progress)
        loss_breakdown = active_loss_breakdown(result.best_params, context)
        print(f"  best_loss = {result.best_loss:.6e}")
        print(f"  best_iter = {result.best_iteration}")
        print(f"  breakdown = { {k: f'{v:.3e}' for k, v in loss_breakdown.items()} }")
        combined[key] = result.best_params
        _update_block_metadata(
            metadata,
            block_index,
            result,
            init_loss,
            init_breakdown,
            loss_breakdown,
            args.warmstart_run_dir,
        )
        records.append(
            {
                "block_index": block_index,
                "block_qubits": list(cfg.blocks[block_index - 1]),
                "target_bit": int(cfg.target_bits[block_index - 1]),
                "best_loss": float(result.best_loss),
                "best_iteration": int(result.best_iteration),
                "initial_loss": float(init_loss),
                "loss_history": result.loss_history.tolist(),
            }
        )

    combined["blocks"] = np.asarray(cfg.blocks, dtype=int)
    combined["target_bits"] = np.asarray(cfg.target_bits, dtype=int)
    combined["theta_sizes"] = np.asarray([len(combined[_params_key(i)]) for i in range(1, cfg.block_count + 1)], dtype=int)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_parent = args.output_dir if args.output_dir is not None else args.base_run_dir.parent
    out_dir = out_parent / f"warmstart_blocks_heisenberg_{args.preset}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    params_path = out_dir / "params.npz"
    np.savez(params_path, **combined)

    metadata.update(
        {
            "artifact_type": "warmstart_blocks_heisenberg",
            "timestamp": timestamp,
            "preset": args.preset,
            "run_dir": str(out_dir),
            "warmstart_run_dir": str(args.warmstart_run_dir),
            "base_run_dir": str(args.base_run_dir),
            "warmstart_block_indices": block_indices,
            "n_qubits": cfg.n_qubits,
            "block_qubits": cfg.blocks,
            "target_bits": cfg.target_bits,
            "radius": cfg.radius,
            "lightcone_mode": cfg.lightcone_mode,
            "loss_mode": cfg.loss_mode,
            "ansatz": cfg.ansatz,
            "block_only_ansatz": cfg.block_only_ansatz,
            "loss_function": cfg.loss_function,
            "iterations": cfg.iterations,
            "lr": cfg.lr,
            "fd_eps": args.fd_eps,
            "params_file": params_path.name,
            "metadata_file": "metadata.json",
        }
    )
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(_json_ready(metadata), indent=2), encoding="utf-8")

    trajectory_path = out_dir / "loss_trajectories.json"
    trajectory_path.write_text(json.dumps(_json_ready(records), indent=2), encoding="utf-8")
    _save_loss_plot(out_dir, records, timestamp)

    print(f"\nSaved run directory: {out_dir}")
    print(f"Saved params: {params_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved loss trajectories: {trajectory_path}")


if __name__ == "__main__":
    main()
