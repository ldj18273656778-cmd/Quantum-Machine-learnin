"""Resume Heisenberg Pauli training for specific blocks while keeping others fixed.

Usage:
    python task2_code/resume_training_blocks.py \
        --run-dir "task2_code/data/task2_training_n12_3blocks_heisenberg_heisenberg_pauli_20260530_152212" \
        --block-indices 2,3 \
        --extra-restarts 6 \
        --iterations 150 \
        --success-threshold 0.01 \
        --seed-offset 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from task2_code.experiment_config import (
    ExperimentConfig,
    N12_3BLOCKS_HEISENBERG,
    N20_5BLOCKS_HEISENBERG,
    N24_6BLOCKS_HEISENBERG,
    N28_7BLOCKS_HEISENBERG,
    N32_8BLOCKS_HEISENBERG,
)
from task2_code.module_e_training import (
    AdamConfig,
    build_target_objective_context,
    multi_restart_train,
    residual_operator_for_context,
    sum_block_loss,
)
from task2_code.loss_registry import (
    active_loss_breakdown,
    set_active_loss_function,
)
from task2_code.superoperator_registry import set_active_superop
from task2_code.target_factory import build_target_from_seed, target_metadata


PRESETS: dict[str, ExperimentConfig] = {
    "n12_3blocks_heisenberg": N12_3BLOCKS_HEISENBERG,
    "n20_5blocks_heisenberg": N20_5BLOCKS_HEISENBERG,
    "n24_6blocks_heisenberg": N24_6BLOCKS_HEISENBERG,
    "n28_7blocks_heisenberg": N28_7BLOCKS_HEISENBERG,
    "n32_8blocks_heisenberg": N32_8BLOCKS_HEISENBERG,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume block training with extra restarts.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None, help="override metadata preset")
    parser.add_argument("--block-indices", type=str, required=True,
                        help="comma-separated 1-based block indices to resume, e.g. 2,3")
    parser.add_argument("--extra-restarts", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--success-threshold", type=float, default=0.01)
    parser.add_argument("--seed-offset", type=int, default=5000,
                        help="added to training seeds to avoid repeating previous restarts")
    parser.add_argument("--no-progress", action="store_true", default=False)
    return parser.parse_args()


def _resolve_config(args: argparse.Namespace, metadata: dict[str, Any]) -> ExperimentConfig:
    preset = args.preset or str(metadata.get("preset", "n12_3blocks_heisenberg"))
    if preset not in PRESETS:
        raise ValueError(f"resume_training_blocks supports Heisenberg presets only, got {preset!r}")
    cfg = PRESETS[preset]
    if str(metadata.get("loss_function", cfg.loss_function)) != "heisenberg_pauli":
        raise ValueError("resume_training_blocks expects heisenberg_pauli metadata")
    return cfg


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


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    block_indices = [int(x.strip()) for x in args.block_indices.split(",")]

    params_path = run_dir / "params.npz"
    metadata_path = run_dir / "metadata.json"
    if not params_path.exists():
        raise FileNotFoundError(f"params.npz not found: {params_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")

    # Load existing data
    arrays = dict(np.load(params_path, allow_pickle=False))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    cfg = _resolve_config(args, metadata)
    set_active_loss_function(cfg.loss_function)
    invalid = [idx for idx in block_indices if idx < 1 or idx > cfg.block_count]
    if invalid:
        raise ValueError(f"block indices out of range 1..{cfg.block_count}: {invalid}")

    # Build target once
    target = build_target_from_seed(cfg.n_qubits, cfg.target_seed, cfg.time_k)
    adam_cfg = AdamConfig(iterations=args.iterations, lr=cfg.lr)

    print(f"Resume from: {run_dir}")
    print(f"Block indices to resume: {block_indices}")
    print(f"Extra restarts per block: {args.extra_restarts}")
    print(f"Iterations per restart: {args.iterations}")
    print(f"Success threshold: {args.success_threshold}")
    print(f"Seed offset: {args.seed_offset}")

    _latest_results: dict[int, Any] = {}

    for bi_1based in block_indices:
        bi = bi_1based - 1  # 0-based
        block_qubits = cfg.blocks[bi]
        target_bit = cfg.target_bits[bi]
        params_key = f"best_params_block_{bi_1based}"

        print(f"\n{'='*60}")
        print(f"Block {bi_1based}: qubits {block_qubits}, target_bit {target_bit}")
        print(f"{'='*60}")

        context, context_meta = build_target_objective_context(
            cfg.n_qubits, block_qubits, target_bit,
            cfg.radius, cfg.target_seed, cfg.time_k,
            lightcone_mode=cfg.lightcone_mode,
            loss_mode=cfg.loss_mode,
            require_unitary=False,
            max_n_qubits=cfg.n_qubits,
            max_hilbert_dim=4096,
            ansatz=cfg.ansatz,
            block_only_ansatz=cfg.block_only_ansatz,
        )
        loss_fn = lambda theta, ctx=context: sum_block_loss(theta, ctx)

        # Use a different seed so new restarts don't repeat the old ones
        seed = cfg.training_seed_for_block(bi) + args.seed_offset
        rng = np.random.default_rng(seed)

        result = multi_restart_train(
            loss_fn,
            restarts=args.extra_restarts,
            rng=rng,
            config=adam_cfg,
            n_qubits=context.ansatz_qubits,
            ansatz=context.ansatz,
            show_progress=not args.no_progress,
            success_threshold=args.success_threshold,
        )
        _latest_results[bi_1based] = result

        arrays[params_key] = result.best_params
        loss_breakdown = active_loss_breakdown(result.best_params, context)

        print(f"  selected_restart = {result.best_restart + 1}/{args.extra_restarts}")
        print(f"  completed_restarts = {len(result.restart_results)}/{args.extra_restarts}")
        print(f"  best_loss = {result.best_loss:.12g}")
        print(f"  loss_breakdown = { {k: f'{v:.3e}' for k, v in loss_breakdown.items()} }")

    # Save to a new run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_run_dir = run_dir.parent / f"{run_dir.name}_resumed_{timestamp}"
    new_run_dir.mkdir(parents=True, exist_ok=False)

    new_params_path = new_run_dir / "params.npz"
    new_metadata_path = new_run_dir / "metadata.json"

    # Update theta_sizes
    arrays["theta_sizes"] = np.asarray(
        [len(arrays[f"best_params_block_{i}"]) for i in range(1, cfg.block_count + 1)],
        dtype=int,
    )
    np.savez(new_params_path, **arrays)

    metadata["timestamp"] = timestamp
    metadata["run_dir"] = str(new_run_dir)
    metadata["resumed_from"] = str(run_dir)
    for item in metadata["blocks"]:
        item["extra_restarts_run"] = args.extra_restarts
        item["success_threshold"] = args.success_threshold

    new_metadata_path.write_text(json.dumps(_json_ready(metadata), indent=2), encoding="utf-8")

    # Save loss trajectories (all restart histories) for resumed blocks
    loss_trajectories_path = new_run_dir / "loss_trajectories.json"
    restart_records: list[dict[str, Any]] = []
    all_loss_histories: list[list[float]] = []
    all_block_labels: list[str] = []
    for bi_1based in range(1, cfg.block_count + 1):
        bi = bi_1based - 1
        label = f"best_params_block_{bi_1based}"
        if bi_1based in block_indices:
            restart_records.append({
                "block_index": bi_1based,
                "block_qubits": list(cfg.blocks[bi]),
                "target_bit": int(cfg.target_bits[bi]),
                "best_loss": float(metadata["blocks"][bi].get("best_loss", 0)),
                "all_restart_loss_histories": [
                    rr.loss_history.tolist() for rr in _latest_results[bi_1based].restart_results
                ],
            })
            best_hist = _latest_results[bi_1based].restart_results[
                _latest_results[bi_1based].best_restart
            ].loss_history.tolist()
            all_loss_histories.append(best_hist)
            all_block_labels.append(f"Block {bi_1based}")
    loss_trajectories_path.write_text(json.dumps(restart_records, indent=2), encoding="utf-8")

    # Save loss trajectory plot for resumed blocks
    if all_loss_histories:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_blocks = len(all_loss_histories)
        n_cols = min(n_blocks, 3)
        n_rows = (n_blocks + 2) // 3
        fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
        axes = np.asarray(axes_grid).ravel()
        for idx in range(n_blocks):
            ax = axes[idx]
            steps = list(range(len(all_loss_histories[idx])))
            ax.plot(steps, all_loss_histories[idx], linewidth=1)
            ax.set_title(all_block_labels[idx])
            ax.set_xlabel("iteration")
            ax.set_ylabel("loss")
            ax.grid(alpha=0.3)
        for idx in range(n_blocks, len(axes)):
            axes[idx].set_visible(False)
        fig.suptitle(f"Resumed training  n=12  heisenberg_pauli  {timestamp}", fontsize=14)
        fig.tight_layout()
        plot_path = new_run_dir / "loss_trajectory.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  loss_plot: {plot_path}")

    print(f"\nSaved: {new_run_dir}")
    print(f"  params: {new_params_path}")
    print(f"  metadata: {new_metadata_path}")
    print(f"  loss_trajectories: {loss_trajectories_path}")


if __name__ == "__main__":
    main()
