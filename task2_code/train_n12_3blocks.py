"""Train three 4-qubit blocks on n=12 and produce sum_block_123 plot.

Usage:
    python task2_code/train_n12_3blocks.py

The experiment configuration lives in
``task2_code.experiment_config.N12_3BLOCKS``.  To change seeds, time_k,
lr or output paths, edit that preset instead of hunting through scripts.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
import sys
from pathlib import Path

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from task2_code.ansatz_registry import random_ansatz_theta
from task2_code.experiment_config import N12_3BLOCKS, ExperimentConfig
from task2_code.module_e_training import (
    AdamConfig,
    adam_optimize,
    build_target_objective_context,
    sum_block_loss,
)
from task2_code.loss_registry import active_loss_breakdown, loss_function_uses_superoperator, set_active_loss_function
from task2_code.superoperator_registry import set_active_superop

# ── configuration ────────────────────────────────────────────────────
config: ExperimentConfig = N12_3BLOCKS
HILBERT_GUARD = 1 << 12  # 4096 — guard replicated from existing behaviour

# Activate the configured superoperator so all internal loss calls
# (sum_block_loss, target_bit_loss, etc.) use it automatically.
set_active_loss_function(config.loss_function)
USES_SUPEROPERATOR = loss_function_uses_superoperator(config.loss_function)
if USES_SUPEROPERATOR:
    set_active_superop(config.superoperator)
print(f"loss_function = {config.loss_function!r}")
if USES_SUPEROPERATOR:
    print(f"superoperator = {config.superoperator!r}")
else:
    print("superoperator = not used by this loss_function")

# ── derived paths ────────────────────────────────────────────────────
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = config.output_dir
DATA_DIR = config.data_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMAGE = OUTPUT_DIR / f"sum_block_123_{RUN_TIMESTAMP}.png"
PARAMS_PATH = DATA_DIR / f"sum_block_123_best_params_{RUN_TIMESTAMP}.npz"
METADATA_PATH = DATA_DIR / f"sum_block_123_best_params_{RUN_TIMESTAMP}.json"

# ── training ─────────────────────────────────────────────────────────
all_step_records: list[list[float]] = []
all_block_losses: list[dict[int, float]] = []
overall_best_losses: list[float] = []
all_best_params: list[NDArray[np.float64]] = []
block_metadata: list[dict[str, object]] = []

adam_cfg = AdamConfig(iterations=config.iterations, lr=config.lr)

for bi, (block_qubits, target_bit) in enumerate(zip(config.blocks, config.target_bits)):
    print(f"\nBlock {bi + 1}: qubits {block_qubits}, target_bit {target_bit}")
    rng = np.random.default_rng(config.training_seed_for_block(bi))
    context, _ = build_target_objective_context(
        config.n_qubits,
        block_qubits,
        target_bit,
        config.radius,
        config.target_seed,
        config.time_k,
        lightcone_mode=config.lightcone_mode,
        loss_mode=config.loss_mode,
        require_unitary=False,
        max_n_qubits=config.n_qubits,
        max_hilbert_dim=HILBERT_GUARD,
        ansatz=config.ansatz,
        block_only_ansatz=config.block_only_ansatz,
    )
    print(f"  lightcone: {context.lightcone_qubits}")

    best_restart_idx = -1
    best_step_losses: list[float] = []
    best_loss = float("inf")
    best_per_bit: dict[int, float] = {}
    best_params = None

    for restart_idx in range(config.max_restarts):
        print(f"  restart {restart_idx + 1}/{config.max_restarts}")
        theta0 = np.asarray(
            random_ansatz_theta(
                context.ansatz,
                rng,
                low=0.0,
                high=2 * np.pi,
                n_qubits=context.ansatz_qubits,
            ),
            dtype=float,
        )
        init_loss = sum_block_loss(theta0, context)
        print(f"    init loss: {init_loss:.6f}")

        step_losses: list[float] = []

        def step_cb(step: int, theta: NDArray[np.float64], loss: float) -> None:
            step_losses.append(float(loss))

        loss_fn = lambda t: sum_block_loss(t, context)
        result = adam_optimize(loss_fn, theta0, adam_cfg, show_progress=True, step_callback=step_cb)

        final_loss = step_losses[-1] if step_losses else result.best_loss
        per_bit_vals = active_loss_breakdown(result.best_params, context)
        print(f"    best loss: {result.best_loss:.6f}")

        if USES_SUPEROPERATOR:
            success_threshold_reached = all(v < config.success_threshold for v in per_bit_vals.values())
            success_message = f"all per-bit losses below {config.success_threshold}"
        else:
            success_threshold_reached = result.best_loss < config.success_threshold
            success_message = f"best loss below {config.success_threshold}"
        if final_loss < best_loss:
            best_restart_idx = restart_idx
            best_step_losses = list(step_losses)
            best_loss = final_loss
            best_per_bit = per_bit_vals
            best_params = result.best_params.copy()
            print(f"    ** new best **")
        if success_threshold_reached:
            print(f"    {success_message} -> stopping restarts")
            break

    if best_params is None:
        raise RuntimeError(f"block {bi + 1} did not produce best_params")
    all_step_records.append(best_step_losses)
    all_block_losses.append(best_per_bit)
    overall_best_losses.append(best_loss)
    all_best_params.append(best_params)

    print(f"  best restart: {best_restart_idx + 1}  best loss: {best_loss:.6f}")
    for q, val in sorted(best_per_bit.items()):
        print(f"    qubit {q}: {val:.6f}")

    block_metadata.append(
        {
            "block_index": bi + 1,
            "block_qubits": list(block_qubits),
            "target_bit": target_bit,
            "lightcone_qubits": list(context.lightcone_qubits),
            "ansatz": context.ansatz,
            "block_only_ansatz": context.block_only_ansatz,
            "ansatz_qubits": context.ansatz_qubits,
            "theta_size": context.theta_size,
            "best_loss": best_loss,
            "per_bit_losses": {str(k): v for k, v in best_per_bit.items()},
        }
    )

# ── save ─────────────────────────────────────────────────────────────
np.savez(
    PARAMS_PATH,
    **{f"best_params_block_{bi + 1}": all_best_params[bi] for bi in range(config.block_count)},
)
metadata = {
    "timestamp": RUN_TIMESTAMP,
    "n_qubits": config.n_qubits,
    "radius": config.radius,
    "lightcone_mode": config.lightcone_mode,
    "loss_mode": config.loss_mode,
    "ansatz": config.ansatz,
    "block_only_ansatz": config.block_only_ansatz,
    "loss_function": config.loss_function,
    "loss_function_uses_superoperator": USES_SUPEROPERATOR,
    "iterations": config.iterations,
    "lr": config.lr,
    "success_threshold": config.success_threshold,
    "max_restarts": config.max_restarts,
    "blocks": block_metadata,
}
if USES_SUPEROPERATOR:
    metadata["superoperator"] = config.superoperator
with open(METADATA_PATH, "w", encoding="utf-8") as fh:
    json.dump(metadata, fh, indent=2)
print(f"\nSaved params: {PARAMS_PATH}")
print(f"Saved metadata: {METADATA_PATH}")

# ── plot ─────────────────────────────────────────────────────────────
fig, axes_grid = plt.subplots(2, 3, figsize=(18, 10))
axes = np.asarray(axes_grid).ravel()
for bi in range(config.block_count):
    ax = axes[bi]
    steps = list(range(len(all_step_records[bi])))
    ax.plot(steps, all_step_records[bi], linewidth=1)
    ax.set_title(f"Block {bi + 1}  best loss={overall_best_losses[bi]:.4f}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
for bi in range(config.block_count, len(axes)):
    axes[bi].set_visible(False)
fig.suptitle(f"sum_block_123  n={config.n_qubits}  restarts up to {config.max_restarts}  {RUN_TIMESTAMP}", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved: {OUTPUT_IMAGE}")
