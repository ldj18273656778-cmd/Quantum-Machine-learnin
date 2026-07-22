"""Print per-qubit losses before and after Module E ADAM training.

The training objective follows Eq. S.2.3: sum over block qubits of squared
Frobenius norm.  After training this script prints each qubit's loss at the
initial, final, and best parameters, and displays a matplotlib plot of the
per-qubit loss trajectories.

For evaluation (Eq. S.2.4) the max per-qubit loss is also reported.

Example:
    python task2_code/check_module_e_per_bit_losses.py --iterations 150
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from task2_code.experiment_config import DEFAULT_SEED as default_seed
from task2_code.module_e_training import (
    AdamConfig,
    FloatArray,
    ObjectiveContext,
    build_target_objective_context,
    multi_restart_train,
    residual_operator_for_context,
    target_bit_loss,
)
from task2_code.loss_registry import set_active_loss_function
from task2_code.superoperator_registry import get_active_superop, set_active_superop


def _parse_block(value: str) -> list[int]:
    items = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(items) != 4:
        raise argparse.ArgumentTypeError("block must contain exactly four comma-separated labels")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check all per-bit losses around target-bit ADAM training.")
    parser.add_argument("--n-qubits", type=int, default=12)
    parser.add_argument("--block", type=_parse_block, default=[8, 9, 10, 11])
    parser.add_argument("--target-bit", type=int, default=9)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--lightcone-mode", choices=["circuit", "radius"], default="circuit")
    parser.add_argument("--loss-mode", choices=["lightcone", "full_system"], default="lightcone")
    parser.add_argument("--time-k", type=int, default=5)
    parser.add_argument("--target-seed", type=int, default=default_seed)
    parser.add_argument("--training-seed", type=int, default=default_seed + 1000)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--fd-eps", type=float, default=1e-5)
    parser.add_argument("--ansatz", default="default_5layer_cz")
    parser.add_argument("--block-only-ansatz", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loss-function", default="edge_quantum_channel")
    parser.add_argument("--superoperator", default="superoperator_from_mix")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()

def all_block_losses(theta: NDArray[np.float64], context: ObjectiveContext) -> dict[int, float]:
    residual, loss_qubits = residual_operator_for_context(theta, context)
    return get_active_superop()(
        residual,
        context.block_qubits,
        loss_qubits,
        target_bits=None,
    )


def print_loss_table(
    block_qubits: tuple[int, ...],
    target_bit: int,
    initial_losses: dict[int, float],
    final_losses: dict[int, float],
    best_losses: dict[int, float],
) -> None:
    print("\nPer-bit loss comparison")
    print("q\tinitial\t\tfinal\t\tbest\t\tfinal_delta\tbest_delta")
    for q in block_qubits:
        initial = initial_losses[q]
        final = final_losses[q]
        best = best_losses[q]
        marker = ""
        print(
            f"{q}\t{initial:.12g}\t{final:.12g}\t{best:.12g}\t"
            f"{final - initial:+.12g}\t{best - initial:+.12g}{marker}"
        )

    all_final_decreased = all(final_losses[q] < initial_losses[q] for q in block_qubits)
    all_best_decreased = all(best_losses[q] < initial_losses[q] for q in block_qubits)
    max_final = max(final_losses[q] for q in block_qubits)
    print(f"\nall_final_losses_decreased = {all_final_decreased}")
    print(f"all_best_losses_decreased = {all_best_decreased}")
    print(f"max_per_qubit_loss (S.2.4 evaluation) = {max_final:.12g}  target delta=0.01")
    print("Note: training objective is sum over block qubits (Eq. S.2.3).")


def plot_per_bit_trajectories(
    block_qubits: tuple[int, ...],
    iterations: int,
    losses_over_time: dict[int, list[float]],
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    steps = list(range(len(losses_over_time[block_qubits[0]])))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for q, color in zip(block_qubits, colors):
        ax1.plot(steps, losses_over_time[q], color=color, linewidth=1.2, label=f"qubit {q}")
    ax1.set_xlabel("ADAM iteration")
    ax1.set_ylabel("per-bit loss")
    ax1.set_title("Per-qubit loss trajectories")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    sum_traj = [
        sum(losses_over_time[q][i] for q in block_qubits)
        for i in range(len(steps))
    ]
    ax2.plot(steps, sum_traj, color="black", linewidth=1.4, label="sum-block loss (objective, Eq. S.2.3)")
    for q, color in zip(block_qubits, colors):
        ax2.plot(steps, losses_over_time[q], color=color, linewidth=0.4, alpha=0.5)
    ax2.set_xlabel("ADAM iteration")
    ax2.set_ylabel("loss")
    ax2.set_title("sum-block objective (Eq. S.2.3) vs per-qubit losses")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    set_active_loss_function(args.loss_function)
    set_active_superop(args.superoperator)
    context, metadata = build_target_objective_context(
        n_qubits=args.n_qubits,
        block_qubits=args.block,
        target_bit=args.target_bit,
        radius=args.radius,
        target_seed=args.target_seed,
        time_k=args.time_k,
        lightcone_mode=args.lightcone_mode,
        loss_mode=args.loss_mode,
        require_unitary=False,
        max_n_qubits=12,
        max_hilbert_dim=4096,
        ansatz=args.ansatz,
        block_only_ansatz=args.block_only_ansatz,
    )
    loss_fn = lambda theta: target_bit_loss(theta, context)
    config = AdamConfig(iterations=args.iterations, lr=args.lr, fd_eps=args.fd_eps)
    training_rng = np.random.default_rng(args.training_seed)

    # collect per-qubit losses at each ADAM step
    losses_over_time: dict[int, list[float]] = {q: [] for q in context.block_qubits}

    def record_losses(step: int, theta: FloatArray, _scalar_loss: float) -> None:
        block_losses = all_block_losses(theta, context)
        for q in context.block_qubits:
            losses_over_time[q].append(block_losses[q])

    result = multi_restart_train(
        loss_fn,
        args.restarts,
        training_rng,
        config,
        n_qubits=context.ansatz_qubits,
        ansatz=context.ansatz,
        show_progress=True,
        step_callback=record_losses,
    )
    best_run = result.restart_results[result.best_restart]

    initial_theta = best_run.initial_params
    final_theta = best_run.final_params
    best_theta = best_run.best_params
    initial_losses = all_block_losses(initial_theta, context)
    final_losses = all_block_losses(final_theta, context)
    best_losses = all_block_losses(best_theta, context)

    print("Module E per-bit loss check")
    print(f"n_qubits = {metadata['n_qubits']}")
    print(f"block_qubits = {context.block_qubits}")
    print(f"target_bit = {context.target_bit}")
    print(f"lightcone_qubits = {context.lightcone_qubits}")
    print(f"lightcone_mode = {metadata['lightcone_mode']}")
    print(f"loss_mode = {metadata['loss_mode']}")
    print(f"loss_semantics = {metadata['loss_semantics']}")
    print(f"ansatz_qubits = {context.ansatz_qubits}")
    print(f"theta_size = {context.theta_size}")
    print(f"loss_qubits = {context.loss_qubits}")
    print(f"target_seed = {args.target_seed}")
    print(f"training_seed = {args.training_seed}")
    print(f"iterations = {args.iterations}")
    print(f"restarts = {args.restarts}")
    print(f"lr = {args.lr}")
    print(f"fd_eps = {args.fd_eps}")
    print(f"loss_function = {args.loss_function}")
    print(f"superoperator = {args.superoperator}")
    print("loss_objective = sum over block qubits (Eq. S.2.3)")
    print(f"best_restart = {result.best_restart}")
    print(f"best_iteration = {result.best_iteration}")
    print(f"best_sum_block_loss = {result.best_loss:.12g}")
    print_loss_table(context.block_qubits, context.target_bit, initial_losses, final_losses, best_losses)

    print(f"\nrecorded {len(losses_over_time[context.block_qubits[0]])} snapshots (0..{args.iterations})")
    if not args.no_plot:
        plot_per_bit_trajectories(context.block_qubits, args.iterations, losses_over_time)


if __name__ == "__main__":
    main()
