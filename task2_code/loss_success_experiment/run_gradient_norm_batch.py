"""Compute one Heisenberg-loss gradient norm for a saved best parameter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from numpy.typing import NDArray

from task2_code.hpc_parallel_training.hpc_block_flow import PRESETS, atomic_write_json
from task2_code.loss_registry import set_active_loss_function
from task2_code.module_e_training import build_target_objective_context, finite_difference_gradient, sum_block_loss


FloatArray = NDArray[np.float64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute finite-difference gradient norm for one paired best parameter.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True, help="1..800: 1-400 Heisenberg-only, 401-800 warm-start")
    parser.add_argument("--fd-eps", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def _load_manifest(experiment_root: Path) -> dict[str, Any]:
    path = experiment_root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    return data


def _context_from_manifest(manifest: dict[str, Any]):
    preset = str(manifest["preset"])
    cfg = PRESETS[preset]
    context, _context_meta = build_target_objective_context(
        int(manifest["n_qubits"]),
        [int(q) for q in manifest["block_qubits"]],
        int(manifest["target_bit"]),
        int(manifest["radius"]),
        int(manifest["target_seed"]),
        int(manifest["time_k"]),
        lightcone_mode=str(manifest["lightcone_mode"]),
        loss_mode=str(manifest["loss_mode"]),
        require_unitary=False,
        max_n_qubits=int(manifest["n_qubits"]),
        max_hilbert_dim=4096,
        ansatz=str(manifest.get("ansatz", cfg.ansatz)),
        block_only_ansatz=bool(manifest.get("block_only_ansatz", cfg.block_only_ansatz)),
    )
    return context


def _load_theta(experiment_root: Path, task_index: int) -> tuple[str, int, FloatArray, float, bool]:
    summary_path = experiment_root / "summary" / "paired_best_params.npz"
    if not summary_path.exists():
        raise FileNotFoundError(f"paired best params not found: {summary_path}")
    data = np.load(summary_path)
    pair_count = int(np.asarray(data["pair_indices"]).size)
    if task_index < 1 or task_index > 2 * pair_count:
        raise ValueError(f"--task-index must be in 1..{2 * pair_count}, got {task_index}")
    if task_index <= pair_count:
        row = task_index - 1
        training_type = "heisenberg_only"
        theta = np.asarray(data["heisenberg_best_params"][row], dtype=float)
        best_loss = float(data["heisenberg_best_losses"][row])
        success = bool(data["heisenberg_success"][row])
    else:
        row = task_index - pair_count - 1
        training_type = "warmstart"
        theta = np.asarray(data["warmstart_best_params"][row], dtype=float)
        best_loss = float(data["warmstart_best_losses"][row])
        success = bool(data["warmstart_success"][row])
    pair_index = int(data["pair_indices"][row])
    return training_type, pair_index, theta, best_loss, success


def main() -> None:
    args = parse_args()
    manifest = _load_manifest(args.experiment_root)
    task_index = int(args.task_index)
    output_dir = args.experiment_root / "summary" / "gradient_norms"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"gradient_norm_{task_index:06d}.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"gradient norm result exists; pass --overwrite to replace it: {output_path}")

    training_type, pair_index, theta, best_loss, success = _load_theta(args.experiment_root, task_index)
    if theta.shape != (int(manifest["theta_size"]),):
        raise ValueError(f"theta shape mismatch: {theta.shape}, expected ({manifest['theta_size']},)")

    set_active_loss_function("heisenberg_pauli")
    context = _context_from_manifest(manifest)
    loss_fn = lambda value, ctx=context: sum_block_loss(value, ctx)
    fd_eps = float(args.fd_eps if args.fd_eps is not None else manifest.get("fd_eps", 1e-5))

    started = perf_counter()
    loss_at_theta = float(loss_fn(theta))
    grad = finite_difference_gradient(loss_fn, theta, fd_eps)
    elapsed = perf_counter() - started
    grad_norm = float(np.linalg.norm(grad))
    grad_inf_norm = float(np.linalg.norm(grad, ord=np.inf))
    payload = {
        "task_index": task_index,
        "training_type": training_type,
        "pair_index": pair_index,
        "fd_eps": fd_eps,
        "best_loss_recorded": best_loss,
        "loss_at_theta": loss_at_theta,
        "loss_difference": float(loss_at_theta - best_loss),
        "success": success,
        "grad_norm": grad_norm,
        "grad_inf_norm": grad_inf_norm,
        "theta_size": int(theta.size),
        "elapsed_seconds": float(elapsed),
    }
    atomic_write_json(output_path, payload)
    print(f"Wrote {output_path}")
    print(f"{training_type} pair={pair_index} loss={loss_at_theta:.8g} grad_norm={grad_norm:.8g} elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
