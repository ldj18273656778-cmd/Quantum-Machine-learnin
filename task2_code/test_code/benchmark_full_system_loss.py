"""Benchmark one full-system local-inversion loss evaluation.

This script compares the current light-cone loss with a direct full-system
residual-channel loss.  The full-system path embeds the light-cone ansatz into
all ``n`` qubits, forms ``U_full @ U_trial_full.conj().T``, and then computes
the per-bit reduced-superoperator loss on the full ``n``-qubit residual.

Default setup: n=10, block [4,5,6,7].  For the current Task 2 target circuit,
the circuit-derived light cone is typically eight qubits, so this is a useful
small-system proxy for the n=12, n_C=8 case without requiring a 4096x4096 full
residual matrix.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cirq
import numpy as np
from numpy.typing import NDArray

from task2_code.ansatz import ansatz_unitary, random_theta
from task2_code.experiment_config import DEFAULT_SEED as default_seed
from task2_code.lightcone import backward_block_lightcone_from_circuit
from task2_code.local_loss import embed_block_unitary_in_lightcone
from task2_code.module_e_training import build_target_objective_context, target_bit_loss
from task2_code.superoperator import per_bit_losses_from_V
from task2_code.U_target import build_h_diag, build_u_target_circuit


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _parse_block(value: str) -> list[int]:
    items = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(items) != 4:
        raise argparse.ArgumentTypeError("block must contain exactly four comma-separated labels")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one full-system local-inversion loss.")
    parser.add_argument("--n-qubits", type=int, default=10)
    parser.add_argument("--block", type=_parse_block, default=[4, 5, 6, 7])
    parser.add_argument("--target-bit", type=int, default=5)
    parser.add_argument("--time-k", type=int, default=5)
    parser.add_argument("--target-seed", type=int, default=default_seed)
    parser.add_argument("--theta-seed", type=int, default=default_seed + 2000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--skip-context-loss", action="store_true")
    return parser.parse_args()


def _time_call(label: str, fn: Callable[[], object]) -> tuple[object, float]:
    start = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - start
    print(f"{label}_seconds = {elapsed:.6f}")
    return value, elapsed


def _full_system_loss(
    U_full: ComplexArray,
    U_trial_C: ComplexArray,
    lightcone_qubits: list[int],
    block_qubits: list[int],
    n_qubits: int,
) -> tuple[float, dict[int, float], dict[str, float]]:
    full_qubits = list(range(n_qubits))
    timings: dict[str, float] = {}

    embedded, timings["embed_trial"] = _time_call(
        "embed_trial",
        lambda: embed_block_unitary_in_lightcone(U_trial_C, full_qubits, lightcone_qubits),
    )
    U_trial_full = np.asarray(embedded, dtype=complex)

    residual, timings["full_residual_matmul"] = _time_call(
        "full_residual_matmul",
        lambda: U_full @ U_trial_full.conj().T,
    )
    residual_full = np.asarray(residual, dtype=complex)

    start = time.perf_counter()
    per_bit = per_bit_losses_from_V(residual_full, block_qubits, full_qubits, target_bits=None)
    timings["full_per_bit_losses"] = time.perf_counter() - start
    print(f"full_per_bit_losses_seconds = {timings['full_per_bit_losses']:.6f}")
    return float(sum(per_bit.values())), per_bit, timings


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError(f"repeats must be positive, got {args.repeats}")
    if args.warmups < 0:
        raise ValueError(f"warmups must be non-negative, got {args.warmups}")

    n_qubits = int(args.n_qubits)
    block = [int(q) for q in args.block]
    t_value = 3.0 * np.pi / 40.0 * int(args.time_k) + 0.001
    rng_target = np.random.default_rng(int(args.target_seed))

    _, h_arr = build_h_diag(n_qubits, rng_target)
    target_circuit = build_u_target_circuit(n_qubits, rng_target, h_arr, t_value)
    lightcone = backward_block_lightcone_from_circuit(target_circuit, block)

    qubit_order = list(cirq.LineQubit.range(n_qubits))
    U_full_obj, build_full_seconds = _time_call(
        "build_full_unitary",
        lambda: target_circuit.unitary(qubit_order=qubit_order),
    )
    U_full = np.asarray(U_full_obj, dtype=complex)

    theta = random_theta(np.random.default_rng(int(args.theta_seed)), n_qubits=len(lightcone))
    U_trial_obj, build_ansatz_seconds = _time_call(
        "build_ansatz_unitary",
        lambda: ansatz_unitary(theta, n_qubits=len(lightcone)),
    )
    U_trial_C = np.asarray(U_trial_obj, dtype=complex)

    print("\nConfiguration")
    print(f"n_qubits = {n_qubits}")
    print(f"block_qubits = {tuple(block)}")
    print(f"target_bit = {args.target_bit}")
    print(f"lightcone_qubits = {tuple(lightcone)}")
    print(f"n_C = {len(lightcone)}")
    print(f"theta_size = {theta.size}")
    print(f"full_dim = {1 << n_qubits}")
    print(f"lightcone_dim = {1 << len(lightcone)}")
    print(f"U_full_bytes = {U_full.nbytes}")
    print(f"U_trial_C_bytes = {U_trial_C.nbytes}")

    if not args.skip_context_loss:
        context, _ = build_target_objective_context(
            n_qubits=n_qubits,
            block_qubits=block,
            target_bit=int(args.target_bit),
            radius=0,
            target_seed=int(args.target_seed),
            time_k=int(args.time_k),
            lightcone_mode="circuit",
            require_unitary=False,
            max_n_qubits=12,
            max_hilbert_dim=4096,
        )
        for _ in range(args.warmups):
            _ = target_bit_loss(theta, context)
        lightcone_times = []
        lightcone_loss = 0.0
        for _ in range(args.repeats):
            start = time.perf_counter()
            lightcone_loss = target_bit_loss(theta, context)
            lightcone_times.append(time.perf_counter() - start)
        print("\nContext loss")
        print(f"loss = {lightcone_loss:.12g}")
        print(f"avg_seconds = {sum(lightcone_times) / len(lightcone_times):.6f}")
        print(f"min_seconds = {min(lightcone_times):.6f}")
        print(f"max_seconds = {max(lightcone_times):.6f}")

    for _ in range(args.warmups):
        _ = _full_system_loss(U_full, U_trial_C, lightcone, block, n_qubits)

    full_times = []
    full_loss = 0.0
    full_per_bit: dict[int, float] = {}
    print("\nManual full-system residual loss")
    for repeat_idx in range(args.repeats):
        start = time.perf_counter()
        full_loss, full_per_bit, _ = _full_system_loss(U_full, U_trial_C, lightcone, block, n_qubits)
        elapsed = time.perf_counter() - start
        full_times.append(elapsed)
        print(f"repeat_{repeat_idx}_total_seconds = {elapsed:.6f}")

    print("\nSummary")
    print(f"build_full_unitary_seconds = {build_full_seconds:.6f}")
    print(f"build_ansatz_unitary_seconds = {build_ansatz_seconds:.6f}")
    print(f"full_system_loss = {full_loss:.12g}")
    print(f"full_system_per_bit = {full_per_bit}")
    print(f"full_system_avg_seconds = {sum(full_times) / len(full_times):.6f}")
    print(f"full_system_min_seconds = {min(full_times):.6f}")
    print(f"full_system_max_seconds = {max(full_times):.6f}")


if __name__ == "__main__":
    main()
