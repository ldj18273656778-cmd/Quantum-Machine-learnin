"""Shared U_target construction helpers for Task 2.

The training and comparison pipelines both use this module so that target
reconstruction consumes random numbers in exactly one place: sample h_arr first,
then build V_scr with the same RNG inside build_u_target_circuit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping
from typing import Any

import cirq
import numpy as np
from numpy.typing import NDArray

from task2_code_auto.U_target import build_u_target_circuit


@dataclass(frozen=True)
class TargetSpec:
    n_qubits: int
    target_seed: int
    time_k: int
    time: float
    h_arr: NDArray[np.float64]
    circuit: cirq.Circuit


def target_time(time_k: int = 5, time: float | None = None) -> float:
    return float(time if time is not None else 3.0 * math.pi / 40.0 * int(time_k) + 0.001)


def build_target_from_seed(
    n_qubits: int,
    target_seed: int = 42,
    time_k: int = 5,
    time: float | None = None,
) -> TargetSpec:
    n_val = int(n_qubits)
    seed_val = int(target_seed)
    time_k_val = int(time_k)
    time_val = target_time(time_k_val, time)
    rng = np.random.default_rng(seed_val)
    h_arr = rng.uniform(-1.0, 1.0, size=n_val)
    circuit = build_u_target_circuit(n_val, rng, h_arr, time_val)
    return TargetSpec(n_val, seed_val, time_k_val, time_val, np.asarray(h_arr, dtype=float), circuit)


def build_target_from_metadata(metadata: Mapping[str, Any]) -> TargetSpec:
    n_qubits = int(metadata["n_qubits"])
    target_seed = int(metadata.get("target_seed", 42))
    time_k = int(metadata.get("time_k", 5))
    time = metadata.get("time")
    spec = build_target_from_seed(n_qubits, target_seed, time_k, None if time is None else float(time))
    if "h_arr" in metadata:
        stored = np.asarray(metadata["h_arr"], dtype=float)
        if stored.shape != spec.h_arr.shape or not np.allclose(stored, spec.h_arr):
            raise ValueError("metadata h_arr does not match target_seed/time reconstruction")
    return spec


def target_metadata(spec: TargetSpec) -> dict[str, Any]:
    return {
        "n_qubits": spec.n_qubits,
        "target_seed": spec.target_seed,
        "time_k": spec.time_k,
        "time": spec.time,
        "h_arr": spec.h_arr.tolist(),
    }
