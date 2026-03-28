"""Bitstring generation for arXiv:2509.09033 (IDQNN mapping).

This module implements:
1) A shallow 2D IDQNN sampler (Algorithm 1 in Appendix C.3.a).
2) A mapped (1+1)D deep sampler (Algorithm 2 in Appendix C.3.a).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cirq
import numpy as np


@dataclass(frozen=True)
class IDQNNConfig:
    n1: int
    m: int
    spatial_edges: tuple[tuple[int, int], ...] # Tuple of spatial edges in the graph, where each edge is a tuple (a, b) with 0 <= a < b < m.
    include_temporal_edges: bool = True# Whether to include temporal edges between layers in the shallow model. The deep model always includes temporal edges.

    @property
    def n(self) -> int:# Total number of qubits = n1 * m.
        return self.n1 * self.m# Total number of spatial edges = n1 * len(spatial_edges).


def _default_spatial_edges(m: int) -> tuple[tuple[int, int], ...]:# Default spatial edges for a 1D chain of m qubits: (0, 1), (1, 2), ..., (m-2, m-1).
    return tuple((i, i + 1) for i in range(m - 1))


def make_config(
    n1: int,
    m: int,
    spatial_edges: Iterable[tuple[int, int]] | None = None,
    include_temporal_edges: bool = True,
) -> IDQNNConfig:
    if n1 < 2:
        raise ValueError("n1 must be >= 2.")
    if m < 1:
        raise ValueError("m must be >= 1.")
    if spatial_edges is None:
        edges = _default_spatial_edges(m)
    else:
        edges = tuple(tuple(sorted(e)) for e in spatial_edges)
        for a, b in edges:
            if a < 0 or b < 0 or a >= m or b >= m or a == b:
                raise ValueError(f"Invalid spatial edge: {(a, b)} for m={m}.")
    return IDQNNConfig(n1=n1, m=m, spatial_edges=edges, include_temporal_edges=include_temporal_edges)


def _validate_inputs(bitstring_x: str, theta: np.ndarray, cfg: IDQNNConfig) -> None:
    if len(bitstring_x) != cfg.n:
        raise ValueError(f"len(bitstring_x) must be {cfg.n}, got {len(bitstring_x)}.")
    if any(ch not in ("0", "1") for ch in bitstring_x):
        raise ValueError("bitstring_x must contain only '0' and '1'.")
    if theta.shape != (cfg.n1, cfg.m):
        raise ValueError(f"theta must have shape ({cfg.n1}, {cfg.m}), got {theta.shape}.")


def _bit_at(bitstring_x: str, t: int, q: int, m: int) -> str:
    return bitstring_x[t * m + q]


def _samples_to_strings(samples: np.ndarray) -> list[str]:
    return ["".join(str(int(b)) for b in row.tolist()) for row in samples]


def _extract_meas_bit(arr: np.ndarray) -> int:
    a = np.asarray(arr)
    if a.ndim == 0:
        return int(a)
    if a.ndim == 1:
        return int(a[0])
    return int(a[0, 0])


def build_shallow_circuit(bitstring_x: str, theta: np.ndarray, cfg: IDQNNConfig, measure_x: bool) -> tuple[cirq.Circuit, list[cirq.GridQubit]]:
    _validate_inputs(bitstring_x, theta, cfg)
    qubits = [cirq.GridQubit(t, q) for t in range(cfg.n1) for q in range(cfg.m)]
    qb = {(t, q): cirq.GridQubit(t, q) for t in range(cfg.n1) for q in range(cfg.m)}
    circuit = cirq.Circuit()

    # Step 2 in Algorithm 1: apply H^(1-x) on all qubits.
    for t in range(cfg.n1):
        for q in range(cfg.m):
            if _bit_at(bitstring_x, t, q, cfg.m) == "0":
                circuit.append(cirq.H(qb[(t, q)]))

    # Step 3 in Algorithm 1: apply Rz(theta_t,q).
    for t in range(cfg.n1):
        for q in range(cfg.m):
            circuit.append(cirq.rz(float(theta[t, q]))(qb[(t, q)]))

    # Step 4 in Algorithm 1: CZ on neighbors in graph G.
    for t in range(cfg.n1):
        for a, b in cfg.spatial_edges:
            circuit.append(cirq.CZ(qb[(t, a)], qb[(t, b)]))

    if cfg.include_temporal_edges:
        for t in range(cfg.n1 - 1):
            for q in range(cfg.m):
                circuit.append(cirq.CZ(qb[(t, q)], qb[(t + 1, q)]))

    if measure_x:
        for t in range(cfg.n1):
            for q in range(cfg.m):
                qubit = qb[(t, q)]
                circuit.append(cirq.H(qubit))
                circuit.append(cirq.measure(qubit, key=f"y_{t}_{q}"))

    return circuit, qubits


def sample_shallow_idqnn(
    bitstring_x: str,
    theta: np.ndarray,
    cfg: IDQNNConfig,
    shots: int,
    seed: int = 0,
) -> np.ndarray:
    circuit, _ = build_shallow_circuit(bitstring_x=bitstring_x, theta=theta, cfg=cfg, measure_x=True)
    sim = cirq.Simulator(seed=seed)
    result = sim.run(circuit, repetitions=shots)
    samples = np.zeros((shots, cfg.n), dtype=np.int8)
    col = 0
    for t in range(cfg.n1):
        for q in range(cfg.m):
            key = f"y_{t}_{q}"
            samples[:, col] = result.measurements[key][:, 0].astype(np.int8)
            col += 1
    return samples


def sample_deep_mapped_idqnn(
    bitstring_x: str,
    theta: np.ndarray,
    cfg: IDQNNConfig,
    shots: int,
    seed: int = 0,
) -> np.ndarray:
    """Implements Algorithm 2 in Appendix C.3.a."""
    _validate_inputs(bitstring_x, theta, cfg)
    rng = np.random.default_rng(seed)
    qubits = cirq.LineQubit.range(cfg.m)
    all_samples = np.zeros((shots, cfg.n), dtype=np.int8)

    for shot in range(shots):
        y = np.zeros((cfg.n1, cfg.m), dtype=np.int8)
        sim = cirq.Simulator(seed=int(rng.integers(0, 2**31 - 1)))
        state: np.ndarray | int = 0

        # Prepare the first slice from x.
        init_ops = []
        for q in range(cfg.m):
            if _bit_at(bitstring_x, 0, q, cfg.m) == "0":
                init_ops.append(cirq.H(qubits[q]))
        if init_ops:
            state = sim.simulate(cirq.Circuit(init_ops), initial_state=state, qubit_order=qubits).final_state_vector

        def apply_layer(t: int, cur_state: np.ndarray | int) -> np.ndarray:
            ops = []
            for qq in range(cfg.m):
                ops.append(cirq.rz(float(theta[t, qq]))(qubits[qq]))
            for a, b in cfg.spatial_edges:
                ops.append(cirq.CZ(qubits[a], qubits[b]))
            return sim.simulate(cirq.Circuit(ops), initial_state=cur_state, qubit_order=qubits).final_state_vector

        # Apply layer 1 in the deep model.
        state = apply_layer(0, state)

        # Transition from layer t to layer t+1, generating y_t.
        for t in range(cfg.n1 - 1):
            for q in range(cfg.m):
                x_next = _bit_at(bitstring_x, t + 1, q, cfg.m)
                if x_next == "0":
                    state = sim.simulate(cirq.Circuit(cirq.H(qubits[q])), initial_state=state, qubit_order=qubits).final_state_vector
                    y[t, q] = int(rng.integers(0, 2))
                    if y[t, q] == 1:
                        state = sim.simulate(cirq.Circuit(cirq.Z(qubits[q])), initial_state=state, qubit_order=qubits).final_state_vector
                else:
                    meas = sim.simulate(
                        cirq.Circuit(cirq.measure(qubits[q], key="mz")),
                        initial_state=state,
                        qubit_order=qubits,
                    )
                    y[t, q] = _extract_meas_bit(meas.measurements["mz"])
                    state = meas.final_state_vector

            # Apply next circuit layer after the teleport/measure transition.
            state = apply_layer(t + 1, state)

        # Final X-basis measurement gives y_{T,q}.
        for q in range(cfg.m):
            meas_x = sim.simulate(
                cirq.Circuit(cirq.H(qubits[q]), cirq.measure(qubits[q], key="mx")),
                initial_state=state,
                qubit_order=qubits,
            )
            y[cfg.n1 - 1, q] = _extract_meas_bit(meas_x.measurements["mx"])
            state = meas_x.final_state_vector

        all_samples[shot, :] = y.reshape(-1)

    return all_samples


def exact_probs_shallow(bitstring_x: str, theta: np.ndarray, cfg: IDQNNConfig) -> np.ndarray:
    """Exact probability vector over n-bit outputs for the shallow model."""
    circuit, qubits = build_shallow_circuit(bitstring_x=bitstring_x, theta=theta, cfg=cfg, measure_x=False)
    # X-basis measurement equals H + Z-basis measurement.
    circuit_x = cirq.Circuit(circuit)
    circuit_x.append(cirq.H(q) for q in qubits)
    sim = cirq.Simulator()
    psi = sim.simulate(circuit_x, qubit_order=qubits).final_state_vector
    probs = np.abs(psi) ** 2
    return probs / probs.sum()


def linear_xeb(samples: np.ndarray, probs: np.ndarray) -> float:
    n = samples.shape[1]
    weights = 1 << np.arange(n - 1, -1, -1, dtype=np.int64)
    ids = (samples.astype(np.int64) * weights).sum(axis=1)
    return float(np.mean((2**n) * probs[ids] - 1.0))


def empirical_distribution(samples: np.ndarray) -> dict[str, float]:
    strings = _samples_to_strings(samples)
    counts: dict[str, int] = {}
    for s in strings:
        counts[s] = counts.get(s, 0) + 1
    total = len(strings)
    return {k: v / total for k, v in counts.items()}


def tv_distance(dist_a: dict[str, float], dist_b: dict[str, float]) -> float:
    keys = set(dist_a) | set(dist_b)
    return 0.5 * sum(abs(dist_a.get(k, 0.0) - dist_b.get(k, 0.0)) for k in keys)
