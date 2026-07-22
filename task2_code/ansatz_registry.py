"""Ansatz builder registry.

Maps stable string names to ansatz construction functions so
experiment configs can switch circuit architectures without
editing training / sewing code.

Each registered callable has the same signature as ``build_ansatz``:

    (theta, qubits, n_qubits) -> cirq.Circuit

Usage:
    from task2_code.ansatz_registry import ANSATZ_REGISTRY, resolve_ansatz

    builder = resolve_ansatz("default_5layer_cz")
    circuit = builder(theta, qubits=q, n_qubits=n)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import cirq
import numpy as np
from numpy.typing import NDArray

from task2_code.ansatz import (
    build_ansatz,
    build_ansatz_cnot,
    build_scoped_ansatz_circuit,
    scoped_ansatz_unitary_on_lightcone,
    theta_count,
)

DEFAULT_ANSATZ = "default_5layer_cz"

# ── public type ───────────────────────────────────────────────────────
AnsatzBuilder = Callable[..., cirq.Circuit]
ThetaCountFn = Callable[[int], int]


@dataclass(frozen=True)
class AnsatzSpec:
    builder: AnsatzBuilder
    theta_count_fn: ThetaCountFn = theta_count

# ── registry ──────────────────────────────────────────────────────────
ANSATZ_SPECS: dict[str, AnsatzSpec] = {
    DEFAULT_ANSATZ: AnsatzSpec(build_ansatz),
    "default_5layer_cnot": AnsatzSpec(build_ansatz_cnot),
}

# Backward-compatible builder-only view used by older scripts.
ANSATZ_REGISTRY: dict[str, AnsatzBuilder] = {name: spec.builder for name, spec in ANSATZ_SPECS.items()}


def resolve_ansatz(name: str) -> AnsatzBuilder:
    """Look up an ansatz builder by registry key.

    Raises ``KeyError`` with available key names for unrecognised inputs.
    """
    try:
        return resolve_ansatz_spec(name).builder
    except KeyError:
        raise KeyError(
            f"unknown ansatz '{name}'.  Available: {sorted(set(ANSATZ_SPECS) | set(ANSATZ_REGISTRY))}"
        ) from None


def resolve_ansatz_spec(name: str) -> AnsatzSpec:
    """Look up the full ansatz spec, including parameter-count semantics."""
    try:
        return ANSATZ_SPECS[name]
    except KeyError:
        if name in ANSATZ_REGISTRY:
            return AnsatzSpec(ANSATZ_REGISTRY[name], theta_count)
        raise KeyError(
            f"unknown ansatz '{name}'.  Available: {sorted(set(ANSATZ_SPECS) | set(ANSATZ_REGISTRY))}"
        ) from None


def register_ansatz(
    name: str,
    builder: AnsatzBuilder,
    theta_count_fn: ThetaCountFn = theta_count,
) -> None:
    """Register a new ansatz builder under *name* (overwrites existing)."""
    if not callable(builder):
        raise TypeError(f"ansatz builder must be callable, got {type(builder).__name__}")
    if not callable(theta_count_fn):
        raise TypeError(f"theta_count_fn must be callable, got {type(theta_count_fn).__name__}")
    ANSATZ_SPECS[name] = AnsatzSpec(builder, theta_count_fn)
    ANSATZ_REGISTRY[name] = builder


def ansatz_theta_count(name: str, n_qubits: int) -> int:
    """Return trainable parameter count for an ansatz registry key."""
    return int(resolve_ansatz_spec(name).theta_count_fn(int(n_qubits)))


def random_ansatz_theta(
    name: str,
    rng: np.random.Generator | None = None,
    low: float = 0.0,
    high: float = 2.0 * np.pi,
    *,
    n_qubits: int,
) -> NDArray[np.float64]:
    """Return random parameters sized by the selected ansatz spec."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(low, high, size=ansatz_theta_count(name, n_qubits))


def build_registered_scoped_ansatz_circuit(
    name: str,
    theta: object,
    lightcone_qubits: Sequence[int],
    block_qubits: Sequence[int],
    *,
    block_only_ansatz: bool = False,
) -> cirq.Circuit:
    """Build a registered ansatz on the selected lightcone/block scope."""
    return build_scoped_ansatz_circuit(
        theta,
        lightcone_qubits,
        block_qubits,
        block_only_ansatz=block_only_ansatz,
        builder=resolve_ansatz(name),
    )


def registered_ansatz_unitary_on_lightcone(
    name: str,
    theta: object,
    lightcone_qubits: Sequence[int],
    block_qubits: Sequence[int],
    *,
    block_only_ansatz: bool = False,
) -> NDArray[np.complex128]:
    """Return a registered trial ansatz as a lightcone-ordered unitary."""
    return scoped_ansatz_unitary_on_lightcone(
        theta,
        lightcone_qubits,
        block_qubits,
        block_only_ansatz=block_only_ansatz,
        builder=resolve_ansatz(name),
    )
