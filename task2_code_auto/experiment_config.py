"""Canonical experiment configurations for Task 2 multi-block training.

Every hard-coded constant that was duplicated across training scripts
now lives in one frozen dataclass.  Presets for common experiments are
provided as module-level constants.

Usage:
    from task2_code.experiment_config import N4_SINGLE_BLOCK, N12_3BLOCKS, N20_5BLOCKS

    config = N12_3BLOCKS
    # config.n_qubits, config.blocks, config.target_bits, ...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable descriptor for one multi-block training experiment."""

    # ── system ──
    n_qubits: int
    blocks: list[list[int]]
    target_bits: list[int]

    # ── target construction ──
    target_seed: int = 42
    time_k: int = 5
    lightcone_mode: str = "circuit"
    loss_mode: str = "lightcone"
    radius: int = 2

    # ── algorithm selection (registry keys) ──
    ansatz: str = "default_5layer_cz"
    block_only_ansatz: bool = False
    superoperator: str = "superoperator_from_mix"
    loss_function: str = "edge_quantum_channel"

    # ── training ──
    iterations: int = 150
    lr: float = 0.1
    training_seed_start: int = 1042
    max_restarts: int = 5
    success_threshold: float = 0.01

    # ── output ──
    output_dir: Path = Path("report/task2")
    data_dir: Path = Path("task2_code/data")

    # ── derived ──
    @property
    def block_count(self) -> int:
        return len(self.blocks)

    def training_seed_for_block(self, block_index: int) -> int:
        """Deterministic per-block training seed."""
        return self.training_seed_start + block_index * 100


# ════════════════════════════════════════════════════════════════════════
#  Presets
# ════════════════════════════════════════════════════════════════════════

N4_SINGLE_BLOCK = ExperimentConfig(
    n_qubits=4,
    blocks=[[0, 1, 2, 3]],
    target_bits=[1],
    target_seed=42,
    time_k=5,
    lightcone_mode="circuit",
    loss_mode="lightcone",
    radius=0,
    iterations=150,
    lr=0.1,
    training_seed_start=1042,
    max_restarts=1,
    success_threshold=0.01,
    ansatz="default_5layer_cz",
    superoperator="superoperator_from_mix",
    loss_function="edge_quantum_channel",
    output_dir=Path("report/task2"),
    data_dir=Path("task2_code/data"),
)


# ════════════════════════════════════════════════════════════════════════
#  Legacy module-level constants (from task2_code/config.py)
# ════════════════════════════════════════════════════════════════════════

DEFAULT_SEED = 42

# PhXZ gate parameters x, z, a (sampled uniformly from [0, 4]).
PHXZ_X_RANGE: tuple[float, float] = (0.0, 4.0)
PHXZ_Z_RANGE: tuple[float, float] = (0.0, 4.0)
PHXZ_A_RANGE: tuple[float, float] = (0.0, 4.0)

# Diagonal Hamiltonian H_diag = sum_j h_j Z_j (h_j sampled from [-1, 1]).
H_RANGE: tuple[float, float] = (-1.0, 1.0)

# Time evolution short-time subset.
T_SHORT_K_START: int = 0
T_SHORT_K_END: int = 10

# Unitarity verification tolerance.
VERIFY_TOLERANCE: float = 1e-10

N12_3BLOCKS = ExperimentConfig(
    n_qubits=12,
    blocks=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
    target_bits=[1, 5, 9],
    target_seed=42,
    time_k=5,
    lightcone_mode="circuit",
    loss_mode="lightcone",
    radius=2,
    iterations=150,
    lr=0.1,
    training_seed_start=1042,
    max_restarts=5,
    success_threshold=0.01,
    ansatz="default_5layer_cz",
    superoperator="superoperator_from_mix",
    output_dir=Path("report/task2"),
    data_dir=Path("task2_code/data"),
)

N12_3BLOCKS_CNOT_MIXED = ExperimentConfig(
    n_qubits=12,
    blocks=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
    target_bits=[1, 5, 9],
    target_seed=42,
    time_k=5,
    lightcone_mode="circuit",
    loss_mode="lightcone",
    radius=2,
    iterations=150,
    lr=0.1,
    training_seed_start=1042,
    max_restarts=5,
    success_threshold=0.01,
    ansatz="default_5layer_cnot",
    block_only_ansatz=False,
    superoperator="superoperator_from_mix",
    loss_function="heisenberg_pauli",
    output_dir=Path("report/task2"),
    data_dir=Path("task2_code/data"),
)

N32_8BLOCKS = ExperimentConfig(
    n_qubits=32,
    blocks=[
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23],
        [24, 25, 26, 27],
        [28, 29, 30, 31],
    ],
    target_bits=[1, 5, 9, 13, 17, 21, 25, 29],
    target_seed=42,
    time_k=5,
    lightcone_mode="circuit",
    loss_mode="lightcone",
    radius=2,
    iterations=150,
    lr=0.1,
    training_seed_start=1042,
    max_restarts=5,
    success_threshold=0.01,
    ansatz="default_5layer_cz",
    superoperator="superoperator_from_mix",
    output_dir=Path("report/task2"),
    data_dir=Path("task2_code/data"),
)

N32_8BLOCKS_HEISENBERG = ExperimentConfig(
    n_qubits=32,
    blocks=[
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23],
        [24, 25, 26, 27],
        [28, 29, 30, 31],
    ],
    target_bits=[1, 5, 9, 13, 17, 21, 25, 29],
    target_seed=42,
    time_k=5,
    lightcone_mode="circuit",
    loss_mode="lightcone",
    radius=2,
    iterations=150,
    lr=0.1,
    training_seed_start=1042,
    max_restarts=5,
    success_threshold=3.0,
    ansatz="default_5layer_cz",
    block_only_ansatz=False,
    superoperator="superoperator_from_mix",
    loss_function="heisenberg_pauli",
    output_dir=Path("report/task2"),
    data_dir=Path("task2_code/data"),
)

N12_3BLOCKS_ZERO = ExperimentConfig(
    n_qubits=12,
    blocks=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
    target_bits=[1, 5, 9],
    target_seed=43,
    time_k=5,
    lightcone_mode="circuit",
    loss_mode="lightcone",
    radius=2,
    iterations=150,
    lr=0.1,
    training_seed_start=1042,
    max_restarts=5,
    success_threshold=0.01,
    ansatz="default_5layer_cz",
    superoperator="superoperator_from_zero",
    output_dir=Path("report/task2"),
    data_dir=Path("task2_code/data"),
)

N12_3BLOCKS_ONE = ExperimentConfig(
    n_qubits=12,
    blocks=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
    target_bits=[1, 5, 9],
    target_seed=43,
    time_k=5,
    lightcone_mode="circuit",
    loss_mode="lightcone",
    radius=2,
    iterations=150,
    lr=0.1,
    training_seed_start=1042,
    max_restarts=5,
    success_threshold=0.01,
    ansatz="default_5layer_cz",
    superoperator="superoperator_from_one",
    output_dir=Path("report/task2"),
    data_dir=Path("task2_code/data"),
)


# ── Heisenberg Pauli presets ──────────────────────────────────────────
# Uses the operator-norm Heisenberg picture loss (sum over block qubits
# of ||W^† P_q W - P_q||^2 for P∈{X,Y,Z}).  Does NOT use superoperator
# sub-modes; the ``superoperator`` field below is a harmless placeholder.

N4_SINGLE_BLOCK_HEISENBERG = ExperimentConfig(
    n_qubits=4,
    blocks=[[0, 1, 2, 3]],
    target_bits=[1],
    target_seed=42,
    time_k=5,
    lightcone_mode="circuit",
    loss_mode="lightcone",
    radius=0,
    iterations=150,
    lr=0.1,
    training_seed_start=1042,
    max_restarts=5,
    success_threshold=1.0,
    ansatz="default_5layer_cz",
    superoperator="superoperator_from_mix",
    loss_function="heisenberg_pauli",
    output_dir=Path("report/task2"),
    data_dir=Path("task2_code/data"),
)

N12_3BLOCKS_HEISENBERG = ExperimentConfig(
    n_qubits=12,
    blocks=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
    target_bits=[1, 5, 9],
    target_seed=42,
    time_k=5,
    lightcone_mode="circuit",
    loss_mode="lightcone",
    radius=2,
    iterations=150,
    lr=0.1,
    training_seed_start=1042,
    max_restarts=5,
    success_threshold=3.0,
    ansatz="default_5layer_cz",
    superoperator="superoperator_from_mix",
    loss_function="heisenberg_pauli",
    output_dir=Path("report/task2"),
    data_dir=Path("task2_code/data"),
)


N20_5BLOCKS = ExperimentConfig(
    n_qubits=20,
    blocks=[
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
    ],
    target_bits=[1, 5, 9, 13, 17],
    target_seed=42,
    time_k=5,
    lightcone_mode="circuit",
    loss_mode="lightcone",
    radius=2,
    iterations=150,
    lr=0.1,
    training_seed_start=1042,
    max_restarts=5,
    success_threshold=0.01,
    ansatz="default_5layer_cz",
    superoperator="superoperator_from_mix",
    output_dir=Path("report/task2"),
    data_dir=Path("task2_code/data"),
)

N20_5BLOCKS_HEISENBERG = ExperimentConfig(
    n_qubits=20,
    blocks=[
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
    ],
    target_bits=[1, 5, 9, 13, 17],
    target_seed=42,
    time_k=5,
    lightcone_mode="circuit",
    loss_mode="lightcone",
    radius=2,
    iterations=150,
    lr=0.1,
    training_seed_start=1042,
    max_restarts=5,
    success_threshold=3.0,
    ansatz="default_5layer_cz",
    block_only_ansatz=False,
    superoperator="superoperator_from_mix",
    loss_function="heisenberg_pauli",
    output_dir=Path("report/task2"),
    data_dir=Path("task2_code/data"),
)

N24_6BLOCKS = ExperimentConfig(
    n_qubits=24,
    blocks=[
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23],
    ],
    target_bits=[1, 5, 9, 13, 17, 21],
    target_seed=42,
    time_k=5,
    lightcone_mode="circuit",
    loss_mode="lightcone",
    radius=2,
    iterations=150,
    lr=0.1,
    training_seed_start=1042,
    max_restarts=5,
    success_threshold=0.01,
    ansatz="default_5layer_cz",
    superoperator="superoperator_from_mix",
    output_dir=Path("report/task2"),
    data_dir=Path("task2_code/data"),
)

N24_6BLOCKS_HEISENBERG = ExperimentConfig(
    n_qubits=24,
    blocks=[
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23],
    ],
    target_bits=[1, 5, 9, 13, 17, 21],
    target_seed=42,
    time_k=5,
    lightcone_mode="circuit",
    loss_mode="lightcone",
    radius=2,
    iterations=150,
    lr=0.1,
    training_seed_start=1042,
    max_restarts=5,
    success_threshold=3.0,
    ansatz="default_5layer_cz",
    block_only_ansatz=False,
    superoperator="superoperator_from_mix",
    loss_function="heisenberg_pauli",
    output_dir=Path("report/task2"),
    data_dir=Path("task2_code/data"),
)

N28_7BLOCKS = ExperimentConfig(
    n_qubits=28,
    blocks=[
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23],
        [24, 25, 26, 27],
    ],
    target_bits=[1, 5, 9, 13, 17, 21, 25],
    target_seed=42,
    time_k=5,
    lightcone_mode="circuit",
    loss_mode="lightcone",
    radius=2,
    iterations=150,
    lr=0.1,
    training_seed_start=1042,
    max_restarts=5,
    success_threshold=0.01,
    ansatz="default_5layer_cz",
    superoperator="superoperator_from_mix",
    output_dir=Path("report/task2"),
    data_dir=Path("task2_code/data"),
)

N28_7BLOCKS_HEISENBERG = ExperimentConfig(
    n_qubits=28,
    blocks=[
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23],
        [24, 25, 26, 27],
    ],
    target_bits=[1, 5, 9, 13, 17, 21, 25],
    target_seed=42,
    time_k=5,
    lightcone_mode="circuit",
    loss_mode="lightcone",
    radius=2,
    iterations=150,
    lr=0.1,
    training_seed_start=1042,
    max_restarts=5,
    success_threshold=3.0,
    ansatz="default_5layer_cz",
    block_only_ansatz=False,
    superoperator="superoperator_from_mix",
    loss_function="heisenberg_pauli",
    output_dir=Path("report/task2"),
    data_dir=Path("task2_code/data"),
)
