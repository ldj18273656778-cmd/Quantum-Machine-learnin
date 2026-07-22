"""Per-bit local-inversion loss test for n=12, block {5,6,7,8}.

Uses a direct per-qubit reduced-channel computation that avoids building
the full 4^s x 4^s superoperator, making larger light cones tractable.

Run from the repository root:

    python task2_code/test_code/test_per_bit_loss.py
"""

import math
import os
import sys
from pathlib import Path

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from task2_code.ansatz import ansatz_unitary, random_theta
from task2_code.experiment_config import DEFAULT_SEED as seed
from task2_code.lightcone import extract_target_lightcone_operator
from task2_code.local_loss import embed_block_unitary_in_lightcone
from task2_code.superoperator import per_bit_losses_from_V
from task2_code.U_target import build_h_diag, u_target_unitary


def main():
    n_qubits = 12
    block_qubits = [5, 6, 7, 8]
    radius = 2

    print(f"n={n_qubits}  block={block_qubits}  radius={radius}")

    rng = np.random.default_rng(seed)

    # U_target(t)
    _, h_arr = build_h_diag(n_qubits, rng)
    t = 3 * math.pi / 40 * 5 + 0.001
    U_full = u_target_unitary(n_qubits, rng, h_arr, t)
    print(f"\nU_target(t={t:.4f})  {U_full.shape}")

    # Light-cone extraction
    result = extract_target_lightcone_operator(
        U_full,
        block_qubits=block_qubits,
        n_qubits=n_qubits,
        radius=radius,
        require_unitary=False,
        max_n_qubits=12,
        max_hilbert_dim=4096,
    )
    cone = list(result.lightcone_qubits)
    print(f"light cone: {cone}  (|S_j|={len(cone)})")
    print(f"block positions in cone: {list(result.block_positions)}")
    print(f"semantics: {result.semantics}")
    diag = result.diagnostics
    print(f"unitarity: left={diag.left_unitarity_error:.3e}  "
          f"right={diag.right_unitarity_error:.3e}  "
          f"leakage={diag.max_column_leakage:.3e}")

    # Random U_trial
    theta = random_theta(rng)
    U_trial = ansatz_unitary(theta)
    print(theta.shape)
    print(f"\ntheta[:6] = {theta[:6].round(4)} ...")

    # Embed and compute residual
    trial_tilde = embed_block_unitary_in_lightcone(U_trial, cone, block_qubits)
    V = result.operator @ trial_tilde.conj().T

    # Per-bit loss
    bit_losses = per_bit_losses_from_V(V, block_qubits, cone)

    total = sum(bit_losses.values())
    print(f"\nTotal loss (Sigma Frobenius): {total:.6f}\n")
    for q in sorted(bit_losses):
        print(f"  qubit {q}:  {bit_losses[q]:.6f}")


if __name__ == "__main__":
    main()
