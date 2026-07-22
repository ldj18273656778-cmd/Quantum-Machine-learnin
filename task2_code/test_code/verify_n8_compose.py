"""Verify if two independently trained 4-qubit blocks can compose to a global inverse on n=8.

Tests: train block [0,1,2,3] and block [4,5,6,7] independently on n=8,
then compose the two trained ansatzes and compare to U_target.
"""

from __future__ import annotations
import os, sys
from pathlib import Path
os.environ["GRPC_VERBOSITY"] = "ERROR"; os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from task2_code.module_e_training import (
    AdamConfig, adam_optimize, build_target_objective_context, sum_block_loss,
)
from task2_code.superoperator import per_bit_losses_from_V
from task2_code.local_loss import embed_block_unitary_in_lightcone
from task2_code.ansatz import ansatz_unitary, random_theta
from task2_code.U_target import build_h_diag, u_target_unitary

# ── config ──────────────────────────────────────────
N_QUBITS = 8
BLOCKS = [[0, 1, 2, 3], [4, 5, 6, 7]]
TARGET_BITS = [1, 5]
RADIUS = 0       # closed light cones, no overlap
TIME_K = 5
TARGET_SEED = 42
ITERATIONS = 150
LR = 0.1

# ═══════════════════════════════════════════════════════
# 1.  Build global U_target
# ═══════════════════════════════════════════════════════
rng = np.random.default_rng(TARGET_SEED)
_, h_arr = build_h_diag(N_QUBITS, rng)
t_val = 3.0 * np.pi / 40.0 * TIME_K + 0.001
U_global = u_target_unitary(N_QUBITS, rng, h_arr, t_val)
print(f"U_global shape: {U_global.shape}")

# ═══════════════════════════════════════════════════════
# 2.  Train each block independently
# ═══════════════════════════════════════════════════════
trained_thetas = []
trained_lightcones = []

cfg = AdamConfig(iterations=ITERATIONS, lr=LR)

for bi, (block, tbit) in enumerate(zip(BLOCKS, TARGET_BITS)):
    print(f"\n{'='*50}\nBlock {bi+1}: qubits {block}  target_bit {tbit}\n{'='*50}")
    context, meta = build_target_objective_context(
        N_QUBITS, block, tbit, RADIUS, TARGET_SEED, TIME_K,
        require_unitary=False, max_n_qubits=N_QUBITS, max_hilbert_dim=1 << N_QUBITS,
    )
    cone = list(context.lightcone_qubits)
    print(f"  lightcone: {cone}  |cone|={len(cone)}")
    trained_lightcones.append(cone)

    rng_train = np.random.default_rng(1042 + bi * 100)
    theta0 = np.asarray(random_theta(rng_train, low=0.0, high=2 * np.pi), dtype=float)

    class Rec:
        losses = []
    def loss_fn(t):
        l = sum_block_loss(t, context)
        Rec.losses.append(float(l))
        return l

    init = loss_fn(theta0)
    print(f"  init sum-block loss: {init:.6f}")
    result = adam_optimize(loss_fn, theta0, cfg, show_progress=True)
    print(f"  best sum-block loss: {result.best_loss:.6f} (iter {result.best_iteration})")

    trained_thetas.append(result.best_params)

    # per-qubit at best
    trial = ansatz_unitary(result.best_params)
    emb = embed_block_unitary_in_lightcone(trial, context.lightcone_qubits, context.block_qubits)
    residual = context.target_operator @ emb.conj().T
    pb = per_bit_losses_from_V(residual, context.block_qubits, context.lightcone_qubits, target_bits=None)
    for q, v in pb.items():
        print(f"    qubit {q}: {v:.6e}")

# ═══════════════════════════════════════════════════════
# 3.  Compose the two trained ansatzes in the full 8-qubit space
# ═══════════════════════════════════════════════════════
print(f"\n{'='*50}\nComposing trained ansatzes\n{'='*50}")

# Build full 8-qubit U_trial_total = U_trial_2 ⊗ I_4 · U_trial_1
# U_trial_1 acts on [0,1,2,3], identity on [4,5,6,7]
# U_trial_2 acts on [4,5,6,7], identity on [0,1,2,3]

FULL_DIM = 1 << N_QUBITS

U1_4q = ansatz_unitary(trained_thetas[0])
U2_4q = ansatz_unitary(trained_thetas[1])

# embed U1 on block [0,1,2,3] into 8-qubit space (global qubit order 0..7)
U1_full = np.eye(FULL_DIM, dtype=complex)
block_bits_1 = 4  # qubits 0-3
env_bits_1 = N_QUBITS - block_bits_1
U1_block = U1_4q
I_env = np.eye(1 << env_bits_1, dtype=complex)
# block on low qubits, env on high qubits (right-to-left tensor order)
U1_full = np.kron(I_env, U1_block)  # env ⊗ block → qubits [4-7] ⊗ [0-3]

# embed U2 on block [4,5,6,7] into 8-qubit space
U2_block = U2_4q
I_env2 = np.eye(1 << block_bits_1, dtype=complex)
U2_full = np.kron(U2_block, I_env2)  # block ⊗ env → qubits [4-7] ⊗ [0-3]

# Compose: first U1 then U2 (right-to-left)
U_trial_total = U2_full @ U1_full

# ═══════════════════════════════════════════════════════
# 4.  Compare with U_global
# ═══════════════════════════════════════════════════════

# Frobenius distance
diff = U_global - U_trial_total
frob_dist = np.linalg.norm(diff, ord="fro")
frob_norm_U = np.linalg.norm(U_global, ord="fro")
rel_error = frob_dist / frob_norm_U

# Phase-align: U_target† @ U_trial_total → should be ~eiφ I
overlap = np.trace(U_global.conj().T @ U_trial_total) / FULL_DIM
phase = overlap / abs(overlap) if abs(overlap) > 1e-12 else 1.0
phased_diff = phase * U_global - U_trial_total
phased_frob = np.linalg.norm(phased_diff, ord="fro")
phased_rel = phased_frob / frob_norm_U

print(f"\nGlobal comparison (n={N_QUBITS}):")
print(f"  ||U_target||_F          = {frob_norm_U:.6f}")
print(f"  ||U_target - U_trial||_F = {frob_dist:.6f}")
print(f"  relative error           = {rel_error:.6f}")
print(f"  ||U_target·eiφ - U_trial||_F = {phased_frob:.6f}")
print(f"  phase-aligned rel error  = {phased_rel:.6f}")

# Per-bit losses on the composed global residual
# First check the full-global reduced channels
residual_global = U_global @ U_trial_total.conj().T
all_qubits = list(range(N_QUBITS))
pb_all = per_bit_losses_from_V(residual_global, all_qubits, all_qubits, target_bits=None)
print(f"\nPer-bit losses of composed global residual:")
for q in range(N_QUBITS):
    status = "✓" if pb_all[q] < 0.01 else "✗"
    print(f"  qubit {q}: {pb_all[q]:.6e}  {status}")
print(f"  sum  = {sum(pb_all.values()):.6f}")
print(f"  max  = {max(pb_all.values()):.6f}  (S.2.4 threshold δ=0.01)")
