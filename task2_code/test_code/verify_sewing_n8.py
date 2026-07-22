"""Verify sewing on n=8: train 2 blocks independently, sew them, compare to U_target.

Uses state-vector approach to avoid 65536x65536 matrix multiplication.
"""

from __future__ import annotations
import os, sys
from pathlib import Path
os.environ["GRPC_VERBOSITY"] = "ERROR"; os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from task2_code.module_e_training import AdamConfig, adam_optimize, build_target_objective_context, sum_block_loss
from task2_code.superoperator import per_bit_losses_from_V
from task2_code.local_loss import embed_block_unitary_in_lightcone
from task2_code.ansatz import ansatz_unitary, random_theta
from task2_code.U_target import build_h_diag, u_target_unitary

# ── config ──
N = 8
BLOCKS = [[0,1,2,3], [4,5,6,7]]
TARGET_BITS = [1, 5]
RADIUS = 0
TIME_K = 5
TARGET_SEED = 42
ITERATIONS = 150
LR = 0.1

d = 1 << N           # 256
NN = 2 * N           # 16 qubits
dim = 1 << NN        # 65536

# ═══════════════════════════════════════════════════════════
# 1. Build global U_target
# ═══════════════════════════════════════════════════════════
rng = np.random.default_rng(TARGET_SEED)
_, h_arr = build_h_diag(N, rng)
t_val = 3.0 * np.pi / 40.0 * TIME_K + 0.001
U_global = u_target_unitary(N, rng, h_arr, t_val)
print(f"U_global shape: {U_global.shape}")

# ═══════════════════════════════════════════════════════════
# 2. Train two blocks independently
# ═══════════════════════════════════════════════════════════
cfg = AdamConfig(iterations=ITERATIONS, lr=LR)
trained_thetas = []

for bi, (block, tbit) in enumerate(zip(BLOCKS, TARGET_BITS)):
    print(f"\n--- Block {bi+1}: qubits {block} ---")
    context, _ = build_target_objective_context(
        N, block, tbit, RADIUS, TARGET_SEED, TIME_K,
        require_unitary=False, max_n_qubits=N, max_hilbert_dim=1 << N,
    )
    train_rng = np.random.default_rng(1042 + bi * 100)
    theta0 = np.asarray(random_theta(train_rng), dtype=float)
    class R:
        losses = []
    def loss_fn(t):
        l = sum_block_loss(t, context)
        R.losses.append(float(l))
        return l

    init = loss_fn(theta0)
    print(f"  init loss: {init:.4f}")
    result = adam_optimize(loss_fn, theta0, cfg, show_progress=True)
    print(f"  best loss: {result.best_loss:.4f}  (iter {result.best_iteration})")

    # per-qubit at best
    trial = ansatz_unitary(result.best_params)
    emb = embed_block_unitary_in_lightcone(trial, context.lightcone_qubits, context.block_qubits)
    residual = context.target_operator @ emb.conj().T
    pb = per_bit_losses_from_V(residual, context.block_qubits, context.lightcone_qubits, target_bits=None)
    for q, v in pb.items():
        print(f"    q{q}: {v:.4e}")
    trained_thetas.append(result.best_params)

# ═══════════════════════════════════════════════════════════
# 3. Build V_full for each block (embed 4q ansatz into n-qubit system)
# ═══════════════════════════════════════════════════════════
def embed_block_into_n_qubits(U_block_4q, block_qubits, n_qubits):
    """Embed 4-qubit unitary into n-qubit space (identity on other qubits).
    
    Qubit order: q0⊗q1⊗...⊗q_{n-1} (standard big-endian).
    block_qubits define which positions get the 4-qubit unitary.
    """
    U = U_block_4q
    d_n = 1 << n_qubits
    result = np.zeros((d_n, d_n), dtype=complex)
    block_bits = 4
    env_bits = n_qubits - block_bits
    
    # Map: for each (row_env, row_block) and (col_env, col_block)
    # result = kron(I_env, U_block) IF block qubits are in the low positions
    # But block qubits may not be contiguous and at position 0.
    # For our case: block [0,1,2,3] means they ARE the low 4 qubits.
    # block [4,5,6,7] means they ARE the high 4 qubits.
    
    if block_qubits == [0, 1, 2, 3]:
        # block in low qubits: result = kron(I_env, U_block)
        I_env = np.eye(1 << env_bits, dtype=complex)
        result = np.kron(I_env, U)
    elif block_qubits == [4, 5, 6, 7]:
        # block in high qubits: result = kron(U_block, I_env)
        I_env = np.eye(1 << env_bits, dtype=complex)
        result = np.kron(U, I_env)
    else:
        # Generic case — build explicitly via basis states
        for env_row in range(1 << env_bits):
            env_row_bits = [(env_row >> i) & 1 for i in range(env_bits)]
            for block_row in range(block_bits):
                for block_col in range(block_bits):
                    val = U[block_row, block_col]
                    if abs(val) < 1e-15:
                        continue
                    for env_col in range(1 << env_bits):
                        # Build full basis indices
                        pass  # skip generic case for now
    
    return result

U1_4q = ansatz_unitary(trained_thetas[0])
U2_4q = ansatz_unitary(trained_thetas[1])

V0_full = embed_block_into_n_qubits(U1_4q, BLOCKS[0], N)  # acts on [0,1,2,3]
V1_full = embed_block_into_n_qubits(U2_4q, BLOCKS[1], N)  # acts on [4,5,6,7]

print(f"\nV0_full shape: {V0_full.shape}, unitary: {np.allclose(V0_full @ V0_full.conj().T, np.eye(d))}")
print(f"V1_full shape: {V1_full.shape}, unitary: {np.allclose(V1_full @ V1_full.conj().T, np.eye(d))}")

# For sewing, each qubit i=0..3 in block 1 uses V0_full as its local inversion.
# Each qubit i=4..7 in block 2 uses V1_full as its local inversion.
V_list = [V0_full] * 4 + [V1_full] * 4

# ═══════════════════════════════════════════════════════════
# 4. Apply U_sew to basis states (state-vector approach)
# ═══════════════════════════════════════════════════════════
print("\nApplying sewing (state-vector method)...")

# Precompute swap permutations for S_i and S_full
def swap_perm(i, n):
    """Permutation vector for S_i: swap system qubit i with ancilla qubit i."""
    NN = 2 * n
    perm = np.arange(1 << NN, dtype=np.int64)
    for idx in range(1 << NN):
        bits = [(idx >> b) & 1 for b in range(NN)]
        bits[i], bits[i + n] = bits[i + n], bits[i]
        new_idx = sum(b << p for p, b in enumerate(bits))
        perm[idx] = new_idx
    return perm

def full_swap_perm(n):
    """Permutation for S_full: swap all system with all ancilla."""
    NN = 2 * n
    perm = np.arange(1 << NN, dtype=np.int64)
    for idx in range(1 << NN):
        bits = [(idx >> b) & 1 for b in range(NN)]
        for i in range(n):
            bits[i], bits[i + n] = bits[i + n], bits[i]
        new_idx = sum(b << p for p, b in enumerate(bits))
        perm[idx] = new_idx
    return perm

S_i_perm = [swap_perm(i, N) for i in range(N)]
S_full_perm = full_swap_perm(N)

def apply_V_tens(state, V_full, n):
    """Apply kron(V_full, I_n) to a 2n-qubit state vector.
    
    V_full acts on the first n qubits (system), identity on last n (ancilla).
    kron(V, I_n): for each ancilla basis state, V acts on system.
    """
    d = 1 << n
    result = np.zeros_like(state)
    for anc in range(d):
        # Extract system block for this ancilla value
        sys_slice = state[anc::d]  # stride by d for same anc state
        # Apply V to system block
        result_sys = V_full @ sys_slice
        # Scatter back
        result[anc::d] = result_sys
    return result

def apply_S_i(state, perm):
    """Apply swap permutation to state vector."""
    return state[perm]

def apply_U_sew(state, V_list, n):
    """Apply the full sewn unitary to a 2n-qubit state.
    
    U_sew = S_full · term_{n-1} · ... · term_0
    term_i = (V_i ⊗ I) · S_i · (V_i† ⊗ I)
    """
    d = 1 << n
    # Apply terms right to left (term_0 first in time)
    for i in range(n):
        V = V_list[i]
        Vdag = V.conj().T
        # term_i = V_tens @ S_i @ Vdag_tens
        state = apply_V_tens(state, Vdag, n)  # V_i† ⊗ I
        state = apply_S_i(state, S_i_perm[i])  # S_i
        state = apply_V_tens(state, V, n)      # V_i ⊗ I
    # S_full at end
    state = apply_S_i(state, S_full_perm)
    return state

# ════════════════════════════════════════════════════
# 5. Kraus + direct fidelity (no 65536x65536 Choi)
# ════════════════════════════════════════════════════
print("Building Kraus operators (state-vector method) ...")

Kraus = [np.zeros((d, d), dtype=complex) for _ in range(d)]

for c in range(d):
    if c % 64 == 0:
        print(f"  column {c}/{d}")
    psi = np.zeros(dim, dtype=complex)
    psi[c] = 1.0  # ancilla=0, system=c
    psi = apply_U_sew(psi, V_list, N)
    for k in range(d):
        for r in range(d):
            idx = r + k * d
            Kraus[k][r, c] = psi[idx]

# Completeness
comp = sum(A.conj().T @ A for A in Kraus)
print(f"  Kraus completeness error: {np.linalg.norm(comp - np.eye(d)):.2e}")

# Process fidelity via Kraus: F = (1/d²) Σ_k |Tr(A_k · U_target†)|²
Udag = U_global.conj().T
fid_sum = 0.0
for A in Kraus:
    fid_sum += abs(np.trace(A @ Udag)) ** 2
process_fid = fid_sum / (d * d)
print(f"\n  Sewn channel process fidelity: {process_fid:.6f}")

# Random state fidelities
print("\nRandom state fidelities:")
for trial in range(5):
    psi_rand = rng.normal(size=d) + 1j * rng.normal(size=d)
    psi_rand /= np.linalg.norm(psi_rand)
    rho = np.outer(psi_rand, psi_rand.conj())
    rho_sewn = sum(A @ rho @ A.conj().T for A in Kraus)
    rho_target = U_global @ rho @ U_global.conj().T
    f = np.real(np.trace(rho_sewn @ rho_target))
    print(f"  trial {trial}: {f:.6f}")

avg_fid = np.mean([np.real(np.trace(
    sum(A @ rho @ A.conj().T for A in Kraus) @ (U_global @ rho @ U_global.conj().T)
)) for rho in [
    np.outer(psi, psi.conj()) for psi in [
        (rng.normal(size=d) + 1j * rng.normal(size=d)) / np.linalg.norm(rng.normal(size=d) + 1j * rng.normal(size=d))
        for _ in range(10)
    ]
]])
print(f"  average state fidelity (10 trials): {avg_fid:.6f}")

# Also check: direct composition of the two trained unitaries
U1_full = embed_block_into_n_qubits(U1_4q, BLOCKS[0], N)
U2_full = embed_block_into_n_qubits(U2_4q, BLOCKS[1], N)
U_direct = U2_full @ U1_full
diff_direct = np.linalg.norm(U_direct - U_global, 'fro')
# phase align
ov = np.trace(U_global.conj().T @ U_direct) / d
phi = ov / abs(ov) if abs(ov) > 1e-12 else 1.0
diff_phased = np.linalg.norm(U_direct - phi * U_global, 'fro')
print(f"\nDirect composition (no sewing):")
print(f"  ||U_direct - U_target||_F = {diff_direct:.4f}")
print(f"  ||U_direct - phase*U_target|| = {diff_phased:.4f}")

print(f"\n{'='*60}")
print(f"Sewing on n={N}: process fidelity = {process_fid:.4f}")
print(f"Direct composition error = {diff_phased:.4f} (vs Frobenius norm)")
