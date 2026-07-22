# task2_code/lightcone_test.py
# Light cone analysis for Task 2 Mode 1.

import os
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import cirq
from numpy.linalg import norm
from functools import reduce

from task2_code.U_target import (
    build_v_scr_unitary, build_v_scr_circuit,
    build_h_diag, build_h_target,
    u_target_unitary, u_target_expm, is_unitary,
)
from task2_code.experiment_config import DEFAULT_SEED as seed

I2 = np.eye(2, dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)


# ═══════════════════════════════════════════════════════════════
#  Dense commutator method (n <= 10 only)
# ═══════════════════════════════════════════════════════════════

def z_operator(j, n):
    ops = [I2] * n; ops[j] = Z
    return reduce(np.kron, ops)


def backward_lightcone_dense(U_target, q, n, tol=1e-6):
    """O_q = U^† Z_q U; j in cone if ||[O_q, Z_j]|| > tol."""
    Oq = U_target.conj().T @ z_operator(q, n) @ U_target#逆时间演化
    cone = []
    for j in range(n):
        comm = Oq @ z_operator(j, n) - z_operator(j, n) @ Oq
        if norm(comm) > tol:
            cone.append(j)
    return cone


def forward_lightcone_dense(U_target, j, n, tol=1e-6):
    """F_j = U Z_j U^†; q in cone if ||[F_j, Z_q]|| > tol."""
    Fj = U_target @ z_operator(j, n) @ U_target.conj().T#正时间演化
    cone = []
    for q in range(n):
        comm = Fj @ z_operator(q, n) - z_operator(q, n) @ Fj
        if norm(comm) > tol:
            cone.append(q)
    return cone


# ═══════════════════════════════════════════════════════════════
#  Circuit-based method (any n)
# ═══════════════════════════════════════════════════════════════

def _gate_adds_to_lightcone(op, current_qubits):
    """Given a Cirq operation and a set of currently-affected qubits,
    return the expanded set after this gate is applied in reverse."""
    qs = {q.x for q in op.qubits}
    if qs & current_qubits:
        return current_qubits | qs
    return current_qubits


def backward_lightcone_circuit(circuit, q, n):
    """Trace circuit backwards from output qubit q to find input light cone.

    Start from {q}, apply gates in REVERSE time order.
    Whenever a gate touches any qubit in the current set, add ALL its qubits.
    """
    affected = {q}
    for op in reversed(list(circuit.all_operations())):
        qs = {_q.x for _q in op.qubits}
        if qs & affected:
            affected |= qs#类似于a+=b的集合版本，更新受影响的集合,|=是集合的并集操作
    return sorted(affected)


def forward_lightcone_circuit(circuit, j, n):
    """Trace circuit forwards from input qubit j to find output light cone.

    Start from {j}, apply gates in FORWARD time order.
    Whenever a gate touches any qubit in the current set, add ALL its qubits.
    """
    affected = {j}
    for op in circuit.all_operations():
        qs = {_q.x for _q in op.qubits}
        if qs & affected:
            affected |= qs
    return sorted(affected)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import math

    N = 10
    rng = np.random.default_rng(seed)
    print(f"=== Light Cone Test (n={N}) ===\n")

    # Physical system
    V = build_v_scr_unitary(N, rng)
    Hd, hv = build_h_diag(N, rng)
    Ht = build_h_target(V, Hd)

    rng_cache = np.random.default_rng(seed)
    t = 3 * math.pi / 40 * 10 + 0.001
    Ut = u_target_unitary(N, rng_cache, hv, t)

    circ_v = build_v_scr_circuit(N, rng_cache)
    # Build full U_target circuit for circuit-based method
    circ_u = cirq.Circuit()
    circ_u += circ_v
    circ_u.append([cirq.rz(+2*h*t).on(cirq.LineQubit(i)) for i, h in enumerate(hv)])
    # V_scr^dagger: invert + reverse
    for op in reversed(list(circ_v.all_operations())):
        circ_u.append(cirq.inverse(op))

    # ── backward light cone ──
    print("=== Backward Light Cone (output qubit q ← input qubits) ===")
    print(f"{'q':>4} {'dense':>20}  {'circuit':>20}  {'match'}")
    print("-" * 55)
    for q in [0, 4, 9]:
        d_back = backward_lightcone_dense(Ut, q, N)
        c_back = backward_lightcone_circuit(circ_u, q, N)
        ok = "OK" if d_back == c_back else f"MISMATCH dense={d_back} circ={c_back}"
        print(f"{q:>4} {str(d_back):>20}  {str(c_back):>20}  {ok}")

    # ── forward light cone ──
    print("\n=== Forward Light Cone (input qubit j → output qubits) ===")
    print(f"{'j':>4} {'dense':>20}  {'circuit':>20}  {'match'}")
    print("-" * 55)
    for j in [0, 4, 9]:
        d_fwd = forward_lightcone_dense(Ut, j, N)
        c_fwd = forward_lightcone_circuit(circ_u, j, N)
        ok = "OK" if d_fwd == c_fwd else f"MISMATCH dense={d_fwd} circ={c_fwd}"
        print(f"{j:>4} {str(d_fwd):>20}  {str(c_fwd):>20}  {ok}")

    # ── test circuit method on larger n ──
    print(f"\n=== Circuit Method on n=20 (no dense comparison) ===")
    rng20 = np.random.default_rng(seed)
    circ_v20 = build_v_scr_circuit(20, rng20)
    circ_u20 = cirq.Circuit()
    circ_u20 += circ_v20
    hv20 = rng20.uniform(-1, 1, 20)
    circ_u20.append([cirq.rz(+2*h*t).on(cirq.LineQubit(i)) for i, h in enumerate(hv20)])
    for op in reversed(list(circ_v20.all_operations())):
        circ_u20.append(cirq.inverse(op))

    for q in [0, 9, 19]:
        c = backward_lightcone_circuit(circ_u20, q, 20)
        print(f"  backward q={q:>2}: cone={c}  (size={len(c)})")
    for j in [0, 9, 19]:
        c = forward_lightcone_circuit(circ_u20, j, 20)
        print(f"  forward  j={j:>2}: cone={c}  (size={len(c)})")

    print("\nDone.")
