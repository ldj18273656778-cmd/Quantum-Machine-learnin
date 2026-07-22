# task2_code/U_target.py
# Construction of the target unitary U_target(t) for Task 2 using Cirq.
#
# Implements:
#   H_diag   = sum_j h_j Z_j
#   H_target = V_scr^dagger H_diag V_scr
#   U_target(t) = V_scr^dagger * exp(-i H_diag t) * V_scr
#
# Dependencies: numpy, scipy, cirq

import numpy as np
import cirq
from numpy import array, eye, zeros, diag
from numpy.linalg import norm

# ======================================================================
#  Pauli matrices (kept for H_diag construction)
# ======================================================================
I2 = eye(2, dtype=complex)
Z  = array([[1, 0], [0, -1]], dtype=complex)


# ======================================================================
#  PhasedXZ – using cirq's native PhasedXZGate
# ======================================================================

def random_phxz(rng):
    """Return a cirq.PhasedXZGate with parameter range [0, 4/pi].

    cirq.PhasedXZGate uses exponent convention where exponent=1
    means a rotation of pi radians.  Sampling from [0, 4/pi] gives
    an effective rotation angle in [0, 4] radians, matching the
    paper's parameter range for PhXZ(x, z, a).
    """
    max_val = 4.0 / np.pi          # ~1.273
    return cirq.PhasedXZGate(
        x_exponent=rng.uniform(0, max_val),
        z_exponent=rng.uniform(0, max_val),
        axis_phase_exponent=rng.uniform(0, max_val),
    )


# ======================================================================
#  V_scr – 1D random brickwork scrambling unitary
# ======================================================================

def build_v_scr_circuit(n, rng):
    """Return a Cirq circuit implementing V_scr on n qubits.

    Structure (Eq. S.2.6):
        PhXZ(all) -> CZ(even) -> PhXZ(all) -> CZ(odd) -> PhXZ(all)

    Each PhXZ is a single cirq.PhasedXZGate (1 moment).
    """
    qubits = cirq.LineQubit.range(n)
    circuit = cirq.Circuit()

    # Layer 0: PhXZ on every qubit
    circuit.append([random_phxz(rng).on(qubits[i]) for i in range(n)])

    # Layer 1: CZ on even pairs
    circuit.append([cirq.CZ(qubits[i], qubits[i + 1])
                    for i in range(0, n - 1, 2)])

    # Layer 2: PhXZ on every qubit
    circuit.append([random_phxz(rng).on(qubits[i]) for i in range(n)])

    # Layer 3: CZ on odd pairs
    circuit.append([cirq.CZ(qubits[i], qubits[i + 1])
                    for i in range(1, n - 1, 2)])

    # Layer 4: PhXZ on every qubit
    circuit.append([random_phxz(rng).on(qubits[i]) for i in range(n)])

    return circuit


def build_v_scr_unitary(n, rng):
    """Return the dense unitary matrix of V_scr."""
    return cirq.unitary(build_v_scr_circuit(n, rng))


# ======================================================================
#  H_diag, H_target
# ======================================================================

def build_h_diag(n, rng):
    h_arr = rng.uniform(-1.0, 1.0, size=n)
    H = zeros((2 ** n, 2 ** n), dtype=complex)
    for j in range(n):
        ops = [I2] * n; ops[j] = Z
        Zj = ops[0]
        for g in ops[1:]: Zj = np.kron(Zj, g)
        H += h_arr[j] * Zj
    return H, h_arr


def build_h_target(V_scr, H_diag):
    return V_scr.conj().T @ H_diag @ V_scr


# ======================================================================
#  U_target(t)
# ======================================================================

def build_u_target_circuit(n, rng, h_arr, t):
    """Return U_target(t) Cirq circuit.

    Circuit time-order (paper: "U, Rz, U†"):
        V_scr -> R_z layer -> V_scr^dagger

    V_scr^dagger is built via cirq.inverse().
    """
    qubits = cirq.LineQubit.range(n)

    circ_v = build_v_scr_circuit(n, rng)

    # V_scr^dagger
    ops_dag = [cirq.inverse(op)
               for op in reversed(list(circ_v.all_operations()))]
    circ_vdag = cirq.Circuit(ops_dag)

    # R_z layer: exp(-i H_diag t) = kron_j exp(-i h_j Z_j t)
    rz_ops = [cirq.rz(+2 * h * t).on(qubits[i]) for i, h in enumerate(h_arr)]

    circ = cirq.Circuit()
    circ += circ_v
    circ.append(rz_ops)
    circ += circ_vdag
    return circ


def u_target_unitary(n, rng, h_arr, t):
    circ = build_u_target_circuit(n, rng, h_arr, t)
    return cirq.unitary(circ)


def u_target_expm(H_target, t):
    from scipy.linalg import expm
    return expm(-1.0j * H_target * t)


# ======================================================================
#  Helpers
# ======================================================================

def is_unitary(U, tol=1e-10):
    dim = U.shape[0]
    return norm(U @ U.conj().T - eye(dim)) < tol


# ======================================================================
#  Verification
# ======================================================================
if __name__ == "__main__":
    from config import seed
    import math

    # ---- 数值验证 ----
    for test_n in [4, 6]:
        print(f"\n=== Testing n = {test_n} ===")

        rng = np.random.default_rng(seed)
        V = build_v_scr_unitary(test_n, rng)
        print(f"  V_scr unitary: {is_unitary(V)}")

        Hd, h_vec = build_h_diag(test_n, rng)
        print(f"  h_j: {np.round(h_vec, 3)}")
        Ht = build_h_target(V, Hd)
        ei_d = np.sort(np.linalg.eigvalsh(Hd).real)
        ei_t = np.sort(np.linalg.eigvalsh(Ht).real)
        print(f"  eig H_diag == H_target: {np.allclose(ei_d, ei_t)}")

        for k_t in [0, 5, 10]:
            t = 3 * math.pi / 40 * k_t + 0.001
            rng_u = np.random.default_rng(seed)
            U_cirq = u_target_unitary(test_n, rng_u, h_vec, t)
            U_ex   = u_target_expm(Ht, t)
            err = norm(U_cirq - U_ex)
            print(f"  t={t:.4f}: "
                  f"||U_cirq-U_expm||={err:.2e} "
                  f"unitary={is_unitary(U_cirq)}")

    print("\nDone.")
