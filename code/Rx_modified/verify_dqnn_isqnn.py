"""Verify: Can modified DQNN (measure x=1 after Rx+Rz+CZ) match ISQNN?

Compares 3 circuits on a 2x2 grid (n1=2, m=2, n=4):
  A) ISQNN (4 qubits, all gates + final measurement)
  B) Original DQNN (x=1 measured before Rx+Rz+CZ)  
  C) Modified DQNN (x=1 measured after Rx+Rz+CZ)

Input x, fixed theta, Rx=pi/4. Computes output distribution P(y).
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

import numpy as np
import cirq

np.set_printoptions(precision=4, suppress=True)
n1, m = 2, 2
n = n1 * m
rx_angle = np.pi / 4


def isqnn_full_distribution(x_str, theta, rx):
    """ISQNN circuit: all n1*m qubits, all gates upfront, final measurement.
    Returns full P(y) distribution as array[2^n]."""
    rows, cols = n1, m
    q = [[cirq.GridQubit(r, c) for c in range(cols)] for r in range(rows)]
    sim = cirq.Simulator()
    circ = cirq.Circuit()

    # Encode
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if x_str[idx] == "0":
                circ.append(cirq.H(q[r][c]))
    # Rx
    if rx != 0:
        for r in range(rows):
            for c in range(cols):
                circ.append(cirq.rx(rx)(q[r][c]))
    # Rz
    for r in range(rows):
        for c in range(cols):
            circ.append(cirq.rz(theta[r * cols + c])(q[r][c]))
    # CZ intra
    for r in range(rows):
        if r % 2 == 0:
            for c in range(0, cols - 1, 2):
                circ.append(cirq.CZ(q[r][c], q[r][c + 1]))
        else:
            for c in range(1, cols - 1, 2):
                circ.append(cirq.CZ(q[r][c], q[r][c + 1]))
    # CZ inter
    for r in range(rows - 1):
        for c in range(cols):
            circ.append(cirq.CZ(q[r][c], q[r + 1][c]))
    # H for X-measurement
    flat_q = [q[r][c] for r in range(rows) for c in range(cols)]
    for qq in flat_q:
        circ.append(cirq.H(qq))

    result = sim.simulate(circ, qubit_order=flat_q)
    probs = np.abs(result.final_state_vector) ** 2
    return probs


def dqnn_distribution(x_str, theta, rx, modified=False):
    """DQNN circuit: m qubits reused across n1 blocks.

    modified=True: x=1 qubits measured AFTER Rx+Rz+CZ (not before).
    Returns full P(y) distribution via sampling (N samples per path).
    """
    qubits = cirq.LineQubit.range(m)
    sim = cirq.Simulator()

    # Block 0: initial encoding + Rx + Rz + CZ
    circ_full = cirq.Circuit()
    for i in range(m):
        if x_str[i] == "0":
            circ_full.append(cirq.H(qubits[i]))
    if rx != 0:
        for i in range(m):
            circ_full.append(cirq.rx(rx)(qubits[i]))
    for i in range(m):
        circ_full.append(cirq.rz(theta[i])(qubits[i]))
    if m == 2:
        circ_full.append(cirq.CZ(qubits[0], qubits[1]))
    result = sim.simulate(circ_full)
    state = result.final_state_vector

    y_output = []

    # Block 1
    for i in range(m):
        bit_idx = m + i
        if not modified:
            # ORIGINAL: measure x=1 before Rx+Rz+CZ
            if x_str[bit_idx] == "1":
                hc = cirq.Circuit(cirq.H(qubits[i]), cirq.measure(qubits[i], key="m"))
                res = sim.simulate(hc, initial_state=state, qubit_order=qubits)
                y_output.append(int(res.measurements["m"][0]))
                state = res.final_state_vector
                rc = cirq.Circuit(cirq.reset(qubits[i]))
                res = sim.simulate(rc, initial_state=state, qubit_order=qubits)
                state = res.final_state_vector
            else:
                hc = cirq.Circuit(cirq.H(qubits[i]))
                res = sim.simulate(hc, initial_state=state, qubit_order=qubits)
                state = res.final_state_vector
                meas = np.random.randint(0, 2)
                y_output.append(meas)
                if meas == 1:
                    xc = cirq.Circuit(cirq.X(qubits[i]))
                    res = sim.simulate(xc, initial_state=state, qubit_order=qubits)
                    state = res.final_state_vector

    # Block operations (Rz + CZ for block 1)
    bc = cirq.Circuit()
    if rx != 0:
        for i in range(m):
            bc.append(cirq.rx(rx)(qubits[i]))
    for i in range(m):
        bc.append(cirq.rz(theta[m + i])(qubits[i]))
    if m == 2 and 1 % 2 == 0:
        bc.append(cirq.CZ(qubits[0], qubits[1]))
    result = sim.simulate(bc, initial_state=state, qubit_order=qubits)
    state = result.final_state_vector

    if modified:
        # MODIFIED: measure x=1 AFTER Rx+Rz+CZ
        for i in range(m):
            bit_idx = m + i
            if x_str[bit_idx] == "1":
                hc = cirq.Circuit(cirq.H(qubits[i]), cirq.measure(qubits[i], key="m"))
                res = sim.simulate(hc, initial_state=state, qubit_order=qubits)
                y_output.append(int(res.measurements["m"][0]))
                state = res.final_state_vector
                rc = cirq.Circuit(cirq.reset(qubits[i]))
                res = sim.simulate(rc, initial_state=state, qubit_order=qubits)
                state = res.final_state_vector
        # x=0 bits in block 1 still need y output (random, as before)
        for i in range(m):
            bit_idx = m + i
            if x_str[bit_idx] == "0":
                hc = cirq.Circuit(cirq.H(qubits[i]))
                res = sim.simulate(hc, initial_state=state, qubit_order=qubits)
                state = res.final_state_vector
                meas = np.random.randint(0, 2)
                y_output.append(meas)
                if meas == 1:
                    xc = cirq.Circuit(cirq.X(qubits[i]))
                    res = sim.simulate(xc, initial_state=state, qubit_order=qubits)
                    state = res.final_state_vector

    # Final measure: all m qubits in X-basis
    fm = cirq.Circuit()
    for i in range(m):
        fm.append(cirq.H(qubits[i]))
        fm.append(cirq.measure(qubits[i], key=f"f{i}"))
    res = sim.simulate(fm, initial_state=state, qubit_order=qubits)
    for i in range(m):
        y_output.append(int(res.measurements[f"f{i}"][0]))

    return None, y_output  # returns (circuit, y) like original


def sample_dqnn(x_str, theta, rx, modified, num_samples=20000):
    """Sample DQNN multiple times to estimate P(y)."""
    probs = np.zeros(2**n)
    for _ in range(num_samples):
        _, y = dqnn_distribution(x_str, theta, rx, modified)
        y_idx = sum(y[j] << (n - 1 - j) for j in range(n))
        probs[y_idx] += 1
    return probs / num_samples


# ===== Main comparison =====
np.random.seed(0)

# Test input: middle qubit is the x=1 barrier
x_str = "0100"  # q0=|+⟩, q1=|0⟩, q2=|+⟩, q3=|+⟩
theta = np.array([0.5, 1.2, 0.8, 2.1])

print(f"Input x: {x_str}")
print(f"Theta: {theta}")
print(f"Rx angle: {rx_angle:.4f}")
print(f"Grid: n1={n1}, m={m}\n")

# A) ISQNN (exact)
p_isqnn = isqnn_full_distribution(x_str, theta, rx_angle)
print("=== A) ISQNN (exact) ===")
top = np.argsort(p_isqnn)[::-1][:4]
for idx in top:
    bits = "".join("1" if (idx >> (n - 1 - j)) & 1 else "0" for j in range(n))
    print(f"  |{bits}>  P={p_isqnn[idx]:.4f}")

# B) Original DQNN (sampled)
print("\n=== B) Original DQNN (x=1 measured BEFORE Rx+Rz+CZ) ===")
p_dqnn_old = sample_dqnn(x_str, theta, rx_angle, modified=False, num_samples=20000)
top = np.argsort(p_dqnn_old)[::-1][:4]
for idx in top:
    bits = "".join("1" if (idx >> (n - 1 - j)) & 1 else "0" for j in range(n))
    print(f"  |{bits}>  P={p_dqnn_old[idx]:.4f}")

# C) Modified DQNN (sampled)
print("\n=== C) Modified DQNN (x=1 measured AFTER Rx+Rz+CZ) ===")
p_dqnn_new = sample_dqnn(x_str, theta, rx_angle, modified=True, num_samples=20000)
top = np.argsort(p_dqnn_new)[::-1][:4]
for idx in top:
    bits = "".join("1" if (idx >> (n - 1 - j)) & 1 else "0" for j in range(n))
    print(f"  |{bits}>  P={p_dqnn_new[idx]:.4f}")

# Comparison metrics
print("\n=== Comparison ===")
tv_old = 0.5 * np.sum(np.abs(p_isqnn - p_dqnn_old))
tv_new = 0.5 * np.sum(np.abs(p_isqnn - p_dqnn_new))
kl_old = np.sum(p_isqnn[p_isqnn > 0] * np.log(p_isqnn[p_isqnn > 0] / (p_dqnn_old[p_isqnn > 0] + 1e-15)))
kl_new = np.sum(p_isqnn[p_isqnn > 0] * np.log(p_isqnn[p_isqnn > 0] / (p_dqnn_new[p_isqnn > 0] + 1e-15)))
print(f"Total variation distance (old DQNN vs ISQNN): {tv_old:.4f}")
print(f"Total variation distance (new DQNN vs ISQNN): {tv_new:.4f}")
print(f"KL divergence          (old DQNN vs ISQNN): {kl_old:.4f}")
print(f"KL divergence          (new DQNN vs ISQNN): {kl_new:.4f}")

if tv_new < tv_old:
    imp = (tv_old - tv_new) / tv_old * 100
    print(f"\n[+] Modified DQNN is {imp:.1f}% CLOSER to ISQNN than original")
else:
    print(f"\n[-] Modified DQNN is WORSE")
