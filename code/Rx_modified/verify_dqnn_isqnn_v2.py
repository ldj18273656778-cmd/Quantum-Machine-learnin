"""Compare ISQNN vs 3 DQNN variants under Rx(pi/4) on a 2x2 grid.

Variants:
  A) ISQNN (exact, 4 qubits, all gates then final measurement)
  B) Original DQNN (random.randint for x=0, measure+reset for x=1)
  C) DQNN with TRUE quantum measurement for x=0 (original x=1 handling)
  D) DQNN with TRUE x=0 measurement + x=1 measured AFTER Rx+Rz+CZ

Input: x="0010" (bit2=1, rest=0). n1=2, m=2, rx=pi/4.
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


# ============================================================
# A) ISQNN
# ============================================================
def isqnn_distribution(x_str, theta, rx):
    rows, cols = n1, m
    q = [[cirq.GridQubit(r, c) for c in range(cols)] for r in range(rows)]
    sim = cirq.Simulator()
    circ = cirq.Circuit()
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if x_str[idx] == "0":
                circ.append(cirq.H(q[r][c]))
    if rx != 0:
        for r in range(rows):
            for c in range(cols):
                circ.append(cirq.rx(rx)(q[r][c]))
    for r in range(rows):
        for c in range(cols):
            circ.append(cirq.rz(theta[r * cols + c])(q[r][c]))
    # intra CZ
    for r in range(rows):
        if r % 2 == 0:
            for c in range(0, cols - 1, 2):
                circ.append(cirq.CZ(q[r][c], q[r][c + 1]))
        else:
            for c in range(1, cols - 1, 2):
                circ.append(cirq.CZ(q[r][c], q[r][c + 1]))
    # inter CZ
    for r in range(rows - 1):
        for c in range(cols):
            circ.append(cirq.CZ(q[r][c], q[r + 1][c]))
    flat = [q[r][c] for r in range(rows) for c in range(cols)]
    for qq in flat:
        circ.append(cirq.H(qq))
    result = sim.simulate(circ, qubit_order=flat)
    return np.abs(result.final_state_vector) ** 2


# ============================================================
# DQNN variants (sampled because of mid-circuit measurements)
# ============================================================
def dqnn_one_sample(x_str, theta, rx, variant="original"):
    """Generate ONE sample y from DQNN.

    variant:
      "original": random.randint for x=0, measure+reset for x=1 (before Rx+Rz+CZ)
      "true_x0":  cirq.measure for x=0, measure+reset for x=1 (before Rx+Rz+CZ)
      "true_x0_x1defer": cirq.measure for x=0, x=1 measured AFTER Rx+Rz+CZ
    """
    qubits = cirq.LineQubit.range(m)
    sim = cirq.Simulator()

    # Block 0: encode + Rx + Rz + CZ
    circ = cirq.Circuit()
    for i in range(m):
        if x_str[i] == "0":
            circ.append(cirq.H(qubits[i]))
    if rx != 0:
        for i in range(m):
            circ.append(cirq.rx(rx)(qubits[i]))
    for i in range(m):
        circ.append(cirq.rz(theta[i])(qubits[i]))
    # CZ for block 0 (even block)
    for i in range(m // 2):
        circ.append(cirq.CZ(qubits[2 * i], qubits[2 * i + 1]))
    result = sim.simulate(circ, qubit_order=qubits)
    state = result.final_state_vector

    y_out = []
    deferred_x1 = []  # (qubit_idx, theta_idx, bit_pos) for variant "true_x0_x1defer"

    # Blocks 1..n1-1
    for block in range(1, n1):
        for i in range(m):
            bit_idx = block * m + i
            x_bit = x_str[bit_idx]

            if x_bit == "0":
                # ---- x=0 ----
                hc = cirq.Circuit(cirq.H(qubits[i]))
                res = sim.simulate(hc, initial_state=state, qubit_order=qubits)
                state = res.final_state_vector

                if variant == "original":
                    meas = np.random.randint(0, 2)
                else:
                    mc = cirq.Circuit(cirq.measure(qubits[i], key="m"))
                    res = sim.simulate(mc, initial_state=state, qubit_order=qubits)
                    state = res.final_state_vector
                    meas = int(res.measurements["m"][0])

                y_out.append(meas)
                if meas == 1:
                    xc = cirq.Circuit(cirq.X(qubits[i]))
                    res = sim.simulate(xc, initial_state=state, qubit_order=qubits)
                    state = res.final_state_vector

            else:  # x_bit == "1"
                if variant == "true_x0_x1defer":
                    # Defer measurement: note it for later
                    deferred_x1.append((i, block * m + i, len(y_out)))
                    y_out.append(0)  # placeholder, will be overwritten
                else:
                    # ---- x=1: measure + reset ----
                    hc = cirq.Circuit(cirq.H(qubits[i]))
                    res = sim.simulate(hc, initial_state=state, qubit_order=qubits)
                    state = res.final_state_vector

                    mc = cirq.Circuit(cirq.measure(qubits[i], key="m"))
                    res = sim.simulate(mc, initial_state=state, qubit_order=qubits)
                    state = res.final_state_vector
                    meas = int(res.measurements["m"][0])
                    y_out.append(meas)

                    rc = cirq.Circuit(cirq.reset(qubits[i]))
                    res = sim.simulate(rc, initial_state=state, qubit_order=qubits)
                    state = res.final_state_vector

        # ---- Block Rx+Rz+CZ ----
        if block <= n1 - 1:
            bc = cirq.Circuit()
            if rx != 0:
                for i in range(m):
                    bc.append(cirq.rx(rx)(qubits[i]))
            for i in range(m):
                bc.append(cirq.rz(theta[block * m + i])(qubits[i]))
            if block % 2 == 0:
                for i in range(m // 2):
                    bc.append(cirq.CZ(qubits[2 * i], qubits[2 * i + 1]))
            else:
                for i in range((m - 1) // 2):
                    bc.append(cirq.CZ(qubits[2 * i + 1], qubits[2 * i + 2]))
            res = sim.simulate(bc, initial_state=state, qubit_order=qubits)
            state = res.final_state_vector

    # ---- Process deferred x=1 measurements (for "true_x0_x1defer") ----
    for qi, th_idx, y_pos in deferred_x1:
        hc = cirq.Circuit(cirq.H(qubits[qi]))
        res = sim.simulate(hc, initial_state=state, qubit_order=qubits)
        state = res.final_state_vector
        mc = cirq.Circuit(cirq.measure(qubits[qi], key="m"))
        res = sim.simulate(mc, initial_state=state, qubit_order=qubits)
        state = res.final_state_vector
        meas = int(res.measurements["m"][0])
        y_out[y_pos] = meas
        rc = cirq.Circuit(cirq.reset(qubits[qi]))
        res = sim.simulate(rc, initial_state=state, qubit_order=qubits)
        state = res.final_state_vector

    # ---- Final measurement ----
    fm = cirq.Circuit()
    for i in range(m):
        fm.append(cirq.H(qubits[i]))
        fm.append(cirq.measure(qubits[i], key=f"f{i}"))
    res = sim.simulate(fm, initial_state=state, qubit_order=qubits)
    for i in range(m):
        y_out.append(int(res.measurements[f"f{i}"][0]))

    return y_out


def estimate_distribution(x_str, theta, rx, variant, num_samples=5000):
    probs = np.zeros(2**n)
    for _ in range(num_samples):
        y = dqnn_one_sample(x_str, theta, rx, variant)
        idx = sum(y[j] << (n - 1 - j) for j in range(n))
        probs[idx] += 1
    return probs / num_samples


# ============================================================
# Main comparison
# ============================================================
np.random.seed(0)
x_str = "0010"  # bit 2 is x=1, in block 1
theta = np.array([0.5, 1.2, 0.8, 2.1])

print(f"x={x_str}, theta={theta}, rx={rx_angle:.4f}")
print()

# A) ISQNN
p_isqnn = isqnn_distribution(x_str, theta, rx_angle)
print("A) ISQNN (exact):")
for idx in np.argsort(p_isqnn)[::-1][:6]:
    bits = "".join("1" if (idx >> (n - 1 - j)) & 1 else "0" for j in range(n))
    print(f"  |{bits}>  P={p_isqnn[idx]:.5f}")

# B) Original DQNN
    p_orig = estimate_distribution(x_str, theta, rx_angle, "original")
print("\nB) Original DQNN (random.randint for x=0):")
for idx in np.argsort(p_orig)[::-1][:6]:
    bits = "".join("1" if (idx >> (n - 1 - j)) & 1 else "0" for j in range(n))
    print(f"  |{bits}>  P={p_orig[idx]:.5f}")

# C) DQNN with true x=0 measurement
p_tx0 = estimate_distribution(x_str, theta, rx_angle, "true_x0")
print("\nC) DQNN true-x0-measure (cirq.measure for x=0):")
for idx in np.argsort(p_tx0)[::-1][:6]:
    bits = "".join("1" if (idx >> (n - 1 - j)) & 1 else "0" for j in range(n))
    print(f"  |{bits}>  P={p_tx0[idx]:.5f}")

# D) DQNN with true x=0 + deferred x=1
p_defer = estimate_distribution(x_str, theta, rx_angle, "true_x0_x1defer")
print("\nD) DQNN true-x0 + x1-deferred (x=1 after Rx+Rz+CZ):")
for idx in np.argsort(p_defer)[::-1][:6]:
    bits = "".join("1" if (idx >> (n - 1 - j)) & 1 else "0" for j in range(n))
    print(f"  |{bits}>  P={p_defer[idx]:.5f}")

# Comparison
print("\n======= Total Variation Distance vs ISQNN =======")
tv_orig = 0.5 * np.sum(np.abs(p_isqnn - p_orig))
tv_tx0 = 0.5 * np.sum(np.abs(p_isqnn - p_tx0))
tv_defer = 0.5 * np.sum(np.abs(p_isqnn - p_defer))
print(f"B) Original:  TV={tv_orig:.4f}")
print(f"C) true-x0:   TV={tv_tx0:.4f}  ({(1-tv_tx0/tv_orig)*100:+.1f}% vs original)")
print(f"D) x0+x1def:  TV={tv_defer:.4f}  ({(1-tv_defer/tv_orig)*100:+.1f}% vs original)")

# Per-bit marginal comparison
print("\n======= Per-bit Marginal P(y_j=1) =======")
for name, p_dist in [("ISQNN", p_isqnn), ("Orig", p_orig), ("C)tx0", p_tx0), ("D)defer", p_defer)]:
    margins = np.zeros(n)
    for j in range(n):
        mask = (np.arange(2**n) >> (n - 1 - j)) & 1
        margins[j] = np.sum(p_dist[mask == 1])
    print(f"  {name:8s}: {margins}")
