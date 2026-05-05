"""Test Rx gate effect on IDQNN global information propagation.

Compares output distributions with rx_angle=0 vs rx_angle=pi/4
for a 3x3 grid with a |0>-encoded barrier in the input.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
from ISQNN_generate_y_rx import idqnn_connectivity


def compute_marginals(probs: np.ndarray, n: int):
    """Compute per-qubit P(y_j=1) from full distribution."""
    margins = np.zeros(n)
    for j in range(n):
        mask = (np.arange(2**n) >> j) & 1
        margins[j] = np.sum(probs[mask == 1])
    return margins


def compute_pairwise(probs: np.ndarray, n: int):
    """Compute pairwise correlations E[y_i*y_j] - E[y_i]*E[y_j]."""
    margins = compute_marginals(probs, n)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mask_i = (np.arange(2**n) >> i) & 1
            mask_j = (np.arange(2**n) >> j) & 1
            if i == j:
                corr[i, j] = margins[i] * (1 - margins[i])
            else:
                joint = np.sum(probs[(mask_i == 1) & (mask_j == 1)])
                corr[i, j] = joint - margins[i] * margins[j]
    return margins, corr


def analyze_circuit(n1, m, bitstring, theta_list, rx_angle):
    """Run ISQNN circuit and return full output distribution."""
    try:
        import cirq
    except Exception:
        return None, None

    n = n1 * m
    rows, cols = n1, m
    qubits = [[cirq.GridQubit(r, c) for c in range(cols)] for r in range(rows)]
    sim = cirq.Simulator()

    theta = theta_list
    if isinstance(theta, (int, float)):
        theta = [theta] * n
    else:
        theta = list(theta)
        if len(theta) < n:
            theta.extend([theta[-1]] * (n - len(theta)))

    circ = cirq.Circuit()
    # Step 1: Encode
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx < len(bitstring) and bitstring[idx] == "0":
                circ.append(cirq.H(qubits[r][c]))

    # Step 1.5: Rx
    if rx_angle != 0.0:
        for r in range(rows):
            for c in range(cols):
                circ.append(cirq.rx(rx_angle)(qubits[r][c]))

    # Step 2: Rz
    for r in range(rows):
        for c in range(cols):
            circ.append(cirq.rz(theta[r * cols + c])(qubits[r][c]))

    # CZ intra-slice
    for r in range(rows):
        if r % 2 == 0:
            for c in range(0, cols - 1, 2):
                circ.append(cirq.CZ(qubits[r][c], qubits[r][c + 1]))
        else:
            for c in range(1, cols - 1, 2):
                circ.append(cirq.CZ(qubits[r][c], qubits[r][c + 1]))

    # CZ inter-slice
    for r in range(rows - 1):
        for c in range(cols):
            circ.append(cirq.CZ(qubits[r][c], qubits[r + 1][c]))

    # H for X-measurement
    for r in range(rows):
        for c in range(cols):
            circ.append(cirq.H(qubits[r][c]))

    result = sim.simulate(circ)
    sv = result.final_state_vector
    probs = np.abs(sv) ** 2
    return probs, circ


def print_grid(label, data, n1, m, fmt=".3f"):
    """Print data as n1 x m grid."""
    print(f"  {label}:")
    grid = np.array(data).reshape(n1, m)
    for row in grid:
        line = "".join(f"{v:{fmt}} " for v in row)
        print(f"    [{line}]")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # ===== 手动参数区 =====
    n1 = 3
    m = 3
    n = n1 * m
    rx_test_angle = np.pi / 4
    # Input with a |0> barrier in the middle row
    # x = "000"  (row0: all |+>)
    #     "111"  (row1: all |0>, creates barrier!)
    #     "000"  (row2: all |+>)
    # Without Rx, row1 blocks CZ between row0 and row2
    bitstring = "000" + "111" + "000"
    theta = np.random.RandomState(42).uniform(0, 2 * np.pi, n)
    # ====================

    print("=" * 60)
    print(" Rx Gate Effect Test — 3x3 Grid with |0> Barrier")
    print("=" * 60)
    print(f"  n1={n1}, m={m}, n={n}")
    print(f"  Input: x[0]='{bitstring[:m]}' | x[1]='{bitstring[m:2*m]}' | x[2]='{bitstring[2*m:]}'")
    print(f"  Barrier: middle row is all '1' encoded as |0>_j (CZ eigenstate)")
    print()

    # Analyze without Rx
    print("--- rx_angle = 0 (original circuit) ---")
    probs0, _ = analyze_circuit(n1, m, bitstring, theta, rx_angle=0.0)
    if probs0 is not None:
        margins0, corr0 = compute_pairwise(probs0, n)
        supp0 = int(np.sum(probs0 > 1e-10))
        ent0 = -np.sum(probs0[probs0 > 1e-10] * np.log2(probs0[probs0 > 1e-10]))
        print_grid("P(y_j=1)", margins0, n1, m)
        print(f"  Support size: {supp0}/{2**n}")
        print(f"  Entropy: {ent0:.4f} bits (max={n:.1f})")
        print()

    # Analyze with Rx(pi/4)
    print(f"--- rx_angle = {rx_test_angle:.4f} (Rx-modified circuit) ---")
    probs_rx, _ = analyze_circuit(n1, m, bitstring, theta, rx_angle=rx_test_angle)
    if probs_rx is not None:
        margins_rx, corr_rx = compute_pairwise(probs_rx, n)
        supp_rx = int(np.sum(probs_rx > 1e-10))
        ent_rx = -np.sum(probs_rx[probs_rx > 1e-10] * np.log2(probs_rx[probs_rx > 1e-10]))
        print_grid("P(y_j=1)", margins_rx, n1, m)
        print(f"  Support size: {supp_rx}/{2**n}")
        print(f"  Entropy: {ent_rx:.4f} bits (max={n:.1f})")
        print()

    # Compare correlations
    if probs0 is not None and probs_rx is not None:
        print("--- Correlation Comparison ---")
        print("  Without Rx -- inter-row correlations:")
        # Show key inter-row correlations (row0-row1, row1-row2, row0-row2)
        for ri in range(n1):
            for rj in range(n1):
                if ri != rj:
                    avg_corr = np.mean([
                        corr0[ri * m + ci, rj * m + cj]
                        for ci in range(m) for cj in range(m)
                    ])
                    print(f"    avg corr row{ri}-row{rj}: {avg_corr:.6f}")

        print()
        print("  With Rx(pi/4) -- inter-row correlations:")
        for ri in range(n1):
            for rj in range(n1):
                if ri != rj:
                    avg_corr = np.mean([
                        corr_rx[ri * m + ci, rj * m + cj]
                        for ci in range(m) for cj in range(m)
                    ])
                    print(f"    avg corr row{ri}-row{rj}: {avg_corr:.6f}")

        print()
        # Check if Rx changes the marginal distribution
        margin_diff = np.max(np.abs(margins_rx - margins0))
        print(f"  Max marginal difference (|dP|): {margin_diff:.6f}")

        # Check support expansion
        print(f"  Support change: {supp0} -> {supp_rx} (+{supp_rx - supp0})")

        print()
        print("=" * 60)
        print("  Interpretation:")
        if supp_rx > supp0:
            print(f"  [+] Rx expanded output support (+{supp_rx - supp0} states)")
        if margin_diff > 0.01:
            print(f"  [+] Rx changed marginal probabilities (max delta={margin_diff:.4f})")
        else:
            print("  [-] Marginal probs nearly unchanged")
        print("=" * 60)
