"""Quick MNIST inference test with Rx-modified DQNN.

Runs on a small subset (100 samples) to verify the pipeline works.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
from tqdm import tqdm

from DQNN_generate_y_rx import DQNN_generate_y_rx
from sampling.ISQNN_generate_y import idqnn_connectivity
from MNIST.Encode_0to9 import encode_diagonal

# ===== 手动参数区 =====
NUM_TEST_SAMPLES = 100
rx_angle = np.pi / 4
threshold = 0.4
n1 = 10
m = 10
# ====================

RX_DIR = Path(__file__).resolve().parent
MNIST_DIR = ROOT / "code" / "MNIST"

# Load test data
X_test = np.load(MNIST_DIR / "data" / "test_MNIST_10x10_binarize0.5.npy")
Y_test = np.load(MNIST_DIR / "data" / "test_MNIST_labels.npy")
X_test = X_test[:NUM_TEST_SAMPLES].reshape(NUM_TEST_SAMPLES, -1).astype(int)
Y_test = Y_test[:NUM_TEST_SAMPLES]

X_flip = 1 - X_test
x_test_bits = np.array(["".join(map(str, row)) for row in X_flip], dtype=str)
y_test_encoded = np.array([encode_diagonal(label) for label in Y_test])

# Load theta (prefer Rx-estimated, fallback to original)
theta_path = RX_DIR / "data" / f"estimate_theta_rx_pi4_binarized{threshold}.npz"
if not theta_path.exists():
    theta_path = MNIST_DIR / "data" / f"estimate_theta_binarized{threshold}.npz"
    print(f"Using original theta from: {theta_path}")
else:
    print(f"Using Rx-estimated theta from: {theta_path}")
with np.load(theta_path) as data:
    theta_hat_matrix = data["theta_hat_matrix"]
theta_hat_flat = theta_hat_matrix.reshape(-1)

n = n1 * m
print(f"n1={n1}, m={m}, n={n}")
print(f"Test samples: {NUM_TEST_SAMPLES}")
print(f"Rx angle: {rx_angle:.4f}")
print()

# ---- Inference WITHOUT Rx ----
print("Generating y WITHOUT Rx...")
y_no_rx = []
for bits in tqdm(x_test_bits, desc="No Rx"):
    _, y = DQNN_generate_y_rx(bits, n1, m, theta_hat_flat, rx_angle=0.0)
    y_no_rx.append(y)
y_no_rx = np.array(y_no_rx)

# ---- Inference WITH Rx(pi/4) ----
print(f"Generating y WITH Rx({rx_angle:.4f})...")
y_with_rx = []
for bits in tqdm(x_test_bits, desc="With Rx"):
    _, y = DQNN_generate_y_rx(bits, n1, m, theta_hat_flat, rx_angle=rx_angle)
    y_with_rx.append(y)
y_with_rx = np.array(y_with_rx)

# ---- Analysis ----
print()
print("=" * 60)
print("  Results")
print("=" * 60)

# Per-bit agreement with test labels
y_target = y_test_encoded.reshape(NUM_TEST_SAMPLES, -1)
acc_no_rx = np.mean(y_no_rx == y_target)
acc_with_rx = np.mean(y_with_rx == y_target)
print(f"  Bit-level accuracy (no Rx):   {acc_no_rx:.4f}")
print(f"  Bit-level accuracy (with Rx):  {acc_with_rx:.4f}")

# Check if output distribution changes
# Average Hamming distance between outputs
from scipy.spatial.distance import hamming
avg_hamm_no_rx = np.mean([hamming(y_target[i], y_no_rx[i]) for i in range(NUM_TEST_SAMPLES)])
avg_hamm_with_rx = np.mean([hamming(y_target[i], y_with_rx[i]) for i in range(NUM_TEST_SAMPLES)])
print(f"  Avg Hamming dist (no Rx):     {avg_hamm_no_rx:.4f}")
print(f"  Avg Hamming dist (with Rx):    {avg_hamm_with_rx:.4f}")

# Per-qubit marginal difference
marginals_no = np.mean(y_no_rx, axis=0)
marginals_rx = np.mean(y_with_rx, axis=0)
marginals_target = np.mean(y_target, axis=0)
print(f"  |dP| per qubit (max):           {np.max(np.abs(marginals_rx - marginals_no)):.4f}")
print(f"  |dP| per qubit (mean):          {np.mean(np.abs(marginals_rx - marginals_no)):.4f}")

# Spatial correlation change
corr_no = np.corrcoef(y_no_rx.T)
corr_rx = np.corrcoef(y_with_rx.T)
corr_target = np.corrcoef(y_target.T)
corr_diff = np.mean(np.abs(corr_rx - corr_no))
corr_no_target = np.mean(np.abs(corr_no - corr_target))
corr_rx_target = np.mean(np.abs(corr_rx - corr_target))
print(f"  Avg |corr_diff| (Rx vs no-Rx):  {corr_diff:.6f}")
print(f"  Avg |corr to target| (no Rx):   {corr_no_target:.6f}")
print(f"  Avg |corr to target| (with Rx): {corr_rx_target:.6f}")

print()
if avg_hamm_with_rx < avg_hamm_no_rx:
    print(f"  [+] Rx IMPROVED Hamming distance ({(avg_hamm_no_rx - avg_hamm_with_rx):.4f})")
else:
    print(f"  [-] Rx did NOT improve Hamming distance")

if corr_rx_target < corr_no_target:
    print(f"  [+] Rx IMPROVED correlation to target")
else:
    print(f"  [-] Rx did NOT improve correlation to target")

# Save results
output_npz = RX_DIR / "data" / f"mnist_quick_rx_pi4_N{NUM_TEST_SAMPLES}.npz"
np.savez(
    output_npz,
    y_no_rx=y_no_rx,
    y_with_rx=y_with_rx,
    y_target=y_target,
    acc_no_rx=acc_no_rx,
    acc_with_rx=acc_with_rx,
    avg_hamm_no_rx=avg_hamm_no_rx,
    avg_hamm_with_rx=avg_hamm_with_rx,
    rx_angle=rx_angle,
    n1=n1, m=m, n=n,
    NUM_TEST_SAMPLES=NUM_TEST_SAMPLES,
)
print(f"\nSaved results to: {output_npz}")
