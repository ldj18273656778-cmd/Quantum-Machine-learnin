"""Visualize MNIST inference: input image vs inferred label vs true label.

Usage: python visualize_inference.py

Displays a few test samples side by side without saving files.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
import matplotlib.pyplot as plt

from DQNN_generate_y_rx import DQNN_generate_y_rx
from MNIST.Encode_0to9 import encode_diagonal

# ===== 手动参数区 =====
NUM_SHOW = 8          # 显示样本数
rx_angle = np.pi / 4  # 推理用 Rx 角
threshold = 0.4        # 训练二值化阈值
n1, m = 10, 10
n = n1 * m
# ====================

RX_DIR = Path(__file__).resolve().parent
MNIST_DIR = ROOT / "code" / "MNIST"

# Load theta (prefer Rx_modified estimate, fallback to original)
theta_path = RX_DIR / "data" / f"estimate_theta_rx_pi4_binarized{threshold}.npz"
if not theta_path.exists():
    theta_path = MNIST_DIR / "data" / f"estimate_theta_binarized{threshold}.npz"
with np.load(theta_path) as data:
    theta_hat_matrix = data["theta_hat_matrix"]
theta_hat_flat = theta_hat_matrix.reshape(-1)

# Load test data
X_test = np.load(MNIST_DIR / "data" / "test_MNIST_10x10_binarize0.5.npy")
Y_test = np.load(MNIST_DIR / "data" / "test_MNIST_labels.npy")
X_test = X_test[:NUM_SHOW].reshape(NUM_SHOW, n1, m)
Y_test = Y_test[:NUM_SHOW]

# Flip (MNIST has black background=0, we need white=1 on black=0)
X_flip = 1 - X_test

# Run inference
print(f"Running inference for {NUM_SHOW} samples (Rx={rx_angle:.4f})...")
y_inferred_flat = []
for i in range(NUM_SHOW):
    bits = "".join(map(str, X_flip[i].reshape(-1)))
    _, y = DQNN_generate_y_rx(bits, n1, m, theta_hat_flat, rx_angle=rx_angle)
    y_inferred_flat.append(y)
y_inferred = np.array(y_inferred_flat).reshape(NUM_SHOW, n1, m)

# Encode true labels
y_true_encoded = np.array([encode_diagonal(label) for label in Y_test])

# Plot
fig, axes = plt.subplots(NUM_SHOW, 3, figsize=(8, 2.5 * NUM_SHOW))
if NUM_SHOW == 1:
    axes = axes.reshape(1, -1)

col_titles = ["Input (flipped)", "Inferred label", "True label"]
for c, title in enumerate(col_titles):
    axes[0, c].set_title(title, fontsize=11, fontweight="bold")

for i in range(NUM_SHOW):
    # Input image
    axes[i, 0].imshow(X_flip[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])
    if i == 0:
        axes[i, 0].set_ylabel(f"  Sample", fontsize=10)

    # Inferred label
    axes[i, 1].imshow(y_inferred[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[i, 1].set_xticks([])
    axes[i, 1].set_yticks([])

    # True label
    axes[i, 2].imshow(y_true_encoded[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[i, 2].set_xticks([])
    axes[i, 2].set_yticks([])

    # Show digit label on left
    axes[i, 0].set_ylabel(f"  {Y_test[i]}", fontsize=12, rotation=0, labelpad=15)

plt.tight_layout()
plt.show()
