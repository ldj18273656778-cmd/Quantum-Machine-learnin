import os
import random
import numpy as np
import matplotlib.pyplot as plt
from probility_distribution import generate_probability_distribution


def total_variation_distance(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p_sum = np.sum(p)
    q_sum = np.sum(q)
    if p_sum == 0 or q_sum == 0:
        raise ValueError("Input distributions must have positive total mass.")

    p = p / p_sum
    q = q / q_sum
    return 0.5 * np.sum(np.abs(p - q))


<<<<<<< HEAD
seed = 44
=======
seed = 42
>>>>>>> 0d5e358849638861c71eeab1e1643ea0d2712d23
num_samples = 8000
output_dir = "output_images"
bin_width = 20

random.seed(seed)
bitstring = "100010110001"  # n = n1*m #在单层的情况下，二者输出的y分布几乎相同；没有bug.
n1 = 3
m = 4
# bitstring = "0000"
# n1 = 2
# m = 2
n = n1 * m
theta_test = [random.uniform(0, 1) * np.pi for _ in range(n)]

dqnn_samples, isqnn_samples, dqnn_hist, isqnn_hist, bins = generate_probability_distribution(
    bitstring, n1, m, theta_test, num_samples=num_samples, bin_width=bin_width
)

tvd = total_variation_distance(dqnn_hist, isqnn_hist)

print("DQNN sample decimal y:", dqnn_samples[:20], "...")
print("ISQNN sample decimal y:", isqnn_samples[:20], "...")
print("DQNN hist:", dqnn_hist)
print("ISQNN hist:", isqnn_hist)
print("TVD:", tvd)

fig, ax = plt.subplots(figsize=(8.8, 4.8))
ax.hist(dqnn_samples, bins=bins, alpha=0.6, label="DQNN")
ax.hist(isqnn_samples, bins=bins, alpha=0.6, label="ISQNN")
ax.set_xlabel("y (decimal)")
ax.set_ylabel("count")
ax.set_title(
    f"DQNN/ISQNN Histogram (Decimal) for bitstring: {bitstring}\nTVD={tvd:.6f}"
)
ax.legend()
ax.grid(True)
ax.text(
    0.98,
    0.98,
    f"samples-number: {num_samples}\nrandom-seed: {seed}",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9,
    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
)
fig.tight_layout()

os.makedirs(output_dir, exist_ok=True)
filename = f"seed_{seed}_samples_{num_samples}_bitstring_{bitstring}_tvd_{tvd:.6f}.png"
save_path = os.path.join(output_dir, filename)
fig.savefig(save_path, dpi=300, bbox_inches="tight")
print("Saved figure to:", save_path)

plt.show()
