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

seed = 42
num_samples = 3000
output_dir = "output_images"

random.seed(seed)
# bitstring = "100010110001"  # n = n1*m #在单层的情况下，二者输出的y分布几乎相同；没有bug.
# n1 = 1
# m = 12
bitstring = "0000" 
n1=2
m=2
n = n1 * m
theta_test = [random.uniform(0, 1) * np.pi for _ in range(n)]

dqnn_samples, isqnn_samples, dqnn_hist, isqnn_hist, bins = generate_probability_distribution(
    bitstring, n1, m, theta_test, num_samples=num_samples, bin_width=1
)

tvd = total_variation_distance(dqnn_hist, isqnn_hist)

print("DQNN sample decimal y:", dqnn_samples[:20], "...")
print("ISQNN sample decimal y:", isqnn_samples[:20], "...")
print("DQNN hist:", dqnn_hist)
print("ISQNN hist:", isqnn_hist)
print("TVD:", tvd)

plt.figure(figsize=(8, 4))
plt.hist(dqnn_samples, bins=bins, alpha=0.6, label="DQNN")
plt.hist(isqnn_samples, bins=bins, alpha=0.6, label="ISQNN")
plt.xlabel("y (decimal)")
plt.ylabel("count")
plt.title(f"DQNN/ISQNN Histogram (Decimal) for bitstring: {bitstring}\nTVD={tvd:.6f}")
plt.legend()
plt.grid(True)

os.makedirs(output_dir, exist_ok=True)
filename = f"seed_{seed}_samples_{num_samples}_bitstring_{bitstring}_tvd_{tvd:.6f}.png"
save_path = os.path.join(output_dir, filename)
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print("Saved figure to:", save_path)

plt.show()
