import random
import numpy as np
import matplotlib.pyplot as plt
from probility_distribution import generate_probability_distribution

random.seed(42)
# bitstring = "100010110001"  # n = n1*m #在单层的情况下，二者输出的y分布几乎相同；没有bug.
# n1 = 1
# m = 12
bitstring = "0001" 
n1=2
m=2
n = n1 * m
theta_test = [random.uniform(0, 1) * np.pi for _ in range(n)]

dqnn_samples, isqnn_samples, dqnn_hist, isqnn_hist, bins = generate_probability_distribution(
    bitstring, n1, m, theta_test, num_samples=3000, bin_width=1
)

print("DQNN sample decimal y:", dqnn_samples[:20], "...")
print("ISQNN sample decimal y:", isqnn_samples[:20], "...")
print("DQNN hist:", dqnn_hist)
print("ISQNN hist:", isqnn_hist)

plt.figure(figsize=(8, 4))
plt.hist(dqnn_samples, bins=bins, alpha=0.6, label="DQNN")
plt.hist(isqnn_samples, bins=bins, alpha=0.6, label="ISQNN")
plt.xlabel("y (decimal)")
plt.ylabel("count")
plt.title(f"DQNN/ISQNN Histogram (Decimal) for bitstring: {bitstring}")
plt.legend()
plt.grid(True)
# plt.savefig('dqnn_isqnn_histogram.png')
plt.show()