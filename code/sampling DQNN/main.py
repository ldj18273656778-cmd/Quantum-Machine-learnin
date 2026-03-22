#from cirq_circuit import build_circuit
import numpy as np
from DQNN_generate_y import DQNN_generate_y
from ISQNN_generate_y import ISQNN_generate_y
import random


print("Starting main.py...")

# bitstring = "0011"
# theta = np.pi / 2
# layers = 1  # 可以调整层数

# circuit = build_circuit(bitstring, theta, layers, measure=True)

bitstring = "00001111000000110000"  # n = n1*m
n1 = 4
m = 5
n = n1 * m
theta = [random.uniform(0, 1)*np.pi for _ in range(n)]
print(f"Generated theta list with {len(theta)} elements")

# 可以使用标量theta（会自动扩展）或theta_list
print("Calling DQNN_generate_y...")
dqnn_circuit, y = DQNN_generate_y(bitstring, n1, m, theta)
print("DQNN_generate_y call completed")

print("Theta values:", theta[:5], "...")
print("Generated y:", y)
# print("\n=== DQNN 完整的量子电路 ===")
# print(dqnn_circuit)

print("Calling ISQNN_generate_y...")
isqnn_circuit, y1 = ISQNN_generate_y(bitstring, n1, m, theta)
print("ISQNN_generate_y call completed")

print("Theta values:", theta[:5], "...")
print("Generated y:", y1)