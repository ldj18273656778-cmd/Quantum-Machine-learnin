import numpy as np
from DQNN_generate_y import DQNN_generate_y
from ISQNN_generate_y import ISQNN_generate_y
import random

print("Starting simplified main.py...")

random.seed(42)
bitstring = "111111111111"  
n1 = 3
m = 4
n = n1 * m
theta = [random.uniform(0, 1)*np.pi for _ in range(n)]
print(f"Generated theta list with {len(theta)} elements")

print("Calling DQNN_generate_y...")
try:
    circuit, y = DQNN_generate_y(bitstring, n1, m, theta)
    print("DQNN_generate_y call completed")
    print("Generated y:", y)
except Exception as e:
    print(f"DQNN error: {e}")

print("Calling ISQNN_generate_y...")
try:
    circuit, y1 = ISQNN_generate_y(bitstring, n1, m, theta)
    print("ISQNN_generate_y call completed")
    print("Generated y1:", y1)
except Exception as e:
    print(f"ISQNN error: {e}")

print("All done!")