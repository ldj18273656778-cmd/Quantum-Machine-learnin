import random
import numpy as np

from sampling.DQNN_generate_y import DQNN_generate_y
from sampling.ISQNN_generate_y import ISQNN_generate_y


if __name__ == "__main__":
    print("Starting Train/main.py...")

    bitstring = "00001111000000110000"
    n1 = 4
    m = 5
    n = n1 * m
    theta = [random.uniform(0, 1) * np.pi for _ in range(n)]
    print(f"Generated theta list with {len(theta)} elements")

    print("Calling DQNN_generate_y...")
    _, y = DQNN_generate_y(bitstring, n1, m, theta)
    print("DQNN_generate_y call completed")
    print("Theta values:", theta[:5], "...")
    print("Generated y:", y)

    print("Calling ISQNN_generate_y...")
    _, y1 = ISQNN_generate_y(bitstring, n1, m, theta)
    print("ISQNN_generate_y call completed")
    print("Theta values:", theta[:5], "...")
    print("Generated y:", y1)
