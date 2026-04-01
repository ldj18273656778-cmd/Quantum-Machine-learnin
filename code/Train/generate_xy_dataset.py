from __future__ import annotations

"""Generate (x_i, y_i)^N dataset using fixed theta and input distribution D(x).

- x is sampled from D(x) (see generate_Dx.py).
- y is sampled from ISQNN with fixed theta (see sampling/ISQNN_generate_y.py).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


from Train.generate_Dx import sample_dx
from sampling.ISQNN_generate_y import ISQNN_generate_y


def load_theta(theta_path: Path, n1: int, m: int) -> np.ndarray:
    theta = np.load(theta_path)
    if theta.shape != (n1, m):
        raise ValueError(f"theta shape must be ({n1}, {m}), got {theta.shape}.")
    return theta.astype(float)


def generate_xy_dataset(
    n1: int,
    m: int,
    theta: np.ndarray,
    num_samples: int = 1000,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (x,y)^N with fixed theta and D(x).

    Returns:
        x: shape (N,) array of bitstrings (dtype str)
        y: shape (N, n1*m) array of bits (dtype int8)
        comps: shape (N,) component id for D(x)
    """
    n = n1 * m
    if theta.shape != (n1, m):
        raise ValueError(f"theta shape must be ({n1}, {m}), got {theta.shape}.")

    theta_list = theta.reshape(-1).tolist()

    x, comps = sample_dx(n_bits=n, num_samples=num_samples, seed=seed)
    y = np.zeros((num_samples, n), dtype=np.int8)
    for i, xi in enumerate(x.tolist()):
        _, yi = ISQNN_generate_y(xi, n1, m, theta_list)
        y[i] = np.asarray(yi, dtype=np.int8)

    return x, y, comps


if __name__ == "__main__":
    # ======= 手动修改这里的参数 =======
    n1 = 3
    m = 4
    num_samples = 1000
    seed = 7
    theta_path = ROOT / "code" / "Train" / "data" / "theta_demo.npy"
    out_path = ROOT / "code" / "Train" / "data" / "xy_dataset.txt"
    out_npy_path = ROOT / "code" / "Train" / "data" / "xy_dataset.npy"
    # ================================

    n = n1 * m
    theta = load_theta(theta_path, n1, m)
    x, y, comps = generate_xy_dataset(
        n1=n1,
        m=m,
        theta=theta,
        num_samples=num_samples,
        seed=seed,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"n1={n1}, m={m}, n={n}, N={num_samples}\n")
        f.write(f"seed={seed}\n")
        f.write(f"theta_path={theta_path}\n")
        f.write("comp\tx\ty\n")
        for comp, xi, yi in zip(comps.tolist(), x.tolist(), y.tolist()):
            y_str = "".join(str(int(b)) for b in yi)
            f.write(f"{int(comp)}\t{xi}\t{y_str}\n")

    data = {
        "x": x,
        "y": y,
        "comps": comps,
        "theta": theta,
        "n1": n1,
        "m": m,
        "seed": seed,
    }
    np.save(out_npy_path, data, allow_pickle=True)

    print("=== (x,y)^N dataset generated ===")
    print(f"output: {out_path}")
    print(f"output (npy): {out_npy_path}")
    print(f"n1={n1}, m={m}, n={n}, N={num_samples}")
    print(f"theta: {theta_path}")
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    print("first 3 x:", x[:3].tolist())
    print("first 3 y:", ["".join(str(int(b)) for b in row) for row in y[:3]])
