import sys
import time
import traceback
from pathlib import Path
from tqdm import tqdm

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from Train.generate_xy_dataset import load_theta, generate_xy_dataset
from sampling.ISQNN_generate_y import idqnn_connectivity
from Train.find_x_indices_by_graph_condition import build_adjacency, find_indices
from Train.estimate_theta_from_filtered_samples import estimate_theta_from_filtered_samples

def main() -> int:
    # ===== 调试参数区 =====
    n1 = 3
    m = 4
    num_samples = 10000
    seed = 7
    theta_path = ROOT / "code" / "Train" / "data" / "theta_demo.npy"
    # =====================

    t0 = time.perf_counter()
    print("=== test_train start ===")
    print(f"python: {sys.executable}")
    print(f"cwd: {Path.cwd()}")
    print(f"theta_path: {theta_path}")

    n = n1 * m
    print(f"n1={n1}, m={m}, n={n}, num_samples={num_samples}, seed={seed}")

    if not theta_path.exists():
        raise FileNotFoundError(f"theta file not found: {theta_path}")

    theta = load_theta(theta_path, n1, m)
    print(theta)
    print(f"theta loaded, shape={theta.shape}, dtype={theta.dtype}")

    x, y, comps = generate_xy_dataset(
        n1=n1,
        m=m,
        theta=theta,
        num_samples=num_samples,
        seed=seed,
    )

    print(f"x.shape={x.shape}, y.shape={y.shape}, comps.shape={comps.shape}")
    print("first 3 x:", x[:3].tolist())
    print("first 3 y:", ["".join(str(int(b)) for b in row) for row in y[:3]])
    print("first 3 comps:", comps[:3].tolist())
    print(f"elapsed={time.perf_counter() - t0:.3f}s")
    print("=== test_train done ===")

    
    G = idqnn_connectivity(n1, m)
    target_bit = 5
    adjacency = build_adjacency(n=n, edges=G["all_edges"])#生成每个比特的连通邻居列表
    neighbors = sorted(adjacency[target_bit])

    indices = find_indices(x=x, target_bit=target_bit, adjacency=adjacency)

    print(f"Neighbors (0-based): {neighbors}")
    print(f"Indices of x where target_bit is 1: {indices[:3]}")
    print(f"Adjacency list for target_bit {target_bit}: {adjacency[target_bit]}")

    theta_hat_flat = np.zeros(n, dtype=float)
    records: list[dict] = []

    bit_iter = range(n)
    if tqdm is not None:
        bit_iter = tqdm(bit_iter, total=n, desc="Estimating all thetas", unit="bit")

    for target_bit in bit_iter:
        result = estimate_theta_from_filtered_samples(
            x=x,
            y=y,
            target_bit=target_bit,
            adjacency=adjacency,
            show_progress=False,
        )
        theta_hat_flat[target_bit] = result["theta_hat_rad"]
        records.append(result)

    theta_hat_matrix = theta_hat_flat.reshape(n1, m)

    print("theta_hat matrix (rad):")
    print(theta_hat_matrix)
    return 0



if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("=== test_train error ===")
        traceback.print_exc()
        raise SystemExit(1)
