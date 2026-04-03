from __future__ import annotations

"""Estimate one parameter theta_{i,j} from filtered samples.

流程：
1) 筛选满足 x[target_bit]=0 且邻域位全为1 的样本索引；
2) 对应读取 y[target_bit]，按
   theta_hat = 1/2 * arccos(1 - 2/N_sp * sum(y_t))
   估计参数。
"""

from pathlib import Path
import sys

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from sampling.ISQNN_generate_y import idqnn_connectivity
from Train.find_x_indices_by_graph_condition import build_adjacency, find_indices


def load_xy_from_npy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    data = obj.item() if isinstance(obj, np.ndarray) and obj.shape == () else obj

    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a dict with keys 'x' and 'y'.")
    if "x" not in data or "y" not in data:
        raise ValueError(f"{path} must contain keys 'x' and 'y'.")

    x = np.asarray(data["x"]).astype(str)
    y = np.asarray(data["y"])  # shape: (N, n)

    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape={x.shape}")
    if y.ndim != 2:
        raise ValueError(f"y must be 2D, got shape={y.shape}")
    if len(x) != y.shape[0]:
        raise ValueError(f"x/y sample size mismatch: len(x)={len(x)}, y.shape[0]={y.shape[0]}")

    return x, y


def estimate_theta_from_filtered_samples(
    x: np.ndarray,
    y: np.ndarray,
    target_bit: int,
    adjacency: list[set[int]],
    show_progress: bool = True,
) -> dict:
    indices = find_indices(
        x=x,
        target_bit=target_bit,
        adjacency=adjacency,
        show_progress=show_progress,
    )

    n_sp = int(len(indices))
    if n_sp == 0:
        raise ValueError("N_sp = 0，无法估计参数。")

    if not (0 <= target_bit < y.shape[1]):
        raise ValueError(f"target_bit out of range for y: {target_bit}, y.shape={y.shape}")

    y_selected = y[indices, target_bit]
    if not np.isin(y_selected, [0, 1]).all():
        raise ValueError("y_selected 必须是二值 {0,1}。")

    y_sum = int(np.sum(y_selected))
    arg = 1.0 - 2.0 * y_sum / n_sp
    arg = float(np.clip(arg, -1.0, 1.0))
    theta_hat =  float(np.arccos(arg))#没有*0.5；文章又搞错了

    return {
        "target_bit_0based": int(target_bit),
        "N_sp": n_sp,
        "sum_y": y_sum,
        "theta_hat_rad": theta_hat,
        "theta_hat_deg": float(np.degrees(theta_hat)),
        "indices_0based": indices,
    }


if __name__ == "__main__":
    # ===== 手动参数区 =====
    input_path = ROOT / "code" / "Train" / "data" / "xy_dataset.npy"
    output_path = ROOT / "code" / "Train" / "data" / "theta_estimate_all_bits.npy"

    n1 = 3
    m = 4
    target_bit_1based = 5
    # ====================

    x, y = load_xy_from_npy(input_path)
    n = len(x[0])

    G = idqnn_connectivity(n1, m)
    if G["n"] != n:
        raise ValueError(f"Graph size mismatch: G.n={G['n']}, bitstring length={n}")

    adjacency = build_adjacency(n=n, edges=G["all_edges"])

    if n1 * m != n:
        raise ValueError(f"n1*m must equal n, got n1*m={n1*m}, n={n}")

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records_serializable: list[dict] = []
    for r in records:
        bit = int(r["target_bit_0based"])
        records_serializable.append(
            {
                "target_bit_1based": bit + 1,
                "target_bit_0based": bit,
                "neighbors_0based": sorted(adjacency[bit]),
                "N_sp": int(r["N_sp"]),
                "sum_y": int(r["sum_y"]),
                "theta_hat_rad": float(r["theta_hat_rad"]),
                "theta_hat_deg": float(r["theta_hat_deg"]),
                "indices_0based": np.asarray(r["indices_0based"], dtype=int),
            }
        )

    output_data = {
        "input_path": str(input_path),
        "n1": int(n1),
        "m": int(m),
        "n": int(n),
        "theta_hat_matrix_rad": theta_hat_matrix,
        "theta_hat_matrix_deg": np.degrees(theta_hat_matrix),
        "records": np.asarray(records_serializable, dtype=object),
    }
    np.save(output_path, output_data, allow_pickle=True)

    print("=== Done ===")
    print(f"Input: {input_path}")
    print("theta_hat matrix (rad):")
    print(theta_hat_matrix)
    print(f"Output: {output_path}")
