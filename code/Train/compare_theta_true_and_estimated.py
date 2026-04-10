from __future__ import annotations

"""Compare true theta and estimated theta.

默认比较：
- true theta: 从 xy_dataset.npy 中读取键 `theta`
- estimated theta: 从 theta_estimate_all_bits.npy 中读取键 `theta_hat_matrix_rad`
"""

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_true_theta(path: Path) -> np.ndarray:
    obj = np.load(path, allow_pickle=True)
    data = obj.item() if isinstance(obj, np.ndarray) and obj.shape == () else obj

    if isinstance(data, dict) and "theta" in data:
        theta = np.asarray(data["theta"], dtype=float)
        if theta.ndim != 2:
            raise ValueError(f"true theta must be 2D, got shape={theta.shape}")
        return theta

    # 兼容直接保存二维数组的情况
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 2:
        return arr

    raise ValueError(f"Cannot load true theta from {path}")


def load_estimated_theta(path: Path) -> np.ndarray:
    obj = np.load(path, allow_pickle=True)
    data = obj.item() if isinstance(obj, np.ndarray) and obj.shape == () else obj

    if isinstance(data, dict):
        if "theta_hat_matrix_rad" in data:
            theta_hat = np.asarray(data["theta_hat_matrix_rad"], dtype=float)
            if theta_hat.ndim != 2:
                raise ValueError(f"estimated theta must be 2D, got shape={theta_hat.shape}")
            return theta_hat
        if "theta_hat_matrix" in data:
            theta_hat = np.asarray(data["theta_hat_matrix"], dtype=float)
            if theta_hat.ndim != 2:
                raise ValueError(f"estimated theta must be 2D, got shape={theta_hat.shape}")
            return theta_hat

    # 兼容直接保存二维数组的情况
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 2:
        return arr

    raise ValueError(f"Cannot load estimated theta from {path}")


def compare_theta(theta_true: np.ndarray, theta_hat: np.ndarray) -> dict:
    if theta_true.shape != theta_hat.shape:
        raise ValueError(
            f"shape mismatch: true={theta_true.shape}, estimated={theta_hat.shape}"
        )

    d_cos = np.cos(theta_hat) - np.cos(theta_true)
    abs_err = np.abs(d_cos)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(d_cos**2)))
    max_abs = float(np.max(abs_err))

    return {
        "theta_true": theta_true,
        "theta_hat": theta_hat,
        "d_cos": d_cos,
        "abs_err": abs_err,
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
    }


if __name__ == "__main__":
    # ===== 手动参数区 =====
    true_theta_path = ROOT / "code" / "Train" / "data" / "xy_dataset.npy"
    estimated_theta_path = ROOT / "code" / "Train" / "data" / "theta_estimate_all_bits.npy"
    # ====================

    theta_true = load_true_theta(true_theta_path)
    theta_hat = load_estimated_theta(estimated_theta_path)

    result = compare_theta(theta_true, theta_hat)

    print("=== Theta comparison done ===")
    print(f"true_theta_path: {true_theta_path}")
    print(f"estimated_theta_path: {estimated_theta_path}")
    print(f"Shape: {result['theta_true'].shape}")
    print(f"MAE: {result['mae']}")
    print(f"RMSE: {result['rmse']}")
    print(f"MAX_ABS: {result['max_abs']}")
    print("\ntrue_theta(rad):")
    print(result["theta_true"])
    print("\nestimated_theta(rad):")
    print(result["theta_hat"])
    print("\nd_cos = cos(estimated) - cos(true):")
    print(result["d_cos"])
