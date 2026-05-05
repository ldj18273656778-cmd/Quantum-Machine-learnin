"""Estimate Rx-modified ISQNN theta parameters from the LPN training dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import Learning_Parity_with_Noise.config as config
from Learning_Parity_with_Noise.estimate_theta_lpn import rows_to_bitstrings
from Rx_modified.ISQNN_generate_y_rx import idqnn_connectivity
from Rx_modified.estimate_theta_rx import estimate_theta_rx
from Train.find_x_indices_by_graph_condition import build_adjacency


def estimate_theta_from_training_data_rx(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    n1: int,
    m: int,
    rx_angle: float,
) -> dict[str, Any]:
    """Estimate all Rx-modified ISQNN theta parameters from training samples."""
    X_train = np.asarray(X_train, dtype=np.uint8)
    Y_train = np.asarray(Y_train, dtype=np.uint8)
    n_bits = n1 * m

    if X_train.ndim != 2 or X_train.shape[1] != n_bits:
        raise ValueError(
            f"X_train must have shape (num_train, {n_bits}), got {X_train.shape}."
        )
    if Y_train.ndim != 2 or Y_train.shape != X_train.shape:
        raise ValueError(
            "Y_train must be 2D and have the same shape as X_train; "
            f"got X_train={X_train.shape}, Y_train={Y_train.shape}."
        )

    x_bitstrings = rows_to_bitstrings(X_train)
    graph = idqnn_connectivity(n1, m)
    adjacency = build_adjacency(n=n_bits, edges=graph["all_edges"])

    theta_hat_flat = np.full(n_bits, np.nan, dtype=float)
    records: list[dict[str, Any]] = []
    failed_bits: list[int] = []

    for target_bit in range(n_bits):
        try:
            result = estimate_theta_rx(
                x=x_bitstrings,
                y=Y_train,
                target_bit=target_bit,
                adjacency=adjacency,
                rx_angle=rx_angle,
                show_progress=False,
            )
            theta_hat_flat[target_bit] = result["theta_hat_rad"]
            records.append(
                {
                    "target_bit_0based": int(target_bit),
                    "target_bit_1based": int(target_bit + 1),
                    "neighbors_0based": np.asarray(sorted(adjacency[target_bit]), dtype=int),
                    "N_sp": int(result["N_sp"]),
                    "sum_y": int(result["sum_y"]),
                    "p_hat": float(result["p_hat"]),
                    "alpha_d": float(result["alpha_d"]),
                    "n_neighbors": int(result["n_neighbors"]),
                    "rx_angle": float(result["rx_angle"]),
                    "theta_hat_rad": float(result["theta_hat_rad"]),
                    "theta_hat_deg": float(result["theta_hat_deg"]),
                    "indices_0based": np.asarray(result["indices_0based"], dtype=int),
                    "status": "ok",
                }
            )
        except ValueError as exc:
            failed_bits.append(target_bit)
            records.append(
                {
                    "target_bit_0based": int(target_bit),
                    "target_bit_1based": int(target_bit + 1),
                    "neighbors_0based": np.asarray(sorted(adjacency[target_bit]), dtype=int),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    return {
        "theta_hat_flat_rad": theta_hat_flat,
        "theta_hat_matrix_rad": theta_hat_flat.reshape(n1, m),
        "theta_hat_flat_deg": np.degrees(theta_hat_flat),
        "theta_hat_matrix_deg": np.degrees(theta_hat_flat.reshape(n1, m)),
        "records": np.asarray(records, dtype=object),
        "failed_bits_0based": np.asarray(failed_bits, dtype=int),
        "n1": int(n1),
        "m": int(m),
        "n_bits": int(n_bits),
        "rx_angle": float(rx_angle),
        "model": "rx_modified_isqnn",
    }


def save_theta_estimates_rx(estimates: dict[str, Any], output_path: Path) -> None:
    """Save Rx theta estimates to an npz file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **estimates)


def print_estimation_summary_rx(estimates: dict[str, Any], output_path: Path) -> None:
    """Print a concise summary of Rx theta estimation results."""
    theta = np.asarray(estimates["theta_hat_flat_rad"])
    failed = np.asarray(estimates["failed_bits_0based"])

    print("Rx-modified ISQNN theta estimation from LPN training data")
    print(f"saved: {output_path}")
    print(f"n1={estimates['n1']}, m={estimates['m']}, n_bits={estimates['n_bits']}")
    print(f"rx_angle={estimates['rx_angle']}")
    print(f"estimated bits: {int(np.sum(~np.isnan(theta)))}/{theta.size}")
    print(f"failed bits: {failed.tolist()}")
    print("theta_hat_matrix_rad:")
    print(estimates["theta_hat_matrix_rad"])


def main() -> None:
    config.validate_config()
    config.ensure_directories()

    if not config.DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {config.DATASET_PATH}. Run generate_lpn_dataset.py first."
        )

    dataset = np.load(config.DATASET_PATH, allow_pickle=True)
    estimates = estimate_theta_from_training_data_rx(
        X_train=dataset["X_train"],
        Y_train=dataset["Y_train_noisy"],
        n1=int(dataset["n1"]),
        m=int(dataset["m"]),
        rx_angle=config.rx_angle,
    )
    estimates.update(
        {
            "dataset_path": str(config.DATASET_PATH),
            "theta_path": str(config.THETA_RX_PATH),
            "n_x": int(dataset["n_x"]),
            "n_y": int(dataset["n_y"]),
            "num_train": int(dataset["num_train"]),
            "noise_rate": float(dataset["noise_rate"]),
            "seed": int(dataset["seed"]),
        }
    )

    save_theta_estimates_rx(estimates, config.THETA_RX_PATH)
    print_estimation_summary_rx(estimates, config.THETA_RX_PATH)


if __name__ == "__main__":
    main()
