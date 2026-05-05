"""Generate ISQNN predictions for the LPN test inputs using learned theta."""

from __future__ import annotations

from typing import Iterable

import numpy as np

import Learning_Parity_with_Noise.config as config
from Learning_Parity_with_Noise.estimate_theta_lpn import rows_to_bitstrings
from sampling.ISQNN_generate_y import ISQNN_generate_y

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def generate_predictions(
    X_test: np.ndarray,
    theta: np.ndarray,
    n1: int,
    m: int,
    show_progress: bool = True,
) -> np.ndarray:
    """Run original ISQNN on every test input and return Y_pred."""
    X_test = np.asarray(X_test, dtype=np.uint8)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    n_bits = n1 * m

    if X_test.ndim != 2 or X_test.shape[1] != n_bits:
        raise ValueError(f"X_test must have shape (num_test, {n_bits}), got {X_test.shape}.")
    if theta.shape[0] != n_bits:
        raise ValueError(f"theta must have length {n_bits}, got {theta.shape[0]}.")
    if np.isnan(theta).any():
        raise ValueError("theta contains NaN. Check failed bits in theta_lpn.npz first.")

    x_bitstrings = rows_to_bitstrings(X_test)
    Y_pred = np.zeros((len(x_bitstrings), n_bits), dtype=np.uint8)

    iterator: Iterable[tuple[int, str]] = enumerate(x_bitstrings)
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, total=len(x_bitstrings), desc="ISQNN test inference", unit="sample")

    theta_list = theta.tolist()
    for sample_idx, bitstring in iterator:
        _, y = ISQNN_generate_y(bitstring=bitstring, n1=n1, m=m, theta_list=theta_list)
        Y_pred[sample_idx] = np.asarray(y, dtype=np.uint8)

    return Y_pred


def save_predictions(dataset, theta_data, Y_pred: np.ndarray) -> None:
    """Save ISQNN test predictions with test targets for later evaluation."""
    config.PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        config.PREDICTION_PATH,
        X_test=dataset["X_test"],
        Y_pred=Y_pred,
        Y_test_clean=dataset["Y_test_clean"],
        Y_test_noisy=dataset["Y_test_noisy"],
        S=dataset["S"],
        theta_hat_flat_rad=theta_data["theta_hat_flat_rad"],
        theta_hat_matrix_rad=theta_data["theta_hat_matrix_rad"],
        n_x=int(dataset["n_x"]),
        n_y=int(dataset["n_y"]),
        n1=int(theta_data["n1"]),
        m=int(theta_data["m"]),
        n_bits=int(theta_data["n_bits"]),
        num_test=int(dataset["num_test"]),
        noise_rate=float(dataset["noise_rate"]),
        seed=int(dataset["seed"]),
        model="original_isqnn",
    )


def print_prediction_summary(Y_pred: np.ndarray) -> None:
    """Print a concise summary after inference finishes."""
    print("ISQNN test inference finished")
    print(f"Y_pred shape: {Y_pred.shape}")
    print(f"saved: {config.PREDICTION_PATH}")


def main() -> None:
    config.validate_config()
    config.ensure_directories()

    if not config.DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {config.DATASET_PATH}. Run generate_lpn_dataset.py first."
        )
    if not config.THETA_PATH.exists():
        raise FileNotFoundError(
            f"Theta file not found: {config.THETA_PATH}. Run estimate_theta_lpn.py first."
        )

    dataset = np.load(config.DATASET_PATH, allow_pickle=True)
    theta_data = np.load(config.THETA_PATH, allow_pickle=True)

    Y_pred = generate_predictions(
        X_test=dataset["X_test"],
        theta=theta_data["theta_hat_flat_rad"],
        n1=int(theta_data["n1"]),
        m=int(theta_data["m"]),
        show_progress=True,
    )
    save_predictions(dataset, theta_data, Y_pred)
    print_prediction_summary(Y_pred)


if __name__ == "__main__":
    main()
