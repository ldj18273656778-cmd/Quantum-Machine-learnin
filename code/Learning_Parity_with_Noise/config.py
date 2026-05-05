"""Shared configuration for matrix Learning Parity with Noise experiments.

The first implementation keeps the LPN input and output dimensions equal to
the number of ISQNN bits so that each LPN sample can be used directly as an
ISQNN input/output bitstring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# Project paths.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_IMAGE_DIR = PROJECT_DIR / "output_images"

DATASET_PATH = DATA_DIR / "lpn_dataset.npz"
THETA_PATH = DATA_DIR / "theta_lpn.npz"
THETA_RX_PATH = DATA_DIR / "theta_lpn_rx.npz"
PREDICTION_PATH = DATA_DIR / "lpn_test_predictions.npz"
PREDICTION_RX_PATH = DATA_DIR / "lpn_test_predictions_rx.npz"


# Matrix-LPN dimensions: x in {0,1}^{n_x}, S in {0,1}^{n_x x n_y},
# y = x S XOR e. For a dataset, X in {0,1}^{num_samples x n_x} and
# Y = X S XOR E.
n_x = 16
n_y = 16


# ISQNN dimensions. First-pass direct mapping requires n_x == n_y == n1 * m.
n1 = 4
m = 4
n_bits = n1 * m


# Dataset parameters.
num_train = 60000
num_test = 10000
noise_rate = 0.0
seed = 47


# Rx-modified ISQNN parameter.
rx_angle = np.pi / 4


def ensure_directories() -> None:
    """Create output directories used by the LPN pipeline."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def validate_config() -> None:
    """Validate the default direct-mapping setup."""
    if n_x != n_bits or n_y != n_bits:
        raise ValueError(
            "First-pass LPN-to-ISQNN mapping requires n_x == n_y == n1 * m; "
            f"got n_x={n_x}, n_y={n_y}, n1*m={n_bits}."
        )
    if not 0.0 <= noise_rate < 0.5:
        raise ValueError(f"noise_rate must satisfy 0 <= eta < 0.5; got {noise_rate}.")
    if num_train <= 0 or num_test <= 0:
        raise ValueError("num_train and num_test must be positive.")
