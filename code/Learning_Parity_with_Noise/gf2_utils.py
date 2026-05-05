"""Utilities for matrix Learning Parity with Noise over GF(2)."""

from __future__ import annotations

from typing import Any

import numpy as np


BinaryArray = np.ndarray


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy random generator."""
    return np.random.default_rng(seed)


def as_binary_array(values: Any, *, name: str = "array") -> BinaryArray:
    """Convert values to a uint8 binary array and reject non-binary entries."""
    arr = np.asarray(values, dtype=np.uint8)
    if np.any((arr != 0) & (arr != 1)):
        raise ValueError(f"{name} must contain only 0/1 values.")
    return arr


def sample_secret(n_x: int, n_y: int, rng: np.random.Generator) -> BinaryArray:
    """Sample a secret matrix S in {0,1}^{n_x x n_y}."""
    _validate_positive_int(n_x, "n_x")
    _validate_positive_int(n_y, "n_y")
    return rng.integers(0, 2, size=(n_x, n_y), dtype=np.uint8)


def sample_inputs(num_samples: int, n_x: int, rng: np.random.Generator) -> BinaryArray:
    """Sample input matrix X in {0,1}^{num_samples x n_x}."""
    _validate_positive_int(num_samples, "num_samples")
    _validate_positive_int(n_x, "n_x")
    return rng.integers(0, 2, size=(num_samples, n_x), dtype=np.uint8)


def gf2_matmul(x: Any, s: Any) -> BinaryArray:
    """Compute matrix product x @ S over GF(2)."""
    x_arr = as_binary_array(x, name="x")
    s_arr = as_binary_array(s, name="s")
    if x_arr.ndim != 2 or s_arr.ndim != 2:
        raise ValueError("gf2_matmul expects two 2D arrays.")
    if x_arr.shape[1] != s_arr.shape[0]:
        raise ValueError(
            "Incompatible matrix shapes for multiplication: "
            f"{x_arr.shape} and {s_arr.shape}."
        )
    return ((x_arr @ s_arr) % 2).astype(np.uint8)


def generate_labels(x: Any, s: Any) -> BinaryArray:
    """Generate clean matrix-LPN labels Y = X S mod 2."""
    return gf2_matmul(x, s)


def add_bernoulli_noise(
    labels: Any,
    noise_rate: float,
    rng: np.random.Generator,
) -> tuple[BinaryArray, BinaryArray]:
    """Flip each label bit independently with probability noise_rate.

    Returns (noisy_labels, noise_matrix).
    """
    clean = as_binary_array(labels, name="labels")
    if not 0.0 <= noise_rate <= 1.0:
        raise ValueError(f"noise_rate must be in [0, 1]; got {noise_rate}.")
    noise = (rng.random(size=clean.shape) < noise_rate).astype(np.uint8)
    noisy = np.bitwise_xor(clean, noise).astype(np.uint8)
    return noisy, noise


def generate_lpn_split(
    num_samples: int,
    s: Any,
    noise_rate: float,
    rng: np.random.Generator,
) -> dict[str, BinaryArray]:
    """Generate one matrix-LPN split using a fixed secret matrix."""
    s_arr = as_binary_array(s, name="s")
    if s_arr.ndim != 2:
        raise ValueError("s must be a 2D secret matrix.")
    x = sample_inputs(num_samples, s_arr.shape[0], rng)
    y_clean = generate_labels(x, s_arr)
    y_noisy, noise = add_bernoulli_noise(y_clean, noise_rate, rng)
    return {
        "x": x,
        "y_clean": y_clean,
        "y_noisy": y_noisy,
        "noise": noise,
    }


def bit_accuracy(predicted: Any, target: Any) -> float:
    """Return mean bitwise agreement between two binary arrays."""
    pred = as_binary_array(predicted, name="predicted")
    true = as_binary_array(target, name="target")
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: predicted {pred.shape}, target {true.shape}.")
    return float(np.mean(pred == true))


def sample_accuracy(predicted: Any, target: Any) -> float:
    """Return fraction of rows whose full output bitstring is correct."""
    pred = as_binary_array(predicted, name="predicted")
    true = as_binary_array(target, name="target")
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: predicted {pred.shape}, target {true.shape}.")
    if pred.ndim != 2:
        raise ValueError("sample_accuracy expects 2D arrays.")
    return float(np.mean(np.all(pred == true, axis=1)))


def hamming_distance(predicted: Any, target: Any) -> BinaryArray:
    """Return row-wise Hamming distance between two binary matrices."""
    pred = as_binary_array(predicted, name="predicted")
    true = as_binary_array(target, name="target")
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: predicted {pred.shape}, target {true.shape}.")
    if pred.ndim != 2:
        raise ValueError("hamming_distance expects 2D arrays.")
    return np.sum(pred != true, axis=1)


def _validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
