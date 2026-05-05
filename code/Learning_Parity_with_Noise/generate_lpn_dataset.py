"""Generate train/test data for matrix Learning Parity with Noise."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import Learning_Parity_with_Noise.config as config
from Learning_Parity_with_Noise.gf2_utils import (
    BinaryArray,
    generate_lpn_split,
    get_rng,
    sample_secret,
)


def generate_dataset(
    n_x: int,
    n_y: int,
    num_train: int,
    num_test: int,
    noise_rate: float,
    seed: int,
) -> dict[str, Any]:
    """Generate a full matrix-LPN dataset.

    The hidden secret matrix S is shared by train and test splits:
    Y_clean = X S mod 2, Y_noisy = Y_clean XOR noise.
    """
    rng = get_rng(seed)
    S = sample_secret(n_x, n_y, rng)

    train = generate_lpn_split(num_train, S, noise_rate, rng)
    test = generate_lpn_split(num_test, S, noise_rate, rng)

    return {
        "S": S,
        "X_train": train["x"],
        "Y_train_clean": train["y_clean"],
        "Y_train_noisy": train["y_noisy"],
        "train_noise": train["noise"],
        "X_test": test["x"],
        "Y_test_clean": test["y_clean"],
        "Y_test_noisy": test["y_noisy"],
        "test_noise": test["noise"],
        "n_x": n_x,
        "n_y": n_y,
        "num_train": num_train,
        "num_test": num_test,
        "noise_rate": noise_rate,
        "seed": seed,
    }


def save_dataset(dataset: dict[str, Any], output_path: Path) -> None:
    """Save a generated dataset to an npz file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **dataset)


def print_dataset_summary(dataset: dict[str, Any], output_path: Path) -> None:
    """Print shapes and empirical noise rates for quick inspection."""
    train_noise = _as_array(dataset["train_noise"])
    test_noise = _as_array(dataset["test_noise"])

    print("Matrix LPN dataset generated")
    print(f"saved: {output_path}")
    print(f"seed={dataset['seed']}, noise_rate={dataset['noise_rate']}")
    print(f"n_x={dataset['n_x']}, n_y={dataset['n_y']}")
    print(f"S shape: {_as_array(dataset['S']).shape}")
    print(f"X_train shape: {_as_array(dataset['X_train']).shape}")
    print(f"Y_train_clean shape: {_as_array(dataset['Y_train_clean']).shape}")
    print(f"Y_train_noisy shape: {_as_array(dataset['Y_train_noisy']).shape}")
    print(f"X_test shape: {_as_array(dataset['X_test']).shape}")
    print(f"Y_test_clean shape: {_as_array(dataset['Y_test_clean']).shape}")
    print(f"Y_test_noisy shape: {_as_array(dataset['Y_test_noisy']).shape}")
    print(
        "train flipped bits: "
        f"{int(train_noise.sum())}/{train_noise.size} "
        f"({float(train_noise.mean()):.4f})"
    )
    print(
        "test flipped bits: "
        f"{int(test_noise.sum())}/{test_noise.size} "
        f"({float(test_noise.mean()):.4f})"
    )


def main() -> None:
    config.validate_config()
    config.ensure_directories()

    dataset = generate_dataset(
        n_x=config.n_x,
        n_y=config.n_y,
        num_train=config.num_train,
        num_test=config.num_test,
        noise_rate=config.noise_rate,
        seed=config.seed,
    )

    dataset.update(
        {
            "n1": config.n1,
            "m": config.m,
            "n_bits": config.n_bits,
        }
    )
    save_dataset(dataset, config.DATASET_PATH)
    print_dataset_summary(dataset, config.DATASET_PATH)


def _as_array(value: Any) -> BinaryArray:
    return np.asarray(value)


if __name__ == "__main__":
    main()
