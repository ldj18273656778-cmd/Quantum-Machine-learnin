"""Majority-vote experiment for one LPN test sample using learned ISQNN theta."""

from __future__ import annotations

from typing import Iterable

import numpy as np

import Learning_Parity_with_Noise.config as config
from Learning_Parity_with_Noise.estimate_theta_lpn import rows_to_bitstrings
from Learning_Parity_with_Noise.gf2_utils import generate_labels
from sampling.ISQNN_generate_y import ISQNN_generate_y

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def majority_vote_for_sample(
    x_sample: np.ndarray,
    theta: np.ndarray,
    n1: int,
    m: int,
    num_shots: int,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the same x multiple times and return samples, vote, and one-counts."""
    if num_shots <= 0:
        raise ValueError(f"num_shots must be positive, got {num_shots}.")
    if num_shots % 2 == 0:
        raise ValueError("num_shots should be odd to avoid majority-vote ties.")

    x_sample = np.asarray(x_sample, dtype=np.uint8).reshape(1, -1)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    n_bits = n1 * m

    if x_sample.shape[1] != n_bits:
        raise ValueError(f"x_sample must have length {n_bits}, got {x_sample.shape[1]}.")
    if theta.shape[0] != n_bits:
        raise ValueError(f"theta must have length {n_bits}, got {theta.shape[0]}.")
    if np.isnan(theta).any():
        raise ValueError("theta contains NaN. Check theta estimation failures first.")

    bitstring = rows_to_bitstrings(x_sample)[0]
    samples = np.zeros((num_shots, n_bits), dtype=np.uint8)

    iterator: Iterable[int] = range(num_shots)
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, total=num_shots, desc="Majority vote shots", unit="shot")

    theta_list = theta.tolist()
    for shot_idx in iterator:
        _, y = ISQNN_generate_y(bitstring=bitstring, n1=n1, m=m, theta_list=theta_list)
        samples[shot_idx] = np.asarray(y, dtype=np.uint8)

    one_counts = np.sum(samples, axis=0)
    vote = (one_counts > num_shots / 2).astype(np.uint8)
    return samples, vote, one_counts


def print_majority_vote_result(
    sample_idx: int,
    x_sample: np.ndarray,
    y_true: np.ndarray,
    samples: np.ndarray,
    vote: np.ndarray,
    one_counts: np.ndarray,
) -> None:
    """Print one-sample majority-vote diagnostics."""
    num_shots = samples.shape[0]
    per_shot_bit_acc = np.mean(samples == y_true, axis=1)
    vote_bit_accuracy = float(np.mean(vote == y_true))
    vote_hamming_distance = int(np.sum(vote != y_true))

    print("Single-sample majority vote experiment")
    print(f"sample_idx: {sample_idx}")
    print(f"num_shots: {num_shots}")
    print(f"x_test: {x_sample.tolist()}")
    print(f"y_true = x_test @ S mod 2: {y_true.tolist()}")
    print(f"y_vote: {vote.tolist()}")
    print(f"vote bit accuracy: {vote_bit_accuracy:.6f}")
    print(f"vote hamming distance: {vote_hamming_distance}/{len(y_true)}")
    print(f"mean single-shot bit accuracy: {float(np.mean(per_shot_bit_acc)):.6f}")
    print(f"best single-shot bit accuracy: {float(np.max(per_shot_bit_acc)):.6f}")
    print(f"worst single-shot bit accuracy: {float(np.min(per_shot_bit_acc)):.6f}")
    print()
    print("bit  true  vote  ones/num_shots")
    for bit_idx, (true_bit, vote_bit, ones) in enumerate(zip(y_true, vote, one_counts), start=1):
        print(f"{bit_idx:>3}  {int(true_bit):>4}  {int(vote_bit):>4}  {int(ones):>4}/{num_shots}")


def main() -> None:
    # ===== 手动参数区 =====
    target_sample_idx = 0
    num_shots = 11
    # ====================

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

    X_test = dataset["X_test"]
    if not 0 <= target_sample_idx < len(X_test):
        raise ValueError(
            f"target_sample_idx must be in [0, {len(X_test) - 1}], got {target_sample_idx}."
        )

    x_sample = np.asarray(X_test[target_sample_idx], dtype=np.uint8)
    y_true = generate_labels(x_sample.reshape(1, -1), dataset["S"])[0]

    samples, vote, one_counts = majority_vote_for_sample(
        x_sample=x_sample,
        theta=theta_data["theta_hat_flat_rad"],
        n1=int(theta_data["n1"]),
        m=int(theta_data["m"]),
        num_shots=num_shots,
        show_progress=True,
    )

    print_majority_vote_result(
        sample_idx=target_sample_idx,
        x_sample=x_sample,
        y_true=y_true,
        samples=samples,
        vote=vote,
        one_counts=one_counts,
    )


if __name__ == "__main__":
    main()
