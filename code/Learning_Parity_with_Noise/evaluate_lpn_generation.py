"""Evaluate ISQNN-generated test labels against the clean LPN labels."""

from __future__ import annotations

import numpy as np

import Learning_Parity_with_Noise.config as config
from Learning_Parity_with_Noise.gf2_utils import (
    bit_accuracy,
    generate_labels,
    hamming_distance,
    sample_accuracy,
)


def evaluate_predictions(Y_pred: np.ndarray, Y_target: np.ndarray) -> dict:
    """Compute generation quality metrics against a target label matrix."""
    Y_pred = np.asarray(Y_pred, dtype=np.uint8)
    Y_target = np.asarray(Y_target, dtype=np.uint8)
    distances = hamming_distance(Y_pred, Y_target)

    return {
        "bit_accuracy": bit_accuracy(Y_pred, Y_target),
        "sample_accuracy": sample_accuracy(Y_pred, Y_target),
        "mean_hamming_distance": float(np.mean(distances)),
        "median_hamming_distance": float(np.median(distances)),
        "max_hamming_distance": int(np.max(distances)),
        "per_output_bit_accuracy": np.mean(Y_pred == Y_target, axis=0),
        "hamming_distance_per_sample": distances,
    }


def save_metrics(metrics: dict, output_path) -> None:
    """Save metrics to npz and a readable text report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **metrics)

    txt_path = output_path.with_suffix(".txt")
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("LPN ISQNN generation evaluation\n")
        f.write(f"bit_accuracy = {metrics['bit_accuracy']:.6f}\n")
        f.write(f"sample_accuracy = {metrics['sample_accuracy']:.6f}\n")
        f.write(f"mean_hamming_distance = {metrics['mean_hamming_distance']:.6f}\n")
        f.write(f"median_hamming_distance = {metrics['median_hamming_distance']:.6f}\n")
        f.write(f"max_hamming_distance = {metrics['max_hamming_distance']}\n")
        f.write("\nper_output_bit_accuracy:\n")
        for idx, acc in enumerate(metrics["per_output_bit_accuracy"], start=1):
            f.write(f"y_{idx}\t{float(acc):.6f}\n")


def print_metrics(metrics: dict, output_path) -> None:
    """Print evaluation metrics."""
    print("LPN ISQNN generation evaluation")
    print(f"saved npz: {output_path}")
    print(f"saved txt: {output_path.with_suffix('.txt')}")
    print(f"bit_accuracy: {metrics['bit_accuracy']:.6f}")
    print(f"sample_accuracy: {metrics['sample_accuracy']:.6f}")
    print(f"mean_hamming_distance: {metrics['mean_hamming_distance']:.6f}")
    print(f"median_hamming_distance: {metrics['median_hamming_distance']:.6f}")
    print(f"max_hamming_distance: {metrics['max_hamming_distance']}")
    print("per_output_bit_accuracy:")
    for idx, acc in enumerate(metrics["per_output_bit_accuracy"], start=1):
        print(f"  y_{idx}: {float(acc):.6f}")


def main() -> None:
    config.validate_config()
    config.ensure_directories()

    if not config.PREDICTION_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {config.PREDICTION_PATH}. Run generate_test_y_lpn.py first."
        )

    pred_data = np.load(config.PREDICTION_PATH, allow_pickle=True)

    Y_pred = pred_data["Y_pred"]
    X_test = pred_data["X_test"]
    S = pred_data["S"]

    # Recompute the clean LPN labels from the hidden S to make the comparison explicit.
    Y_from_S = generate_labels(X_test, S)

    if "Y_test_clean" in pred_data.files:
        Y_test_clean = pred_data["Y_test_clean"]
        if not np.array_equal(Y_from_S, Y_test_clean):
            raise ValueError("Y_test_clean does not match X_test @ S mod 2.")

    metrics = evaluate_predictions(Y_pred, Y_from_S)
    metrics.update(
        {
            "Y_pred": Y_pred,
            "Y_from_S": Y_from_S,
            "X_test": X_test,
            "S": S,
            "prediction_path": str(config.PREDICTION_PATH),
            "model": "original_isqnn",
        }
    )

    output_path = config.DATA_DIR / "lpn_generation_metrics.npz"
    save_metrics(metrics, output_path)
    print_metrics(metrics, output_path)


if __name__ == "__main__":
    main()
