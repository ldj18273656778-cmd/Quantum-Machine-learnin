from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MNIST_DIR = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from MNIST.Encode_0to9 import encode_diagonal


def distance(x: np.ndarray, y: np.ndarray) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)

    if x_array.shape != y_array.shape:
        raise ValueError(f"Shape mismatch: {x_array.shape} vs {y_array.shape}")

    return float(np.linalg.norm(x_array.ravel() - y_array.ravel()))


def build_distance_table(target_grid: np.ndarray) -> pd.DataFrame:
    rows = target_grid.shape[0]
    records: list[dict[str, float | int]] = []

    for digit in range(10):
        reference_grid = encode_diagonal(digit, n=rows)
        records.append(
            {
                "Digit": digit,
                "Distance": distance(target_grid, reference_grid),
            }
        )

    df = pd.DataFrame(records).sort_values("Distance", ascending=True).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)
    return df[["Rank", "Digit", "Distance"]]


def load_test_sample(
    sample_index: int,
    threshold: float,
    n1: int,
    m: int,
) -> tuple[np.ndarray, int, Path]:
    test_path = MNIST_DIR / "data" / f"test_MNIST_{n1}x{m}_binarize{threshold}.npy"
    label_path = MNIST_DIR / "data" / "test_MNIST_labels.npy"

    x_test = np.load(test_path).reshape(-1, n1, m).astype(float)
    y_test = np.load(label_path)

    if not (0 <= sample_index < len(x_test)):
        raise IndexError(f"sample_index={sample_index} is out of range for {len(x_test)} samples.")

    return x_test[sample_index], int(y_test[sample_index]), test_path


def load_generated_mean(
    sample_index: int,
    threshold: float,
    n1: int,
    m: int,
    num_trials: int,
) -> tuple[np.ndarray, Path]:
    generated_path = (
        MNIST_DIR
        / "data"
        / "distance_related"
        / f"y_generated_mean_{sample_index}th_number{num_trials}times_binarized{threshold}.npy"
    )

    if not generated_path.exists():
        raise FileNotFoundError(
            "Generated mean file not found. "
            f"Expected: {generated_path}"
        )

    return np.load(generated_path).reshape(n1, m).astype(float), generated_path


def get_distance_target(
    source_mode: str,
    sample_grid: np.ndarray,
    generated_mean_grid: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    if source_mode == "test_image":
        return sample_grid, "Selected test image"

    if source_mode == "generated_mean":
        if generated_mean_grid is None:
            raise ValueError("generated_mean source mode requires generated_mean_grid.")
        return generated_mean_grid, "Averaged model output"

    raise ValueError(
        f"Unsupported source_mode={source_mode!r}. Use 'test_image' or 'generated_mean'."
    )


def plot_distance_bars(
    distance_label: str,
    sample_index: int,
    true_label: int,
    num_trials: int,
    distance_table: pd.DataFrame,
    output_path: Path,
) -> None:
    best_digit = int(distance_table.iloc[0]["Digit"])

    plt.rcParams.update(
        {
            "axes.facecolor": "#fffdf8",
            "figure.facecolor": "#f7f3eb",
            "savefig.facecolor": "#f7f3eb",
            "font.size": 13,
        }
    )

    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.12, right=0.96)

    fig.suptitle(
        f"Distance from Sample #{sample_index} to Label Encodings",
        fontsize=22,
        fontweight="bold",
        color="#111827",
        y=0.98,
    )
    fig.text(
        0.5,
        0.92,
        f"true label = {true_label}   |   source = {distance_label}   |   "
        "distance = ||source - encode_diagonal(k)||_2",
        ha="center",
        fontsize=12,
        color="#57534e",
    )

    sorted_digits = distance_table["Digit"].astype(int).tolist()
    sorted_distances = distance_table["Distance"].astype(float).tolist()
    bar_labels = [f"digit {digit}" for digit in sorted_digits]
    y_positions = np.arange(len(sorted_digits))

    bar_colors: list[str] = []
    for digit in sorted_digits:
        if digit == true_label and digit == best_digit:
            bar_colors.append("#16a34a")
        elif digit == true_label:
            bar_colors.append("#dc2626")
        elif digit == best_digit:
            bar_colors.append("#2563eb")
        else:
            bar_colors.append("#d6d3d1")

    ax.barh(
        y_positions,
        sorted_distances,
        color=bar_colors,
        edgecolor="white",
        linewidth=1.2,
        height=0.72,
    )
    ax.set_yticks(y_positions, bar_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Euclidean distance", color="#57534e")
    ax.grid(axis="x", color="#e7e5e4", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#a8a29e")
    ax.spines["bottom"].set_color("#a8a29e")

    distance_padding = (max(sorted_distances) - min(sorted_distances)) * 0.2 + 0.02
    ax.set_xlim(0, max(sorted_distances) + distance_padding * 1.7)

    for y_pos, digit, dist_value in zip(y_positions, sorted_digits, sorted_distances):
        label_suffix = ""
        if digit == true_label and digit == best_digit:
            label_suffix = "  true + nearest"
        elif digit == true_label:
            label_suffix = "  true label"
        elif digit == best_digit:
            label_suffix = "  nearest"

        ax.text(
            dist_value + distance_padding * 0.06,
            y_pos,
            f"{dist_value:.4f}{label_suffix}",
            va="center",
            ha="left",
            fontsize=12,
            color="#374151",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved distance bar figure to: {output_path}")

    if "agg" in plt.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    threshold = 0.4
    n1 = 10
    m = 10
    sample_index = 99
    source_mode = "generated_mean"
    num_trials = 10000

    sample_grid, true_label, test_path = load_test_sample(
        sample_index=sample_index,
        threshold=threshold,
        n1=n1,
        m=m,
    )

    generated_mean_grid = None
    generated_path: Path | None = None
    if source_mode == "generated_mean":
        generated_mean_grid, generated_path = load_generated_mean(
            sample_index=sample_index,
            threshold=threshold,
            n1=n1,
            m=m,
            num_trials=num_trials,
        )

    distance_grid, distance_label = get_distance_target(
        source_mode=source_mode,
        sample_grid=sample_grid,
        generated_mean_grid=generated_mean_grid,
    )
    distance_table = build_distance_table(distance_grid)

    print(f"Loaded test sample from: {test_path}")
    if generated_path is not None:
        print(f"Loaded generated mean from: {generated_path}")
    print(f"Selected sample index: {sample_index}")
    print(f"True label: {true_label}")
    print(f"Distance source: {distance_label}")
    print(distance_table.to_string(index=False, formatters={"Distance": "{:.4f}".format}))

    output_path = (
        ROOT
        / "output_images"
        / (
            f"distance_bars_sample_{sample_index}_{source_mode}_"
            f"threshold_{threshold}.png"
        )
    )
    plot_distance_bars(
        distance_label=distance_label,
        sample_index=sample_index,
        true_label=true_label,
        num_trials=num_trials,
        distance_table=distance_table,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
