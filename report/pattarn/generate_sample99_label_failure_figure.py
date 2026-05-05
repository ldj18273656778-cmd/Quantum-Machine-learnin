from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow


SAMPLE_INDEX = 99
THRESHOLD = 0.4


def encode_diagonal(label: int, n: int = 10) -> np.ndarray:
    grid = np.zeros((n, n), dtype=int)
    for row in range(n):
        col = (row + label) % n
        grid[row, col] = 1
    return grid


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "code" / "MNIST" / "data"
    output_dir = repo_root / "report" / "pattarn" / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    x_test = np.load(data_dir / f"test_MNIST_10x10_binarize{THRESHOLD}.npy")
    y_test = np.load(data_dir / "test_MNIST_labels.npy")

    sample = x_test[SAMPLE_INDEX, 0].astype(int)
    sample_inverted = 1 - sample
    label = int(y_test[SAMPLE_INDEX])
    encoded_label = encode_diagonal(label, n=sample.shape[0])

    fig = plt.figure(figsize=(12.5, 5.4), facecolor="white")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.48, 1.25], wspace=0.02)

    ax_left = fig.add_subplot(grid[0, 0])
    ax_mid = fig.add_subplot(grid[0, 1])
    ax_right = fig.add_subplot(grid[0, 2])

    ax_left.imshow(sample_inverted, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_left.set_title(f"Input Sample #99: {label}", fontsize=18)
    ax_left.set_xticks(range(0, 10, 2))
    ax_left.set_yticks(range(0, 10, 2))
    ax_left.tick_params(labelsize=12, width=1.0)
    for spine in ax_left.spines.values():
        spine.set_linewidth(1.0)

    ax_right.imshow(encoded_label, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_right.set_title(f"Encoded True Label: {label}", fontsize=18)
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    for spine in ax_right.spines.values():
        spine.set_visible(False)

    ax_mid.set_axis_off()
    arrow = FancyArrow(
        0.08,
        0.5,
        0.84,
        0.0,
        width=0.10,
        head_width=0.20,
        head_length=0.18,
        length_includes_head=True,
        facecolor="#4c78c5",
        edgecolor="#1d3f82",
        linewidth=1.0,
        transform=ax_mid.transAxes,
    )
    ax_mid.add_patch(arrow)
    ax_mid.plot(
        [0.36, 0.66],
        [0.68, 0.32],
        color="#4c78c5",
        linewidth=26,
        solid_capstyle="projecting",
        transform=ax_mid.transAxes,
        zorder=5,
    )
    ax_mid.plot(
        [0.36, 0.66],
        [0.68, 0.32],
        color="#1d3f82",
        linewidth=1.2,
        transform=ax_mid.transAxes,
        zorder=6,
    )

    output_path = output_dir / "sample99_vs_encoded_label_failure.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
