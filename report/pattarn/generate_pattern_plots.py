from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from sampling.probility_distribution import generate_probability_distribution


ASSET_DIR = Path(__file__).resolve().parent / "assets"
PATTERNS = ["101", "1001", "10001"]
BASE_SEED = 42
NUM_SAMPLES = 1000


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    return float(0.5 * np.abs(p - q).sum())


def entropy_bits(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=float)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def format_state(index: int, num_bits: int) -> str:
    return format(index, f"0{num_bits}b")


def build_distribution_plot(
    pattern: str,
    idqnn_probs: np.ndarray,
    support_size: int,
    entropy: float,
    tvd: float,
    top_states: list[tuple[str, float]],
    save_path: Path,
) -> None:
    num_bits = len(pattern)
    labels = [format_state(i, num_bits) for i in range(len(idqnn_probs))]
    x = np.arange(len(idqnn_probs))

    fig = plt.figure(figsize=(11.2, 4.8))
    grid = fig.add_gridspec(1, 2, width_ratios=[4.4, 1.6])
    ax = fig.add_subplot(grid[0, 0])
    side = fig.add_subplot(grid[0, 1])

    ax.bar(x, idqnn_probs, width=0.82, color="#3b82f6", edgecolor="#1d4ed8")
    ax.set_title(f"IDQNN empirical distribution for x = {pattern}", fontsize=13)
    ax.set_xlabel("output bitstring y")
    ax.set_ylabel("estimated probability")
    ax.set_ylim(0, max(0.1, float(idqnn_probs.max()) * 1.18))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7 if num_bits >= 5 else 8)
    ax.grid(axis="y", alpha=0.25)

    side.axis("off")
    summary_lines = [
        f"seed = {BASE_SEED}",
        f"samples = {NUM_SAMPLES}",
        f"n1 = {num_bits}, m = 1",
        "",
        f"support = {support_size}/{2 ** num_bits}",
        f"entropy = {entropy:.3f} bits",
        f"TVD(DQNN, IDQNN) = {tvd:.3f}",
        "",
        "Top IDQNN states:",
    ]
    summary_lines.extend([f"{state}: {prob:.3f}" for state, prob in top_states])
    side.text(
        0.0,
        1.0,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "seed": BASE_SEED,
        "num_samples": NUM_SAMPLES,
        "patterns": [],
    }

    for pattern in PATTERNS:
        random.seed(BASE_SEED)
        np.random.seed(BASE_SEED)

        n1 = len(pattern)
        m = 1
        theta = [random.uniform(0.0, 1.0) * math.pi for _ in range(n1 * m)]

        _, _, dqnn_hist, idqnn_hist, _ = generate_probability_distribution(
            pattern,
            n1,
            m,
            theta,
            num_samples=NUM_SAMPLES,
            bin_width=1,
        )

        dqnn_probs = dqnn_hist / dqnn_hist.sum()
        idqnn_probs = idqnn_hist / idqnn_hist.sum()
        tvd = total_variation_distance(dqnn_probs, idqnn_probs)
        support_size = int(np.count_nonzero(idqnn_hist))
        entropy = entropy_bits(idqnn_probs)

        top_indices = np.argsort(idqnn_probs)[::-1][:5]
        top_states = [
            (format_state(int(index), n1), float(idqnn_probs[index]))
            for index in top_indices
            if idqnn_probs[index] > 0
        ]

        image_path = ASSET_DIR / f"pattern_{pattern}_distribution.png"
        build_distribution_plot(
            pattern=pattern,
            idqnn_probs=idqnn_probs,
            support_size=support_size,
            entropy=entropy,
            tvd=tvd,
            top_states=top_states,
            save_path=image_path,
        )

        summary["patterns"].append(
            {
                "pattern": pattern,
                "n1": n1,
                "m": m,
                "support_size": support_size,
                "state_count": 2**n1,
                "entropy_bits": entropy,
                "tvd_dqnn_idqnn": tvd,
                "top_states": [
                    {"state": state, "probability": probability}
                    for state, probability in top_states
                ],
                "image": image_path.name,
            }
        )
        print(f"Saved {image_path}")

    summary_path = ASSET_DIR / "pattern_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
