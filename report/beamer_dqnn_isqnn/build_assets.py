import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_models(repo_root: Path):
    import sys

    code_dir = repo_root / "code" / "sampling DQNN"
    sys.path.insert(0, str(code_dir))
    from DQNN_generate_y import DQNN_generate_y  # type: ignore
    from ISQNN_generate_y import ISQNN_generate_y  # type: ignore

    return DQNN_generate_y, ISQNN_generate_y


def sample_model(model_fn, x: str, n1: int, m: int, theta_list, shots: int):
    n = n1 * m
    counts = Counter()
    bit_sums = np.zeros(n, dtype=np.int64)
    for _ in range(shots):
        _, y = model_fn(x, n1, m, theta_list)
        y_str = "".join(str(bit) for bit in y)
        counts[y_str] += 1
        bit_sums += np.asarray(y, dtype=np.int64)
    probs = {k: v / shots for k, v in counts.items()}
    bit_means = (bit_sums / shots).tolist()
    return probs, bit_means


def entropy(prob_dist):
    return -sum(p * math.log2(p) for p in prob_dist.values() if p > 0)


def tv_distance(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def topk(prob_dist, k=8):
    return sorted(prob_dist.items(), key=lambda item: item[1], reverse=True)[:k]


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(__file__).resolve().parent / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    DQNN_generate_y, ISQNN_generate_y = _load_models(repo_root)

    # Reproducible seeds for all experiments.
    random.seed(20260323)
    np.random.seed(20260323)

    # Experiment A/B/C settings (small-size for exhaustive empirical statistics).
    n1, m = 3, 2
    n = n1 * m
    shots = 2000
    theta_small = (np.random.uniform(0.0, 1.0, size=n) * np.pi).tolist()
    x_all_zero = "0" * n
    x_all_one = "1" * n
    x_mixed = "000100"

    p_dqnn_zero, bit_dqnn_zero = sample_model(
        DQNN_generate_y, x_all_zero, n1, m, theta_small, shots
    )
    p_dqnn_one, bit_dqnn_one = sample_model(
        DQNN_generate_y, x_all_one, n1, m, theta_small, shots
    )
    p_dqnn_mixed, bit_dqnn_mixed = sample_model(
        DQNN_generate_y, x_mixed, n1, m, theta_small, shots
    )
    p_isqnn_mixed, bit_isqnn_mixed = sample_model(
        ISQNN_generate_y, x_mixed, n1, m, theta_small, shots
    )

    # Experiment D: one-shot run with the same structure as main.py.
    bitstring_main = "00001111000000110000"
    n1_main, m_main = 4, 5
    theta_main = [random.uniform(0.0, 1.0) * np.pi for _ in range(n1_main * m_main)]
    _, y_dqnn_main = DQNN_generate_y(bitstring_main, n1_main, m_main, theta_main)
    _, y_isqnn_main = ISQNN_generate_y(bitstring_main, n1_main, m_main, theta_main)

    summary = {
        "small_case": {
            "n1": n1,
            "m": m,
            "n": n,
            "shots": shots,
            "theta": theta_small,
            "x_all_zero_entropy": entropy(p_dqnn_zero),
            "x_all_one_entropy": entropy(p_dqnn_one),
            "tv_dqnn_vs_isqnn_mixed": tv_distance(p_dqnn_mixed, p_isqnn_mixed),
            "bit_mean_dqnn_x_all_zero": bit_dqnn_zero,
            "bit_mean_dqnn_x_all_one": bit_dqnn_one,
            "bit_mean_dqnn_x_mixed": bit_dqnn_mixed,
            "bit_mean_isqnn_x_mixed": bit_isqnn_mixed,
            "top8_dqnn_x_all_zero": topk(p_dqnn_zero, k=8),
            "top8_dqnn_x_all_one": topk(p_dqnn_one, k=8),
            "top8_dqnn_x_mixed": topk(p_dqnn_mixed, k=8),
            "top8_isqnn_x_mixed": topk(p_isqnn_mixed, k=8),
        },
        "main_like_single_run": {
            "bitstring": bitstring_main,
            "n1": n1_main,
            "m": m_main,
            "theta_first5": theta_main[:5],
            "y_dqnn": y_dqnn_main,
            "y_isqnn": y_isqnn_main,
        },
    }
    save_json(out_dir / "summary.json", summary)

    # Plot 1: bit-wise probabilities for DQNN under x=0...0 and x=1...1.
    x_idx = np.arange(1, n + 1)
    plt.figure(figsize=(8, 3.8))
    plt.plot(x_idx, bit_dqnn_zero, marker="o", label="DQNN, x=0...0")
    plt.plot(x_idx, bit_dqnn_one, marker="s", label="DQNN, x=1...1")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="0.5 baseline")
    plt.ylim(0.0, 1.0)
    plt.xticks(x_idx)
    plt.xlabel("Bit index of y")
    plt.ylabel("Empirical P(y_i = 1)")
    plt.title("Bit-wise output statistics (2000 shots)")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "bitwise_prob_dqnn.png", dpi=180)
    plt.close()

    # Plot 2: top bitstring probabilities for mixed input (DQNN vs ISQNN).
    top_dqnn = topk(p_dqnn_mixed, k=8)
    top_isqnn = topk(p_isqnn_mixed, k=8)
    labels = [item[0] for item in top_dqnn]
    dqnn_vals = [item[1] for item in top_dqnn]
    isqnn_lookup = dict(top_isqnn)
    isqnn_vals = [isqnn_lookup.get(label, 0.0) for label in labels]

    pos = np.arange(len(labels))
    width = 0.4
    plt.figure(figsize=(9, 4))
    plt.bar(pos - width / 2, dqnn_vals, width=width, label="DQNN")
    plt.bar(pos + width / 2, isqnn_vals, width=width, label="ISQNN")
    plt.xticks(pos, labels, rotation=30)
    plt.ylabel("Empirical probability")
    plt.xlabel("Bitstring y (Top-8 of DQNN)")
    plt.title("Mixed input x=000100: top output modes")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "top_modes_compare.png", dpi=180)
    plt.close()

    # Text summary for quick insertion into slides.
    lines = [
        "Reproducible experiment summary",
        "--------------------------------",
        f"small case: n1={n1}, m={m}, n={n}, shots={shots}",
        f"theta_small={theta_small}",
        f"H(y|x=0...0)={entropy(p_dqnn_zero):.4f} (ideal max={n})",
        f"H(y|x=1...1)={entropy(p_dqnn_one):.4f}",
        f"TV(DQNN,ISQNN | x=000100)={tv_distance(p_dqnn_mixed, p_isqnn_mixed):.4f}",
        f"bit_mean_dqnn_x0={[round(v,4) for v in bit_dqnn_zero]}",
        f"bit_mean_dqnn_x1={[round(v,4) for v in bit_dqnn_one]}",
        f"bit_mean_dqnn_xm={[round(v,4) for v in bit_dqnn_mixed]}",
        f"bit_mean_isqnn_xm={[round(v,4) for v in bit_isqnn_mixed]}",
        "",
        "main-like single run (n1=4,m=5):",
        f"theta_first5={[round(v,6) for v in theta_main[:5]]}",
        f"y_dqnn={y_dqnn_main}",
        f"y_isqnn={y_isqnn_main}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
