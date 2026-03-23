from __future__ import annotations

import argparse
import hashlib

import numpy as np

from idqnn_bitstring import (
    make_config,
    sample_deep_mapped_idqnn,
)


def run_experiment(n1: int, m: int, shots: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    cfg = make_config(n1=n1, m=m, spatial_edges=None, include_temporal_edges=True)
    x = "".join(rng.choice(["0", "1"], size=cfg.n, p=[0.7, 0.3]).tolist())
    theta = rng.uniform(0.0, np.pi, size=(n1, m))

    # Reproducibility check: same seed -> same samples.
    deep_samples_a = sample_deep_mapped_idqnn(bitstring_x=x, theta=theta, cfg=cfg, shots=shots, seed=seed + 29)
    deep_samples_b = sample_deep_mapped_idqnn(bitstring_x=x, theta=theta, cfg=cfg, shots=shots, seed=seed + 29)
    deep_samples_c = sample_deep_mapped_idqnn(bitstring_x=x, theta=theta, cfg=cfg, shots=shots, seed=seed + 30)

    same_seed_equal = np.array_equal(deep_samples_a, deep_samples_b)
    diff_seed_diff = not np.array_equal(deep_samples_a, deep_samples_c)
    bit_means = deep_samples_a.mean(axis=0)
    digest = hashlib.sha256(deep_samples_a.tobytes()).hexdigest()[:16]
    head = ["".join(str(int(b)) for b in row) for row in deep_samples_a[:5]]

    print("=== Reproduction: arXiv:2509.09033 bitstring generation (Algorithm 2) ===")
    print(f"n1={n1}, m={m}, n={cfg.n}, shots={shots}, seed={seed}")
    print(f"input x: {x}")
    print(f"sample digest (sha256 first16): {digest}")
    print(f"same seed deterministic: {same_seed_equal}")
    print(f"different seed changes samples: {diff_seed_diff}")
    print(f"first 5 samples: {head}")
    print(f"mean(y_i) first 10 bits: {[round(float(v), 4) for v in bit_means[:10]]}")

    ok = same_seed_equal and diff_seed_diff and deep_samples_a.shape == (shots, cfg.n)
    print(f"pass: {ok}")
    if not ok:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n1", type=int, default=4, help="time-depth dimension")
    parser.add_argument("--m", type=int, default=3, help="qubits per time slice")
    parser.add_argument("--shots", type=int, default=4000, help="number of samples")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(n1=args.n1, m=args.m, shots=args.shots, seed=args.seed)
