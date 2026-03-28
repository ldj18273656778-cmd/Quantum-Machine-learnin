from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

from idqnn_bitstring import (
    exact_probs_shallow,
    make_config,
    sample_deep_mapped_idqnn,
    sample_shallow_idqnn,
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _tv_from_samples(samples: np.ndarray, probs_exact: np.ndarray) -> float:
    n = samples.shape[1]
    shots = samples.shape[0]
    weights = 1 << np.arange(n - 1, -1, -1, dtype=np.int64)
    ids = (samples.astype(np.int64) * weights).sum(axis=1).tolist()
    cnt = Counter(ids)
    tv = 0.0
    for i in range(2**n):
        p_emp = cnt.get(i, 0) / shots
        p_ex = float(probs_exact[i])
        tv += abs(p_emp - p_ex)
    return 0.5 * tv


def test_shapes_and_binary() -> None:
    cfg = make_config(n1=4, m=3)
    x = "010100110001"
    theta = np.linspace(0.01, 1.2, cfg.n).reshape(cfg.n1, cfg.m)
    sh = sample_shallow_idqnn(x, theta, cfg, shots=128, seed=3)
    dp = sample_deep_mapped_idqnn(x, theta, cfg, shots=128, seed=7)
    _assert(sh.shape == (128, cfg.n), f"shallow shape wrong: {sh.shape}")
    _assert(dp.shape == (128, cfg.n), f"deep shape wrong: {dp.shape}")
    _assert(np.all((sh == 0) | (sh == 1)), "shallow has non-binary values")
    _assert(np.all((dp == 0) | (dp == 1)), "deep has non-binary values")


def test_deep_determinism() -> None:
    cfg = make_config(n1=4, m=3)
    x = "011001011000"
    rng = np.random.default_rng(7)
    theta = rng.uniform(0.0, np.pi, size=(cfg.n1, cfg.m))
    a = sample_deep_mapped_idqnn(x, theta, cfg, shots=256, seed=123)
    b = sample_deep_mapped_idqnn(x, theta, cfg, shots=256, seed=123)
    c = sample_deep_mapped_idqnn(x, theta, cfg, shots=256, seed=124)
    _assert(np.array_equal(a, b), "same seed should generate identical deep samples")
    _assert(not np.array_equal(a, c), "different seeds should usually generate different deep samples")


def test_shallow_exact_consistency() -> None:
    cfg = make_config(n1=3, m=2)
    x = "001011"
    theta = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]
    )
    probs = exact_probs_shallow(x, theta, cfg)
    _assert(probs.shape == (2 ** cfg.n,), f"prob shape wrong: {probs.shape}")
    _assert(abs(float(probs.sum()) - 1.0) < 1e-6, "exact probabilities do not sum to 1")

    samples = sample_shallow_idqnn(x, theta, cfg, shots=6000, seed=9)
    tv = _tv_from_samples(samples, probs)
    _assert(tv < 0.08, f"shallow empirical distribution too far from exact: TV={tv:.4f}")


def test_deep_regression_fingerprint() -> None:
    # Fixed case to detect unintended behavior drift after code edits.
    cfg = make_config(n1=4, m=3)
    x = "011001011000"
    rng = np.random.default_rng(7)
    theta = rng.uniform(0.0, np.pi, size=(cfg.n1, cfg.m))
    samples = sample_deep_mapped_idqnn(x, theta, cfg, shots=300, seed=36)
    digest = hashlib.sha256(samples.tobytes()).hexdigest()[:16]
    expected = "fd28556599da2f73"
    _assert(digest == expected, f"deep regression mismatch: got {digest}, expected {expected}")


def main() -> None:
    tests = [
        ("shapes_and_binary", test_shapes_and_binary),
        ("deep_determinism", test_deep_determinism),
        ("shallow_exact_consistency", test_shallow_exact_consistency),
        ("deep_regression_fingerprint", test_deep_regression_fingerprint),
    ]
    for name, fn in tests:
        fn()
        print(f"[PASS] {name}")
    print("All tests passed.")


if __name__ == "__main__":
    main()
