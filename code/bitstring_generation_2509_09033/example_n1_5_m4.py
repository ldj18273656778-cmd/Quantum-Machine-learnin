from __future__ import annotations

import numpy as np

from idqnn_bitstring import (
    make_config,
    sample_deep_mapped_idqnn,
    sample_shallow_idqnn,
)


def main() -> None:
    # Fixed example
    n1 = 5
    m = 4
    shots = 20
    seed = 11

    cfg = make_config(n1=n1, m=m, include_temporal_edges=True)

    # Fixed input x and theta for reproducibility
    rng = np.random.default_rng(seed)
    x = "".join(rng.choice(["0", "1"], size=cfg.n, p=[0.7, 0.3]).tolist())
    theta = rng.uniform(0.0, np.pi, size=(n1, m))

    deep_samples = sample_deep_mapped_idqnn(
        bitstring_x=x,
        theta=theta,
        cfg=cfg,
        shots=shots,
        seed=seed + 100,
    )
    shallow_samples = sample_shallow_idqnn(
        bitstring_x=x,
        theta=theta,
        cfg=cfg,
        shots=shots,
        seed=seed + 200,
    )

    print("=== Concrete Example (n1=5, m=4) ===")
    print(f"n1={n1}, m={m}, n={cfg.n}, shots={shots}")
    print("input x bitstring:")
    print(f"x = {x}")
    print(f"len(x) = {len(x)}")
    print("x by time-slice (t=0..n1-1):")
    for t in range(n1):
        print(f"  t={t}: {x[t * m:(t + 1) * m]}")
    print("theta[0]=", np.round(theta[0], 4).tolist())
    print("deep first 5 samples:")
    for row in deep_samples[:5]:
        print("".join(str(int(b)) for b in row))
    print("shallow first 5 samples:")
    for row in shallow_samples[:5]:
        print("".join(str(int(b)) for b in row))


if __name__ == "__main__":
    main()
