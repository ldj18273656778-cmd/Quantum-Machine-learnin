"""Small sanity test for matrix Learning Parity with Noise data generation."""

from __future__ import annotations

import numpy as np

# import config
import Learning_Parity_with_Noise.config as config
from Learning_Parity_with_Noise.gf2_utils import (
# from gf2_utils import (
    add_bernoulli_noise,
    generate_labels,
    get_rng,
    sample_inputs,
    sample_secret,
)


def main() -> None:
    config.validate_config()
    config.ensure_directories()

    num_samples = 8
    demo_noise_rate = 0.10
    rng = get_rng(config.seed)
    print(f"Using random seed: {config.seed}")

    # Matrix LPN: x in {0,1}^{n_x}, S in {0,1}^{n_x x n_y}, y = x S mod 2.
    S = sample_secret(config.n_x, config.n_y, rng)
    X = sample_inputs(num_samples, config.n_x, rng)
    print("Sampled secret matrix S and input matrix X:", S, X)
    print(S.shape, X.shape)

    Y_clean = generate_labels(X, S)
    print("Generated clean labels Y_clean:", Y_clean)
    print(Y_clean.shape)
    Y_noisy, noise = add_bernoulli_noise(Y_clean, demo_noise_rate, rng)

    output_path = config.DATA_DIR / "test_lpn_demo.npz"
    np.savez(
        output_path,
        S=S,
        X=X,
        Y_clean=Y_clean,
        Y_noisy=Y_noisy,
        noise=noise,
        n_x=config.n_x,
        n_y=config.n_y,
        num_samples=num_samples,
        noise_rate=demo_noise_rate,
        seed=config.seed,
    )

    print("Matrix LPN sanity test")
    print(f"n_x={config.n_x}, n_y={config.n_y}, num_samples={num_samples}")
    print(f"noise_rate={demo_noise_rate}, flipped_bits={int(noise.sum())}/{noise.size}")
    print(f"S shape: {S.shape}")
    print(f"X shape: {X.shape}")
    print(f"Y_clean shape: {Y_clean.shape}")
    print(f"Y_noisy shape: {Y_noisy.shape}")
    print(f"saved: {output_path}")

    print("\nFirst 3 samples:")
    for i in range(min(3, num_samples)):
        print(f"sample {i}")
        print(f"x       = {X[i].tolist()}")
        print(f"y_clean = {Y_clean[i].tolist()}")
        print(f"noise   = {noise[i].tolist()}")
        print(f"y_noisy = {Y_noisy[i].tolist()}")


if __name__ == "__main__":
    main()
