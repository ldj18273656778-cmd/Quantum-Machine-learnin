"""Count how many LPN training samples enter each ISQNN theta estimate."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import Learning_Parity_with_Noise.config as config
from Learning_Parity_with_Noise.estimate_theta_lpn import rows_to_bitstrings
from sampling.ISQNN_generate_y import idqnn_connectivity
from Train.find_x_indices_by_graph_condition import build_adjacency, find_indices


def count_samples_for_each_theta(
    X_train: np.ndarray,
    n1: int,
    m: int,
) -> list[dict]:
    """Count N_sp for every target bit in the original ISQNN estimator.

    N_sp is the number of samples satisfying:
    x[target_bit] = 0 and all CZ-neighbor bits of target_bit equal 1.
    """
    X_train = np.asarray(X_train, dtype=np.uint8)
    n_bits = n1 * m

    if X_train.ndim != 2 or X_train.shape[1] != n_bits:
        raise ValueError(
            f"X_train must have shape (num_train, {n_bits}), got {X_train.shape}."
        )

    x_bitstrings = rows_to_bitstrings(X_train)
    graph = idqnn_connectivity(n1, m)
    adjacency = build_adjacency(n=n_bits, edges=graph["all_edges"])

    records = []
    for target_bit in range(n_bits):
        neighbors = sorted(adjacency[target_bit])
        indices = find_indices(
            x=x_bitstrings,
            target_bit=target_bit,
            adjacency=adjacency,
            show_progress=False,
        )
        records.append(
            {
                "target_bit_0based": int(target_bit),
                "target_bit_1based": int(target_bit + 1),
                "slice_idx": int(target_bit // m),
                "position_in_slice": int(target_bit % m),
                "neighbors_0based": np.asarray(neighbors, dtype=int),
                "n_neighbors": int(len(neighbors)),
                "N_sp": int(len(indices)),
                "fraction": float(len(indices) / len(X_train)),
                "indices_0based": indices,
            }
        )
    return records


def save_counts(records: list[dict], output_path) -> None:
    """Save full records to npz and a readable text table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, records=np.asarray(records, dtype=object))

    txt_path = output_path.with_suffix(".txt")
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("target_1based\ttarget_0based\tslice\tpos\tn_neighbors\tN_sp\tfraction\tneighbors_0based\n")
        for record in records:
            neighbors = record["neighbors_0based"].tolist()
            f.write(
                f"{record['target_bit_1based']}\t"
                f"{record['target_bit_0based']}\t"
                f"{record['slice_idx']}\t"
                f"{record['position_in_slice']}\t"
                f"{record['n_neighbors']}\t"
                f"{record['N_sp']}\t"
                f"{record['fraction']:.6f}\t"
                f"{neighbors}\n"
            )


def print_counts_table(records: list[dict], output_path) -> None:
    """Print a compact count table."""
    counts = np.asarray([record["N_sp"] for record in records], dtype=int)

    print("Theta estimation sample counts")
    print(f"saved npz: {output_path}")
    print(f"saved txt: {output_path.with_suffix('.txt')}")
    print(f"min N_sp: {int(counts.min())}")
    print(f"max N_sp: {int(counts.max())}")
    print(f"mean N_sp: {float(counts.mean()):.2f}")
    print()
    print("target  slice  pos  deg  N_sp  fraction  neighbors")
    for record in records:
        print(
            f"{record['target_bit_1based']:>6}  "
            f"{record['slice_idx']:>5}  "
            f"{record['position_in_slice']:>3}  "
            f"{record['n_neighbors']:>3}  "
            f"{record['N_sp']:>4}  "
            f"{record['fraction']:.6f}  "
            f"{record['neighbors_0based'].tolist()}"
        )


def plot_counts(records: list[dict]) -> None:
    """Show a bar chart of N_sp for each theta parameter."""
    target_bits = [record["target_bit_1based"] for record in records]
    counts = [record["N_sp"] for record in records]

    plt.figure(figsize=(10, 4.8))
    plt.bar(target_bits, counts, color="#4C72B0")
    plt.xlabel("theta index (1-based)")
    plt.ylabel("N_sp")
    plt.title("Samples Used for Each ISQNN Theta Estimate")
    plt.xticks(target_bits)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.show()


def main() -> None:
    config.validate_config()
    config.ensure_directories()

    if not config.DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {config.DATASET_PATH}. Run generate_lpn_dataset.py first."
        )

    dataset = np.load(config.DATASET_PATH, allow_pickle=True)
    records = count_samples_for_each_theta(
        X_train=dataset["X_train"],
        n1=int(dataset["n1"]),
        m=int(dataset["m"]),
    )

    output_path = config.DATA_DIR / "theta_estimation_sample_counts.npz"
    save_counts(records, output_path)
    print_counts_table(records, output_path)
    plot_counts(records)


if __name__ == "__main__":
    main()
