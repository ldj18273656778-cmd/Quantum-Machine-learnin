from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MNIST_DIR = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from sampling.ISQNN_generate_y import idqnn_connectivity
from Encode_0to9 import encode_diagonal
from connectivity_of_testbitstring import (
    bit_to_coord,
    get_selected_components,
    print_component_report,
    style_grid_axis,
)


threshold = 0.4
n1 = 10
m = 10
sample_indices = [164]
selected_bit_value = 0


def draw_connectivity(
    ax: plt.Axes,
    bit_grid: np.ndarray,
    components: list[list[int]],
    selected_edges: list[tuple[int, int]],
) -> None:
    rows, cols = bit_grid.shape
    ax.imshow(bit_grid, cmap="gray", alpha=0.25, interpolation="nearest")
    style_grid_axis(ax, rows, cols)

    multi_node_components = [component for component in components if len(component) > 1]
    palette = plt.colormaps["tab20"](
        np.linspace(0, 1, max(len(multi_node_components), 1))
    )
    palette_index = 0

    for component in components:
        if len(component) > 1:
            color = palette[palette_index % len(palette)]
            palette_index += 1
        else:
            color = "#9ca3af"

        component_set = set(component)

        for a, b in selected_edges:
            if a not in component_set or b not in component_set:
                continue
            row_a, col_a = bit_to_coord(a, cols)
            row_b, col_b = bit_to_coord(b, cols)
            ax.plot(
                [col_a, col_b],
                [row_a, row_b],
                color=color,
                linewidth=2.8,
                alpha=0.95,
                zorder=2,
            )

        rows_in_component = [bit_to_coord(bit_index, cols)[0] for bit_index in component]
        cols_in_component = [bit_to_coord(bit_index, cols)[1] for bit_index in component]
        marker_size = 250 if len(component) > 1 else 140

        ax.scatter(
            cols_in_component,
            rows_in_component,
            s=marker_size,
            c=[color],
            marker="s",
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )


def overlay_reference_diagonal(
    ax: plt.Axes,
    reference_grid: np.ndarray,
    label_value: int,
) -> list[int]:
    diagonal_indices = np.flatnonzero(reference_grid.reshape(-1) == 1).tolist()
    diagonal_coords = [bit_to_coord(bit_index, reference_grid.shape[1]) for bit_index in diagonal_indices]
    rows = [row for row, _ in diagonal_coords]
    cols = [col for _, col in diagonal_coords]

    ax.scatter(
        cols,
        rows,
        s=360,
        facecolors="none",
        edgecolors="#dc2626",
        linewidths=2.2,
        marker="s",
        zorder=4,
        label=f"encode_diagonal({label_value})",
    )
    ax.scatter(
        cols,
        rows,
        s=90,
        c="#dc2626",
        marker="x",
        linewidths=1.8,
        zorder=5,
    )
    return diagonal_indices


def print_diagonal_report(
    diagonal_indices: list[int],
    overlap_indices: list[int],
    width: int,
    label_value: int,
    selected_bit_value_local: int,
) -> None:
    diagonal_coords = [bit_to_coord(bit_index, width) for bit_index in diagonal_indices]
    overlap_coords = [bit_to_coord(bit_index, width) for bit_index in overlap_indices]

    print(f"Reference label for diagonal encoding: {label_value}")
    print(f"encode_diagonal({label_value}) bits: {diagonal_indices}")
    print(f"encode_diagonal({label_value}) coords: {diagonal_coords}")
    print(f"Overlap with selected {selected_bit_value_local}-bits: {len(overlap_indices)}")
    print(f"Overlap bit indices: {overlap_indices}")
    print(f"Overlap coords: {overlap_coords}")


def plot_connectivity_with_reference_diagonal(
    bit_grid: np.ndarray,
    sample_index: int,
    sample_label: int,
    components: list[list[int]],
    selected_edges: list[tuple[int, int]],
    selected_nodes: list[int],
    reference_grid: np.ndarray,
    output_path: Path,
) -> tuple[list[int], list[int]]:
    rows, cols = bit_grid.shape
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    ax_image, ax_graph, ax_overlay = axes

    ax_image.imshow(bit_grid, cmap="gray", interpolation="nearest")
    ax_image.set_title(f"test x bitstring #{sample_index} | label={sample_label}")
    style_grid_axis(ax_image, rows, cols)

    draw_connectivity(
        ax=ax_graph,
        bit_grid=bit_grid,
        components=components,
        selected_edges=selected_edges,
    )
    ax_graph.set_title(
        f"ISQNN connectivity on {selected_bit_value}-bits\n"
        f"selected bits={len(selected_nodes)}, "
        f"selected edges={len(selected_edges)}, components={len(components)}"
    )

    draw_connectivity(
        ax=ax_overlay,
        bit_grid=bit_grid,
        components=components,
        selected_edges=selected_edges,
    )
    diagonal_indices = overlay_reference_diagonal(
        ax=ax_overlay,
        reference_grid=reference_grid,
        label_value=sample_label,
    )
    overlap_indices = sorted(set(selected_nodes) & set(diagonal_indices))
    ax_overlay.set_title(
        f"Connectivity + encode_diagonal({sample_label})\n"
        f"diagonal ones={len(diagonal_indices)}, overlap={len(overlap_indices)}"
    )
    ax_overlay.legend(loc="upper right", frameon=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved connectivity+diagonal figure to: {output_path}")
    if "agg" in plt.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()

    return diagonal_indices, overlap_indices


def main() -> None:
    if n1 != m:
        raise ValueError("encode_diagonal overlay requires n1 == m for a square grid.")

    test_path = MNIST_DIR / "data" / f"test_MNIST_10x10_binarize{threshold}.npy"
    label_path = MNIST_DIR / "data" / "test_MNIST_labels.npy"

    x_test = np.load(test_path).reshape(-1, n1 * m).astype(int)
    y_test = np.load(label_path)
    x_flip = 1 - x_test

    if not sample_indices:
        raise ValueError("No sample indices were provided.")

    unique_indices: list[int] = []
    seen: set[int] = set()
    for sample_index in sample_indices:
        if not (0 <= sample_index < len(x_flip)):
            raise IndexError(
                f"sample_index={sample_index} is out of range for {len(x_flip)} samples."
            )
        if sample_index not in seen:
            seen.add(sample_index)
            unique_indices.append(sample_index)

    graph = idqnn_connectivity(n1, m)

    print(f"Loaded: {test_path}")
    print(f"Number of test samples: {len(x_flip)}")
    print(f"Sample indices from script: {unique_indices}")
    print(f"Selected bit value: {selected_bit_value}")

    for sample_index in unique_indices:
        sample_grid = x_flip.reshape(-1, n1, m)[sample_index]
        sample_label = int(y_test[sample_index])
        reference_grid = encode_diagonal(sample_label, n=n1)

        selected_nodes, selected_edges, components = get_selected_components(
            bit_grid=sample_grid,
            edges=graph["all_edges"],
            selected_bit_value=selected_bit_value,
        )

        print()
        print("=" * 72)
        print(f"Selected sample index: {sample_index}")
        print(f"Selected sample label: {sample_label}")
        print(f"Reference label for diagonal encoding: {sample_label}")
        print(f"Number of {selected_bit_value}-bits: {len(selected_nodes)}")
        print(
            f"Edges among {selected_bit_value}-bits under ISQNN connectivity: "
            f"{len(selected_edges)}"
        )
        print_component_report(
            components=components,
            width=m,
            selected_bit_value=selected_bit_value,
        )

        output_path = (
            ROOT
            / "output_images"
            / (
                f"mnist_test_{sample_index}_connectivity_bit_{selected_bit_value}"
                f"_with_label_{sample_label}_diagonal_threshold_{threshold}.png"
            )
        )
        diagonal_indices, overlap_indices = plot_connectivity_with_reference_diagonal(
            bit_grid=sample_grid,
            sample_index=sample_index,
            sample_label=sample_label,
            components=components,
            selected_edges=selected_edges,
            selected_nodes=selected_nodes,
            reference_grid=reference_grid,
            output_path=output_path,
        )
        print_diagonal_report(
            diagonal_indices=diagonal_indices,
            overlap_indices=overlap_indices,
            width=m,
            label_value=sample_label,
            selected_bit_value_local=selected_bit_value,
        )


if __name__ == "__main__":
    main()
