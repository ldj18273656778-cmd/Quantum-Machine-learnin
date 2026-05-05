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


threshold = 0.4
n1 = 10
m = 10
test_number = 99


def build_adjacency(n: int, edges: list[tuple[int, int]]) -> list[set[int]]:
    adjacency: list[set[int]] = [set() for _ in range(n)]
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency


def get_selected_components(
    bit_grid: np.ndarray,
    edges: list[tuple[int, int]],
    selected_bit_value: int,
) -> tuple[list[int], list[tuple[int, int]], list[list[int]]]:
    flat_bits = bit_grid.reshape(-1)
    selected_nodes = np.flatnonzero(flat_bits == selected_bit_value).tolist()
    selected_set = set(selected_nodes)
    selected_edges = [(a, b) for a, b in edges if a in selected_set and b in selected_set]
    adjacency = build_adjacency(flat_bits.size, selected_edges)

    components: list[list[int]] = []
    visited: set[int] = set()

    for node in selected_nodes:
        if node in visited:
            continue

        stack = [node]
        visited.add(node)
        component: list[int] = []

        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        components.append(sorted(component))

    components.sort(key=lambda component: (-len(component), component[0]))
    return selected_nodes, selected_edges, components


def bit_to_coord(bit_index: int, width: int) -> tuple[int, int]:
    return divmod(bit_index, width)


def print_component_report(
    components: list[list[int]],
    width: int,
    selected_bit_value: int,
) -> None:
    if not components:
        print(f"No {selected_bit_value}-bits were found in the selected sample.")
        return

    print(f"Connected components among {selected_bit_value}-bits (0-based indices):")
    for component_id, component in enumerate(components, start=1):
        coords = [bit_to_coord(bit_index, width) for bit_index in component]
        print(
            f"  Component {component_id:02d} | size={len(component):2d} "
            f"| bits={component} | coords={coords}"
        )


def style_grid_axis(ax: plt.Axes, rows: int, cols: int) -> None:
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect("equal")


def plot_connectivity(
    bit_grid: np.ndarray,
    label: int,
    components: list[list[int]],
    selected_edges: list[tuple[int, int]],
    selected_bit_value: int,
    output_path: Path,
) -> None:
    rows, cols = bit_grid.shape
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    ax_image, ax_graph = axes

    ax_image.imshow(bit_grid, cmap="gray", interpolation="nearest")
    ax_image.set_title(f"test x bitstring #{test_number} | label={label}")
    style_grid_axis(ax_image, rows, cols)

    ax_graph.imshow(bit_grid, cmap="gray", alpha=0.25, interpolation="nearest")
    style_grid_axis(ax_graph, rows, cols)

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
            ax_graph.plot(
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

        ax_graph.scatter(
            cols_in_component,
            rows_in_component,
            s=marker_size,
            c=[color],
            marker="s",
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )

    ax_graph.set_title(
        f"ISQNN connectivity on {selected_bit_value}-bits\n"
        f"selected bits={sum(len(component) for component in components)}, "
        f"selected edges={len(selected_edges)}, components={len(components)}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved connectivity figure to: {output_path}")
    if "agg" in plt.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    test_path = MNIST_DIR / "data" / f"test_MNIST_10x10_binarize{threshold}.npy"
    label_path = MNIST_DIR / "data" / "test_MNIST_labels.npy"
    selected_bit_value = 0

    x_test = np.load(test_path).reshape(-1, n1 * m).astype(int)
    y_test = np.load(label_path)
    x_flip = 1 - x_test

    if not (0 <= test_number < len(x_flip)):
        raise IndexError(f"test_number={test_number} is out of range for {len(x_flip)} samples.")

    sample_grid = x_flip.reshape(-1, n1, m)[test_number]
    label = int(y_test[test_number])

    graph = idqnn_connectivity(n1, m)
    selected_nodes, selected_edges, components = get_selected_components(
        bit_grid=sample_grid,
        edges=graph["all_edges"],
        selected_bit_value=selected_bit_value,
    )

    print(f"Loaded: {test_path}")
    print(f"Number of test samples: {len(x_flip)}")
    print(f"Selected sample index: {test_number}")
    print(f"Selected label: {label}")
    print(f"Selected bit value: {selected_bit_value}")
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
            f"mnist_test_{test_number}_connectivity_"
            f"bit_{selected_bit_value}_threshold_{threshold}.png"
        )
    )
    plot_connectivity(
        bit_grid=sample_grid,
        label=label,
        components=components,
        selected_edges=selected_edges,
        selected_bit_value=selected_bit_value,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
