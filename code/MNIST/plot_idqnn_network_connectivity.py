from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from sampling.ISQNN_generate_y import idqnn_connectivity


def ensure_interactive_backend() -> None:
    current_backend = matplotlib.get_backend().lower()
    if "agg" not in current_backend:
        return

    for candidate in ("qtagg", "tkagg"):
        try:
            plt.switch_backend(candidate)
            print(f"Switched matplotlib backend to: {plt.get_backend()}")
            return
        except Exception:
            continue

    raise RuntimeError(
        "Matplotlib is using a non-interactive backend, and no interactive backend "
        "could be enabled. Please run in a local desktop Python environment."
    )


def display_index_to_coord(bit_index: int, m: int) -> tuple[int, int]:
    return divmod(bit_index, m)


def style_axis(ax: plt.Axes, n1: int, m: int) -> None:
    ax.set_xticks(np.arange(m))
    ax.set_yticks(np.arange(n1))
    ax.set_xticks(np.arange(-0.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n1, 1), minor=True)
    ax.grid(which="minor", color="#d4d4d8", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlim(-0.5, m - 0.5)
    ax.set_ylim(n1 - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("column")
    ax.set_ylabel("row")


def draw_edges(
    ax: plt.Axes,
    edges: list[tuple[int, int]],
    m: int,
    color: str,
    linewidth: float,
) -> None:
    for a, b in edges:
        row_a, col_a = display_index_to_coord(a, m)
        row_b, col_b = display_index_to_coord(b, m)
        ax.plot(
            [col_a, col_b],
            [row_a, row_b],
            color=color,
            linewidth=linewidth,
            alpha=0.9,
            zorder=1,
        )


def annotate_nodes(ax: plt.Axes, n1: int, m: int) -> None:
    font_size = float(np.clip(14 - 0.4 * max(n1, m), 5, 11))
    pad = float(np.clip(0.28 - 0.01 * max(n1, m), 0.14, 0.24))

    for bit_index in range(n1 * m):
        row, col = display_index_to_coord(bit_index, m)
        ax.text(
            col,
            row,
            str(bit_index),
            ha="center",
            va="center",
            fontsize=font_size,
            color="black",
            zorder=3,
            bbox={
                "boxstyle": f"round,pad={pad}",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 0.9,
            },
        )


def plot_network_connectivity(n1: int, m: int) -> None:
    ensure_interactive_backend()

    graph = idqnn_connectivity(n1, m)
    intra_edges = graph["intra_slice_edges"]
    inter_edges = graph["inter_slice_edges"]

    fig_width = float(np.clip(m + 2.5, 7, 18))
    fig_height = float(np.clip(n1 + 2.0, 6, 18))
    line_width = float(np.clip(2.6 - 0.05 * max(n1, m), 1.3, 2.4))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    style_axis(ax, n1, m)

    draw_edges(
        ax=ax,
        edges=intra_edges,
        m=m,
        color="#2563eb",
        linewidth=line_width,
    )
    draw_edges(
        ax=ax,
        edges=inter_edges,
        m=m,
        color="#f97316",
        linewidth=line_width,
    )
    annotate_nodes(ax, n1, m)

    legend_handles = [
        Line2D([0], [0], color="#2563eb", lw=line_width, label="intra-slice edges"),
        Line2D([0], [0], color="#f97316", lw=line_width, label="inter-slice edges"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
        ncol=2,
    )
    ax.set_title(
        f"IDQNN Connectivity Structure ({n1} x {m})\n"
        "bit indices increase left-to-right, then top-to-bottom, starting from 0"
    )

    print(f"n1={n1}, m={m}, total bits={graph['n']}")
    print(f"intra-slice edges: {len(graph['intra_slice_edges'])}")
    print(f"inter-slice edges: {len(graph['inter_slice_edges'])}")
    print(
        "Displayed bit index rule: bit = row * m + column "
        "(left-to-right first, 0-based)."
    )

    plt.show(block=True)


def main() -> None:
    plot_network_connectivity(n1=n1, m=m)


if __name__ == "__main__":
    n1 = 10
    m = 10
    print(f"matplotlib backend: {plt.get_backend()}")
    main()
