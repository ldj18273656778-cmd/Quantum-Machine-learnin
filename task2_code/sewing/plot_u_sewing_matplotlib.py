"""Render U_sewing circuits as Matplotlib PNG images.

Usage:
    python task2_code/sewing/plot_u_sewing_matplotlib.py

The text circuit produced by Cirq can be too wide to read in a plain `.txt`
file.  This script rebuilds the same circuit as `block_sewing.py` and saves it
as a single PNG by default.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from task2_code.sewing.block_sewing import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PARAMS,
    build_block_sew_circuit,
    load_block_specs,
    resolve_order,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot U_sewing circuit with Matplotlib.")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--order", choices=["odd-even", "metadata", "reverse"], default="odd-even")
    parser.add_argument("--dagger-convention", choices=["trained", "inverse"], default="inverse")
    parser.add_argument("--max-columns", type=int, default=150)
    parser.add_argument("--paginate", action="store_true")
    parser.add_argument("--font-size", type=float, default=5.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _split_diagram(diagram: str, max_columns: int) -> list[str]:
    if max_columns <= 20:
        raise ValueError(f"max_columns must be greater than 20, got {max_columns}")
    lines = diagram.splitlines()
    if not lines:
        return [""]
    width = max(len(line) for line in lines)
    pages = []
    for start in range(0, width, max_columns):
        stop = start + max_columns
        page_lines = [line[start:stop].rstrip() for line in lines]
        pages.append("\n".join(page_lines))
    return pages


def _save_page(text: str, path: Path, title: str, font_size: float, dpi: int) -> None:
    lines = text.splitlines() or [""]
    max_len = max(len(line) for line in lines)
    width = max(12.0, min(40.0, max_len * font_size / 45.0))
    height = max(8.0, min(60.0, len(lines) * font_size / 8.0 + 1.5))
    fig = plt.figure(figsize=(width, height))
    fig.suptitle(title, fontsize=10)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(
        0.01,
        0.99,
        text,
        family="monospace",
        fontsize=font_size,
        va="top",
        ha="left",
        transform=ax.transAxes,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    n_qubits, specs, metadata = load_block_specs(args.params, args.metadata)
    ordered_specs = resolve_order(specs, args.order)
    circuit = build_block_sew_circuit(n_qubits, ordered_specs, args.dagger_convention)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_timestamp = str(metadata.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S")))
    stem = f"U_sewing_block-sew_{args.order}_{args.dagger_convention}_{source_timestamp}"
    diagram = circuit.to_text_diagram(transpose=False)
    pages = _split_diagram(diagram, args.max_columns) if args.paginate else [diagram]

    output_paths = []
    for index, page in enumerate(pages, start=1):
        suffix = f"_page_{index:03d}_of_{len(pages):03d}" if args.paginate else ""
        path = args.output_dir / f"{stem}{suffix}.png"
        title = f"{stem}  page {index}/{len(pages)}"
        _save_page(page, path, title, args.font_size, args.dpi)
        output_paths.append(path)

    print("Rendered {0} image(s) for block-sew circuit".format(len(output_paths)))
    print(f"operation_count = {sum(1 for _ in circuit.all_operations())}, moments = {len(circuit)}")
    for path in output_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
