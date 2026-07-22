"""Create publication-style figures for local observable comparison results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from numpy.typing import NDArray


PAULI_ORDER = ["X", "Y", "Z"]
PAULI_COLORS = {"X": "#1f77b4", "Y": "#d62728", "Z": "#2ca02c"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot academic figures for merged local observable comparisons.")
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to merged-json parent")
    parser.add_argument("--prefix", default=None, help="output filename prefix")
    parser.add_argument("--title", default="N=32 local observable comparison")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", default="png,pdf", help="comma-separated output formats")
    return parser.parse_args()


def _load_payload(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"merged JSON must contain an object: {path}")
    if "rows" not in data or not isinstance(data["rows"], list):
        raise ValueError("merged JSON must contain a rows list")
    return data


def _matrix(rows: list[dict[str, Any]], field: str, qubits: list[int], paulis: list[str]) -> NDArray[np.float64]:
    lookup = {(int(row["qubit"]), str(row["pauli"])): float(row[field]) for row in rows}
    arr = np.empty((len(paulis), len(qubits)), dtype=float)
    for pi, pauli in enumerate(paulis):
        for qi, qubit in enumerate(qubits):
            arr[pi, qi] = lookup[(qubit, pauli)]
    return arr


def _save(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.22, linewidth=0.6)


def plot_expectations(
    qubits: list[int],
    paulis: list[str],
    target: NDArray[np.float64],
    sewing: NDArray[np.float64],
    title: str,
) -> plt.Figure:
    fig, axes = plt.subplots(len(paulis), 1, figsize=(9.0, 6.8), sharex=True)
    if len(paulis) == 1:
        axes = np.asarray([axes])
    for idx, pauli in enumerate(paulis):
        ax = axes[idx]
        color = PAULI_COLORS.get(pauli, "#1f77b4")
        ax.plot(qubits, target[idx], marker="o", markersize=3.8, linewidth=1.4, color=color, label="Target")
        ax.plot(qubits, sewing[idx], marker="s", markersize=3.4, linewidth=1.2, linestyle="--", color="#333333", label="Sewing")
        ax.set_ylabel(rf"$\langle {pauli}_i \rangle$")
        ax.set_ylim(-1.05, 1.05)
        _style_axes(ax)
        ax.legend(frameon=False, loc="best", fontsize=8)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Qubit index $i$")
    axes[-1].set_xticks(qubits)
    fig.tight_layout()
    return fig


def plot_abs_errors(qubits: list[int], paulis: list[str], abs_diff: NDArray[np.float64], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    for idx, pauli in enumerate(paulis):
        ax.plot(
            qubits,
            abs_diff[idx],
            marker="o",
            markersize=3.6,
            linewidth=1.3,
            color=PAULI_COLORS.get(pauli, None),
            label=rf"$|\Delta\langle {pauli}_i\rangle|$",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Qubit index $i$")
    ax.set_ylabel("Absolute deviation")
    ax.set_xticks(qubits)
    ax.set_title(title)
    _style_axes(ax)
    ax.legend(frameon=False, ncol=min(3, len(paulis)), fontsize=8)
    fig.tight_layout()
    return fig


def plot_heatmap(qubits: list[int], paulis: list[str], abs_diff: NDArray[np.float64], title: str) -> plt.Figure:
    positive = abs_diff[abs_diff > 0]
    vmin = max(float(np.min(positive)) if positive.size else 1e-8, 1e-8)
    vmax = max(float(np.max(abs_diff)), vmin * 10.0)
    fig, ax = plt.subplots(figsize=(9.0, 2.6))
    im = ax.imshow(abs_diff, aspect="auto", cmap="magma", norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.set_yticks(np.arange(len(paulis)))
    ax.set_yticklabels([rf"$|\Delta\langle {p}_i\rangle|$" for p in paulis])
    ax.set_xticks(np.arange(len(qubits)))
    ax.set_xticklabels([str(q) for q in qubits], fontsize=8)
    ax.set_xlabel("Qubit index $i$")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("Absolute deviation")
    fig.tight_layout()
    return fig


def plot_summary(summary: dict[str, Any], paulis: list[str], title: str) -> plt.Figure:
    labels = [p for p in paulis if p in summary] + ["all"]
    metrics = ["max_abs_diff", "mean_abs_diff", "rms_abs_diff"]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for offset, metric in enumerate(metrics):
        values = [float(summary[label][metric]) for label in labels]
        ax.bar(x + (offset - 1) * width, values, width=width, label=metric.replace("_", " "))
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([rf"${label}$" if label != "all" else "all" for label in labels])
    ax.set_ylabel("Deviation")
    ax.set_title(title)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    payload = _load_payload(args.merged_json)
    rows = [dict(row) for row in payload["rows"]]
    qubits = sorted({int(row["qubit"]) for row in rows})
    paulis = [p for p in PAULI_ORDER if p in {str(row["pauli"]) for row in rows}]
    if not paulis:
        raise ValueError("no supported Pauli rows found")

    target = _matrix(rows, "target_expectation", qubits, paulis)
    sewing = _matrix(rows, "sewing_expectation", qubits, paulis)
    abs_diff = _matrix(rows, "abs_diff", qubits, paulis)
    output_dir = args.output_dir or args.merged_json.parent
    prefix = args.prefix or args.merged_json.stem
    formats = [fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip()]

    saved: list[Path] = []
    saved += _save(
        plot_expectations(qubits, paulis, target, sewing, f"{args.title}: target vs sewing"),
        output_dir,
        f"{prefix}_expectations",
        formats,
        args.dpi,
    )
    saved += _save(
        plot_abs_errors(qubits, paulis, abs_diff, f"{args.title}: local observable deviations"),
        output_dir,
        f"{prefix}_abs_error_curves",
        formats,
        args.dpi,
    )
    saved += _save(
        plot_heatmap(qubits, paulis, abs_diff, f"{args.title}: deviation heatmap"),
        output_dir,
        f"{prefix}_abs_error_heatmap",
        formats,
        args.dpi,
    )
    saved += _save(
        plot_summary(dict(payload.get("summary", {})), paulis, f"{args.title}: aggregate deviations"),
        output_dir,
        f"{prefix}_summary",
        formats,
        args.dpi,
    )

    for path in saved:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
