"""Plot two-point joint-correlation deviations for two n=12 sewing runs.

The two-point value here is the joint expectation <X_i X_j>, not the product
of independently measured single-qubit expectations <X_i> * <X_j>.

Examples:
    python task2_code/compare_two_point_heatmap.py --run-dir1 <warmstart_run_dir>
    python task2_code/compare_two_point_heatmap.py --run-dir1 <warmstart_run_dir> --output task2_code/data/two_point_heatmap.png
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import NamedTuple

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cirq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from numpy.typing import NDArray

from task2_code.run_sew_and_compare import (
    _basis_state,
    _ghz_state,
)
from task2_code.sewing.block_sewing import (
    build_block_sew_circuit,
    load_block_specs,
    resolve_order,
)
from task2_code.target_factory import build_target_from_metadata

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

DEFAULT_RUN_DIR2 = Path("task2_code/data/task2_training_n12_3blocks_superoperator_from_mix_20260530_150036")
DEFAULT_OUTPUT_NAME = "two_point_correlation_deviation_heatmap.png"
PAULI_ORDER = ("X", "Y", "Z")


class RunHeatmap(NamedTuple):
    title: str
    matrix: FloatArray
    max_delta: float
    mean_delta: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare n=12 two-point joint-correlation deviations for two U_sewing runs."
    )
    parser.add_argument(
        "--run-dir1",
        type=Path,
        required=True,
        help="Heisenberg warmstart run directory.",
    )
    parser.add_argument(
        "--run-dir2",
        type=Path,
        default=DEFAULT_RUN_DIR2,
        help="Mixed reduced-channel run directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path; defaults to <run-dir1>/two_point_correlation_deviation_heatmap.png.",
    )
    parser.add_argument(
        "--input-state",
        default="ghz",
        help="Input state: zero, ones, ghz, or basis:<int>.",
    )
    parser.add_argument(
        "--dagger-convention",
        choices=("trained", "inverse"),
        default="trained",
    )
    return parser.parse_args()


def _prepare_system_state(input_state: str, n_qubits: int) -> ComplexArray:
    value = input_state.strip().lower()
    if value == "zero":
        return _basis_state(0, n_qubits)
    if value == "ones":
        return _basis_state((1 << n_qubits) - 1, n_qubits)
    if value == "ghz":
        return _ghz_state(n_qubits)
    if value.startswith("basis:"):
        return _basis_state(int(value.split(":", 1)[1]), n_qubits)
    raise ValueError("input-state must be 'zero', 'ones', 'ghz', or 'basis:<int>'")


def _labels(n_qubits: int) -> list[str]:
    return [f"X{qubit}" for qubit in range(n_qubits)]


def _x_pair_expectation_matrix(state: ComplexArray, n_qubits: int, total_qubits: int) -> FloatArray:
    flat_state = state.reshape(-1)
    indices = np.arange(flat_state.size, dtype=np.int64)
    bit_masks = [1 << (total_qubits - 1 - qubit) for qubit in range(n_qubits)]

    matrix = np.empty((n_qubits, n_qubits), dtype=np.float64)
    for i, mask_i in enumerate(bit_masks):
        for j, mask_j in enumerate(bit_masks):
            mask = mask_i ^ mask_j
            if mask == 0:
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = float(np.vdot(flat_state, flat_state[indices ^ mask]).real)
    return matrix


def _load_run_heatmap(
    run_dir: Path,
    title: str,
    input_state: str,
    dagger_convention: str,
) -> RunHeatmap:
    params_path = run_dir / "params.npz"
    metadata_path = run_dir / "metadata.json"
    n_qubits, specs, metadata = load_block_specs(params_path, metadata_path)
    if n_qubits != 12:
        raise ValueError(f"expected n=12 run at {run_dir}, got n={n_qubits}")

    print(f"  Building circuits ...", flush=True)
    target = build_target_from_metadata(metadata)
    ansatz_name = str(metadata.get("ansatz", "default_5layer_cz"))
    sewing_circuit = build_block_sew_circuit(
        n_qubits,
        resolve_order(specs, "odd-even"),
        dagger_convention,
        ansatz_name=ansatz_name,
    )

    system_state = _prepare_system_state(input_state, n_qubits)
    ancilla_state = _basis_state(0, n_qubits)
    sewing_initial = np.kron(system_state, ancilla_state)

    simulator = cirq.Simulator(dtype=np.complex128)
    print(f"  Simulating U_target (n={n_qubits}) ...", flush=True)
    target_result = simulator.simulate(
        target.circuit,
        qubit_order=list(cirq.LineQubit.range(n_qubits)),
        initial_state=system_state,
    )
    target_state = np.asarray(target_result.final_state_vector, dtype=np.complex128)
    print(f"  Computing target <X_i X_j> matrix ...", flush=True)
    target_corr = _x_pair_expectation_matrix(target_state, n_qubits, n_qubits)
    del target_result, target_state
    gc.collect()

    print(f"  Simulating U_sew (n={2*n_qubits}) ...", flush=True)
    sewing_result = simulator.simulate(
        sewing_circuit,
        qubit_order=list(cirq.LineQubit.range(2 * n_qubits)),
        initial_state=sewing_initial,
    )
    sewing_state = np.asarray(sewing_result.final_state_vector, dtype=np.complex128)
    print(f"  Computing sewing <X_i X_j> matrix ...", flush=True)
    sewing_corr = _x_pair_expectation_matrix(sewing_state, n_qubits, 2 * n_qubits)
    del sewing_result, sewing_state, sewing_initial
    gc.collect()

    matrix = np.abs(target_corr - sewing_corr)
    return RunHeatmap(title, matrix, float(np.max(matrix)), float(np.mean(matrix)))


def _add_block_boundaries(ax: plt.Axes, n_qubits: int, block_sizes: tuple[int, ...] = (4, 4, 4)) -> None:
    cum = 0
    for size in block_sizes[:-1]:
        cum += size
        index = cum - 0.5
        ax.axhline(index, color="white", linewidth=0.8)
        ax.axvline(index, color="white", linewidth=0.8)
        ax.axhline(index, color="black", linewidth=0.35, alpha=0.7)
        ax.axvline(index, color="black", linewidth=0.35, alpha=0.7)


def plot_heatmaps(results: list[RunHeatmap], labels: list[str], output_path: Path, input_state: str) -> None:
    vmax = max(result.max_delta for result in results)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    images = []
    for ax, result in zip(axes, results):
        matrix_clipped = np.maximum(result.matrix, 1e-8)
        image = ax.imshow(matrix_clipped, origin="upper", cmap="magma",
                         norm=LogNorm(vmin=1e-8, vmax=vmax))
        images.append(image)
        ax.set_title(f"{result.title}\nmax $|\\Delta|$={result.max_delta:.3g}, mean $|\\Delta|$={result.mean_delta:.3g}")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel(r"$X_j$")
        ax.set_ylabel(r"$X_i$")
        _add_block_boundaries(ax, len(labels))

    colorbar = fig.colorbar(images[0], ax=axes, shrink=0.86, pad=0.02)
    colorbar.set_label(r"$|\Delta\langle X_i X_j\rangle|$  (log scale)")
    fig.suptitle(f"Two-point joint-correlation deviations on {input_state} input (n=12)", fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_path = args.output or (args.run_dir1 / DEFAULT_OUTPUT_NAME)

    print(f"Input state: {args.input_state}, dagger convention: {args.dagger_convention}")
    results = []
    for i, (run_dir, title) in enumerate([
        (args.run_dir1, "Heisenberg warmstart"),
        (args.run_dir2, "Mixed reduced-channel"),
    ]):
        print(f"\n[{i+1}/2] Loading {title}: {run_dir}")
        results.append(_load_run_heatmap(run_dir, title, args.input_state, args.dagger_convention))

    print(f"\nComputing 12x12 X-X correlation matrices ...", flush=True)
    labels = _labels(12)
    print(f"Plotting heatmaps ...", flush=True)
    plot_heatmaps(results, labels, output_path, args.input_state)

    for result in results:
        print(f"{result.title}: max |Δ|={result.max_delta:.8g}, mean |Δ|={result.mean_delta:.8g}")
    print(f"Plot saved: {output_path}")

    json_path = output_path.with_suffix(".json")
    json_data = {
        "input_state": args.input_state,
        "dagger_convention": args.dagger_convention,
        "correlation_type": "joint_expectation_<X_i_X_j>",
        "labels": labels,
        "runs": [
            {
                "title": result.title,
                "matrix": result.matrix.tolist(),
                "max_delta": float(result.max_delta),
                "mean_delta": float(result.mean_delta),
            }
            for result in results
        ],
    }
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"Data saved: {json_path}")


if __name__ == "__main__":
    main()
