"""Identify ansatz parameters that cannot affect the block-local loss."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import sys

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from numpy.typing import NDArray

from task2_code_auto.ansatz import CZ_LAYER_PARITIES, N_LAYERS, PARAMS_PER_ROTATION, cz_pairs_for_layer


IntArray = NDArray[np.int64]
Manifest = Mapping[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find ansatz parameters invisible to a block-local loss.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment-root", type=Path, help="Experiment directory containing manifest.json.")
    source.add_argument("--manifest", type=Path, help="Path to a manifest JSON file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <experiment-root>/summary/invisible_parameter_analysis.json.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, object]:
    data: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    if not all(isinstance(key, str) for key in data):
        raise ValueError(f"manifest keys must be strings: {path}")
    return {str(key): value for key, value in data.items()}


def _required_int(manifest: Manifest, key: str) -> int:
    value = manifest[key]
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _as_int_list(value: object, name: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"{name} must be a sequence")
    result: list[int] = []
    for item in value:
        if not isinstance(item, int):
            raise ValueError(f"{name} must contain integers")
        result.append(item)
    return result


def block_local_positions(manifest: Manifest) -> set[int]:
    block_qubits = _as_int_list(manifest["block_qubits"], "block_qubits")
    if bool(manifest.get("block_only_ansatz", False)):
        return set(range(len(block_qubits)))

    lightcone_qubits = _as_int_list(manifest["lightcone_qubits"], "lightcone_qubits")
    missing = [q for q in block_qubits if q not in lightcone_qubits]
    if missing:
        raise ValueError(f"block qubits not in lightcone: {missing}")
    return {lightcone_qubits.index(q) for q in block_qubits}


def analyze_invisible_parameters(manifest: Manifest) -> tuple[IntArray, IntArray, dict[str, object]]:
    """Return visible and invisible 1-based coordinates with metadata.

    Rx/Ry parameters use ordinary backward causal connectivity. Rz is treated
    separately because it is the last gate in each Rx-Ry-Rz block and commutes
    with the immediately following CZ layer, so it cannot use that CZ layer to
    spread support toward the measured block.
    """
    n_qubits = _required_int(manifest, "ansatz_qubits")
    theta_size = _required_int(manifest, "theta_size")
    expected = N_LAYERS * n_qubits * PARAMS_PER_ROTATION
    if theta_size != expected:
        raise ValueError(f"theta_size={theta_size} is inconsistent with {N_LAYERS} layers and {n_qubits} qubits")

    active = set(block_local_positions(manifest))
    layer_active_sets: list[set[int]] = []
    for layer in range(N_LAYERS - 1, -1, -1):
        layer_active_sets.append(set(active))
        if layer > 0:
            for a, b in cz_pairs_for_layer(n_qubits, layer - 1):
                if a in active or b in active:
                    active.add(a)
                    active.add(b)
    layer_active_sets.reverse()

    visible = np.zeros(theta_size, dtype=bool)
    layer_z_active_sets: list[set[int]] = []
    for layer, layer_active in enumerate(layer_active_sets):
        z_active = layer_active_sets[layer + 1] if layer + 1 < N_LAYERS else layer_active
        layer_z_active_sets.append(set(z_active))

        for local in layer_active:
            start = (layer * n_qubits + local) * PARAMS_PER_ROTATION
            visible[start] = True
            visible[start + 1] = True
        for local in z_active:
            start = (layer * n_qubits + local) * PARAMS_PER_ROTATION
            visible[start + 2] = True

    coords = np.arange(1, theta_size + 1, dtype=np.int64)
    visible_coords = coords[visible]
    invisible_coords = coords[~visible]
    metadata: dict[str, object] = {
        "rule": "backward causal connectivity with Rz-CZ commutation: Rx/Ry use current active set; Rz skips the immediately following CZ layer",
        "ansatz_qubits": n_qubits,
        "theta_size": theta_size,
        "cz_layer_parities": list(CZ_LAYER_PARITIES),
        "block_local_positions": sorted(block_local_positions(manifest)),
        "layer_active_before_rotation": [sorted(layer_active) for layer_active in layer_active_sets],
        "layer_z_active_before_rotation": [sorted(layer_active) for layer_active in layer_z_active_sets],
        "visible_count": int(visible_coords.size),
        "invisible_count": int(invisible_coords.size),
        "visible_coordinates": visible_coords.tolist(),
        "invisible_coordinates": invisible_coords.tolist(),
    }
    return visible_coords, invisible_coords, metadata


def default_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    if args.experiment_root is None:
        raise ValueError("--output is required when using --manifest")
    return args.experiment_root / "summary" / "invisible_parameter_analysis.json"


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest if args.manifest is not None else args.experiment_root / "manifest.json"
    manifest = load_manifest(manifest_path)
    _, invisible_coords, metadata = analyze_invisible_parameters(manifest)
    output_path = default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Removed {len(invisible_coords)} invisible parameters: {invisible_coords.tolist()}")
    print(f"Kept {metadata['visible_count']} visible parameters")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
