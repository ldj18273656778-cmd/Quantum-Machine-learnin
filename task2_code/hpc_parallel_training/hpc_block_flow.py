"""Shared helpers for HPC-friendly per-block Task 2 training.

The parallel flow keeps block jobs independent and writes a final bundle that
matches the legacy ``params.npz`` / ``metadata.json`` sewing contract.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from task2_code.experiment_config import (
    ExperimentConfig,
    N4_SINGLE_BLOCK,
    N4_SINGLE_BLOCK_HEISENBERG,
    N12_3BLOCKS,
    N12_3BLOCKS_CNOT_MIXED,
    N12_3BLOCKS_HEISENBERG,
    N12_3BLOCKS_ONE,
    N12_3BLOCKS_ZERO,
    N20_5BLOCKS,
    N20_5BLOCKS_HEISENBERG,
    N24_6BLOCKS,
    N24_6BLOCKS_HEISENBERG,
    N28_7BLOCKS,
    N28_7BLOCKS_HEISENBERG,
    N32_8BLOCKS,
    N32_8BLOCKS_HEISENBERG,
)
from task2_code.loss_registry import loss_function_uses_superoperator


PRESETS: dict[str, ExperimentConfig] = {
    "n4_single_block": N4_SINGLE_BLOCK,
    "n4_single_block_heisenberg": N4_SINGLE_BLOCK_HEISENBERG,
    "n12_3blocks": N12_3BLOCKS,
    "n12_3blocks_cnot_mixed": N12_3BLOCKS_CNOT_MIXED,
    "n12_3blocks_heisenberg": N12_3BLOCKS_HEISENBERG,
    "n12_3blocks_zero": N12_3BLOCKS_ZERO,
    "n12_3blocks_one": N12_3BLOCKS_ONE,
    "n20_5blocks": N20_5BLOCKS,
    "n20_5blocks_heisenberg": N20_5BLOCKS_HEISENBERG,
    "n24_6blocks": N24_6BLOCKS,
    "n24_6blocks_heisenberg": N24_6BLOCKS_HEISENBERG,
    "n28_7blocks": N28_7BLOCKS,
    "n28_7blocks_heisenberg": N28_7BLOCKS_HEISENBERG,
    "n32_8blocks": N32_8BLOCKS,
    "n32_8blocks_heisenberg": N32_8BLOCKS_HEISENBERG,
}


def safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    tmp.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    tmp.replace(path)


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.{uuid4().hex}.tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parallel_experiment_name(preset: str, cfg: ExperimentConfig, stamp: str) -> str:
    parts = ["task2_parallel", safe_stem(preset), safe_stem(cfg.ansatz), safe_stem(cfg.loss_function)]
    if loss_function_uses_superoperator(cfg.loss_function):
        parts.append(safe_stem(cfg.superoperator))
    parts.append(stamp)
    return "_".join(parts)


def bundle_name(preset: str, cfg: ExperimentConfig, stamp: str) -> str:
    parts = ["task2_training", safe_stem(preset), safe_stem(cfg.ansatz), safe_stem(cfg.loss_function)]
    if loss_function_uses_superoperator(cfg.loss_function):
        parts.append(safe_stem(cfg.superoperator))
    parts.append(stamp)
    return "_".join(parts)


def config_to_manifest(preset: str, cfg: ExperimentConfig) -> dict[str, Any]:
    data = asdict(cfg)
    data["output_dir"] = str(cfg.output_dir)
    data["data_dir"] = str(cfg.data_dir)
    data["preset"] = preset
    data["loss_function_uses_superoperator"] = loss_function_uses_superoperator(cfg.loss_function)
    return data


def config_from_manifest(manifest: dict[str, Any]) -> ExperimentConfig:
    preset = str(manifest["preset"])
    if preset not in PRESETS:
        raise ValueError(f"unknown preset in manifest: {preset!r}")
    cfg = PRESETS[preset]
    updates: dict[str, Any] = {}
    for field in [
        "n_qubits",
        "blocks",
        "target_bits",
        "target_seed",
        "time_k",
        "lightcone_mode",
        "loss_mode",
        "radius",
        "ansatz",
        "block_only_ansatz",
        "superoperator",
        "loss_function",
        "iterations",
        "lr",
        "training_seed_start",
        "max_restarts",
        "success_threshold",
    ]:
        if field in manifest:
            updates[field] = manifest[field]
    return replace(cfg, **updates)


def load_manifest(experiment_root: Path) -> dict[str, Any]:
    path = experiment_root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    return data


def block_dir(experiment_root: Path, block_index: int) -> Path:
    return experiment_root / "blocks" / f"block_{int(block_index):02d}"


def block_result_paths(experiment_root: Path, block_index: int) -> tuple[Path, Path]:
    directory = block_dir(experiment_root, block_index)
    return directory / "best.npz", directory / "result.json"


def parse_block_index(value: int, block_count: int) -> int:
    idx = int(value)
    if idx < 1 or idx > block_count:
        raise ValueError(f"block index must be in 1..{block_count}, got {idx}")
    return idx
