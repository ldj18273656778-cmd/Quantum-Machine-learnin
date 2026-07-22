"""Launch one rematerialized JAX CPU block-training job."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


THREAD_LIMITS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
TRAIN_BLOCK_PATH = Path(__file__).resolve().with_name("train_block.py")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch one rematerialized JAX CPU block-training job.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--block-index", type=int, required=True)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--restarts", type=int)
    parser.add_argument("--seed-offset", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--success-threshold", type=float)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--plot-loss", action="store_true")
    parser.add_argument("--telemetry-path", type=Path)
    parser.add_argument("--append-telemetry", action="store_true")
    return parser.parse_args(argv)


def _validated_block_directory(root: Path, block_index: int) -> Path:
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise ValueError(f"manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid manifest: {manifest_path}") from error
    if not isinstance(payload, dict):
        raise ValueError("manifest must contain a JSON object")
    block_specs = payload.get("block_specs")
    if not isinstance(block_specs, list) or not block_specs:
        raise ValueError("manifest block_specs must be a non-empty list")
    if not 1 <= block_index <= len(block_specs):
        raise ValueError(f"block-index must be in 1..{len(block_specs)}")
    directory = root / "blocks" / f"block_{block_index:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _reserve_default_telemetry(block_directory: Path) -> Path:
    telemetry_directory = block_directory / "telemetry"
    telemetry_directory.mkdir(parents=True, exist_ok=True)
    while True:
        path = telemetry_directory / f"jax_{uuid.uuid4()}.jsonl"
        try:
            with path.open("x", encoding="utf-8"):
                pass
        except FileExistsError:
            continue
        return path


def _telemetry_path(args: argparse.Namespace, block_directory: Path) -> Path:
    if args.append_telemetry and args.telemetry_path is None:
        raise ValueError("--append-telemetry requires --telemetry-path")
    if args.telemetry_path is None:
        return _reserve_default_telemetry(block_directory)
    path: Path = args.telemetry_path
    if not path.parent.is_dir():
        raise ValueError(f"telemetry parent does not exist: {path.parent}")
    if path.is_dir():
        raise ValueError(f"telemetry path is a directory: {path}")
    if path.exists() and not args.append_telemetry:
        raise ValueError("existing telemetry requires --append-telemetry")
    if not path.exists():
        with path.open("x", encoding="utf-8"):
            pass
    return path


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in THREAD_LIMITS:
        environment.setdefault(name, "1")
    return environment


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    block_directory = _validated_block_directory(args.experiment_root, args.block_index)
    telemetry_path = _telemetry_path(args, block_directory)
    command = [sys.executable, str(TRAIN_BLOCK_PATH), "--experiment-root", str(args.experiment_root), "--block-index", str(args.block_index), "--gradient-backend", "jax", "--jax-memory-mode", "rematerialized", "--memory-telemetry-path", str(telemetry_path)]
    for flag, value in (("--iterations", args.iterations), ("--restarts", args.restarts), ("--seed-offset", args.seed_offset), ("--lr", args.lr), ("--success-threshold", args.success_threshold)):
        if value is not None:
            command.extend((flag, str(value)))
    for flag, enabled in (("--no-progress", args.no_progress), ("--plot-loss", args.plot_loss)):
        if enabled:
            command.append(flag)
    return int(subprocess.run(args=command, env=_child_environment(), check=False).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
