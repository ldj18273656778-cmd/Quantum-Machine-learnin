"""Plot mixed-pretraining losses before Heisenberg warmstart."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mixed-edge loss vs Heisenberg loss for mixed-trained best parameters."
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--allow-partial", action="store_true", default=False)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _group_files(experiment_root: Path) -> list[Path]:
    files = sorted((experiment_root / "groups").glob("group_*_mixed_warmstart.json"))
    if not files:
        raise FileNotFoundError(f"no group_*_mixed_warmstart.json files found in {experiment_root / 'groups'}")
    return files


def _collect_trials(manifest: dict[str, Any], files: list[Path], *, allow_partial: bool) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    expected_groups = set(range(1, int(manifest["group_count"]) + 1))
    seen_groups: set[int] = set()

    for path in files:
        payload = _load_json(path)
        if payload.get("artifact_type") != "mixed_warmstart_group_result":
            raise ValueError(f"unexpected artifact_type in {path}: {payload.get('artifact_type')!r}")
        group_index = int(payload["group_index"])
        if group_index in seen_groups:
            raise ValueError(f"duplicate group result for group {group_index}")
        if group_index not in expected_groups:
            raise ValueError(f"unexpected group_index {group_index} in {path}")
        seen_groups.add(group_index)
        for trial in payload["result"].get("trials", []):
            item = dict(trial)
            item["group_index"] = group_index
            trials.append(item)

    missing_groups = expected_groups - seen_groups
    if missing_groups and not allow_partial:
        raise ValueError(f"missing group results: {sorted(missing_groups)}")
    if not trials:
        raise ValueError("no trials found")
    return trials


def _set_loss_axis_scale(ax, values: np.ndarray[Any, Any], axis: str) -> None:
    if np.all(values > 0):
        if axis == "x":
            ax.set_xscale("log")
        else:
            ax.set_yscale("log")
    else:
        if axis == "x":
            ax.set_xscale("symlog", linthresh=1e-12)
        else:
            ax.set_yscale("symlog", linthresh=1e-12)


def _plot(path: Path, manifest: dict[str, Any], trials: list[dict[str, Any]]) -> None:
    mixed_loss = np.asarray([float(trial["mixed_best_params_mixed_edge_loss"]) for trial in trials], dtype=float)
    heisenberg_loss = np.asarray([float(trial["mixed_best_params_heisenberg_loss"]) for trial in trials], dtype=float)
    success = np.asarray([bool(trial.get("success", False)) for trial in trials], dtype=bool)
    threshold = float(manifest.get("stage2", {}).get("success_threshold", 0.01))

    fig, ax = plt.subplots(figsize=(8, 6))
    if (~success).any():
        ax.scatter(mixed_loss[~success], heisenberg_loss[~success], s=12, alpha=0.35, label="not successful")
    if success.any():
        ax.scatter(mixed_loss[success], heisenberg_loss[success], s=18, alpha=0.8, label="successful")
    ax.axhline(threshold, color="tab:red", linestyle="--", linewidth=1.2, label=f"Heisenberg threshold={threshold:g}")
    _set_loss_axis_scale(ax, mixed_loss, "x")
    _set_loss_axis_scale(ax, heisenberg_loss, "y")
    ax.set_xlabel("Mixed-trained best parameters: mixed edge loss")
    ax.set_ylabel("Mixed-trained best parameters: Heisenberg loss")
    ax.set_title("Mixed channel pretraining result before Heisenberg warmstart")
    ax.grid(alpha=0.3)
    ax.legend()

    info = (
        f"n={len(trials)}\n"
        f"block={manifest.get('block_index')}\n"
        f"success={int(success.sum())}/{len(success)}"
    )
    ax.text(0.02, 0.98, info, transform=ax.transAxes, va="top", ha="left", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest = _load_json(args.experiment_root / "manifest.json")
    if manifest.get("artifact_type") != "mixed_warmstart_experiment":
        raise ValueError(f"unexpected manifest artifact_type: {manifest.get('artifact_type')!r}")
    output_path = args.output or (args.experiment_root / "summary" / "mixed_best_params_loss_scatter.png")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite to replace it: {output_path}")

    trials = _collect_trials(manifest, _group_files(args.experiment_root), allow_partial=bool(args.allow_partial))
    _plot(output_path, manifest, trials)
    print(f"Saved scatter plot: {output_path}")


if __name__ == "__main__":
    main()
