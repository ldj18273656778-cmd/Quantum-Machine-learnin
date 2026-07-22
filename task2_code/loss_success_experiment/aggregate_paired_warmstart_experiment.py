"""Aggregate paired Heisenberg vs mixed-warmstart experiment results."""

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

from task2_code.hpc_parallel_training.hpc_block_flow import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate a paired warmstart experiment.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def main() -> None:
    args = parse_args()
    manifest = _load_json(args.experiment_root / "manifest.json")
    if manifest.get("artifact_type") != "paired_warmstart_experiment":
        raise ValueError(f"unexpected artifact_type: {manifest.get('artifact_type')!r}")
    group_count = int(manifest["group_count"])
    groups_dir = args.experiment_root / "groups"
    missing = []
    groups = []
    for group_index in range(1, group_count + 1):
        path = groups_dir / f"group_{group_index:06d}_paired.json"
        if not path.exists():
            missing.append(group_index)
            continue
        groups.append(_load_json(path))
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} group result files; first missing: {missing[:10]}")

    trials = [trial for group in groups for trial in group.get("trials", [])]
    total = len(trials)
    if total != int(manifest["total_pairs"]):
        raise ValueError(f"expected {manifest['total_pairs']} trials, found {total}")
    heis_success = sum(1 for trial in trials if trial["heisenberg_only"]["success"])
    warm_success = sum(1 for trial in trials if trial["warmstart"]["success"])
    paired = {
        "heisenberg_only_success_warmstart_success": 0,
        "heisenberg_only_success_warmstart_fail": 0,
        "heisenberg_only_fail_warmstart_success": 0,
        "heisenberg_only_fail_warmstart_fail": 0,
    }
    for trial in trials:
        heis = bool(trial["heisenberg_only"]["success"])
        warm = bool(trial["warmstart"]["success"])
        if heis and warm:
            paired["heisenberg_only_success_warmstart_success"] += 1
        elif heis and not warm:
            paired["heisenberg_only_success_warmstart_fail"] += 1
        elif (not heis) and warm:
            paired["heisenberg_only_fail_warmstart_success"] += 1
        else:
            paired["heisenberg_only_fail_warmstart_fail"] += 1

    summary = {
        "experiment_root": args.experiment_root,
        "total_pairs": total,
        "heisenberg_success_count": heis_success,
        "heisenberg_success_rate": float(heis_success / total),
        "warmstart_success_count": warm_success,
        "warmstart_success_rate": float(warm_success / total),
        "paired_outcomes": paired,
        "manifest": manifest,
    }
    output = args.experiment_root / "summary" / "paired_summary.json"
    atomic_write_json(output, summary)
    print(f"Wrote {output}")
    print(f"Heisenberg-only success: {heis_success}/{total} = {heis_success / total:.4f}")
    print(f"Warm-start success: {warm_success}/{total} = {warm_success / total:.4f}")


if __name__ == "__main__":
    main()
