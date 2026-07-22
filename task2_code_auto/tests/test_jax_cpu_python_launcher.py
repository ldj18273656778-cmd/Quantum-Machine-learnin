"""RED contracts for the JAX CPU Python launcher."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import uuid

from task2_code_auto.hpc_parallel_training import run_jax_cpu_training as launcher


THREAD_LIMITS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@dataclass(frozen=True, slots=True)
class Completed:
    returncode: int


def make_experiment_root(directory: Path, block_count: int) -> Path:
    root = directory / "experiment"
    (root / "blocks").mkdir(parents=True)
    block_specs = [{"block_index": index + 1} for index in range(block_count)]
    (root / "manifest.json").write_text(json.dumps({"block_specs": block_specs}), encoding="utf-8")
    return root


def write_manifest(root: Path, payload: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(payload, encoding="utf-8")


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def run_launcher(argv: list[str], returncode: int = 0) -> tuple[int, mock.MagicMock]:
    with mock.patch.object(launcher.subprocess, "run", return_value=Completed(returncode)) as run:
        exit_code = launcher.main(argv)
    return exit_code, run


class JaxCpuPythonLauncherTests(unittest.TestCase):
    def test_child_command_forces_jax_rematerialized_forwards_training_controls_and_propagates_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_experiment_root(Path(directory), block_count=1)

            exit_code, run = run_launcher(
                [
                    "--experiment-root",
                    str(root),
                    "--block-index",
                    "1",
                    "--iterations",
                    "3",
                    "--restarts",
                    "2",
                    "--seed-offset",
                    "4",
                    "--lr",
                    "0.05",
                    "--success-threshold",
                    "1e-6",
                    "--no-progress",
                    "--plot-loss",
                ],
                returncode=17,
            )

        command = run.call_args.kwargs["args"]
        self.assertEqual(exit_code, 17)
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(Path(command[1]).resolve(), launcher.TRAIN_BLOCK_PATH)
        self.assertEqual(option_value(command, "--experiment-root"), str(root))
        self.assertEqual(option_value(command, "--block-index"), "1")
        self.assertEqual(option_value(command, "--gradient-backend"), "jax")
        self.assertEqual(option_value(command, "--jax-memory-mode"), "rematerialized")
        self.assertEqual(option_value(command, "--iterations"), "3")
        self.assertEqual(option_value(command, "--restarts"), "2")
        self.assertEqual(option_value(command, "--seed-offset"), "4")
        self.assertEqual(option_value(command, "--lr"), "0.05")
        self.assertEqual(float(option_value(command, "--success-threshold")), 1e-6)
        self.assertIn("--no-progress", command)
        self.assertIn("--plot-loss", command)

    def test_child_environment_defaults_thread_limits_and_preserves_caller_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_experiment_root(Path(directory), block_count=1)
            with mock.patch.dict(launcher.os.environ, {}, clear=True):
                _exit_code, default_run = run_launcher(["--experiment-root", str(root), "--block-index", "1"])

            existing_values = {
                "OMP_NUM_THREADS": "8",
                "OPENBLAS_NUM_THREADS": "7",
                "MKL_NUM_THREADS": "6",
                "NUMEXPR_NUM_THREADS": "5",
            }
            with mock.patch.dict(launcher.os.environ, existing_values, clear=True):
                _exit_code, preserved_run = run_launcher(["--experiment-root", str(root), "--block-index", "1"])

        default_environment = default_run.call_args.kwargs["env"]
        preserved_environment = preserved_run.call_args.kwargs["env"]
        self.assertEqual({name: default_environment[name] for name in THREAD_LIMITS}, dict.fromkeys(THREAD_LIMITS, "1"))
        self.assertEqual({name: preserved_environment[name] for name in THREAD_LIMITS}, existing_values)

    def test_manifest_json_object_and_block_specs_are_validated_before_spawning(self) -> None:
        cases = (
            ("missing manifest", None, "manifest"),
            ("malformed manifest", "{", "manifest"),
            ("non-object manifest", "[]", "JSON object"),
            ("missing block_specs", "{}", "block_specs"),
            ("non-list block_specs", json.dumps({"block_specs": {}}), "block_specs"),
            ("empty block_specs", json.dumps({"block_specs": []}), "block_specs"),
        )
        for name, payload, message in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory) / "experiment"
                    if payload is not None:
                        write_manifest(root, payload)
                    with mock.patch.object(launcher.subprocess, "run") as run:
                        with self.assertRaisesRegex(ValueError, message):
                            launcher.main(["--experiment-root", str(root), "--block-index", "1"])
                    run.assert_not_called()

    def test_one_based_block_index_bounds_are_validated_before_block_directory_creation(self) -> None:
        for block_index, message in (("0", "block-index"), ("3", "block-index")):
            with self.subTest(block_index=block_index):
                with tempfile.TemporaryDirectory() as directory:
                    root = make_experiment_root(Path(directory), block_count=2)
                    with mock.patch.object(launcher.subprocess, "run") as run:
                        with self.assertRaisesRegex(ValueError, message):
                            launcher.main(["--experiment-root", str(root), "--block-index", block_index])
                    run.assert_not_called()
                    self.assertFalse((root / "blocks" / f"block_{int(block_index):02d}").exists())

    def test_validated_block_directory_and_default_telemetry_are_created_under_selected_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_experiment_root(Path(directory), block_count=2)

            exit_code, run = run_launcher(["--experiment-root", str(root), "--block-index", "2"])

            block_directory = root / "blocks" / "block_02"
            telemetry = Path(option_value(run.call_args.kwargs["args"], "--memory-telemetry-path"))
            self.assertEqual(exit_code, 0)
            self.assertTrue(block_directory.is_dir())
            self.assertEqual(telemetry.parent, block_directory / "telemetry")
            self.assertTrue(telemetry.name.startswith("jax_"))
            self.assertEqual(telemetry.suffix, ".jsonl")
            self.assertTrue(telemetry.is_file())

    def test_default_telemetry_reservation_retries_existing_candidate_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_experiment_root(Path(directory), block_count=1)
            telemetry_directory = root / "blocks" / "block_01" / "telemetry"
            telemetry_directory.mkdir(parents=True)
            first = telemetry_directory / "jax_11111111-1111-1111-1111-111111111111.jsonl"
            first.write_text("existing\n", encoding="utf-8")

            with mock.patch.object(
                launcher.uuid,
                "uuid4",
                side_effect=[
                    uuid.UUID("11111111-1111-1111-1111-111111111111"),
                    uuid.UUID("22222222-2222-2222-2222-222222222222"),
                ],
            ):
                _exit_code, run = run_launcher(["--experiment-root", str(root), "--block-index", "1"])

            telemetry = Path(option_value(run.call_args.kwargs["args"], "--memory-telemetry-path"))
            self.assertEqual(first.read_text(encoding="utf-8"), "existing\n")
            self.assertEqual(telemetry.name, "jax_22222222-2222-2222-2222-222222222222.jsonl")
            self.assertTrue(telemetry.is_file())

    def test_explicit_telemetry_rejects_overwrite_and_append_requires_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_experiment_root(Path(directory), block_count=1)
            telemetry = Path(directory) / "existing.jsonl"
            telemetry.write_text("old\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "--append-telemetry"):
                launcher.main(["--experiment-root", str(root), "--block-index", "1", "--telemetry-path", str(telemetry)])
            with self.assertRaisesRegex(ValueError, "--telemetry-path"):
                launcher.main(["--experiment-root", str(root), "--block-index", "1", "--append-telemetry"])

    def test_append_telemetry_forwards_existing_explicit_path_without_reserving_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_experiment_root(Path(directory), block_count=1)
            telemetry = Path(directory) / "existing.jsonl"
            telemetry.write_text("old\n", encoding="utf-8")

            _exit_code, run = run_launcher(
                [
                    "--experiment-root",
                    str(root),
                    "--block-index",
                    "1",
                    "--telemetry-path",
                    str(telemetry),
                    "--append-telemetry",
                ]
            )

            self.assertEqual(option_value(run.call_args.kwargs["args"], "--memory-telemetry-path"), str(telemetry))
            self.assertEqual(telemetry.read_text(encoding="utf-8"), "old\n")


if __name__ == "__main__":
    unittest.main()
