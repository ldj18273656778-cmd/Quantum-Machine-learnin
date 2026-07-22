"""Compatibility checks for the task2_code_auto package layout."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
IGNORED_PARTS = {"__pycache__", "data", "tests"}


@dataclass(frozen=True, slots=True)
class LegacyImportViolation:
    label: str
    line: int
    kind: str
    module: str

    def diagnostic(self) -> str:
        return f"{self.label}:{self.line}: {self.kind} {self.module}"


def is_task2_code_module(module_name: str) -> bool:
    return module_name == "task2_code" or module_name.startswith("task2_code.")


def legacy_task2_code_imports(source: str, label: str) -> list[LegacyImportViolation]:
    violations: list[LegacyImportViolation] = []
    tree = ast.parse(source)
    importlib_names, import_module_names = dynamic_import_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None and is_task2_code_module(node.module):
            violations.append(LegacyImportViolation(label, node.lineno, "from", node.module))
        if isinstance(node, ast.Import):
            for alias in node.names:
                if is_task2_code_module(alias.name):
                    violations.append(LegacyImportViolation(label, node.lineno, "import", alias.name))
        if isinstance(node, ast.Call):
            dynamic_module = literal_dynamic_import_module(node, importlib_names, import_module_names)
            if dynamic_module is not None and is_task2_code_module(dynamic_module):
                violations.append(LegacyImportViolation(label, node.lineno, "dynamic", dynamic_module))
    return violations


def dynamic_import_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    importlib_names = {"importlib"}
    import_module_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_names.add(alias.asname or alias.name)
        if isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    import_module_names.add(alias.asname or alias.name)
    return importlib_names, import_module_names


def literal_dynamic_import_module(
    node: ast.Call,
    importlib_names: set[str],
    import_module_names: set[str],
) -> str | None:
    if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
        return None
    if isinstance(node.func, ast.Name) and node.func.id in import_module_names | {"__import__"}:
        return node.args[0].value
    if (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "import_module"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in importlib_names
    ):
        return node.args[0].value
    return None


def production_python_paths() -> list[Path]:
    return [
        source_path
        for source_path in sorted(PACKAGE_ROOT.rglob("*.py"))
        if source_path.is_file()
        and IGNORED_PARTS.isdisjoint(source_path.relative_to(PACKAGE_ROOT).parts)
    ]


def jax_backend_python_paths() -> list[Path]:
    return [
        source_path
        for source_path in sorted((PACKAGE_ROOT / "jax_backend").rglob("*.py"))
        if source_path.is_file()
    ]


def imported_modules(source: str) -> set[str]:
    modules: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def is_forbidden_jax_backend_dependency(module_name: str) -> bool:
    return (
        module_name == "numpy"
        or module_name.startswith("numpy.")
        or module_name == "cirq"
        or module_name.startswith("cirq.")
        or is_task2_code_module(module_name)
    )


class Task2CodeAutoLayoutCompatibilityTests(unittest.TestCase):
    def test_package_root_is_regular_package(self) -> None:
        # Given: task2_code_auto should not rely on namespace-package resolution.
        import importlib.util

        # When: the package spec is resolved.
        spec = importlib.util.find_spec("task2_code_auto")

        # Then: it has a concrete package initializer.
        if spec is None:
            self.fail("expected task2_code_auto package spec")
        origin = spec.origin
        if origin is None:
            self.fail("expected task2_code_auto package origin")
        self.assertEqual(Path(origin).name, "__init__.py")

    def test_import_guard_rejects_bare_task2_code_imports(self) -> None:
        # Given: synthetic imports cover legacy package forms and the auto package.
        source = """
import task2_code
import task2_code.module_e_training
from task2_code import module_e_training
from task2_code.module_e_training import AdamConfig
import task2_code_auto
from task2_code_auto import ansatz
import importlib
importlib.import_module("task2_code")
importlib.import_module("task2_code.module_e_training")
import importlib as il
il.import_module("task2_code.ansatz")
from importlib import import_module
import_module("task2_code.loss_registry")
__import__("task2_code")
__import__("task2_code.module_e_training")
"""

        # When: imports are inspected by the package-independence guard.
        violations = legacy_task2_code_imports(source, "fixture.py")

        # Then: task2_code forms are rejected and task2_code_auto imports are allowed.
        self.assertCountEqual(
            [(violation.line, violation.kind, violation.module) for violation in violations],
            [
                (2, "import", "task2_code"),
                (3, "import", "task2_code.module_e_training"),
                (4, "from", "task2_code"),
                (5, "from", "task2_code.module_e_training"),
                (9, "dynamic", "task2_code"),
                (10, "dynamic", "task2_code.module_e_training"),
                (12, "dynamic", "task2_code.ansatz"),
                (14, "dynamic", "task2_code.loss_registry"),
                (15, "dynamic", "task2_code"),
                (16, "dynamic", "task2_code.module_e_training"),
            ],
        )
        self.assertTrue(all(violation.label == "fixture.py" for violation in violations))

    def test_public_core_modules_import_without_task2_code_available(self) -> None:
        # Given: a clean interpreter denies imports of the legacy package.
        module_orders = [
            [
                "task2_code_auto",
                "task2_code_auto.experiment_config",
                "task2_code_auto.ansatz",
                "task2_code_auto.ansatz_registry",
                "task2_code_auto.local_loss",
                "task2_code_auto.loss_registry",
                "task2_code_auto.module_e_training",
            ],
            [
                "task2_code_auto.module_e_training",
                "task2_code_auto.loss_registry",
                "task2_code_auto.local_loss",
                "task2_code_auto.ansatz_registry",
                "task2_code_auto.ansatz",
                "task2_code_auto.target_factory",
                "task2_code_auto.superoperator_registry",
                "task2_code_auto.superoperator",
                "task2_code_auto.U_target",
                "task2_code_auto.lightcone",
            ],
        ]
        script = textwrap.dedent(
            """
            import importlib
            import importlib.abc
            import json
            import sys

            class LegacyTask2CodeBlocker(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "task2_code" or fullname.startswith("task2_code."):
                        raise AssertionError(f"legacy import attempted: {fullname}")
                    return None

            sys.meta_path.insert(0, LegacyTask2CodeBlocker())
            for module_name in json.loads(sys.argv[1]):
                importlib.import_module(module_name)
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPOSITORY_ROOT)

        # When/Then: public core modules import in multiple orders without legacy fallback.
        for module_order in module_orders:
            with self.subTest(module_order=module_order):
                result = subprocess.run(
                    [sys.executable, "-c", script, json.dumps(module_order)],
                    cwd=REPOSITORY_ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_ansatz_script_runs_when_invoked_by_file_path(self) -> None:
        # Given: users historically launch the ansatz module by file path.
        env = os.environ.copy()
        _ = env.pop("PYTHONPATH", None)

        # When: ansatz.py is invoked directly.
        result = subprocess.run(
            [sys.executable, "task2_code_auto/ansatz.py"],
            cwd=REPOSITORY_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        # Then: the script reaches its existing smoke output.
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("A.6.1  zero-parameter unitary check:", result.stdout)

    def test_production_python_does_not_import_task2_code(self) -> None:
        # Given: task2_code_auto must be import-independent from task2_code.
        production_paths = production_python_paths()
        self.assertTrue(production_paths, "expected production Python paths under task2_code_auto")

        violations: list[LegacyImportViolation] = []
        for source_path in production_paths:
            source = source_path.read_text(encoding="utf-8-sig")
            label = source_path.relative_to(REPOSITORY_ROOT).as_posix()

            # When: production Python is scanned without importing heavy quantum modules.
            violations.extend(legacy_task2_code_imports(source, label))

        # Then: diagnostics include visible paths for every direct legacy import.
        self.assertEqual(
            violations,
            [],
            "direct task2_code imports found:\n" + "\n".join(violation.diagnostic() for violation in violations),
        )

    def test_jax_backend_production_imports_no_numpy_cirq_or_legacy_task2_code(self) -> None:
        # Given: the JAX backend must stay isolated from NumPy, Cirq, and legacy task2_code modules.
        backend_paths = jax_backend_python_paths()
        self.assertTrue(backend_paths, "expected production Python paths under task2_code_auto/jax_backend")

        violations: list[str] = []
        for source_path in backend_paths:
            source = source_path.read_text(encoding="utf-8-sig")
            label = source_path.relative_to(REPOSITORY_ROOT).as_posix()
            for module_name in imported_modules(source):
                if is_forbidden_jax_backend_dependency(module_name):
                    violations.append(f"{label}: imports {module_name}")

        # Then: the backend production dependency boundary stays JAX-only.
        self.assertEqual(violations, [], "forbidden jax_backend imports found:\n" + "\n".join(violations))


if __name__ == "__main__":
    _ = unittest.main()
