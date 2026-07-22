"""Parity contracts for the isolated JAX backend."""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
import textwrap
import time
from typing import Protocol, runtime_checkable
import unittest

import numpy as np
from numpy.typing import NDArray

from task2_code_auto.ansatz import ansatz_unitary, theta_count
from task2_code_auto.experiment_config import N12_3BLOCKS_HEISENBERG
from task2_code_auto.lightcone import backward_block_lightcone_from_circuit, build_lightcone_target_unitary
from task2_code_auto.loss_registry import heisenberg_pauli_loss
from task2_code_auto.module_e_training import ObjectiveContext, finite_difference_gradient
from task2_code_auto.target_factory import build_target_from_seed


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

BLOCK_LIGHTCONE_N4 = (0, 1, 2, 3)
BLOCK_N12_SECOND = (4, 5, 6, 7)
FD_EPS = 1e-5
LIGHTCONE_N12_SECOND = (2, 3, 4, 5, 6, 7, 8, 9)
LOCAL_N_QUBITS = 4
RANDOM_SEED = 20240720
TARGET_SEED = 314159
TARGET_BIT_N12_SECOND = 5
THETA_SIZE_N4 = 60
THETA_SIZE_N12_LIGHTCONE = 120
UNITARY_SHAPE_N4 = (16, 16)
UNITARY_SHAPE_N12_LIGHTCONE = (256, 256)

_ = os.environ.setdefault("JAX_ENABLE_X64", "1")


@runtime_checkable
class JaxBackendModule(Protocol):
    SUPPORTED_LIGHTCONE_WIDTHS: tuple[int, ...]

    def ansatz_unitary(self, theta: FloatArray, *, n_qubits: int) -> ComplexArray: ...

    def heisenberg_pauli_loss(self, theta: FloatArray, context: ObjectiveContext) -> float: ...

    def value_and_grad_heisenberg_pauli_loss(
        self,
        theta: FloatArray,
        context: ObjectiveContext,
    ) -> tuple[float, FloatArray]: ...

    def heisenberg_pauli_loss_rematerialized(self, theta: FloatArray, context: ObjectiveContext) -> float: ...

    def value_and_grad_heisenberg_pauli_loss_rematerialized(
        self,
        theta: FloatArray,
        context: ObjectiveContext,
    ) -> tuple[float, FloatArray]: ...


class HeisenbergEigvalshDiagnostic(Protocol):
    loss: float
    minimum_eigengap: float
    near_degenerate: bool


@runtime_checkable
class JaxExplicitEigvalshLossModule(Protocol):
    def heisenberg_pauli_loss_eigvalsh(
        self,
        theta: FloatArray,
        target_operator: ComplexArray,
        *,
        block_qubits: tuple[int, ...],
        lightcone_qubits: tuple[int, ...],
    ) -> float: ...


@runtime_checkable
class JaxExplicitEigvalshDiagnosticsModule(Protocol):
    def heisenberg_pauli_eigvalsh_diagnostics(
        self,
        theta: FloatArray,
        target_operator: ComplexArray,
        *,
        block_qubits: tuple[int, ...],
        lightcone_qubits: tuple[int, ...],
    ) -> HeisenbergEigvalshDiagnostic: ...


class JaxConfigModule(Protocol):
    jax_enable_x64: bool

    def update(self, name: str, value: bool) -> None: ...


@runtime_checkable
class JaxRuntimeModule(Protocol):
    config: JaxConfigModule


@runtime_checkable
class JaxNumpyModule(Protocol):
    float64: np.dtype[np.float64]

    def asarray(self, values: list[float]) -> FloatArray: ...


def load_jax_backend() -> JaxBackendModule:
    # Given: JAX x64 must be enabled before backend code can import jax.numpy.
    _ = configure_jax_x64_runtime()
    module = importlib.import_module("task2_code_auto.jax_backend")
    if not isinstance(module, JaxBackendModule):
        raise AssertionError("task2_code_auto.jax_backend does not satisfy the parity API")
    return module


def require_explicit_eigvalsh_loss_backend(jax_backend: JaxBackendModule) -> JaxExplicitEigvalshLossModule:
    if not isinstance(jax_backend, JaxExplicitEigvalshLossModule):
        raise AssertionError("task2_code_auto.jax_backend lacks heisenberg_pauli_loss_eigvalsh")
    return jax_backend


def require_explicit_eigvalsh_diagnostics_backend(
    jax_backend: JaxBackendModule,
) -> JaxExplicitEigvalshDiagnosticsModule:
    if not isinstance(jax_backend, JaxExplicitEigvalshDiagnosticsModule):
        raise AssertionError("task2_code_auto.jax_backend lacks heisenberg_pauli_eigvalsh_diagnostics")
    return jax_backend


def configure_jax_x64_runtime() -> tuple[JaxRuntimeModule, JaxNumpyModule]:
    try:
        jax_module = importlib.import_module("jax")
    except ModuleNotFoundError as exc:
        if exc.name == "jax" and importlib.util.find_spec("task2_code_auto.jax_backend") is None:
            raise ModuleNotFoundError("No module named 'task2_code_auto.jax_backend'") from None
        raise
    if not isinstance(jax_module, JaxRuntimeModule):
        raise AssertionError("jax module does not expose runtime config")
    jax_module.config.update("jax_enable_x64", True)
    jnp_module = importlib.import_module("jax.numpy")
    if not isinstance(jnp_module, JaxNumpyModule):
        raise AssertionError("jax.numpy module does not expose the expected array API")
    return jax_module, jnp_module


def load_jax_runtime() -> tuple[JaxRuntimeModule, JaxNumpyModule]:
    return configure_jax_x64_runtime()


def n4_context() -> ObjectiveContext:
    target_theta = np.random.default_rng(TARGET_SEED).uniform(
        -0.5,
        0.5,
        size=THETA_SIZE_N4,
    )
    target_operator = ansatz_unitary(target_theta, n_qubits=LOCAL_N_QUBITS)
    if target_operator.shape != UNITARY_SHAPE_N4:
        raise AssertionError(f"expected N=4 target shape {UNITARY_SHAPE_N4}, got {target_operator.shape}")
    return ObjectiveContext(
        target_operator=target_operator,
        block_qubits=BLOCK_LIGHTCONE_N4,
        lightcone_qubits=BLOCK_LIGHTCONE_N4,
        target_bit=0,
    )


def deterministic_random_theta() -> FloatArray:
    rng = np.random.default_rng(RANDOM_SEED)
    return np.asarray(rng.uniform(-0.75, 0.75, size=THETA_SIZE_N4), dtype=np.float64)


def deterministic_random_theta_n12_lightcone() -> FloatArray:
    rng = np.random.default_rng(RANDOM_SEED + 12)
    return np.asarray(rng.uniform(-0.75, 0.75, size=THETA_SIZE_N12_LIGHTCONE), dtype=np.float64)


def n12_second_block_context() -> ObjectiveContext:
    config = N12_3BLOCKS_HEISENBERG
    block_qubits = tuple(config.blocks[1])
    target_bit = int(config.target_bits[1])
    target_spec = build_target_from_seed(config.n_qubits, config.target_seed, config.time_k)
    lightcone_qubits = tuple(backward_block_lightcone_from_circuit(target_spec.circuit, block_qubits))
    target_operator = build_lightcone_target_unitary(target_spec.circuit, lightcone_qubits)
    if block_qubits != BLOCK_N12_SECOND:
        raise AssertionError(f"expected N=12 second block {BLOCK_N12_SECOND}, got {block_qubits}")
    if target_bit != TARGET_BIT_N12_SECOND:
        raise AssertionError(f"expected N=12 second target bit {TARGET_BIT_N12_SECOND}, got {target_bit}")
    if lightcone_qubits != LIGHTCONE_N12_SECOND:
        raise AssertionError(f"expected N=12 second lightcone {LIGHTCONE_N12_SECOND}, got {lightcone_qubits}")
    if target_operator.shape != UNITARY_SHAPE_N12_LIGHTCONE:
        raise AssertionError(f"expected N=12 target shape {UNITARY_SHAPE_N12_LIGHTCONE}, got {target_operator.shape}")
    return ObjectiveContext(
        target_operator=target_operator,
        block_qubits=block_qubits,
        lightcone_qubits=lightcone_qubits,
        target_bit=target_bit,
    )


def zero_theta() -> FloatArray:
    return np.zeros(THETA_SIZE_N4, dtype=np.float64)


class JaxBackendParityTests(unittest.TestCase):
    def test_jax_runtime_uses_x64_when_backend_is_loaded(self) -> None:
        # Given: the parity backend is imported after the test enables JAX x64.
        _ = load_jax_backend()
        jax, jnp = load_jax_runtime()

        # When: a default floating-point JAX array is created.
        value = jnp.asarray([1.0])

        # Then: the runtime and array dtype both use x64 precision.
        self.assertTrue(jax.config.jax_enable_x64)
        self.assertEqual(value.dtype, jnp.float64)

    def test_backend_declares_approved_lightcone_widths_and_rejects_unapproved_width(self) -> None:
        # Given: the isolated JAX backend only has parity approval for N=4 and N=12 block-2 lightcones.
        jax_backend = load_jax_backend()

        # When/Then: the production capability contract is explicit and enforces the ansatz boundary.
        self.assertEqual(jax_backend.SUPPORTED_LIGHTCONE_WIDTHS, (4, 8))
        self.assertEqual(
            np.asarray(jax_backend.ansatz_unitary(np.zeros(THETA_SIZE_N12_LIGHTCONE), n_qubits=8)).shape,
            UNITARY_SHAPE_N12_LIGHTCONE,
        )
        with self.assertRaises(ValueError):
            _ = jax_backend.ansatz_unitary(np.zeros(theta_count(5), dtype=np.float64), n_qubits=5)

    def test_ansatz_unitary_matches_numpy_oracle_for_zero_and_random_theta(self) -> None:
        # Given: N=4 theta vectors use layer-major, qubit-major, Rx-Ry-Rz ordering.
        jax_backend = load_jax_backend()
        self.assertEqual(theta_count(LOCAL_N_QUBITS), THETA_SIZE_N4)
        test_thetas = (zero_theta(), deterministic_random_theta())

        for theta in test_thetas:
            with self.subTest(theta_kind="zero" if np.count_nonzero(theta) == 0 else "random"):
                theta_before = theta.copy()

                # When: the JAX backend builds the ansatz unitary.
                actual = np.asarray(jax_backend.ansatz_unitary(theta, n_qubits=LOCAL_N_QUBITS))

                # Then: it matches the existing NumPy/Cirq oracle and leaves theta untouched.
                expected = ansatz_unitary(theta, n_qubits=LOCAL_N_QUBITS)
                self.assertEqual(actual.shape, UNITARY_SHAPE_N4)
                self.assertEqual(expected.shape, UNITARY_SHAPE_N4)
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
                np.testing.assert_array_equal(theta, theta_before)

    def test_ansatz_unitary_rejects_invalid_theta_shape_and_qubit_count(self) -> None:
        # Given: the N=4 backend contract requires a 1D theta with exactly 15*n parameters.
        jax_backend = load_jax_backend()
        invalid_cases = (
            (np.zeros(THETA_SIZE_N4 - 1, dtype=np.float64), LOCAL_N_QUBITS),
            (np.zeros(THETA_SIZE_N4 + 1, dtype=np.float64), LOCAL_N_QUBITS),
            (np.zeros((THETA_SIZE_N4, 1), dtype=np.float64), LOCAL_N_QUBITS),
            (np.zeros(0, dtype=np.float64), 0),
            (np.zeros(0, dtype=np.float64), -1),
        )

        for theta, n_qubits in invalid_cases:
            with self.subTest(theta_shape=theta.shape, n_qubits=n_qubits):
                # When/Then: invalid contracts are rejected at the boundary.
                with self.assertRaises(ValueError):
                    _ = jax_backend.ansatz_unitary(theta, n_qubits=n_qubits)

    def test_heisenberg_scalar_loss_matches_numpy_oracle_for_zero_and_random_theta(self) -> None:
        # Given: a deterministic N=4 ObjectiveContext uses the full block lightcone.
        jax_backend = load_jax_backend()
        context = n4_context()
        test_thetas = (zero_theta(), deterministic_random_theta())

        for theta in test_thetas:
            with self.subTest(theta_kind="zero" if np.count_nonzero(theta) == 0 else "random"):
                theta_before = theta.copy()

                # When: the JAX backend evaluates the scalar Heisenberg Pauli loss.
                actual = float(jax_backend.heisenberg_pauli_loss(theta, context))

                # Then: it matches the existing NumPy oracle and leaves theta untouched.
                expected = heisenberg_pauli_loss(theta, context)
                self.assertAlmostEqual(actual, expected, delta=1e-9)
                np.testing.assert_array_equal(theta, theta_before)

    def test_explicit_eigvalsh_heisenberg_scalar_loss_matches_numpy_oracle_for_n4_random_theta(self) -> None:
        # Given: the Phase 2 API receives only explicit inputs, not an ObjectiveContext or loss registry state.
        jax_backend = load_jax_backend()
        context = n4_context()
        theta = deterministic_random_theta()
        theta_before = theta.copy()
        phase2_backend = require_explicit_eigvalsh_loss_backend(jax_backend)

        # When: the backend evaluates H=(delta+delta^dagger)/2 with eigvalsh semantics.
        actual = float(
            phase2_backend.heisenberg_pauli_loss_eigvalsh(
                theta,
                np.asarray(context.target_operator, dtype=np.complex128),
                block_qubits=context.block_qubits,
                lightcone_qubits=context.lightcone_qubits,
            )
        )

        # Then: it matches the existing NumPy oracle and leaves theta untouched.
        expected = heisenberg_pauli_loss(theta, context)
        self.assertAlmostEqual(actual, expected, delta=1e-9)
        np.testing.assert_array_equal(theta, theta_before)

    def test_explicit_eigvalsh_heisenberg_scalar_loss_matches_numpy_oracle_for_n8_random_theta(self) -> None:
        # Given: the N=12 block-2 lightcone is an eight-qubit explicit-input parity contract.
        jax_backend = load_jax_backend()
        context = n12_second_block_context()
        theta = deterministic_random_theta_n12_lightcone()
        theta_before = theta.copy()
        phase2_backend = require_explicit_eigvalsh_loss_backend(jax_backend)
        self.assertEqual(context.block_qubits, BLOCK_N12_SECOND)
        self.assertEqual(context.lightcone_qubits, LIGHTCONE_N12_SECOND)
        self.assertEqual(theta.shape, (THETA_SIZE_N12_LIGHTCONE,))

        # When: the backend evaluates the scalar loss once without any N=8 finite-difference gradient sweep.
        actual = float(
            phase2_backend.heisenberg_pauli_loss_eigvalsh(
                theta,
                np.asarray(context.target_operator, dtype=np.complex128),
                block_qubits=context.block_qubits,
                lightcone_qubits=context.lightcone_qubits,
            )
        )

        # Then: the explicit eigvalsh scalar agrees with the existing Heisenberg Pauli oracle.
        expected = heisenberg_pauli_loss(theta, context)
        self.assertAlmostEqual(actual, expected, delta=1e-8)
        self.assertTrue(np.isfinite(actual))
        np.testing.assert_array_equal(theta, theta_before)

    def test_explicit_eigvalsh_diagnostics_identify_exact_solution_degeneracy(self) -> None:
        # Given: target and trial are identical, so every Hermitian Pauli residual has degenerate zero spectrum.
        jax_backend = load_jax_backend()
        theta = deterministic_random_theta()
        target_operator = np.asarray(jax_backend.ansatz_unitary(theta, n_qubits=LOCAL_N_QUBITS), dtype=np.complex128)
        phase2_backend = require_explicit_eigvalsh_diagnostics_backend(jax_backend)

        # When: diagnostics are requested for the explicit-input loss contract without smoothing the loss.
        diagnostic = phase2_backend.heisenberg_pauli_eigvalsh_diagnostics(
            theta,
            target_operator,
            block_qubits=BLOCK_LIGHTCONE_N4,
            lightcone_qubits=BLOCK_LIGHTCONE_N4,
        )

        # Then: the diagnostic reports a finite nonnegative eigengap and flags the exact-solution degeneracy.
        self.assertTrue(np.isfinite(float(diagnostic.minimum_eigengap)))
        self.assertGreaterEqual(float(diagnostic.minimum_eigengap), 0.0)
        self.assertTrue(bool(diagnostic.near_degenerate))
        self.assertAlmostEqual(float(diagnostic.loss), 0.0, delta=1e-10)

    def test_value_and_grad_matches_finite_difference_gradient_for_random_theta(self) -> None:
        # Given: a deterministic random N=4 theta and ObjectiveContext.
        jax_backend = load_jax_backend()
        context = n4_context()
        theta = deterministic_random_theta()
        theta_before = theta.copy()

        # When: the JAX backend computes value_and_grad for the Heisenberg objective.
        actual_loss, actual_grad = jax_backend.value_and_grad_heisenberg_pauli_loss(theta, context)

        # Then: the value matches the scalar oracle and the gradient matches central differences.
        expected_loss = heisenberg_pauli_loss(theta, context)
        expected_grad = finite_difference_gradient(
            lambda candidate_theta: heisenberg_pauli_loss(candidate_theta, context),
            theta,
            fd_eps=FD_EPS,
        )
        self.assertEqual(np.asarray(actual_grad).shape, (THETA_SIZE_N4,))
        self.assertEqual(expected_grad.shape, (THETA_SIZE_N4,))
        self.assertAlmostEqual(float(actual_loss), expected_loss, delta=1e-9)
        np.testing.assert_allclose(np.asarray(actual_grad), expected_grad, rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(theta, theta_before)

    def test_value_and_grad_is_finite_and_zero_for_jax_built_exact_solution(self) -> None:
        # Given: target and trial are the same JAX-built unitary, which makes the Pauli residual degenerate.
        jax_backend = load_jax_backend()
        theta = deterministic_random_theta()
        target_operator = jax_backend.ansatz_unitary(theta, n_qubits=LOCAL_N_QUBITS)
        context = ObjectiveContext(
            target_operator=target_operator,
            block_qubits=BLOCK_LIGHTCONE_N4,
            lightcone_qubits=BLOCK_LIGHTCONE_N4,
            target_bit=0,
        )

        # When: the JAX backend differentiates the exact-solution loss.
        actual_loss, actual_grad = jax_backend.value_and_grad_heisenberg_pauli_loss(theta, context)
        grad_array = np.asarray(actual_grad)

        # Then: the scalar and gradient stay finite and the gradient is near zero.
        self.assertTrue(np.isfinite(float(actual_loss)))
        self.assertTrue(np.all(np.isfinite(grad_array)))
        self.assertAlmostEqual(float(actual_loss), 0.0, delta=1e-10)
        np.testing.assert_allclose(grad_array, np.zeros_like(grad_array), rtol=0.0, atol=1e-7)

    def test_n12_second_block_lightcone_matches_numpy_oracle_without_fd_gradient(self) -> None:
        # Given: the n12_3blocks_heisenberg block-2 circuit lightcone is eight qubits, so this is a slow parity contract.
        jax_backend = load_jax_backend()
        context = n12_second_block_context()
        theta = deterministic_random_theta_n12_lightcone()
        theta_before = theta.copy()
        self.assertEqual(N12_3BLOCKS_HEISENBERG.n_qubits, 12)
        self.assertEqual(context.block_qubits, BLOCK_N12_SECOND)
        self.assertEqual(context.target_bit, TARGET_BIT_N12_SECOND)
        self.assertEqual(context.lightcone_qubits, LIGHTCONE_N12_SECOND)
        self.assertEqual(context.ansatz_qubits, 8)
        self.assertEqual(context.theta_size, THETA_SIZE_N12_LIGHTCONE)
        self.assertEqual(theta.shape, (THETA_SIZE_N12_LIGHTCONE,))
        self.assertEqual(context.target_operator.shape, UNITARY_SHAPE_N12_LIGHTCONE)
        self.assertEqual(theta_count(context.ansatz_qubits), THETA_SIZE_N12_LIGHTCONE)

        # When: the JAX backend evaluates the ansatz, scalar loss, and autodiff gradient once.
        started = time.perf_counter()
        actual_unitary = np.asarray(jax_backend.ansatz_unitary(theta, n_qubits=context.ansatz_qubits))
        actual_loss = float(jax_backend.heisenberg_pauli_loss(theta, context))
        actual_grad_loss, actual_grad = jax_backend.value_and_grad_heisenberg_pauli_loss(theta, context)
        elapsed_seconds = time.perf_counter() - started

        # Then: JAX matches the existing Cirq/NumPy oracle without the 120-parameter finite-difference sweep.
        expected_unitary = ansatz_unitary(theta, n_qubits=context.ansatz_qubits)
        expected_loss = heisenberg_pauli_loss(theta, context)
        grad_array = np.asarray(actual_grad)
        self.assertEqual(actual_unitary.shape, UNITARY_SHAPE_N12_LIGHTCONE)
        self.assertEqual(expected_unitary.shape, UNITARY_SHAPE_N12_LIGHTCONE)
        np.testing.assert_allclose(actual_unitary, expected_unitary, rtol=1e-10, atol=1e-10)
        self.assertAlmostEqual(actual_loss, expected_loss, delta=1e-8)
        self.assertAlmostEqual(float(actual_grad_loss), expected_loss, delta=1e-8)
        self.assertEqual(grad_array.shape, (THETA_SIZE_N12_LIGHTCONE,))
        self.assertTrue(np.isfinite(actual_loss))
        self.assertTrue(np.isfinite(float(actual_grad_loss)))
        self.assertTrue(np.all(np.isfinite(grad_array)))
        np.testing.assert_array_equal(theta, theta_before)
        print(f"N=12 block-2 JAX parity runtime: {elapsed_seconds:.3f}s")

    def test_rematerialized_heisenberg_loss_matches_existing_path_for_n4_random_theta(self) -> None:
        # Given: a deterministic N=4 theta and ObjectiveContext already approved for the JAX eigvalsh path.
        jax_backend = load_jax_backend()
        context = n4_context()
        theta = deterministic_random_theta()
        theta_before = theta.copy()

        # When: the rematerialized scalar and autodiff APIs evaluate the same objective.
        actual_loss = float(jax_backend.heisenberg_pauli_loss_rematerialized(theta, context))
        actual_grad_loss, actual_grad = jax_backend.value_and_grad_heisenberg_pauli_loss_rematerialized(theta, context)

        # Then: they match the established JAX path and leave theta untouched.
        expected_loss, expected_grad = jax_backend.value_and_grad_heisenberg_pauli_loss(theta, context)
        np.testing.assert_allclose(actual_loss, float(expected_loss), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(float(actual_grad_loss), float(expected_loss), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.asarray(actual_grad), np.asarray(expected_grad), rtol=1e-8, atol=1e-8)
        np.testing.assert_array_equal(theta, theta_before)

    def test_rematerialized_heisenberg_loss_matches_existing_path_for_n12_second_block(self) -> None:
        # Given: the N=12 block-2 lightcone is the largest approved parity contract for this backend.
        jax_backend = load_jax_backend()
        context = n12_second_block_context()
        theta = deterministic_random_theta_n12_lightcone()
        theta_before = theta.copy()

        # When: the rematerialized scalar and autodiff APIs evaluate the eight-qubit lightcone once.
        actual_loss = float(jax_backend.heisenberg_pauli_loss_rematerialized(theta, context))
        actual_grad_loss, actual_grad = jax_backend.value_and_grad_heisenberg_pauli_loss_rematerialized(theta, context)

        # Then: they stay mathematically identical to the established non-rematerialized path.
        expected_loss, expected_grad = jax_backend.value_and_grad_heisenberg_pauli_loss(theta, context)
        grad_array = np.asarray(actual_grad)
        np.testing.assert_allclose(actual_loss, float(expected_loss), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(float(actual_grad_loss), float(expected_loss), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(grad_array, np.asarray(expected_grad), rtol=1e-6, atol=1e-6)
        self.assertTrue(np.isfinite(actual_loss))
        self.assertTrue(np.isfinite(float(actual_grad_loss)))
        self.assertTrue(np.all(np.isfinite(grad_array)))
        np.testing.assert_array_equal(theta, theta_before)

    def test_backend_import_enables_x64_without_test_side_jax_setup(self) -> None:
        # Given: a clean subprocess without pre-set JAX_ENABLE_X64 imports the backend first.
        script = textwrap.dedent(
            """
            import json
            import task2_code_auto.jax_backend as jax_backend
            import jax
            import jax.numpy as jnp

            theta = jnp.zeros(60)
            unitary = jax_backend.ansatz_unitary(theta, n_qubits=4)
            print(json.dumps({
                "x64": bool(jax.config.jax_enable_x64),
                "float_dtype": str(theta.dtype),
                "complex_dtype": str(unitary.dtype),
            }))
            """
        )
        env = os.environ.copy()
        _ = env.pop("JAX_ENABLE_X64", None)

        # When: the backend is imported before any test-side JAX runtime setup.
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        # Then: backend runtime setup enables x64 and preserves expected dtypes.
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"x64": true', result.stdout)
        self.assertIn('"float_dtype": "float64"', result.stdout)
        self.assertIn('"complex_dtype": "complex128"', result.stdout)


if __name__ == "__main__":
    _ = unittest.main()
