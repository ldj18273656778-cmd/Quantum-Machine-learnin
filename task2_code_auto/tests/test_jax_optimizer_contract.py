"""Phase 3 contracts for the isolated JAX Adam optimizer."""

from __future__ import annotations

from dataclasses import Field, is_dataclass
import importlib
import importlib.util
from pathlib import Path
import sys
from typing import Callable, Protocol, TypeAlias, runtime_checkable
import unittest
from unittest import mock

import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp

from task2_code_auto.tests.test_jax_backend_parity import (
    FD_EPS,
    THETA_SIZE_N4,
    deterministic_random_theta,
    n4_context,
)
from task2_code_auto import module_e_training as legacy_training
from task2_code_auto.module_e_training import ObjectiveContext


FloatArray = NDArray[np.float64]
OPTIMIZER_MODULE_NAME = "task2_code_auto.jax_backend.optimizer"
OPTIMIZER_MODULE_PATH = Path(__file__).resolve().parents[1] / "jax_backend" / "optimizer.py"
AdamLossFunction: TypeAlias = Callable[[FloatArray, ObjectiveContext], float]
AdamFieldValue: TypeAlias = int | float | bool | str | FloatArray
ValueAndGradFunction: TypeAlias = Callable[[FloatArray], tuple[float, FloatArray]]


class DataclassSchema(Protocol):
    __dataclass_fields__: dict[str, Field[AdamFieldValue]]


class AdamConfigLike(Protocol):
    iterations: int
    lr: float
    beta1: float
    beta2: float
    eps: float
    fd_eps: float
    wrap_angles: bool


class AdamConfigDataclass(DataclassSchema, Protocol):
    def __call__(
        self,
        *,
        iterations: int = ...,
        lr: float = ...,
        beta1: float = ...,
        beta2: float = ...,
        eps: float = ...,
        fd_eps: float = ...,
        wrap_angles: bool = ...,
    ) -> AdamConfigLike: ...


class AdamStateLike(Protocol):
    iteration: int
    params: FloatArray
    m: FloatArray
    v: FloatArray


class AdamRunMetadataLike(Protocol):
    gradient_method: str
    optimizer_backend: str
    jax_x64_enabled: bool
    fd_eps: float
    fd_eps_used: bool


class AdamRunResultLike(Protocol):
    initial_params: FloatArray
    final_params: FloatArray
    best_params: FloatArray
    loss_history: FloatArray
    grad_norm_history: FloatArray
    best_loss_history: FloatArray
    best_loss: float
    best_iteration: int
    failed: bool
    failure_reason: str
    metadata: AdamRunMetadataLike


class JaxTransformModule(Protocol):
    def value_and_grad(self, function: Callable[[FloatArray], float]) -> ValueAndGradFunction: ...

    def jit(self, function: ValueAndGradFunction) -> ValueAndGradFunction: ...



@runtime_checkable
class OptimizerModule(Protocol):
    AdamConfig: AdamConfigDataclass
    AdamState: DataclassSchema
    AdamRunResult: DataclassSchema
    AdamRunMetadata: DataclassSchema
    jax: JaxTransformModule

    def adam_optimize(
        self,
        *,
        loss_fn: AdamLossFunction,
        initial_theta: FloatArray,
        context: ObjectiveContext,
        config: AdamConfigLike,
        gradient_backend: str = ...,
    ) -> AdamRunResultLike: ...


def load_optimizer_module() -> OptimizerModule:
    if not OPTIMIZER_MODULE_PATH.is_file():
        raise ModuleNotFoundError(f"No module named '{OPTIMIZER_MODULE_NAME}'")
    spec = importlib.util.spec_from_file_location(OPTIMIZER_MODULE_NAME, OPTIMIZER_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load optimizer module spec from {OPTIMIZER_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[OPTIMIZER_MODULE_NAME] = module
    spec.loader.exec_module(module)
    if not isinstance(module, OptimizerModule):
        raise AssertionError("task2_code_auto.jax_backend.optimizer does not satisfy the Adam API")
    return module


def field_names(dataclass_value: DataclassSchema) -> set[str]:
    return set(dataclass_value.__dataclass_fields__)


def _numpy_adam_reference(
    initial_theta: FloatArray,
    target: FloatArray,
    *,
    config: AdamConfigLike,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, float, int]:
    theta = np.asarray(initial_theta, dtype=np.float64).copy()
    target_array = np.asarray(target, dtype=np.float64)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)

    initial_loss = float(np.sum((theta - target_array) ** 2))
    losses = [initial_loss]
    grad_norms: list[float] = []
    best_losses = [initial_loss]
    best_theta = theta.copy()
    best_loss = initial_loss
    best_iteration = 0

    for step in range(1, config.iterations + 1):
        grad = 2.0 * (theta - target_array)
        grad_norm = float(np.linalg.norm(grad))
        m = config.beta1 * m + (1.0 - config.beta1) * grad
        v = config.beta2 * v + (1.0 - config.beta2) * (grad * grad)
        m_hat = m / (1.0 - config.beta1**step)
        v_hat = v / (1.0 - config.beta2**step)
        theta = theta - config.lr * m_hat / (np.sqrt(v_hat) + config.eps)
        if config.wrap_angles:
            theta = np.mod(theta, 2.0 * np.pi)

        current_loss = float(np.sum((theta - target_array) ** 2))
        losses.append(current_loss)
        grad_norms.append(grad_norm)
        if current_loss < best_loss:
            best_loss = current_loss
            best_theta = theta.copy()
            best_iteration = step
        best_losses.append(best_loss)

    return (
        theta,
        best_theta,
        np.asarray(losses, dtype=np.float64),
        np.asarray(grad_norms, dtype=np.float64),
        np.asarray(best_losses, dtype=np.float64),
        best_loss,
        best_iteration,
    )


class JaxOptimizerContractTests(unittest.TestCase):
    def test_step_callback_reports_initial_and_each_outer_jit_step(self) -> None:
        optimizer = load_optimizer_module()
        context = n4_context()
        theta = deterministic_random_theta()
        observed: list[tuple[int, np.ndarray, float]] = []

        def loss_fn(candidate_theta: FloatArray, objective_context: ObjectiveContext) -> float:
            del objective_context
            return jnp.sum(candidate_theta * candidate_theta)

        result = optimizer.adam_optimize(
            loss_fn=loss_fn,
            initial_theta=theta,
            context=context,
            config=optimizer.AdamConfig(iterations=2, lr=0.05, wrap_angles=False),
            step_callback=lambda step, params, loss: observed.append(
                (step, np.asarray(params, dtype=np.float64), float(loss))
            ),
        )

        self.assertFalse(bool(result.failed), str(result.failure_reason))
        self.assertEqual([step for step, _params, _loss in observed], [0, 1, 2])
        np.testing.assert_array_equal(observed[0][1], theta)
        np.testing.assert_allclose(observed[-1][1], np.asarray(result.final_params))

    def test_adam_config_state_result_and_metadata_define_legacy_compatible_schema(self) -> None:
        # Given: Phase 3 needs an isolated JAX Adam API without importing the legacy optimizer.
        optimizer = load_optimizer_module()

        # When: callers inspect the public optimizer value types.
        config_fields = field_names(optimizer.AdamConfig)
        state_fields = field_names(optimizer.AdamState)
        result_fields = field_names(optimizer.AdamRunResult)
        metadata_fields = field_names(optimizer.AdamRunMetadata)

        # Then: config/state/result preserve legacy names and add explicit JAX metadata.
        self.assertTrue(is_dataclass(optimizer.AdamConfig))
        self.assertTrue(is_dataclass(optimizer.AdamState))
        self.assertTrue(is_dataclass(optimizer.AdamRunResult))
        self.assertTrue(is_dataclass(optimizer.AdamRunMetadata))
        self.assertSetEqual(
            config_fields,
            {"iterations", "lr", "beta1", "beta2", "eps", "fd_eps", "wrap_angles"},
        )
        self.assertSetEqual(state_fields, {"iteration", "params", "m", "v"})
        self.assertSetEqual(
            result_fields,
            {
                "initial_params",
                "final_params",
                "best_params",
                "loss_history",
                "grad_norm_history",
                "best_loss_history",
                "best_loss",
                "best_iteration",
                "failed",
                "failure_reason",
                "metadata",
            },
        )
        self.assertSetEqual(
            metadata_fields,
            {
                "gradient_method",
                "optimizer_backend",
                "jax_x64_enabled",
                "fd_eps",
                "fd_eps_used",
            },
        )

    def test_zero_iteration_run_returns_complete_result_metadata_without_mutating_initial_theta(self) -> None:
        # Given: a pure explicit loss function and context, with fd_eps retained only as historical metadata.
        optimizer = load_optimizer_module()
        context = n4_context()
        theta = deterministic_random_theta()
        theta_before = theta.copy()
        config = optimizer.AdamConfig(iterations=0, lr=0.05, fd_eps=FD_EPS)

        def loss_fn(candidate_theta: FloatArray, objective_context: ObjectiveContext) -> float:
            del objective_context
            return jnp.sum(candidate_theta * candidate_theta)

        # When: a zero-iteration JAX Adam run is requested through the isolated optimizer API.
        result = optimizer.adam_optimize(
            loss_fn=loss_fn,
            initial_theta=theta,
            context=context,
            config=config,
        )

        # Then: the result schema is populated, finite, non-failed, and input theta is preserved.
        loss_history = np.asarray(result.loss_history, dtype=np.float64)
        best_loss_history = np.asarray(result.best_loss_history, dtype=np.float64)
        np.testing.assert_array_equal(theta, theta_before)
        np.testing.assert_array_equal(np.asarray(result.initial_params), theta_before)
        np.testing.assert_array_equal(np.asarray(result.final_params), theta_before)
        np.testing.assert_array_equal(np.asarray(result.best_params), theta_before)
        np.testing.assert_allclose(loss_history, np.asarray([loss_fn(theta_before, context)]))
        self.assertEqual(np.asarray(result.grad_norm_history).shape, (0,))
        np.testing.assert_array_equal(best_loss_history, loss_history)
        self.assertEqual(result.best_loss, loss_history[0])
        self.assertEqual(int(result.best_iteration), 0)
        self.assertFalse(bool(result.failed))
        self.assertEqual(str(result.failure_reason), "")
        self.assertEqual(result.metadata.gradient_method, "jax_value_and_grad")
        self.assertEqual(result.metadata.optimizer_backend, "jax_cpu")
        self.assertTrue(bool(result.metadata.jax_x64_enabled))
        self.assertEqual(float(result.metadata.fd_eps), FD_EPS)
        self.assertFalse(bool(result.metadata.fd_eps_used))

    def test_one_step_n4_jax_adam_decreases_loss_and_never_calls_finite_difference(self) -> None:
        # Given: an N4 random theta and explicit loss/context contract using JAX value_and_grad internally.
        context = n4_context()
        theta = deterministic_random_theta()
        theta_before = theta.copy()

        def loss_fn(candidate_theta: FloatArray, objective_context: ObjectiveContext) -> float:
            del objective_context
            return jnp.sum(candidate_theta * candidate_theta)

        # When: one Adam step is run while the legacy finite-difference helper is poison-patched.
        with mock.patch.object(
            legacy_training,
            "finite_difference_gradient",
            side_effect=AssertionError("JAX Adam must use jax.value_and_grad, not finite differences"),
        ):
            optimizer = load_optimizer_module()
            config = optimizer.AdamConfig(iterations=1, lr=0.05, fd_eps=FD_EPS, wrap_angles=False)
            result = optimizer.adam_optimize(
                loss_fn=loss_fn,
                initial_theta=theta,
                context=context,
                config=config,
            )

        # Then: the JAX Adam step decreases loss, remains finite, preserves input theta, and never uses FD.
        loss_history = np.asarray(result.loss_history, dtype=np.float64)
        grad_norm_history = np.asarray(result.grad_norm_history, dtype=np.float64)
        np.testing.assert_array_equal(theta, theta_before)
        self.assertFalse(bool(result.failed), str(result.failure_reason))
        self.assertEqual(np.asarray(result.initial_params).shape, (THETA_SIZE_N4,))
        self.assertEqual(np.asarray(result.final_params).shape, (THETA_SIZE_N4,))
        self.assertEqual(np.asarray(result.best_params).shape, (THETA_SIZE_N4,))
        self.assertEqual(loss_history.shape, (2,))
        self.assertEqual(grad_norm_history.shape, (1,))
        self.assertEqual(np.asarray(result.best_loss_history).shape, (2,))
        self.assertTrue(np.all(np.isfinite(np.asarray(result.final_params))))
        self.assertTrue(np.all(np.isfinite(loss_history)))
        self.assertTrue(np.all(np.isfinite(grad_norm_history)))
        np.testing.assert_array_less(loss_history[1:], loss_history[:1])
        self.assertTrue(bool(np.all(np.asarray([result.best_loss]) <= loss_history[-1:])))
        self.assertEqual(result.metadata.gradient_method, "jax_value_and_grad")
        self.assertEqual(result.metadata.optimizer_backend, "jax_cpu")
        self.assertTrue(bool(result.metadata.jax_x64_enabled))
        self.assertEqual(float(result.metadata.fd_eps), FD_EPS)
        self.assertFalse(bool(result.metadata.fd_eps_used))

    def test_default_gradient_backend_uses_outer_jit_value_and_grad(self) -> None:
        # Given: a traceable JAX loss and spies around the optimizer's JAX transform factories.
        optimizer = load_optimizer_module()
        context = n4_context()
        theta = np.asarray([0.4, -0.3], dtype=np.float64)
        config = optimizer.AdamConfig(iterations=1, lr=0.05, fd_eps=FD_EPS, wrap_angles=False)
        jit_calls: list[str] = []
        value_and_grad_calls: list[str] = []

        def loss_fn(candidate_theta: FloatArray, objective_context: ObjectiveContext) -> float:
            del objective_context
            return jnp.sum(candidate_theta * candidate_theta)

        original_value_and_grad = optimizer.jax.value_and_grad
        original_jit = optimizer.jax.jit

        def traced_value_and_grad(function: Callable[[FloatArray], float]) -> ValueAndGradFunction:
            value_and_grad_calls.append("value_and_grad")
            return original_value_and_grad(function)

        def traced_jit(function: ValueAndGradFunction) -> ValueAndGradFunction:
            jit_calls.append("jit")
            return original_jit(function)

        # When: Adam runs without an explicit gradient backend override.
        with mock.patch.object(optimizer.jax, "value_and_grad", side_effect=traced_value_and_grad), mock.patch.object(
            optimizer.jax,
            "jit",
            side_effect=traced_jit,
        ):
            result = optimizer.adam_optimize(
                loss_fn=loss_fn,
                initial_theta=theta,
                context=context,
                config=config,
            )

        # Then: the default backend builds one outer-jitted value-and-grad for this run.
        self.assertFalse(bool(result.failed), str(result.failure_reason))
        self.assertEqual(value_and_grad_calls, ["value_and_grad"])
        self.assertEqual(jit_calls, ["jit"])

    def test_explicit_outer_jit_gradient_backend_uses_outer_jit_value_and_grad(self) -> None:
        # Given: a traceable JAX loss and spies around the optimizer's JAX transform factories.
        optimizer = load_optimizer_module()
        context = n4_context()
        theta = np.asarray([0.25, -0.45], dtype=np.float64)
        config = optimizer.AdamConfig(iterations=1, lr=0.05, fd_eps=FD_EPS, wrap_angles=False)
        jit_calls: list[str] = []
        value_and_grad_calls: list[str] = []

        def loss_fn(candidate_theta: FloatArray, objective_context: ObjectiveContext) -> float:
            del objective_context
            return jnp.sum(candidate_theta * candidate_theta)

        original_value_and_grad = optimizer.jax.value_and_grad
        original_jit = optimizer.jax.jit

        def traced_value_and_grad(function: Callable[[FloatArray], float]) -> ValueAndGradFunction:
            value_and_grad_calls.append("value_and_grad")
            return original_value_and_grad(function)

        def traced_jit(function: ValueAndGradFunction) -> ValueAndGradFunction:
            jit_calls.append("jit")
            return original_jit(function)

        # When: Adam runs with the explicit outer-jit backend selection.
        with mock.patch.object(optimizer.jax, "value_and_grad", side_effect=traced_value_and_grad), mock.patch.object(
            optimizer.jax,
            "jit",
            side_effect=traced_jit,
        ):
            result = optimizer.adam_optimize(
                loss_fn=loss_fn,
                initial_theta=theta,
                context=context,
                config=config,
                gradient_backend="outer-jit",
            )

        # Then: explicit outer-jit uses the same per-run cached outer-jitted value-and-grad path.
        self.assertFalse(bool(result.failed), str(result.failure_reason))
        self.assertEqual(value_and_grad_calls, ["value_and_grad"])
        self.assertEqual(jit_calls, ["jit"])

    def test_explicit_eager_gradient_backend_uses_eager_value_and_grad(self) -> None:
        # Given: a traceable JAX loss and spies around the optimizer's JAX transform factories.
        optimizer = load_optimizer_module()
        context = n4_context()
        theta = np.asarray([0.2, -0.1], dtype=np.float64)
        config = optimizer.AdamConfig(iterations=1, lr=0.05, fd_eps=FD_EPS, wrap_angles=False)
        jit_calls: list[str] = []
        value_and_grad_calls: list[str] = []

        def loss_fn(candidate_theta: FloatArray, objective_context: ObjectiveContext) -> float:
            del objective_context
            return jnp.sum(candidate_theta * candidate_theta)

        original_value_and_grad = optimizer.jax.value_and_grad

        def traced_value_and_grad(function: Callable[[FloatArray], float]) -> ValueAndGradFunction:
            value_and_grad_calls.append("value_and_grad")
            return original_value_and_grad(function)

        def rejected_jit(function: ValueAndGradFunction) -> ValueAndGradFunction:
            jit_calls.append("jit")
            raise AssertionError("eager backend must not call jax.jit")

        # When: Adam runs with the explicit eager backend selection.
        with mock.patch.object(optimizer.jax, "value_and_grad", side_effect=traced_value_and_grad), mock.patch.object(
            optimizer.jax,
            "jit",
            side_effect=rejected_jit,
        ):
            result = optimizer.adam_optimize(
                loss_fn=loss_fn,
                initial_theta=theta,
                context=context,
                config=config,
                gradient_backend="eager",
            )

        # Then: eager uses the original non-jitted value-and-grad path.
        self.assertFalse(bool(result.failed), str(result.failure_reason))
        self.assertEqual(value_and_grad_calls, ["value_and_grad"])
        self.assertEqual(jit_calls, [])

    def test_adam_optimize_rejects_invalid_gradient_backend(self) -> None:
        # Given: valid optimizer inputs but an unsupported gradient backend selector.
        optimizer = load_optimizer_module()
        context = n4_context()
        theta = np.asarray([0.1, -0.2], dtype=np.float64)
        config = optimizer.AdamConfig(iterations=1, lr=0.05, fd_eps=FD_EPS, wrap_angles=False)

        def loss_fn(candidate_theta: FloatArray, objective_context: ObjectiveContext) -> float:
            del candidate_theta, objective_context
            raise AssertionError("invalid backend must be rejected before loss evaluation")

        # When/Then: invalid backend selection is rejected before optimization starts.
        with self.assertRaisesRegex(ValueError, "gradient_backend must be 'outer-jit' or 'eager'"):
            _ = optimizer.adam_optimize(
                loss_fn=loss_fn,
                initial_theta=theta,
                context=context,
                config=config,
                gradient_backend="compiled",
            )

    def test_two_step_jax_adam_matches_numpy_reference_when_wrap_angles_is_false(self) -> None:
        # Given: a real JAX quadratic objective and an independently computed NumPy Adam reference.
        optimizer = load_optimizer_module()
        context = n4_context()
        theta = deterministic_random_theta()
        target = np.linspace(-0.35, 0.35, theta.size, dtype=np.float64)
        target_jnp = jnp.asarray(target, dtype=jnp.float64)
        config = optimizer.AdamConfig(iterations=2, lr=0.05, fd_eps=FD_EPS, wrap_angles=False)

        def loss_fn(candidate_theta: FloatArray, objective_context: ObjectiveContext) -> float:
            del objective_context
            return jnp.sum((candidate_theta - target_jnp) ** 2)

        # When: JAX Adam executes two updates without wrapping angles.
        result = optimizer.adam_optimize(
            loss_fn=loss_fn,
            initial_theta=theta,
            context=context,
            config=config,
        )
        reference = _numpy_adam_reference(theta, target, config=config)

        # Then: the JAX optimizer matches the independent NumPy Adam reference exactly in update order.
        final_theta, best_theta, losses, grad_norms, best_losses, best_loss, best_iteration = reference
        np.testing.assert_allclose(np.asarray(result.initial_params), theta, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(result.final_params), final_theta, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(result.best_params), best_theta, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(result.loss_history), losses, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(result.grad_norm_history), grad_norms, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(result.best_loss_history), best_losses, rtol=0.0, atol=1e-12)
        self.assertAlmostEqual(float(result.best_loss), best_loss, places=12)
        self.assertEqual(int(result.best_iteration), best_iteration)
        self.assertFalse(bool(result.failed), str(result.failure_reason))

    def test_wrap_angles_applies_after_update_and_matches_numpy_reference(self) -> None:
        # Given: an initial theta outside the principal interval and a quadratic JAX objective.
        optimizer = load_optimizer_module()
        context = n4_context()
        theta = np.asarray([-0.25, 2.0 * np.pi + 0.50], dtype=np.float64)
        target = np.asarray([0.40, -0.10], dtype=np.float64)
        target_jnp = jnp.asarray(target, dtype=jnp.float64)
        config = optimizer.AdamConfig(iterations=2, lr=0.08, fd_eps=FD_EPS, wrap_angles=True)

        def loss_fn(candidate_theta: FloatArray, objective_context: ObjectiveContext) -> float:
            del objective_context
            return jnp.sum((candidate_theta - target_jnp) ** 2)

        # When: JAX Adam executes with angle wrapping enabled.
        result = optimizer.adam_optimize(
            loss_fn=loss_fn,
            initial_theta=theta,
            context=context,
            config=config,
        )
        wrapped_reference = _numpy_adam_reference(theta, target, config=config)
        prewrapped_reference = _numpy_adam_reference(np.mod(theta, 2.0 * np.pi), target, config=config)

        # Then: wrapping happens after each update, not before gradient evaluation.
        final_theta, best_theta, losses, grad_norms, best_losses, best_loss, best_iteration = wrapped_reference
        np.testing.assert_allclose(np.asarray(result.final_params), final_theta, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(result.best_params), best_theta, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(result.loss_history), losses, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(result.grad_norm_history), grad_norms, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(result.best_loss_history), best_losses, rtol=0.0, atol=1e-12)
        self.assertAlmostEqual(float(result.best_loss), best_loss, places=12)
        self.assertEqual(int(result.best_iteration), best_iteration)
        self.assertFalse(bool(result.failed), str(result.failure_reason))
        self.assertFalse(
            np.allclose(np.asarray(result.final_params), prewrapped_reference[0], rtol=0.0, atol=1e-12),
        )

    def test_adam_optimize_rejects_invalid_config_values(self) -> None:
        # Given: valid JAX loss/context inputs and configuration values that violate the optimizer contract.
        optimizer = load_optimizer_module()
        context = n4_context()
        theta = np.asarray([0.1, -0.2], dtype=np.float64)

        def loss_fn(candidate_theta: FloatArray, objective_context: ObjectiveContext) -> float:
            del objective_context
            return jnp.sum(candidate_theta * candidate_theta)

        # When/Then: each invalid Adam configuration is rejected before any optimization work begins.
        invalid_configs = (
            (optimizer.AdamConfig(iterations=-1), "iterations must be nonnegative"),
            (optimizer.AdamConfig(lr=0.0), "lr must be positive"),
            (optimizer.AdamConfig(beta1=-0.1), "beta1 must satisfy 0 <= beta1 < 1"),
            (optimizer.AdamConfig(beta1=1.0), "beta1 must satisfy 0 <= beta1 < 1"),
            (optimizer.AdamConfig(beta2=-0.1), "beta2 must satisfy 0 <= beta2 < 1"),
            (optimizer.AdamConfig(beta2=1.0), "beta2 must satisfy 0 <= beta2 < 1"),
            (optimizer.AdamConfig(eps=0.0), "eps must be positive"),
            (optimizer.AdamConfig(fd_eps=0.0), "fd_eps must be positive"),
        )
        for invalid_config, message in invalid_configs:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _ = optimizer.adam_optimize(
                        loss_fn=loss_fn,
                        initial_theta=theta,
                        context=context,
                        config=invalid_config,
                    )

    def test_adam_optimize_rejects_invalid_theta_values(self) -> None:
        # Given: valid JAX loss/context inputs and theta values that violate the optimizer contract.
        optimizer = load_optimizer_module()
        context = n4_context()
        config = optimizer.AdamConfig(iterations=1, lr=0.05, fd_eps=FD_EPS, wrap_angles=False)

        def loss_fn(candidate_theta: FloatArray, objective_context: ObjectiveContext) -> float:
            del objective_context
            return jnp.sum(candidate_theta * candidate_theta)

        # When/Then: the optimizer rejects non-vector, empty, and non-finite theta inputs.
        invalid_thetas = (
            (np.asarray([[0.1, 0.2]], dtype=np.float64), "theta must be one-dimensional"),
            (np.asarray([], dtype=np.float64), "theta must contain at least one parameter"),
            (np.asarray([np.nan], dtype=np.float64), "theta must contain only finite values"),
        )
        for invalid_theta, message in invalid_thetas:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _ = optimizer.adam_optimize(
                        loss_fn=loss_fn,
                        initial_theta=invalid_theta,
                        context=context,
                        config=config,
                    )

    def test_adam_optimize_returns_failed_result_when_gradient_is_non_finite(self) -> None:
        # Given: a real JAX objective whose gradient is non-finite at the initial point.
        optimizer = load_optimizer_module()
        context = n4_context()
        theta = np.asarray([0.0], dtype=np.float64)
        config = optimizer.AdamConfig(iterations=1, lr=0.05, fd_eps=FD_EPS, wrap_angles=False)

        def loss_fn(candidate_theta: FloatArray, objective_context: ObjectiveContext) -> float:
            del objective_context
            return jnp.sqrt(candidate_theta[0])

        # When: Adam evaluates the first JAX gradient.
        result = optimizer.adam_optimize(
            loss_fn=loss_fn,
            initial_theta=theta,
            context=context,
            config=config,
        )

        # Then: the optimizer returns a failed result with a non-finite gradient contract violation.
        np.testing.assert_array_equal(np.asarray(result.initial_params), theta)
        np.testing.assert_array_equal(np.asarray(result.final_params), theta)
        np.testing.assert_array_equal(np.asarray(result.best_params), theta)
        self.assertTrue(bool(result.failed))
        self.assertIn("step 1", str(result.failure_reason))
        self.assertIn("non-finite gradient", str(result.failure_reason))
        self.assertEqual(np.asarray(result.loss_history).shape, (1,))
        self.assertEqual(np.asarray(result.grad_norm_history).shape, (0,))
        self.assertEqual(int(result.best_iteration), 0)

    def test_adam_optimize_returns_failed_result_when_loss_is_non_finite(self) -> None:
        # Given: a real JAX objective whose post-update loss becomes non-finite.
        optimizer = load_optimizer_module()
        context = n4_context()
        theta = np.asarray([0.05], dtype=np.float64)
        config = optimizer.AdamConfig(iterations=1, lr=0.5, fd_eps=FD_EPS, wrap_angles=False)

        def loss_fn(candidate_theta: FloatArray, objective_context: ObjectiveContext) -> float:
            del objective_context
            return jnp.where(candidate_theta[0] >= 0.0, candidate_theta[0] * candidate_theta[0], jnp.inf)

        # When: Adam completes the first update and evaluates the new loss.
        result = optimizer.adam_optimize(
            loss_fn=loss_fn,
            initial_theta=theta,
            context=context,
            config=config,
        )

        # Then: the optimizer returns a failed result with a non-finite loss contract violation.
        np.testing.assert_array_equal(np.asarray(result.initial_params), theta)
        self.assertTrue(bool(result.failed))
        self.assertIn("step 1", str(result.failure_reason))
        self.assertIn("non-finite loss", str(result.failure_reason))
        self.assertEqual(np.asarray(result.loss_history).shape, (1,))
        self.assertEqual(np.asarray(result.grad_norm_history).shape, (0,))
        self.assertEqual(int(result.best_iteration), 0)

    def test_untraceable_loss_returns_failed_result_without_surrogate_optimization(self) -> None:
        # Given: a caller loss that can evaluate concretely but cannot be traced by jax.value_and_grad.
        context = n4_context()
        theta = deterministic_random_theta()
        theta_before = theta.copy()

        def loss_fn(candidate_theta: FloatArray, objective_context: ObjectiveContext) -> float:
            del objective_context
            return float(np.sum(candidate_theta * candidate_theta))

        # When: Adam reaches the first autodiff gradient evaluation.
        optimizer = load_optimizer_module()
        config = optimizer.AdamConfig(iterations=1, lr=0.05, fd_eps=FD_EPS, wrap_angles=False)
        result = optimizer.adam_optimize(
            loss_fn=loss_fn,
            initial_theta=theta,
            context=context,
            config=config,
        )

        # Then: the optimizer reports the real tracing error and does not optimize a surrogate objective.
        np.testing.assert_array_equal(theta, theta_before)
        np.testing.assert_array_equal(np.asarray(result.final_params), theta_before)
        np.testing.assert_array_equal(np.asarray(result.best_params), theta_before)
        self.assertTrue(bool(result.failed))
        self.assertIn("step 1", str(result.failure_reason))
        self.assertIn("ConcretizationTypeError", str(result.failure_reason))
        self.assertEqual(np.asarray(result.loss_history).shape, (1,))
        self.assertEqual(np.asarray(result.grad_norm_history).shape, (0,))
        self.assertEqual(result.metadata.gradient_method, "jax_value_and_grad")
        self.assertFalse(bool(result.metadata.fd_eps_used))


if __name__ == "__main__":
    _ = unittest.main()
