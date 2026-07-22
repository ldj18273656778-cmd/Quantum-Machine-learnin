from __future__ import annotations

import importlib
import importlib.util
import io
from collections.abc import Callable
from contextlib import redirect_stdout
from typing import Any, Protocol, runtime_checkable
import unittest
from unittest.mock import patch

from task2_code_auto.module_e_training import ObjectiveContext
from task2_code_auto.jax_backend.runtime import jax as jax_runtime


@runtime_checkable
class BenchmarkOuterJitModule(Protocol):
    AdamLoopConfig: Any
    AdamLoopResult: Any
    BenchmarkReport: Any
    TimingSummary: Any

    def build_n12_second_block_context(self) -> ObjectiveContext: ...

    def build_n12_second_block_theta(self) -> Any: ...

    def make_loss(self, context: ObjectiveContext) -> Callable[[Any], Any]: ...

    def make_eager_value_and_grad(self, loss: Callable[[Any], Any]) -> Callable[[Any], tuple[Any, Any]]: ...

    def make_cached_jitted_value_and_grad(self, loss: Callable[[Any], Any]) -> Callable[[Any], tuple[Any, Any]]: ...

    def run_adam_loop(self, initial_theta: Any, value_and_grad: Callable[[Any], tuple[Any, Any]], config: Any) -> Any: ...

    def assert_result_parity(self, eager_result: Any, jitted_result: Any, *, rtol: float, atol: float) -> None: ...

    def summarize_samples(self, samples: tuple[float, ...]) -> Any: ...

    def candidate_for_future_integration(
        self,
        *,
        strict_five_step_parity: bool,
        eager_one_step_median: float,
        compiled_one_step_median: float,
        eager_five_step_median: float,
        compiled_five_step_median: float,
    ) -> bool: ...

    def build_benchmark_report(self) -> Any: ...

    def main(self) -> None: ...

    def _time_adam_loop(self, initial_theta: Any, value_and_grad: Callable[[Any], tuple[Any, Any]], config: Any) -> Any: ...


def load_benchmark_module() -> BenchmarkOuterJitModule:
    spec = importlib.util.find_spec("task2_code_auto.benchmark_outer_jit")
    assert spec is not None, "task2_code_auto.benchmark_outer_jit must exist"
    module = importlib.import_module("task2_code_auto.benchmark_outer_jit")
    assert isinstance(module, BenchmarkOuterJitModule)
    return module


class BenchmarkOuterJitFixtureTests(unittest.TestCase):
    def test_n12_second_block_context_matches_exact_production_configuration(self) -> None:
        # Given: the benchmark fixture should reuse the approved N12 production helpers.
        module = load_benchmark_module()

        # When: the benchmark context is built from the fixture helper.
        context = module.build_n12_second_block_context()

        # Then: the exact second-block configuration and projected light-cone are preserved.
        self.assertEqual(context.block_qubits, (4, 5, 6, 7))
        self.assertEqual(context.target_bit, 5)
        self.assertEqual(context.lightcone_qubits, (2, 3, 4, 5, 6, 7, 8, 9))
        self.assertEqual(context.target_operator.shape, (256, 256))
        self.assertEqual(context.theta_size, 120)
        self.assertEqual(context.loss_mode, "lightcone")
        self.assertIsNone(context.full_target_operator)
        self.assertIsNone(context.system_qubits)

    def test_n12_second_block_theta_is_deterministic_and_has_shape_120(self) -> None:
        # Given: the benchmark theta helper must be stable across repeated calls.
        module = load_benchmark_module()

        # When: the helper materializes its theta fixture twice.
        theta_one = module.build_n12_second_block_theta()
        theta_two = module.build_n12_second_block_theta()

        # Then: both fixtures are identical and the theta vector has the approved N12 length.
        self.assertEqual(tuple(theta_one.shape), (120,))
        self.assertTrue(bool(jax_runtime.numpy.array_equal(theta_one, theta_two)))

    def test_cached_outer_jit_value_and_grad_matches_uncached_n12_parity(self) -> None:
        # Given: the benchmark fixture exposes eager and cached outer-jitted factories over the same closed-over loss.
        module = load_benchmark_module()
        context = module.build_n12_second_block_context()
        theta = module.build_n12_second_block_theta()

        # When: the eager and cached factories are built from the same fixed context loss.
        loss = module.make_loss(context)
        eager_value_and_grad = module.make_eager_value_and_grad(loss)
        cached_value_and_grad = module.make_cached_jitted_value_and_grad(loss)
        cached_value_and_grad_second = module.make_cached_jitted_value_and_grad(loss)
        expected_value_and_grad = jax_runtime.value_and_grad(loss)
        assert eager_value_and_grad is not None
        assert cached_value_and_grad is not None
        assert cached_value_and_grad_second is not None
        expected_loss, expected_grad = expected_value_and_grad(theta)
        eager_loss, eager_grad = eager_value_and_grad(theta)
        actual_loss, actual_grad = cached_value_and_grad(theta)

        # Then: the cached factory reuses one callable, warms successfully, and matches eager and uncached parity.
        self.assertIs(cached_value_and_grad, cached_value_and_grad_second)
        self.assertIsNot(eager_value_and_grad, cached_value_and_grad)
        getattr(actual_loss, "block_until_ready")()
        getattr(actual_grad, "block_until_ready")()
        getattr(eager_loss, "block_until_ready")()
        getattr(eager_grad, "block_until_ready")()
        getattr(expected_loss, "block_until_ready")()
        getattr(expected_grad, "block_until_ready")()
        self.assertEqual(tuple(actual_grad.shape), (120,))
        self.assertEqual(tuple(expected_grad.shape), (120,))
        self.assertEqual(tuple(eager_grad.shape), (120,))
        self.assertTrue(bool(jax_runtime.numpy.allclose(eager_loss, expected_loss, rtol=1e-10, atol=1e-10)))
        self.assertTrue(bool(jax_runtime.numpy.allclose(eager_grad, expected_grad, rtol=1e-10, atol=1e-10)))
        self.assertTrue(bool(jax_runtime.numpy.allclose(actual_loss, expected_loss, rtol=1e-10, atol=1e-10)))
        self.assertTrue(bool(jax_runtime.numpy.allclose(actual_grad, expected_grad, rtol=1e-10, atol=1e-10)))

    def test_adam_loop_one_step_matches_eager_and_cached_outer_jit(self) -> None:
        eager_result, cached_result = self._run_eager_and_cached_adam_loop(iterations=1)

        module = load_benchmark_module()
        module.assert_result_parity(eager_result, cached_result, rtol=1e-10, atol=1e-10)
        self._assert_matching_success(eager_result, cached_result)

    def test_adam_loop_five_steps_rejects_strict_parameter_parity_mismatch(self) -> None:
        eager_result, cached_result = self._run_eager_and_cached_adam_loop(iterations=5)

        module = load_benchmark_module()
        with self.assertRaises(AssertionError):
            module.assert_result_parity(eager_result, cached_result, rtol=1e-10, atol=1e-10)
        self._assert_matching_success(eager_result, cached_result)
        self.assertTrue(bool(jax_runtime.numpy.allclose(eager_result.loss_history, cached_result.loss_history, rtol=1e-10, atol=1e-10)))
        self.assertTrue(bool(jax_runtime.numpy.allclose(eager_result.gradient_history, cached_result.gradient_history, rtol=1e-10, atol=1e-10)))
        self.assertTrue(bool(jax_runtime.numpy.allclose(eager_result.grad_norm_history, cached_result.grad_norm_history, rtol=1e-10, atol=1e-10)))
        self.assertTrue(bool(jax_runtime.numpy.allclose(eager_result.best_loss_history, cached_result.best_loss_history, rtol=1e-10, atol=1e-10)))
        self.assertFalse(bool(jax_runtime.numpy.allclose(eager_result.best_params, cached_result.best_params, rtol=1e-10, atol=1e-10)))
        self.assertFalse(bool(jax_runtime.numpy.allclose(eager_result.final_params, cached_result.final_params, rtol=1e-10, atol=1e-10)))

    def test_summarize_samples_reports_count_and_distribution(self) -> None:
        # Given: five deterministic timing samples from a warm benchmark loop.
        module = load_benchmark_module()

        # When: the benchmark summarizes them for the CLI report.
        summary = module.summarize_samples((0.030, 0.010, 0.020, 0.050, 0.040))

        # Then: the report exposes sample count, median, mean, min, and max.
        self.assertEqual(summary.samples, 5)
        self.assertEqual(summary.median, 0.030)
        self.assertEqual(summary.mean, 0.030)
        self.assertEqual(summary.minimum, 0.010)
        self.assertEqual(summary.maximum, 0.050)

    def test_candidate_for_future_integration_requires_strict_parity_and_speedup(self) -> None:
        # Given: compiled timing is more than 20% faster, but strict five-step parity fails.
        module = load_benchmark_module()

        # When: candidate status is computed for the known Task 3 strict parity mismatch.
        candidate = module.candidate_for_future_integration(
            strict_five_step_parity=False,
            eager_one_step_median=1.0,
            compiled_one_step_median=0.70,
            eager_five_step_median=1.0,
            compiled_five_step_median=0.70,
        )

        # Then: speed alone is insufficient; strict parity remains mandatory.
        self.assertFalse(candidate)
        self.assertTrue(
            module.candidate_for_future_integration(
                strict_five_step_parity=True,
                eager_one_step_median=1.0,
                compiled_one_step_median=0.80,
                eager_five_step_median=1.0,
                compiled_five_step_median=0.80,
            )
        )
        self.assertFalse(
            module.candidate_for_future_integration(
                strict_five_step_parity=True,
                eager_one_step_median=1.0,
                compiled_one_step_median=0.80,
                eager_five_step_median=1.0,
                compiled_five_step_median=0.81,
                )
        )

    def test_candidate_for_future_integration_rejects_missing_one_step_speedup(self) -> None:
        # Given: strict parity passes and compiled 5-step timing qualifies, but 1-step timing does not.
        module = load_benchmark_module()

        # When: candidate status is computed with insufficient 1-step speedup.
        candidate = module.candidate_for_future_integration(
            strict_five_step_parity=True,
            eager_one_step_median=1.0,
            compiled_one_step_median=0.81,
            eager_five_step_median=1.0,
            compiled_five_step_median=0.70,
        )

        # Then: 5-step speedup alone is insufficient.
        self.assertFalse(candidate)

    def test_time_adam_loop_blocks_at_perf_counter_boundaries(self) -> None:
        # Given: a deterministic loop result whose arrays record readiness boundaries.
        module = load_benchmark_module()
        events: list[str] = []

        class BlockableArray:
            def __init__(self, name: str) -> None:
                self.name: str = name

            def block_until_ready(self) -> None:
                events.append(f"block {self.name}")

        result_array_names = (
            "result.initial_params", "result.final_params", "result.best_params", "result.loss_history",
            "result.gradient_history", "result.grad_norm_history", "result.best_loss_history", "result.best_loss",
        )
        result_arrays = tuple(BlockableArray(name) for name in result_array_names)
        result = module.AdamLoopResult(
            initial_params=result_arrays[0],
            final_params=result_arrays[1],
            best_params=result_arrays[2],
            loss_history=result_arrays[3],
            gradient_history=result_arrays[4],
            grad_norm_history=result_arrays[5],
            best_loss_history=result_arrays[6],
            best_loss=result_arrays[7],
            best_iteration=0,
            failed=False,
            failure_reason="",
        )

        def run_loop(_initial_theta: Any, _value_and_grad: Callable[[Any], tuple[Any, Any]], _config: Any) -> Any:
            events.append("run_adam_loop")
            return result

        def counter() -> float:
            perf_events = [event for event in events if event.startswith("perf_counter")]
            label = "start" if not perf_events else "end"
            events.append(f"perf_counter {label}")
            return 10.0 if label == "start" else 12.5

        # When: the benchmark-local timer wraps one Adam loop execution.
        with patch.object(module, "run_adam_loop", side_effect=run_loop), patch.object(module, "perf_counter", side_effect=counter):
            seconds, actual_result = module._time_adam_loop(
                BlockableArray("initial_theta"),
                lambda theta: (theta, theta),
                module.AdamLoopConfig(iterations=1),
            )

        # Then: input readiness happens before timing, and result readiness happens before the ending sample.
        self.assertEqual(seconds, 2.5)
        self.assertIs(actual_result, result)
        expected_events = ["block initial_theta", "perf_counter start", "run_adam_loop"]
        expected_events.extend(f"block {name}" for name in result_array_names)
        expected_events.append("perf_counter end")
        self.assertEqual(
            events,
            expected_events,
        )

    def test_build_benchmark_report_separates_compile_warmup_and_five_warm_samples(self) -> None:
        # Given: deterministic timing boundary samples for compile warmup plus four warm loop groups.
        module = load_benchmark_module()
        timed_results = [(float(index), f"result-{index}") for index in range(1, 22)]

        # When: the benchmark report is built without touching the expensive real fixture.
        with (
            patch.object(module, "build_n12_second_block_context", return_value="context"),
            patch.object(module, "build_n12_second_block_theta", return_value="theta"),
            patch.object(module, "make_loss", return_value="loss"),
            patch.object(module, "make_eager_value_and_grad", return_value="eager"),
            patch.object(module, "make_cached_jitted_value_and_grad", return_value="compiled"),
            patch.object(module, "run_adam_loop", side_effect=AssertionError("untimed loop execution")),
            patch.object(module, "_time_adam_loop", side_effect=timed_results) as timed_loop,
            patch.object(module, "_has_strict_parity", side_effect=(True, False)),
        ):
            report = module.build_benchmark_report()

        # Then: compile timing is distinct, and each eager/compiled 1-step/5-step loop has exactly five samples.
        self.assertEqual(timed_loop.call_count, 21)
        self.assertEqual(report.compiled_warmup_seconds, 1.0)
        self.assertEqual(report.eager_one_step.samples, 5)
        self.assertEqual(report.compiled_one_step.samples, 5)
        self.assertEqual(report.eager_five_step.samples, 5)
        self.assertEqual(report.compiled_five_step.samples, 5)
        self.assertTrue(report.one_step_strict_parity)
        self.assertFalse(report.five_step_strict_parity)
        self.assertFalse(report.candidate_for_future_integration)

    def test_main_prints_benchmark_report_without_artifacts_or_workflow_changes(self) -> None:
        # Given: a deterministic report from the benchmark runner.
        module = load_benchmark_module()
        summary = module.TimingSummary(samples=5, median=0.20, mean=0.22, minimum=0.18, maximum=0.30)
        report = module.BenchmarkReport(
            compiled_warmup_seconds=1.25,
            eager_one_step=summary,
            compiled_one_step=summary,
            eager_five_step=summary,
            compiled_five_step=summary,
            one_step_strict_parity=True,
            five_step_strict_parity=False,
            candidate_for_future_integration=False,
        )
        stdout = io.StringIO()

        # When: the CLI entrypoint prints the report.
        with patch.object(module, "build_benchmark_report", return_value=report), redirect_stdout(stdout):
            module.main()

        # Then: the CLI surfaces timing, strict parity, and non-integration decisions without artifacts.
        output = stdout.getvalue()
        self.assertIn("compiled warmup: 1.250000 s", output)
        self.assertIn("eager 1-step: samples=5 median=0.200000 s mean=0.220000 s min=0.180000 s max=0.300000 s", output)
        self.assertIn("strict parity 5-step: FAIL", output)
        self.assertIn("candidate_for_future_integration: false", output)
        self.assertIn("No workflow integration changed.", output)
        self.assertIn("No artifacts written.", output)

    def _run_eager_and_cached_adam_loop(self, iterations: int) -> tuple[Any, Any]:
        # Given: eager and cached outer-jitted value-and-grad callables share the same N12 loss fixture.
        module = load_benchmark_module()
        context = module.build_n12_second_block_context()
        initial_theta = module.build_n12_second_block_theta()
        loss = module.make_loss(context)
        config = module.AdamLoopConfig(iterations=iterations, eps=1e-8, wrap_angles=False)
        eager_value_and_grad = module.make_eager_value_and_grad(loss)
        cached_value_and_grad = module.make_cached_jitted_value_and_grad(loss)

        # When: the benchmark-local Adam loop drives both callables for the same number of steps.
        eager_result = module.run_adam_loop(initial_theta, eager_value_and_grad, config)
        cached_result = module.run_adam_loop(initial_theta, cached_value_and_grad, config)

        return eager_result, cached_result

    def _assert_matching_success(self, eager_result: Any, cached_result: Any) -> None:
        self.assertFalse(eager_result.failed)
        self.assertFalse(cached_result.failed)
        self.assertEqual(eager_result.failure_reason, "")
        self.assertEqual(cached_result.failure_reason, "")
        self.assertEqual(eager_result.best_iteration, cached_result.best_iteration)


if __name__ == "__main__":
    _ = unittest.main()
