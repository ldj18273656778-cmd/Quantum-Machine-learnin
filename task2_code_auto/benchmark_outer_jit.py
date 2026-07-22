"""Standalone fixture helpers for the outer JIT benchmark."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from task2_code_auto.experiment_config import N12_3BLOCKS_HEISENBERG
from task2_code_auto.lightcone import backward_block_lightcone_from_circuit, build_lightcone_target_unitary
from task2_code_auto.module_e_training import ObjectiveContext
from task2_code_auto.jax_backend.heisenberg import heisenberg_pauli_loss
from task2_code_auto.jax_backend.runtime import jax
from task2_code_auto.target_factory import build_target_from_seed

jnp = jax.numpy
WARM_SAMPLE_COUNT = 5


@dataclass(slots=True)
class _JaxObjectiveContext:
    target_operator: object
    block_qubits: tuple[int, ...]
    lightcone_qubits: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AdamLoopConfig:
    iterations: int
    lr: float = 0.1
    beta1: float = 0.85
    beta2: float = 0.9995
    eps: float = 1e-8
    wrap_angles: bool = True


@dataclass(frozen=True, slots=True)
class AdamLoopResult:
    initial_params: jax.Array
    final_params: jax.Array
    best_params: jax.Array
    loss_history: jax.Array
    gradient_history: jax.Array
    grad_norm_history: jax.Array
    best_loss_history: jax.Array
    best_loss: jax.Array
    best_iteration: int
    failed: bool
    failure_reason: str


@dataclass(frozen=True, slots=True)
class TimingSummary:
    samples: int
    median: float
    mean: float
    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    compiled_warmup_seconds: float
    eager_one_step: TimingSummary
    compiled_one_step: TimingSummary
    eager_five_step: TimingSummary
    compiled_five_step: TimingSummary
    one_step_strict_parity: bool
    five_step_strict_parity: bool
    candidate_for_future_integration: bool


def build_n12_second_block_context() -> ObjectiveContext:
    config = N12_3BLOCKS_HEISENBERG
    block_qubits = tuple(config.blocks[1])
    target_bit = int(config.target_bits[1])
    target_spec = build_target_from_seed(config.n_qubits, config.target_seed, config.time_k)
    lightcone_qubits = tuple(backward_block_lightcone_from_circuit(target_spec.circuit, block_qubits))
    target_operator = build_lightcone_target_unitary(target_spec.circuit, lightcone_qubits)
    return ObjectiveContext(
        target_operator=target_operator,
        block_qubits=block_qubits,
        lightcone_qubits=lightcone_qubits,
        target_bit=target_bit,
    )


def build_n12_second_block_theta():
    context = build_n12_second_block_context()
    return jax.numpy.zeros(context.theta_size, dtype=jax.numpy.float64)


def make_loss(context: ObjectiveContext):
    jax_context = _JaxObjectiveContext(
        target_operator=context.target_operator,
        block_qubits=context.block_qubits,
        lightcone_qubits=context.lightcone_qubits,
    )

    def loss(theta: object) -> object:
        return heisenberg_pauli_loss(theta, jax_context)

    return loss


def make_eager_value_and_grad(loss):
    return jax.value_and_grad(loss)


@lru_cache(maxsize=None)
def make_cached_jitted_value_and_grad(
    loss,
):
    return jax.jit(jax.value_and_grad(loss))


def run_adam_loop(
    initial_theta: jax.Array,
    value_and_grad: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    config: AdamLoopConfig,
) -> AdamLoopResult:
    theta = jnp.asarray(initial_theta, dtype=jnp.float64)
    initial = jnp.array(theta, copy=True)
    initial_loss, _ = value_and_grad(theta)
    initial_loss = _scalar_loss(initial_loss)
    losses = [initial_loss]
    gradients: list[jax.Array] = []
    grad_norms: list[jax.Array] = []
    best_losses = [initial_loss]
    best_loss = initial_loss
    best_theta = jnp.array(theta, copy=True)
    best_iteration = 0
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)

    def finish(*, failed: bool, failure_reason: str) -> AdamLoopResult:
        return AdamLoopResult(
            initial_params=jnp.array(initial, copy=True),
            final_params=jnp.array(theta, copy=True),
            best_params=jnp.array(best_theta, copy=True),
            loss_history=jnp.asarray(losses, dtype=jnp.float64),
            gradient_history=jnp.asarray(gradients, dtype=jnp.float64),
            grad_norm_history=jnp.asarray(grad_norms, dtype=jnp.float64),
            best_loss_history=jnp.asarray(best_losses, dtype=jnp.float64),
            best_loss=best_loss,
            best_iteration=best_iteration,
            failed=failed,
            failure_reason=failure_reason,
        )

    if not bool(jnp.isfinite(initial_loss)):
        return finish(failed=True, failure_reason=f"initial loss is non-finite: {float(initial_loss)}")

    for step in range(1, config.iterations + 1):
        _, grad = value_and_grad(theta)
        grad = jnp.asarray(grad, dtype=jnp.float64)
        grad_norm = jnp.linalg.norm(grad)
        if not bool(jnp.all(jnp.isfinite(grad))) or not bool(jnp.isfinite(grad_norm)):
            return finish(failed=True, failure_reason=f"step {step}: non-finite gradient")

        m = config.beta1 * m + (1.0 - config.beta1) * grad
        v = config.beta2 * v + (1.0 - config.beta2) * (grad * grad)
        m_hat = m / (1.0 - config.beta1**step)
        v_hat = v / (1.0 - config.beta2**step)
        theta = theta - config.lr * m_hat / (jnp.sqrt(v_hat) + config.eps)
        if config.wrap_angles:
            theta = jnp.mod(theta, 2.0 * jnp.pi)
        if not bool(jnp.all(jnp.isfinite(theta))):
            return finish(failed=True, failure_reason=f"step {step}: non-finite theta")

        current_loss, _ = value_and_grad(theta)
        current_loss = _scalar_loss(current_loss)
        if not bool(jnp.isfinite(current_loss)):
            return finish(failed=True, failure_reason=f"step {step}: non-finite loss")

        losses.append(current_loss)
        gradients.append(grad)
        grad_norms.append(grad_norm)
        if bool(current_loss < best_loss):
            best_loss = current_loss
            best_theta = jnp.array(theta, copy=True)
            best_iteration = step
        best_losses.append(best_loss)

    return finish(failed=False, failure_reason="")


def assert_result_parity(
    eager_result: AdamLoopResult,
    jitted_result: AdamLoopResult,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> None:
    assert eager_result.failed == jitted_result.failed
    assert eager_result.failure_reason == jitted_result.failure_reason
    assert eager_result.best_iteration == jitted_result.best_iteration
    _assert_allclose(eager_result.loss_history, jitted_result.loss_history, rtol=rtol, atol=atol)
    _assert_allclose(eager_result.gradient_history, jitted_result.gradient_history, rtol=rtol, atol=atol)
    _assert_allclose(eager_result.grad_norm_history, jitted_result.grad_norm_history, rtol=rtol, atol=atol)
    _assert_allclose(eager_result.best_loss_history, jitted_result.best_loss_history, rtol=rtol, atol=atol)
    _assert_allclose(eager_result.best_params, jitted_result.best_params, rtol=rtol, atol=atol)
    _assert_allclose(eager_result.final_params, jitted_result.final_params, rtol=rtol, atol=atol)


def summarize_samples(samples: tuple[float, ...]) -> TimingSummary:
    return TimingSummary(
        samples=len(samples),
        median=float(median(samples)),
        mean=float(mean(samples)),
        minimum=float(min(samples)),
        maximum=float(max(samples)),
    )


def candidate_for_future_integration(
    *,
    strict_five_step_parity: bool,
    eager_one_step_median: float,
    compiled_one_step_median: float,
    eager_five_step_median: float,
    compiled_five_step_median: float,
) -> bool:
    return (
        strict_five_step_parity
        and compiled_one_step_median <= eager_one_step_median * 0.8
        and compiled_five_step_median <= eager_five_step_median * 0.8
    )


def build_benchmark_report() -> BenchmarkReport:
    context = build_n12_second_block_context()
    initial_theta = build_n12_second_block_theta()
    loss = make_loss(context)
    eager_value_and_grad = make_eager_value_and_grad(loss)
    compiled_value_and_grad = make_cached_jitted_value_and_grad(loss)
    one_step_config = AdamLoopConfig(iterations=1, eps=1e-8, wrap_angles=False)
    five_step_config = AdamLoopConfig(iterations=5, eps=1e-8, wrap_angles=False)

    compiled_warmup_seconds, _ = _time_adam_loop(initial_theta, compiled_value_and_grad, one_step_config)
    eager_one_step_samples, eager_one_step_result = _sample_adam_loop(initial_theta, eager_value_and_grad, one_step_config)
    compiled_one_step_samples, compiled_one_step_result = _sample_adam_loop(
        initial_theta,
        compiled_value_and_grad,
        one_step_config,
    )
    eager_five_step_samples, eager_five_step_result = _sample_adam_loop(initial_theta, eager_value_and_grad, five_step_config)
    compiled_five_step_samples, compiled_five_step_result = _sample_adam_loop(
        initial_theta,
        compiled_value_and_grad,
        five_step_config,
    )

    one_step_strict_parity = _has_strict_parity(eager_one_step_result, compiled_one_step_result)
    five_step_strict_parity = _has_strict_parity(eager_five_step_result, compiled_five_step_result)
    eager_one_step = summarize_samples(eager_one_step_samples)
    compiled_one_step = summarize_samples(compiled_one_step_samples)
    eager_five_step = summarize_samples(eager_five_step_samples)
    compiled_five_step = summarize_samples(compiled_five_step_samples)

    return BenchmarkReport(
        compiled_warmup_seconds=compiled_warmup_seconds,
        eager_one_step=eager_one_step,
        compiled_one_step=compiled_one_step,
        eager_five_step=eager_five_step,
        compiled_five_step=compiled_five_step,
        one_step_strict_parity=one_step_strict_parity,
        five_step_strict_parity=five_step_strict_parity,
        candidate_for_future_integration=candidate_for_future_integration(
            strict_five_step_parity=five_step_strict_parity,
            eager_one_step_median=eager_one_step.median,
            compiled_one_step_median=compiled_one_step.median,
            eager_five_step_median=eager_five_step.median,
            compiled_five_step_median=compiled_five_step.median,
        ),
    )


def main() -> None:
    report = build_benchmark_report()
    print("Outer JIT benchmark: N=12 second block")
    print(f"compiled warmup: {report.compiled_warmup_seconds:.6f} s")
    _print_summary("eager 1-step", report.eager_one_step)
    _print_summary("compiled 1-step", report.compiled_one_step)
    _print_summary("eager 5-step", report.eager_five_step)
    _print_summary("compiled 5-step", report.compiled_five_step)
    print(f"strict parity 1-step: {_format_pass_fail(report.one_step_strict_parity)}")
    print(f"strict parity 5-step: {_format_pass_fail(report.five_step_strict_parity)}")
    print(f"candidate_for_future_integration: {str(report.candidate_for_future_integration).lower()}")
    print("No workflow integration changed.")
    print("No artifacts written.")


def _sample_adam_loop(
    initial_theta: jax.Array,
    value_and_grad: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    config: AdamLoopConfig,
) -> tuple[tuple[float, ...], AdamLoopResult]:
    samples: list[float] = []
    result: AdamLoopResult | None = None
    for _ in range(WARM_SAMPLE_COUNT):
        seconds, result = _time_adam_loop(initial_theta, value_and_grad, config)
        samples.append(seconds)
    assert result is not None
    return tuple(samples), result


def _time_adam_loop(
    initial_theta: jax.Array,
    value_and_grad: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    config: AdamLoopConfig,
) -> tuple[float, AdamLoopResult]:
    _block_until_ready(initial_theta)
    start = perf_counter()
    result = run_adam_loop(initial_theta, value_and_grad, config)
    _block_result_until_ready(result)
    return perf_counter() - start, result


def _has_strict_parity(eager_result: AdamLoopResult, compiled_result: AdamLoopResult) -> bool:
    try:
        assert_result_parity(eager_result, compiled_result, rtol=1e-10, atol=1e-10)
    except AssertionError:
        return False
    return True


def _print_summary(label: str, summary: TimingSummary) -> None:
    message = (
        f"{label}: samples={summary.samples} median={summary.median:.6f} s "
        f"mean={summary.mean:.6f} s min={summary.minimum:.6f} s max={summary.maximum:.6f} s"
    )
    print(message)


def _format_pass_fail(value: bool) -> str:
    if value:
        return "PASS"
    return "FAIL"


def _block_result_until_ready(result: AdamLoopResult) -> None:
    _block_until_ready(result.initial_params)
    _block_until_ready(result.final_params)
    _block_until_ready(result.best_params)
    _block_until_ready(result.loss_history)
    _block_until_ready(result.gradient_history)
    _block_until_ready(result.grad_norm_history)
    _block_until_ready(result.best_loss_history)
    _block_until_ready(result.best_loss)


def _block_until_ready(value: jax.Array) -> None:
    value.block_until_ready()


def _assert_allclose(actual: jax.Array, expected: jax.Array, *, rtol: float, atol: float) -> None:
    assert bool(jnp.allclose(actual, expected, rtol=rtol, atol=atol))


def _scalar_loss(value: jax.Array) -> jax.Array:
    return jnp.asarray(value, dtype=jnp.float64)


if __name__ == "__main__":
    main()
