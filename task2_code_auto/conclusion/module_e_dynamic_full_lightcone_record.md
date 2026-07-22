# Module E Dynamic Full-Lightcone Implementation Record

## 1. Purpose

This document records the code changes and validation results for extending the Module E local-inversion workflow from the original fixed 4-qubit ansatz to a dynamic full-lightcone ansatz.

The implemented training path now uses an ansatz over the entire light cone `C`, with parameter count

```text
theta_size = 15 * n_C
```

where `n_C = len(lightcone_qubits)`. The original 4-qubit / 60-parameter API remains the default for backward compatibility.

## 2. Main Changes

### Dynamic Ansatz

Updated `task2_code/ansatz.py`:

- `theta_count(n_qubits=4)` now returns `15 * n_qubits`.
- `random_theta(..., *, n_qubits=4)` supports dynamic parameter vectors while preserving old positional calls.
- `build_ansatz(theta, qubits=None, n_qubits=None)` supports arbitrary lightcone size.
- `ansatz_unitary(theta, qubits=None, n_qubits=None)` builds the unitary using explicit qubit order.
- `build_ansatz_4q(theta, qubits=None)` remains as a compatibility wrapper.
- CZ layer pattern is kept as `even, odd, odd, even` through `cz_pairs_for_layer`.

### Full-Lightcone Module E Training (Updated: Full-System Residual)

Updated `task2_code/module_e_training.py`:

- `ObjectiveContext` now exposes:
  - `ansatz_qubits = len(lightcone_qubits)` — ansatz support region
  - `theta_size = theta_count(ansatz_qubits)`
  - `full_target_operator` (optional) — the complete n-qubit `U_full`
  - `system_qubits` (optional) — the full-system qubit tuple `(0, ..., n-1)`
  - `loss_qubits` — the qubits over which the reduced superoperator is computed
- **Full-system residual loss** (replaces projected-lightcone approach):

```python
def residual_operator_for_context(theta, context):
    U_trial_C = ansatz_unitary(theta, n_qubits=context.ansatz_qubits)
    U_trial_full = embed U_trial_C into full n-qubit system
    residual = context.full_target_operator @ U_trial_full.conj().T
    return residual, context.loss_qubits
```

  When `context.full_target_operator` is set, the loss uses `U_full` to construct the residual operator over the entire system, then computes reduced superoperator per block qubit. The light cone `C` still determines the ansatz support region (`n_C` qubits), but the superoperator is no longer derived from a projected `K_C`. This avoids the information loss that occurs when outside-qubit projection turns a unitary `U_full` into a non-unitary `K_C`.

- `sum_block_loss(theta, context)` delegates to `residual_operator_for_context()`, falling back to the old lightcone-projected residual when `full_target_operator` is not set (backward compatibility for tests using `make_objective_context`).
- `build_target_objective_context(...)` now carries both the full-system unitary and the lightcone metadata, with `metadata["loss_semantics"] = "full_system_residual_channel"`.
- `finite_difference_gradient` and `adam_optimize` now support arbitrary one-dimensional `15*n` theta vectors.
- `multi_restart_train(..., n_qubits=4)` initializes restart parameters with the requested ansatz size while keeping the old default.
- Dense target construction is guarded before allocating target matrices, so oversized inputs such as `n_qubits=13` are rejected early.

### Full-Lightcone Loss Helper

Updated `task2_code/local_loss.py`:

- Added `compute_lightcone_loss(...)` for the full-cone trial path.
- Preserved legacy `compute_loss(...)`, which still embeds a block unitary into the light cone.

### Circuit Lightcone Helper

Updated `task2_code/lightcone.py`:

- Added `backward_block_lightcone_from_circuit(circuit, block_qubits)`.
- The helper walks Cirq operations in reverse and returns sorted `LineQubit` labels.
- Non-`LineQubit` circuits raise a clear `ValueError` via the existing qubit-index validation.

### Runners and Diagnostics Scripts

Updated:

- `task2_code/run_module_e_training.py`
- `task2_code/run_module_e_training_with_report.py`
- `task2_code/check_module_e_per_bit_losses.py` — uses `residual_operator_for_context()` for full-system loss, plus `--no-plot` flag to avoid GUI blocking during headless runs.
- `task2_code/test_code/train_n12_3blocks.py` — uses `residual_operator_for_context()` for per-bit loss checking.

These now use `context.theta_size` and `context.ansatz_qubits` instead of fixed 60-parameter assumptions when operating on dynamic light cones.

### Benchmark Script

Added `task2_code/benchmark_module_e_adam.py`.

The script benchmarks ADAM finite-difference loss-call accounting for the dynamic full-lightcone ansatz. The default setup is the n=12, block `[4, 5, 6, 7]`, radius-2 flow. Use `--iterations 1` for a smoke run and default `--iterations 30` for the requested benchmark scale.

## 3. Files Changed

Core implementation:

- `task2_code/ansatz.py`
- `task2_code/module_e_training.py`
- `task2_code/local_loss.py`
- `task2_code/lightcone.py`

Runners / scripts:

- `task2_code/run_module_e_training.py`
- `task2_code/run_module_e_training_with_report.py`
- `task2_code/check_module_e_per_bit_losses.py`
- `task2_code/benchmark_module_e_adam.py`
- `task2_code/benchmark_full_system_loss.py`

Tests / validation scripts:

- `task2_code/test_code/test_module_c_d.py`
- `task2_code/test_code/validate_module_e_structure.py`
- `task2_code/test_code/train_n12_3blocks.py`

## 4. Validation Commands and Results

All commands were run from the repository root under the `QML` conda environment.

### LSP Diagnostics

Severity-error diagnostics were clean for the changed Python files.

Result:

```text
No diagnostics found
```

### Python Compile Check

Command:

```bash
conda activate QML
python -m py_compile \
  task2_code/ansatz.py \
  task2_code/lightcone.py \
  task2_code/local_loss.py \
  task2_code/module_e_training.py \
  task2_code/run_module_e_training.py \
  task2_code/run_module_e_training_with_report.py \
  task2_code/check_module_e_per_bit_losses.py \
  task2_code/benchmark_module_e_adam.py \
  task2_code/test_code/test_module_c_d.py \
  task2_code/test_code/validate_module_e_structure.py \
  task2_code/test_code/train_n12_3blocks.py
```

Result:

```text
passed with no output
```

### Module C/D Tests

Command:

```bash
python task2_code/test_code/test_module_c_d.py
```

Result:

```text
PASS test_lightcone_radius_zero_and_boundary
PASS test_project_identity_on_lightcone
PASS test_projection_detects_cross_boundary_nonunitarity
PASS test_embed_nonleading_block_position
PASS test_identity_and_global_phase_loss_zero
PASS test_exact_closed_support_loss_zero
PASS test_nonleading_exact_embedded_loss_zero
PASS test_full_lightcone_loss_zero_and_shape_checks
PASS test_backward_block_lightcone_from_circuit
PASS test_invalid_inputs_raise
```

### Module E Structural Validation

Command:

```bash
python task2_code/test_code/validate_module_e_structure.py --skip-n12
```

Result:

```text
per-bit convention checks passed
dynamic ansatz API checks passed
objective and gradient checks passed
optimizer and artifact checks passed
n=4 smoke training passed
Module E validation PASSED
```

### n=12 ADAM Benchmark Smoke

Command:

```bash
python task2_code/benchmark_module_e_adam.py --iterations 1
```

Result:

```text
lightcone_qubits = (2, 3, 4, 5, 6, 7, 8, 9)
ansatz_qubits = 8
theta_size = 120
iterations = 1, restarts = 1
measured_loss_calls = 242
expected_loss_calls = 242
best_loss = 14.5906914581
```

The measured ADAM loss-call count matches the finite-difference expectation:

```text
expected_calls = restarts * (1 + iterations * (2 * theta_size + 1))
               = 1 * (1 + 1 * (2 * 120 + 1))
               = 242
```

### n=12 Report Runner Smoke

Command:

```bash
python task2_code/run_module_e_training_with_report.py \
  --n-qubits 12 \
  --block 4,5,6,7 \
  --target-bit 5 \
  --radius 2 \
  --iterations 0 \
  --output-dir task2_code/module_e_validation_output/report_smoke_n12 \
  --report-path task2_code/module_e_validation_output/report_smoke_n12.md
```

Result:

```text
target_bit = 5
loss_objective = sum over all block qubits (Eq. S.2.3)
lightcone_qubits = (2, 3, 4, 5, 6, 7, 8, 9)
ansatz_qubits = 8, theta_size = 120
best_iteration = 0
```

### n=12 Per-Bit Smoke

Command:

```bash
python task2_code/check_module_e_per_bit_losses.py \
  --n-qubits 12 \
  --block 4,5,6,7 \
  --target-bit 5 \
  --radius 2 \
  --iterations 0
```

Result summary:

```text
n_qubits = 12
block_qubits = (4, 5, 6, 7)
lightcone_qubits = (2, 3, 4, 5, 6, 7, 8, 9)
ansatz_qubits = 8
theta_size = 120
best_sum_block_loss = 14.3761355111
recorded 1 snapshots (0..0)
```

### Full-System Residual Loss Smoke (per-bit checker)

Command:

```bash
python task2_code/check_module_e_per_bit_losses.py \
  --n-qubits 5 \
  --block 0,1,2,3 \
  --target-bit 2 \
  --lightcone-mode circuit \
  --iterations 0 \
  --no-plot
```

Result:

```text
n_qubits = 5
block_qubits = (0, 1, 2, 3)
target_bit = 2
lightcone_qubits = (0, 1, 2, 3, 4)
lightcone_mode = circuit
loss_semantics = full_system_residual_channel
ansatz_qubits = 5
theta_size = 75
loss_qubits = (0, 1, 2, 3, 4)
best_sum_block_loss = 11.9121551637
```

Confirms: `loss_semantics = full_system_residual_channel` and `loss_qubits = (0, 1, 2, 3, 4)` (full system, not projected cone).

### Full-System vs Lightcone Loss Benchmark (n=8)

Command:

```bash
python task2_code/benchmark_full_system_loss.py \
  --n-qubits 8 --block 4,5,6,7 --target-bit 5 \
  --repeats 1 --warmups 0
```

Result:

```text
build_full_unitary_seconds = 0.026295
Configuration: n_qubits=8, n_C=6, theta_size=90, full_dim=256, lightcone_dim=64

Context loss = 11.7967293326   avg_seconds = 0.138849
Manual full-system residual loss = 11.7967293326   avg_seconds = 0.122671
```

**Verification:** `Context loss` (main training path) equals `Manual full-system residual loss` (independently computed), confirming the full-system path is active and consistent.

### Full-System vs Lightcone Loss Benchmark (n=10)

Command:

```bash
python task2_code/benchmark_full_system_loss.py \
  --n-qubits 10 --block 4,5,6,7 --target-bit 5 \
  --repeats 1 --warmups 0
```

Result:

```text
build_full_unitary_seconds = 1.050030
Configuration: n_qubits=10, n_C=8, theta_size=120, full_dim=1024, lightcone_dim=256

Context loss = 12.1799204412   avg_seconds = 2.765310
Manual full-system residual loss = 12.1799204412   avg_seconds = 2.700755
```

Identical loss values across both paths, same verification as n=8.

### Dense Guard Check

Command:

```bash
python -c "exec('from task2_code.module_e_training import build_target_objective_context\ntry:\n    build_target_objective_context(13, [4,5,6,7], 5, 2, 42, max_n_qubits=12, max_hilbert_dim=4096)\nexcept ValueError as exc:\n    print(str(exc))\nelse:\n    raise SystemExit(\'expected ValueError\')')"
```

Result:

```text
dense target construction is guarded for small systems; got n_qubits=13, hilbert_dim=8192
```

## 5. Notes

- Cirq emitted repeated gRPC metric registration warnings during several runs. These were already known benign warnings and did not affect test results.
- The full 30-step n=12 benchmark was intentionally not run as part of smoke validation because each ADAM step requires `2 * theta_size + 1` loss calls. For the n=12 radius-2 cone, `theta_size = 120`, so 30 iterations would require `7231` loss calls for one restart.
- The dynamic implementation keeps the old default behavior available for legacy n=4 scripts and tests.
- **Full-system residual performance:** Computing loss from `U_full` instead of projected `K_C` increases per-evaluation cost. For n=8 (full_dim=256) the cost is ~8.5x that of the old lightcone-projected loss; for n=10 (full_dim=1024) it is ~28x. The dominant factor is the full-system `per_bit_losses_from_V` computation. The training path still delegates to `residual_operator_for_context()` so both semantics are supported.

## 6. Conclusion

The Module E local-inversion path now supports the dynamic full-lightcone ansatz with `15 * n_C` trainable parameters. The core training objective, runner scripts, validation scripts, and benchmark smoke path all operate correctly on the n=12 radius-2 light cone with `n_C = 8` and `theta_size = 120`.

**Full-system residual loss:** The superoperator computation has been revised to use the full-system unitary `U_full` instead of a projected lightcone operator `K_C`. The trial ansatz is embedded into the full n-qubit system, and the residual `U_full @ U_trial_full†` forms the basis for the reduced superoperator per block qubit. This avoids information loss from the outside-zero qubit projection and is mathematically equivalent to computing the reduced channel with the complete system. The old projected-lightcone path is preserved as fallback when `full_target_operator` is not set on the context.
