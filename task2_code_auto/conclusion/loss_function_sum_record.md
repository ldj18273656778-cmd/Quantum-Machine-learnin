# Loss-function change: max-block-loss -> sum-block-loss (Eq. S.2.3)

## What changed

The Module E training objective was changed from **max over block qubits** to the paper's **Eq. S.2.3 sum over block qubits**:

```text
old: loss = max_q ||R_q - I_4||_F^2
new: loss = sum_q ||R_q - I_4||_F^2   (Eq. S.2.3)
```

The new training objective matches the paper's Mode 1 deterministic local-inversion loss.  The evaluation criterion (Eq. S.2.4) ? max per-qubit error <= delta=0.01 ? is now printed as a separate diagnostic line.

## Files modified

| File | Change |
|---|---|
| `task2_code/module_e_training.py` | `max_block_loss` renamed to `sum_block_loss`; `max(losses.values())` -> `sum(losses.values())`. Backward-compatible aliases kept. |
| `task2_code/check_module_e_per_bit_losses.py` | Docstring, loss-objective print, marker removed, S.2.4 max evaluation line added. |
| `task2_code/run_module_e_training.py` | Docstring and print message updated. |
| `task2_code/run_module_e_training_with_report.py` | Print message updated. |

## Verification results

- `python task2_code/test_code/validate_module_e_structure.py --skip-n12` -> PASS

150-iteration n=4 run (lr=0.1):

| metric | value |
|---|---|
| best_sum_block_loss (S.2.3) | 0.000777 |
| max_per_qubit_loss (S.2.4) | 0.000294 |
| paper target delta | 0.01 |
| all_final_losses_decreased | True |

| q | initial | final |
|---:|---:|---:|
| 0 | 4.0325 | 0.000126 |
| 1 | 2.5972 | 0.000294 |
| 2 | 3.4896 | 0.000291 |
| 3 | 3.5479 | 0.000066 |

The max per-qubit squared Frobenius error (0.00029) is well below the paper's delta=0.01 threshold, confirming that the sum-based training gradient successfully drives all four qubits to near-zero loss.

## Why sum works better than max

- **Gradient continuity**: sum provides gradient contributions from all four qubits simultaneously, creating a smoother optimization landscape.
- **Paper alignment**: Eq. S.2.3 explicitly uses sum_k, not max_k.
- **S.2.4 is evaluation only**: the paper's max expression in S.2.4 measures post-training success probability, not training loss.

See also: `task2_code/conclusion/loss_function_squared_record.md` (squared F-norm change), `task2_code/conclusion/loss_function_max_record.md` (previous max-loss record).
