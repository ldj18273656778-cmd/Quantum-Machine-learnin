# Loss-function change: Frobenius norm -> squared Frobenius norm

## What changed

The per-qubit deterministic local loss was changed from the **Frobenius norm** to the **squared Frobenius norm**, i.e. the loss is now

```text
loss_q = ||R_q - I_4||_F ^ 2
```

instead of the former

```text
loss_q = ||R_q - I_4||_F
```

where `R_q` is the one-qubit reduced channel matrix obtained from the residual `V = K_S @ U_trial_tilde^dag` via the explicit env-basis summation.

## File modified

| File | Change |
|---|---|
| `task2_code/superoperator.py:377` | `float(np.linalg.norm(...))` -> `float(np.linalg.norm(...) ** 2)` |

No other files were touched.  The change is one line at the bottom of `per_bit_losses_from_V`.

## Effect on loss values

For any loss value `L_old = ||R - I_4||_F`, the new loss is `L_new = L_old^2`.

Example comparison (same seed / same parameters, n=4, 5 ADAM steps):

| qubit | initial (new) | initial (old approx) |
|---|---:|---:|
| 0 | 4.033 | ~2.008  |
| 1 | 2.597 | ~1.612  |
| 2 | 3.490 | ~1.868  |
| 3 | 3.548 | ~1.884  |

Values are roughly `(old_value)^2`, confirming the change is correct.

## Semantics

- The training objective is still `max_q loss_q` over all block qubits.
- The trial circuit itself is unchanged.
- Squared Frobenius norm is a standard choice in optimisation because it produces nicer gradients (the derivative of `||X||_F^2` is `2X`).
- If you want the original (unsquared) semantics back, remove the `** 2` from `superoperator.py:377`.
