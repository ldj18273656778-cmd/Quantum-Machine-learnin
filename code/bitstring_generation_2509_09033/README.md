# Bitstring Generation Reproduction (arXiv:2509.09033)

This folder is fully independent from your existing code.

## What is implemented

- `idqnn_bitstring.py`
  - `sample_shallow_idqnn(...)`: Appendix C.3.a, Algorithm 1.
  - `sample_deep_mapped_idqnn(...)`: Appendix C.3.a, Algorithm 2 (mapped deep process).
  - Metrics/utilities:
    - exact shallow probabilities
    - linear XEB
    - empirical TV distance

- `reproduce_bitstring.py`
  - Runs a reproducibility check for bitstring generation:
    - compares shallow and deep mapped samplers
    - reports linear XEB and TV distance

## Run

```powershell
& "C:\ProgramData\anaconda3\python.exe" "code\bitstring_generation_2509_09033\reproduce_bitstring.py" --n1 4 --m 3 --shots 4000 --seed 7
```

## Understand and modify

- Read: `code/bitstring_generation_2509_09033/GUIDE_CN.md`

## Validate correctness

```powershell
& "C:\ProgramData\anaconda3\python.exe" "code\bitstring_generation_2509_09033\test_idqnn_correctness.py"
```

## Notes

- Input bitstring uses row-major order: `x[t, q]` flattened as `t=0..n1-1`, `q=0..m-1`.
- `x=0` means prepare/operate with `H`; `x=1` means keep/reset to `|0>`, following the paper convention.
- In this machine, `C:\ProgramData\anaconda3\envs\dwave\python.exe` crashes when importing `cirq`; use `C:\ProgramData\anaconda3\python.exe`.
