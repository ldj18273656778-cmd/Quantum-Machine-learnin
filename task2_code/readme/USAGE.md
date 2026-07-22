# Task 2 Usage Notes

This directory is kept compatible with the existing script-based workflow. The
current cleanup does not move source files, saved parameters, plots, or result
directories. Existing imports such as `from task2_code.module_e_training import ...`
remain supported; single-block debugging should use
`python task2_code/check_module_e_per_bit_losses.py`.

## Backup

A fresh timestamped backup should be created before structural edits. For this
cleanup the verified backup is stored at the repository root as:

```text
task2_code_backup_20260526_164753.zip
task2_code_backup_20260526_164753_manifest.json
```

The older `task2_code_copy.zip` file is left untouched. The manifest records the
files included in the backup, their sizes, and SHA256 hashes. Cache directories
such as `__pycache__` are excluded.

## Running From The Repository Root

Run commands from the repository root so relative paths like `task2_code/data`
and `report/task2` resolve the same way as the original scripts.

Common checks:

```powershell
python -m compileall task2_code
python -c "import task2_code; import task2_code.module_e_training; import task2_code.U_target; print('imports ok')"
python task2_code/sewing/sew_saved_n12.py --validate-only
```

Training and diagnostics:

```powershell
python task2_code/check_module_e_per_bit_losses.py --block 4,5,6,7 --target-bit 5 --iterations 150
python task2_code/train_n12_3blocks.py
python task2_code/train_n20_5blocks.py
```

Saved n=12 sewing workflow:

```powershell
python task2_code/sewing/sew_saved_n12.py --validate-only
python task2_code/sewing/plot_u_sewing_matplotlib.py
python task2_code/sewing/compare_all_pauli_expectations.py
python task2_code/sewing/compare_ghz_input.py
```

## Compatibility Rules

- Do not change RNG draw order, target seed defaults, `time_k` defaults, or
  `U_target` gate order unless a target-equivalence check is added.
- Do not rename JSON metadata fields or NPZ keys used by saved training artifacts.
- Do not move historical artifacts under `task2_code/data`,
  `task2_code/module_e_results*`, `task2_code/module_e_validation_output`, or
  `report/task2` without adding compatibility wrappers and updating references.
- If new module-style entry points are added later, keep the current direct script
  commands as thin wrappers over the same implementation.

## Rollback

If a future refactor breaks compatibility, stop editing and restore `task2_code`
from the verified timestamped backup. Keep `task2_code_copy.zip` unchanged unless
the user explicitly requests replacing it.
