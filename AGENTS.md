# AGENTS.md

Quick reference for working in this Quantum Machine Learning repo.

## Package Setup

This is a local editable package. Install with:
```bash
pip install -e .
```
Then import modules via `from sampling import ...` or `from Train import ...`.

## Key Directories

- `code/sampling/` - DQNN/ISQNN generation modules
- `code/Train/` - Parameter estimation and data generation
- `code/MNIST/` - MNIST 10x10 binarized data processing
- `data/` - Input/output data files (`.npy`, `.npz`)
- `output_images/` - Generated visualizations

## Running Scripts

All scripts use "script内参数区" pattern: edit `if __name__ == "__main__":` block variables before running.

Common variables:
- `n1`, `m` - model dimensions (n = n1 * m)
- `input_path`, `output_path` - file paths
- `target_bit` - single bit for debugging

Execute from repository root:
```bash
python code/sampling/main.py
python code/Train/estimate_theta_from_filtered_samples.py
python code/MNIST/test_MNIST.py
```

## Important Formulas

Parameter estimation (from `estimate_theta_from_filtered_samples.py`):
```
hat_theta_j = arccos(1 - (2/N_sp) * sum(y_t))
```
Where N_sp = samples matching condition x_j=0, x_neighbors=1.

## Data Files

- `xy_dataset.npy` - dict with keys: `x`, `y`, `comps`, `theta`, `n1`, `m`, `seed`
- `theta_demo.npy` - parameter vector
- `theta_estimate_all_bits.npy` - estimation output dict

## Dependencies

- **cirq** - quantum computing library (required)
- numpy, random (standard)

## Testing a Single Bit

In script, set `target_bit` in main block, then run. Script prints neighbors, matched indices, and estimated theta for that bit.