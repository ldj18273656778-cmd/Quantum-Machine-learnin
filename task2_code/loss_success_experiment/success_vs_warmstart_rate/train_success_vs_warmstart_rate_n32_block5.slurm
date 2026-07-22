#!/bin/bash
#SBATCH --job-name=warm_rate_n32_b5
#SBATCH --partition=cpu-pgmf
#SBATCH --nodes=10
#SBATCH --ntasks=500
#SBATCH --ntasks-per-node=50
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --output=/home/um202470160/task2_code/loss_success_experiment/success_vs_warmstart_rate/data/success_vs_warmstart_rate_n32_8blocks_block05_heis150_1000/logs/warm_rate_n32_b5_job%j.out
#SBATCH --error=/home/um202470160/task2_code/loss_success_experiment/success_vs_warmstart_rate/data/success_vs_warmstart_rate_n32_8blocks_block05_heis150_1000/logs/warm_rate_n32_b5_job%j.err

set -euo pipefail

EXP_ROOT="/home/um202470160/task2_code/loss_success_experiment/success_vs_warmstart_rate/data/success_vs_warmstart_rate_n32_8blocks_block05_heis150_1000"
GROUP_COUNT=500

mkdir -p "${EXP_ROOT}/logs"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate QML

if [[ ! -f "${EXP_ROOT}/manifest.json" ]]; then
    echo "ERROR: manifest not found: ${EXP_ROOT}/manifest.json" >&2
    exit 1
fi

python - <<PY
import json
from pathlib import Path
root = Path("${EXP_ROOT}")
manifest = json.loads((root / "manifest.json").read_text())
group_count = int(manifest["group_count"])
if group_count != int("${GROUP_COUNT}"):
    raise SystemExit(f"SLURM GROUP_COUNT=${GROUP_COUNT} but manifest group_count={group_count}")
print(
    "Manifest OK: "
    f"warmstart_iterations={manifest['warmstart_iterations']}, "
    f"heisenberg_iterations={manifest['heisenberg_iterations']}, "
    f"trials_per_warmstart={manifest['trials_per_warmstart']}, "
    f"group_count={group_count}"
)
PY

echo "Started at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "Tasks: ${SLURM_NTASKS}"
echo "Experiment root: ${EXP_ROOT}"
echo "Group count: ${GROUP_COUNT}"

run_group() {
    local group_index="$1"
    local log_file="${EXP_ROOT}/logs/warm_rate_n32_b5_job${SLURM_JOB_ID}_task${group_index}.out"

    srun --exclusive -N1 -n1 -c "${SLURM_CPUS_PER_TASK}" bash -lc \
        "source ~/miniconda3/etc/profile.d/conda.sh && conda activate QML && python task2_code/loss_success_experiment/success_vs_warmstart_rate/run_success_vs_warmstart_rate_batch.py --experiment-root '${EXP_ROOT}' --group-index '${group_index}'" \
        > "${log_file}" 2>&1 &
}

for group_index in $(seq 1 "${GROUP_COUNT}"); do
    run_group "${group_index}"
done

wait

echo "Finished all ${GROUP_COUNT} groups at $(date)"
