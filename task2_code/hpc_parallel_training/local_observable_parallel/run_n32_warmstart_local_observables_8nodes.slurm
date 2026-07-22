#!/bin/bash
#SBATCH --job-name=n32_local_obs
#SBATCH --account=um202470160
#SBATCH --partition=cpu-pgmf
#SBATCH --qos=pgmf
#SBATCH --nodes=8
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/home/um202470160/logs/n32_local_obs_%j.out
#SBATCH --error=/home/um202470160/logs/n32_local_obs_%j.err

set -euo pipefail

EXPERIMENT_ROOT=/home/um202470160/task2_code/hpc_parallel_training/data/warmstart_n32_8blocks_heisenberg_threshold001_20260618_222057
BUNDLE=${EXPERIMENT_ROOT}/bundle/task2_training_n32_8blocks_heisenberg_default_5layer_cz_heisenberg_pauli_20260619_201516
OUTDIR=${EXPERIMENT_ROOT}/local_observable_parallel_${SLURM_JOB_ID}
COMPARE_SCRIPT=/home/um202470160/task2_code/hpc_parallel_training/local_observable_parallel/compare_local_observables.py
MERGE_SCRIPT=/home/um202470160/task2_code/hpc_parallel_training/local_observable_parallel/merge_local_observables.py
PYTHON=/home/um202470160/miniconda3/bin/conda

mkdir -p "${OUTDIR}"

echo "job_id=${SLURM_JOB_ID}"
echo "nodes=${SLURM_JOB_NODELIST}"
echo "start=$(date +%FT%T%z)"
echo "bundle=${BUNDLE}"
echo "outdir=${OUTDIR}"
echo "ntasks=${SLURM_NTASKS} cpus_per_task=${SLURM_CPUS_PER_TASK}"

for shard in $(seq 0 31); do
  echo "launch shard ${shard} at $(date +%FT%T%z)"
  srun --exclusive -N1 -n1 -c "${SLURM_CPUS_PER_TASK}" \
    /usr/bin/time -v "${PYTHON}" run -n QML python "${COMPARE_SCRIPT}" \
      --run-dir "${BUNDLE}" \
      --num-shards 32 \
      --shard-index "${shard}" \
      --input-state zero \
      --paulis X,Y,Z \
      --order odd-even \
      --max-cone-qubits 32 \
      --output-dir "${OUTDIR}" \
      > "${OUTDIR}/shard_${shard}.out" \
      2> "${OUTDIR}/shard_${shard}.err" &
done

wait

echo "all shards finished at $(date +%FT%T%z)"

"${PYTHON}" run -n QML python "${MERGE_SCRIPT}" \
  --input-dir "${OUTDIR}" \
  --output-prefix local_observables_n32_zero_XYZ_merged

echo "end=$(date +%FT%T%z)"
