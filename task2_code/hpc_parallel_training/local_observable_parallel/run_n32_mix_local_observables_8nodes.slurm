#!/bin/bash
#SBATCH --job-name=n32_mix_obs
#SBATCH --account=um202470160
#SBATCH --partition=cpu-pgmf
#SBATCH --qos=pgmf
#SBATCH --nodes=8
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/home/um202470160/logs/n32_mix_obs_%j.out
#SBATCH --error=/home/um202470160/logs/n32_mix_obs_%j.err

set -euo pipefail

EXPERIMENT_ROOT=/home/um202470160/task2_code/hpc_parallel_training/data/task2_parallel_n32_8blocks_default_5layer_cz_edge_quantum_channel_superoperator_from_mix_20260613_180437
BUNDLE=${EXPERIMENT_ROOT}/bundle/local_observable_bundle_${SLURM_JOB_ID}
OUTDIR=${EXPERIMENT_ROOT}/local_observable_parallel_${SLURM_JOB_ID}
ASSEMBLE_SCRIPT=/home/um202470160/task2_code/hpc_parallel_training/assemble_parallel_bundle.py
COMPARE_SCRIPT=/home/um202470160/task2_code/hpc_parallel_training/local_observable_parallel/compare_local_observables.py
MERGE_SCRIPT=/home/um202470160/task2_code/hpc_parallel_training/local_observable_parallel/merge_local_observables.py
PYTHON=/home/um202470160/miniconda3/bin/conda

mkdir -p "${OUTDIR}"

echo "job_id=${SLURM_JOB_ID}"
echo "nodes=${SLURM_JOB_NODELIST}"
echo "start=$(date +%FT%T%z)"
echo "experiment_root=${EXPERIMENT_ROOT}"
echo "bundle=${BUNDLE}"
echo "outdir=${OUTDIR}"
echo "ntasks=${SLURM_NTASKS} cpus_per_task=${SLURM_CPUS_PER_TASK}"

echo "assembling bundle at $(date +%FT%T%z)"
"${PYTHON}" run -n QML python "${ASSEMBLE_SCRIPT}" \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --output-dir "${BUNDLE}"

echo "bundle assembled at $(date +%FT%T%z)"

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
  --output-prefix local_observables_n32_mix_zero_XYZ_merged

echo "end=$(date +%FT%T%z)"
