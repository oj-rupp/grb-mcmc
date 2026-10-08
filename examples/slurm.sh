#!/bin/bash
#SBATCH --job-name=grb-mcmc
#SBATCH --array=0-3
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=grb-%A_%a.log
set -euo pipefail
# Submit from the repository root after generating examples/toy_sweep.json.
# Adjust array bounds to the manifest length minus one and activate your env.
source .venv/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
grb-mcmc job runs/sweep/manifest.json "${SLURM_ARRAY_TASK_ID}"
