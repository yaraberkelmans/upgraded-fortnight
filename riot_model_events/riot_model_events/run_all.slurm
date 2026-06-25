#!/bin/bash
#SBATCH --job-name=riot-events
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=riot_events_%j.out
#SBATCH --error=riot_events_%j.err

set -euo pipefail

python run_sobol.py \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --base-n 128 \
  --output-dir data
