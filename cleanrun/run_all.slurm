#!/bin/bash
#SBATCH --job-name=riot_sobol_n64
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=logs/riot_sobol_%j.out
#SBATCH --error=logs/riot_sobol_%j.err

set -euo pipefail

module purge
module load 2025
module load Mesa

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p data plots logs

python run_sobol.py \
  --workers "$SLURM_CPUS_PER_TASK" \
  --output-dir data

python analyze_sobol.py \
  --data-dir data \
  --plots-dir plots

echo "Done. Data: $(pwd)/data"
echo "Plots: $(pwd)/plots"
