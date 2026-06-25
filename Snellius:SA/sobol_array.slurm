#!/bin/bash
#SBATCH --job-name=riot_sobol
#SBATCH --output=logs/sobol_%A_%a.out
#SBATCH --error=logs/sobol_%A_%a.err
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=rome
#SBATCH --array=0-39
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=4G

# N=1024, second-order, D=4 -> 1024*(2*4+2) = 10240 rows.
# CHUNK=256 * 40 array tasks (0-39) = 10240. Keep array size * CHUNK == row count.
set -euo pipefail
CHUNK=256
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))

# single-threaded per task; parallelism comes from the array, not BLAS threads
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0   # adjust to an available Snellius module
source "$HOME/riot_env/bin/activate"        # venv with mesa, numpy, SALib, scikit-learn

mkdir -p logs runs
python run_chunk.py "$START" "$CHUNK"
