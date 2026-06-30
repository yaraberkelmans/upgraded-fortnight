#!/bin/bash
#SBATCH --job-name=sobol_riot
#SBATCH --output=slurm_sobol_%j.out
#SBATCH --error=slurm_sobol_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128

# Single job that runs all five seeds one after another in a bash loop.

# --- Clean base environment on the compute node ---
module purge

# Load the exact Python version the venv was built with
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

# Activate the virtual environment.
source mesa_env/bin/activate

# --- Run each seed sequentially ---
SEEDS=(43 44 45 46 47)

for SEED in "${SEEDS[@]}"; do
    # Give each seed its own output directory so the per-run .npy files
    # (which are named by sample_id, not by seed) don't overwrite each other.
    OUTPUT_DIR="/home/mvbeusekom/data/seed_${SEED}"

    echo "Starting seed=${SEED}, workers=${SLURM_CPUS_PER_TASK}, output=${OUTPUT_DIR}"

    python run_sobol.py \
        --seed "${SEED}" \
        --workers "${SLURM_CPUS_PER_TASK}" \
        --output-dir "${OUTPUT_DIR}"

    echo "Finished seed=${SEED}."
done

echo "All seeds finished.