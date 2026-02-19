#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="deeptactics"
#SBATCH --time=10:00:00
#SBATCH --partition=GPUQ
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --constraint="(p100|v100|a100|h100|h200)&(gpu40g|gpu80g)"
#SBATCH --output=slurm_outputs/output_deeptactics.txt
#SBATCH --error=slurm_outputs/output_deeptactics.err


WORKDIR=../${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"


module purge
module load Anaconda3/2024.02-1

pip install uv --user

uv sync

# Enable W&B logging for Ultralytics

# Run the Python script
uv run python test.py

echo "Job finished"