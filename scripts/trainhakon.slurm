#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="deeptactics"
#SBATCH --time=05-00:00:00
#SBATCH --partition=GPUQ
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --constraint="(a100|h100|h200)"
#SBATCH --output=slurm_outputs/output_deeptactics_connectfour.txt
#SBATCH --error=slurm_outputs/output_deeptactics_connectfour.err


WORKDIR=../${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"


module purge
module load Anaconda3/2024.02-1

pip install uv --user

uv run python main.py connect_four_hakon.yaml

echo "Job finished"
