#!/bin/bash
#SBATCH -Jinfer_test
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=32G
#SBATCH --time=4:59:59
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ychen3302@gatech.edu

# Optional: Load modules or activate your conda environment
# module load cuda/12.1
export HF_HOME=$TMPDIR
source activate vila
cd ~/scratch/VILA

# check smi output
nvidia-smi

python infer_on_test_set.py
