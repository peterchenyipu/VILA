#!/bin/bash
#SBATCH -Jft_15b_vila_test
#SBATCH -N1 --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:8
#SBATCH --mem=200G
#SBATCH --time=1:59:59
#SBATCH --output=logs/%x_%j.out

# module load cuda/12.1

source activate vila
cd ~/scratch/VILA

# check smi output
nvidia-smi

# reduce the global bs or increase the accum steps if facing OOM issues
DEFAULT_RUN_NAME="NVILA-Lite-15B-finetune-1ep" \
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=64 \
DEFAULT_GRADIENT_ACCUMULATION_STEPS=2 \
   bash scripts/NVILA-Lite/sft.sh \
       Efficient-Large-Model/NVILA-Lite-15B \
       EmoStim \
       runs/train/nvila-lite-15b-sft1ep
