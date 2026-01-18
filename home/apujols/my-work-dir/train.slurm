#!/bin/bash
#SBATCH --job-name=train5
#SBATCH --output=logs/training/train_v1.5.out
#SBATCH --error=logs/training/train_v1.5.err
#SBATCH --time=1-12:00:00
#SBATCH --partition=gpu        # or 'cpu' depending on your cluster
#SBATCH --gres=gpu:1           # request 1 GPU, remove if CPU only

module load apptainer/1.3.5

# Activate your environment
export APPTAINER_CACHEDIR=$SCRATCH/apptainer-cache

apptainer exec --nv \
    --bind /data:/data \
    --bind $PWD:/my-work-dir \
    vgg-faiss-gpu.sif \
    python -u /my-work-dir/self_supervised_training.py \
    --scale_dim 64 --backbone_dim 256 --projection_dim 512 --version "1.5"