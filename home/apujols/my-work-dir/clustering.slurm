#!/bin/bash
#SBATCH --job-name=cluster256
#SBATCH --output=logs/cluster_v1.1_256.out
#SBATCH --error=logs/cluster_v1.1_256.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu        # or 'cpu' depending on your cluster
#SBATCH --gres=gpu:1           # request 1 GPU, remove if CPU only

module load apptainer/1.3.5

# Activate your environment
export APPTAINER_CACHEDIR=$SCRATCH/apptainer-cache

apptainer exec --nv \
    --bind /data:/data \
    --bind $PWD:/my-work-dir \
    vgg-faiss-gpu.sif \
    python -u /my-work-dir/self_supervised_clustering.py \
    --backbone_dim 256 --best_clustering 0