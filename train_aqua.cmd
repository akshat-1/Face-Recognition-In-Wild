#!/bin/bash
#PBS -N OccuPose_Train
#PBS -q gpuq
#PBS -l select=2:ncpus=10:ngpus=2
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o train_aqua_live.log

# =====================================================================
# AQUA Cluster Multi-GPU PBS Submission Script for OccuPose-BroadDictNet
# Host: aqua.iitm.ac.in | User: na22b025
# =====================================================================

cd $PBS_O_WORKDIR

echo "=========================================================="
echo "Starting OccuPose-BroadDictNet Training on AQUA Cluster"
echo "Date: $(date)"
echo "Host Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

# Activate Python Virtual Environment on AQUA Cluster
source ~/.venv/bin/activate || source activate torch_env || true

# Launch PyTorch Distributed Data Parallel (DDP) across 4 GPUs
torchrun --nproc_per_node=4 train.py \
    --data_dir ~/Face_Recognition_In_Wild_Data/name_label \
    --unlabeled_dir ~/Face_Recognition_In_Wild_Data/unlabeled \
    --celeba_dir ~/Face_Recognition_In_Wild_Data/celeba \
    --backbone iresnet100 \
    --batch_size 64 \
    --epochs 25 \
    --lr 0.1 \
    --fp16

echo "Training job completed at $(date)"
