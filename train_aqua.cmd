#!/bin/bash
#PBS -N OccuPose_Train
#PBS -q gpuq
#PBS -l select=2:ncpus=10:ngpus=2
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o train_aqua_live.log

# =====================================================================
# AQUA Cluster Single-Node 4-GPU PBS Submission Script for OccuPose-BroadDictNet
# Workspace: ~/Face_Detection_In_Wild/
# Dataset: ~/scratch/Face_Dataset/
# Weights Output: ~/Face_Detection_In_Wild/weights/
# Host: aqua.iitm.ac.in | User: na22b025
# =====================================================================

cd $PBS_O_WORKDIR

echo "=========================================================="
echo "Starting OccuPose-BroadDictNet Training on AQUA Cluster"
echo "Date: $(date)"
echo "Host Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

# Create weights output directory in working space
mkdir -p ~/Face_Detection_In_Wild/weights

# Export system shared library paths for libtiff and CUDA
export LD_LIBRARY_PATH=/lfs/usrhome/btech/na22b025/miniforge3/envs/venv/lib:$LD_LIBRARY_PATH

# Activate PyTorch Conda Environment on AQUA Cluster
source /lfs/usrhome/btech/na22b025/miniforge3/bin/activate venv_gpu

# Auto-detect available GPU count
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=1
fi
echo "Auto-detected GPU count: $NUM_GPUS"

# Launch PyTorch Distributed Data Parallel (DDP) across available GPUs
torchrun --nproc_per_node=$NUM_GPUS train.py \
    --data_dir ~/scratch/Face_Dataset/name_label \
    --unlabeled_dir ~/scratch/Face_Dataset/unlabeled \
    --celeba_dir ~/scratch/Face_Dataset/celeba \
    --checkpoint_dir ~/Face_Detection_In_Wild/weights \
    --backbone iresnet100 \
    --batch_size 32 \
    --epochs 25 \
    --lr 0.1 \
    --fp16

echo "Training job completed at $(date)"
