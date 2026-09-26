#!/bin/bash
#PBS -N OccuPose_Train
#PBS -o train_aqua_live.log
#PBS -e train_aqua.err
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=20:ngpus=2:mem=60gb
#PBS -q gpuq

# =====================================================================
# AQUA Cluster Multi-GPU PBS Submission Script for OccuPose-BroadDictNet
# Workspace: ~/Face_Detection_In_Wild/
# Dataset: ~/scratch/Face_Dataset/
# Weights Output: ~/Face_Detection_In_Wild/weights/
# Host: aqua.iitm.ac.in | User: na22b025
# =====================================================================

cd $PBS_O_WORKDIR || cd $HOME/Face_Detection_In_Wild

# Enable unbuffered Python logging & NCCL environment settings
export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export LD_LIBRARY_PATH=/lfs/usrhome/btech/na22b025/miniforge3/envs/venv/lib:/lfs/usrhome/btech/na22b025/miniforge3/envs/venv_gpu/lib:$LD_LIBRARY_PATH

mkdir -p $HOME/Face_Detection_In_Wild/weights

# Activate Miniforge Conda and venv_gpu Virtual Environment
source $HOME/miniforge3/bin/activate
conda activate venv_gpu

echo "=========================================================="
echo "Starting OccuPose-BroadDictNet Training on AQUA Cluster"
echo "Date: $(date)"
echo "Host Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=1
fi
echo "Auto-detected GPU count on node $(hostname): $NUM_GPUS"

# Launch PyTorch Distributed Data Parallel (DDP) across available GPUs
$HOME/miniforge3/envs/venv_gpu/bin/torchrun --nproc_per_node=$NUM_GPUS train.py \
    --data_dir "$HOME/scratch/Face_Dataset/name_label" \
    --unlabeled_dir "$HOME/scratch/Face_Dataset/unlabeled" \
    --celeba_dir "$HOME/scratch/Face_Dataset/celeba" \
    --checkpoint_dir "$HOME/Face_Detection_In_Wild/weights" \
    --backbone iresnet100 \
    --batch_size 32 \
    --epochs 25 \
    --lr 0.1 \
    --fp16 2>&1 | tee -a "$HOME/Face_Detection_In_Wild/train_aqua_live.log"

echo "Training job completed at $(date)"
