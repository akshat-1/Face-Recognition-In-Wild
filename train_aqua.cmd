#!/bin/bash
#PBS -N OccuPose_Train
#PBS -o train_aqua_live.log
#PBS -e train_aqua.err
#PBS -l walltime=48:00:00
#PBS -l select=2:ncpus=20:ngpus=2:mem=60gb
#PBS -q gpuq

# =====================================================================
# AQUA Cluster 4-GPU Multi-Unit PBS Script for OccuPose-BroadDictNet
# 2 Units x 2 GPUs = 4 GPUs Total Across Nodes
# Workspace: ~/Face_Detection_In_Wild/
# Dataset: ~/scratch/Face_Dataset/
# Weights Output: ~/Face_Detection_In_Wild/weights/
# Host: aqua.iitm.ac.in | User: na22b025
# =====================================================================

cd $PBS_O_WORKDIR || cd $HOME/Face_Detection_In_Wild

# Enable unbuffered Python logging & NCCL environment settings
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0,eth0,ens,enp
export LD_LIBRARY_PATH=/lfs/usrhome/btech/na22b025/miniforge3/envs/venv/lib:/lfs/usrhome/btech/na22b025/miniforge3/envs/venv_gpu/lib:$LD_LIBRARY_PATH

mkdir -p $HOME/Face_Detection_In_Wild/weights

# Activate Miniforge Conda and venv_gpu Virtual Environment
source $HOME/miniforge3/bin/activate
conda activate venv_gpu

MASTER_ADDR=$(head -n 1 $PBS_NODEFILE | cut -d'.' -f1)
MASTER_PORT=29512
NNODES=$(sort -u $PBS_NODEFILE | wc -l)

echo "=========================================================="
echo "Starting OccuPose-BroadDictNet 4-GPU Training on AQUA Cluster"
echo "Date: $(date)"
echo "Master Node: $MASTER_ADDR (Port: $MASTER_PORT)"
echo "Total Allocated Units (Nodes): $NNODES"
echo "Allocated Nodes List:"
cat $PBS_NODEFILE | sort -u
echo "=========================================================="

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NNODES=$NNODES

# Launch multi-node 4-GPU PyTorch DDP via pbsdsh across all allocated nodes
pbsdsh -v $HOME/Face_Detection_In_Wild/run_node_ddp_occupose.sh 2>&1 | tee -a "$HOME/Face_Detection_In_Wild/train_aqua_live.log"

echo "Training job completed at $(date)"
