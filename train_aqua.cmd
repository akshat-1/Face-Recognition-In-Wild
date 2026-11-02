#!/bin/bash
#PBS -N OccuPose_Train
#PBS -o train_aqua_live.log
#PBS -e train_aqua.err
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=10:ngpus=2:mem=30gb
#PBS -q gpuq

# =====================================================================
# AQUA Cluster 4-GPU Multi-Unit PBS Script for OccuPose-BroadDictNet
# 2 Units x 2 GPUs = 4 GPUs Total Across Nodes (gpu005 + gpu015)
# Workspace: ~/Face_Detection_In_Wild/
# Dataset: ~/scratch/Face_Dataset/
# Weights Output: ~/Face_Detection_In_Wild/weights/
# Host: aqua.iitm.ac.in | User: na22b025
# =====================================================================

cd $PBS_O_WORKDIR || cd $HOME/Face_Detection_In_Wild
export PYTHONPATH="$HOME/Face_Detection_In_Wild/.python_deps:${PYTHONPATH:-}"

# Enable unbuffered Python logging & IPv4-only NCCL/GLOO environment settings
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8
export AQUA_TRAIN_DETECTOR_BACKEND=anchor_free
export PYTHONWARNINGS="ignore"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=ib0,eth0,ens,enp
export NCCL_SOCKET_IFNAME=ib0,eth0,ens,enp
export NCCL_IB_DISABLE=0
export LD_LIBRARY_PATH=/lfs/usrhome/btech/na22b025/miniforge3/envs/venv_gpu/lib:/lfs/usrhome/btech/na22b025/miniforge3/envs/venv/lib:${LD_LIBRARY_PATH:-}
if ! command -v stdbuf >/dev/null 2>&1; then
    echo "ERROR: stdbuf is required for live logs" >&2
    exit 1
fi

mkdir -p $HOME/Face_Detection_In_Wild/weights

# Activate Miniforge Conda and venv_gpu Virtual Environment
source $HOME/miniforge3/bin/activate
conda activate venv_gpu

MASTER_NODE_SHORT=$(head -n 1 $PBS_NODEFILE | cut -d'.' -f1)
MASTER_ADDR=$(getent ahostsv4 $MASTER_NODE_SHORT | head -n 1 | awk '{print $1}')
if [ -z "$MASTER_ADDR" ]; then
    MASTER_ADDR=$MASTER_NODE_SHORT
fi
MASTER_PORT=29512
NNODES=$(sort -u $PBS_NODEFILE | wc -l)

echo "=========================================================="
echo "Starting OccuPose-BroadDictNet 4-GPU Training on AQUA Cluster"
echo "Date: $(date)"
echo "Master IPv4: $MASTER_ADDR (Port: $MASTER_PORT)"
echo "Total Allocated Units (Nodes): $NNODES"
echo "Allocated Nodes List:"
cat $PBS_NODEFILE | sort -u
echo "=========================================================="

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NNODES=$NNODES

# Launch one DDP torchrun process per allocated host. The PBS nodefile can
# contain repeated entries for multiple GPUs, so -c prevents duplicate launches.
stdbuf -oL -eL pbsdsh -c -v $HOME/Face_Detection_In_Wild/run_node_ddp_occupose.sh "$PBS_NODEFILE" "$MASTER_ADDR" "$MASTER_PORT" "$NNODES" 2>&1 | stdbuf -oL -eL tee -a "$HOME/Face_Detection_In_Wild/train_aqua_live.log"

echo "Training job completed at $(date)"
