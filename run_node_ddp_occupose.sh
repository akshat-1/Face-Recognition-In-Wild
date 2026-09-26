#!/bin/bash
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=ib0,eth0,ens,enp
export NCCL_SOCKET_IFNAME=ib0,eth0,ens,enp
export NCCL_IB_DISABLE=0
export LD_LIBRARY_PATH=/lfs/usrhome/btech/na22b025/miniforge3/envs/venv_gpu/lib:$LD_LIBRARY_PATH

source /lfs/usrhome/btech/na22b025/miniforge3/bin/activate
conda activate venv_gpu

if ! $HOME/miniforge3/envs/venv_gpu/bin/python -c "import torch; assert torch.cuda.is_available()"; then
    echo "❌ [ERROR] CUDA is not available on $(hostname). Aborting DDP process."
    exit 1
fi

ARG_NODEFILE="$1"
ARG_MASTER="$2"
ARG_PORT="$3"
ARG_NNODES="$4"

if [ -n "$ARG_NODEFILE" ] && [ -f "$ARG_NODEFILE" ]; then
    PBS_NODEFILE="$ARG_NODEFILE"
fi

if [ -z "$PBS_NODEFILE" ] || [ ! -f "$PBS_NODEFILE" ]; then
    PBS_NODEFILE="/tmp/pbs_nodefile_fallback"
fi

if [ -n "$ARG_MASTER" ]; then
    MASTER_ADDR="$ARG_MASTER"
elif [ -f "$PBS_NODEFILE" ]; then
    MASTER_NODE_SHORT=$(head -n 1 $PBS_NODEFILE | cut -d'.' -f1)
    MASTER_ADDR=$(getent hosts $MASTER_NODE_SHORT | awk '{print $1}')
    if [ -z "$MASTER_ADDR" ]; then
        MASTER_ADDR=$MASTER_NODE_SHORT
    fi
else
    MASTER_ADDR=$(hostname | cut -d'.' -f1)
fi

if [ -n "$ARG_PORT" ]; then
    MASTER_PORT="$ARG_PORT"
else
    MASTER_PORT=29512
fi

if [ -n "$ARG_NNODES" ]; then
    NNODES="$ARG_NNODES"
elif [ -f "$PBS_NODEFILE" ]; then
    NNODES=$(sort -u $PBS_NODEFILE | wc -l)
else
    NNODES=1
fi

# Determine node rank dynamically from PBS environment or nodefile
HOSTNAME_SHORT=$(hostname | cut -d'.' -f1)
NODE_RANK=0
if [ -f "$PBS_NODEFILE" ]; then
    UNIQUE_NODES=($(sort -u $PBS_NODEFILE | cut -d'.' -f1))
    for idx in "${!UNIQUE_NODES[@]}"; do
        if [ "${UNIQUE_NODES[$idx]}" == "$HOSTNAME_SHORT" ]; then
            NODE_RANK=$idx
            break
        fi
    done
elif [ -n "$PBS_VNODENUM" ]; then
    NODE_RANK=$PBS_VNODENUM
elif [ -n "$PBS_NODENUM" ]; then
    NODE_RANK=$PBS_NODENUM
fi

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=2
fi

echo "⚡ [OccuPose Unit Rank $NODE_RANK / $NNODES] Launching $NUM_GPUS-GPU torchrun on $HOSTNAME_SHORT (Master IP: $MASTER_ADDR:$MASTER_PORT)..."

/lfs/usrhome/btech/na22b025/miniforge3/envs/venv_gpu/bin/torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NUM_GPUS \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  /lfs/usrhome/btech/na22b025/Face_Detection_In_Wild/train.py \
  --data_dir "/lfs/usrhome/btech/na22b025/scratch/Face_Dataset/name_label" \
  --unlabeled_dir "/lfs/usrhome/btech/na22b025/scratch/Face_Dataset/unlabeled" \
  --celeba_dir "/lfs/usrhome/btech/na22b025/scratch/Face_Dataset/celeba" \
  --checkpoint_dir "/lfs/usrhome/btech/na22b025/Face_Detection_In_Wild/weights" \
  --backbone iresnet100 \
  --batch_size 32 \
  --epochs 100 \
  --resume \
  --retrain_phase2 \
  --lr 0.1 \
  --fp16 2>&1 | tee -a "/lfs/usrhome/btech/na22b025/Face_Detection_In_Wild/train_aqua_live.log"
