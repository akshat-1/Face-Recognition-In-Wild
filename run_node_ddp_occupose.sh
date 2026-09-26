#!/bin/bash
export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export LD_LIBRARY_PATH=/lfs/usrhome/btech/na22b025/miniforge3/envs/venv_gpu/lib:$LD_LIBRARY_PATH

source /lfs/usrhome/btech/na22b025/miniforge3/bin/activate
conda activate venv_gpu

if ! $HOME/miniforge3/envs/venv_gpu/bin/python -c "import torch; assert torch.cuda.is_available()"; then
    echo "❌ [ERROR] CUDA is not available on $(hostname). Aborting DDP process."
    exit 1
fi

HOSTNAME_SHORT=$(hostname | cut -d'.' -f1)

NODE_RANK=0
if [ "$HOSTNAME_SHORT" == "gpu015" ]; then
    NODE_RANK=1
fi

if [ -n "$PBS_VNODENUM" ]; then
    NODE_RANK=$PBS_VNODENUM
elif [ -n "$PBS_NODENUM" ]; then
    NODE_RANK=$PBS_NODENUM
fi

echo "⚡ [OccuPose Node Rank $NODE_RANK / 2] Launching 4-GPU torchrun on $(hostname) (Master: gpu011:29512)..."

if [ "$NODE_RANK" -eq 0 ]; then
    /lfs/usrhome/btech/na22b025/miniforge3/envs/venv_gpu/bin/torchrun \
      --nnodes=2 \
      --nproc_per_node=2 \
      --node_rank=$NODE_RANK \
      --master_addr=gpu011 \
      --master_port=29512 \
      /lfs/usrhome/btech/na22b025/Face_Detection_In_Wild/train.py \
      --data_dir "/lfs/usrhome/btech/na22b025/scratch/Face_Dataset/name_label" \
      --unlabeled_dir "/lfs/usrhome/btech/na22b025/scratch/Face_Dataset/unlabeled" \
      --celeba_dir "/lfs/usrhome/btech/na22b025/scratch/Face_Dataset/celeba" \
      --checkpoint_dir "/lfs/usrhome/btech/na22b025/Face_Detection_In_Wild/weights" \
      --backbone iresnet100 \
      --batch_size 32 \
      --epochs 25 \
      --retrain_phase2 \
      --lr 0.1 \
      --fp16 2>&1 | tee -a "/lfs/usrhome/btech/na22b025/Face_Detection_In_Wild/train_aqua_live.log"
else
    /lfs/usrhome/btech/na22b025/miniforge3/envs/venv_gpu/bin/torchrun \
      --nnodes=2 \
      --nproc_per_node=2 \
      --node_rank=$NODE_RANK \
      --master_addr=gpu011 \
      --master_port=29512 \
      /lfs/usrhome/btech/na22b025/Face_Detection_In_Wild/train.py \
      --data_dir "/lfs/usrhome/btech/na22b025/scratch/Face_Dataset/name_label" \
      --unlabeled_dir "/lfs/usrhome/btech/na22b025/scratch/Face_Dataset/unlabeled" \
      --celeba_dir "/lfs/usrhome/btech/na22b025/scratch/Face_Dataset/celeba" \
      --checkpoint_dir "/lfs/usrhome/btech/na22b025/Face_Detection_In_Wild/weights" \
      --backbone iresnet100 \
      --batch_size 32 \
      --epochs 25 \
      --retrain_phase2 \
      --lr 0.1 \
      --fp16
fi
