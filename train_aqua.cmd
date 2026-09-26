#!/bin/bash
#PBS -N OccuPose_Train
#PBS -o train_aqua_live.log
#PBS -e train_aqua.err
#PBS -l walltime=48:00:00
#PBS -l select=2:ncpus=20:ngpus=2:mem=60gb
#PBS -q gpuq

# =====================================================================
# AQUA Cluster Multi-Node 4-GPU PBS Submission Script for OccuPose-BroadDictNet
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
export LD_LIBRARY_PATH=/lfs/usrhome/btech/na22b025/miniforge3/envs/venv_gpu/lib:$LD_LIBRARY_PATH

mkdir -p $HOME/Face_Detection_In_Wild/weights

# Activate Miniforge Conda and venv_gpu Virtual Environment
source $HOME/miniforge3/bin/activate
conda activate venv_gpu

# 1. Determine Master Node and Node List from PBS_NODEFILE
NODES=($(cat $PBS_NODEFILE | sort -u))
NNODES=${#NODES[@]}
MASTER_ADDR=$(echo "${NODES[0]}" | cut -d'.' -f1)
MASTER_PORT=29512
NODE0_HOST=$(echo "${NODES[0]}" | cut -d'.' -f1)
NODE1_HOST=$(echo "${NODES[1]}" | cut -d'.' -f1)

echo "========================================================================="
echo "🚀 MULTI-NODE 4-GPU DDP OCCUPOSE TRAINING INITIALIZATION"
echo "   Master Node:  $MASTER_ADDR:$MASTER_PORT"
echo "   Total Nodes:  $NNODES (${NODES[*]})"
echo "   Node 0 Host:  $NODE0_HOST"
echo "   Node 1 Host:  $NODE1_HOST"
echo "   GPUs/Node:    2"
echo "   Total GPUs:   4"
echo "=========================================================================\n"

# 2. Create node launcher script for pbsdsh multi-node execution
LAUNCHER="$HOME/Face_Detection_In_Wild/run_node_ddp_occupose.sh"
cat << EOF > "$LAUNCHER"
#!/bin/bash
export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export LD_LIBRARY_PATH=/lfs/usrhome/btech/na22b025/miniforge3/envs/venv_gpu/lib:\$LD_LIBRARY_PATH

source $HOME/miniforge3/bin/activate
conda activate venv_gpu

if ! \$HOME/miniforge3/envs/venv_gpu/bin/python -c "import torch; assert torch.cuda.is_available()"; then
    echo "❌ [ERROR] CUDA is not available on \$(hostname). Aborting DDP process."
    exit 1
fi

HOSTNAME_SHORT=\$(hostname | cut -d'.' -f1)

NODE_RANK=0
if [ "\$HOSTNAME_SHORT" == "$NODE1_HOST" ]; then
    NODE_RANK=1
fi

if [ -n "\$PBS_VNODENUM" ]; then
    NODE_RANK=\$PBS_VNODENUM
elif [ -n "\$PBS_NODENUM" ]; then
    NODE_RANK=\$PBS_NODENUM
fi

echo "⚡ [OccuPose Node Rank \$NODE_RANK / 2] Launching 4-GPU torchrun on \$(hostname) (Master: $MASTER_ADDR:$MASTER_PORT)..."

if [ "\$NODE_RANK" -eq 0 ]; then
    $HOME/miniforge3/envs/venv_gpu/bin/torchrun \\
      --nnodes=2 \\
      --nproc_per_node=2 \\
      --node_rank=\$NODE_RANK \\
      --master_addr=$MASTER_ADDR \\
      --master_port=$MASTER_PORT \\
      $HOME/Face_Detection_In_Wild/train.py \\
      --data_dir "$HOME/scratch/Face_Dataset/name_label" \\
      --unlabeled_dir "$HOME/scratch/Face_Dataset/unlabeled" \\
      --celeba_dir "$HOME/scratch/Face_Dataset/celeba" \\
      --checkpoint_dir "$HOME/Face_Detection_In_Wild/weights" \\
      --backbone iresnet100 \\
      --batch_size 32 \\
      --epochs 25 \\
      --lr 0.1 \\
      --fp16 2>&1 | tee -a "$HOME/Face_Detection_In_Wild/train_aqua_live.log"
else
    $HOME/miniforge3/envs/venv_gpu/bin/torchrun \\
      --nnodes=2 \\
      --nproc_per_node=2 \\
      --node_rank=\$NODE_RANK \\
      --master_addr=$MASTER_ADDR \\
      --master_port=$MASTER_PORT \\
      $HOME/Face_Detection_In_Wild/train.py \\
      --data_dir "$HOME/scratch/Face_Dataset/name_label" \\
      --unlabeled_dir "$HOME/scratch/Face_Dataset/unlabeled" \\
      --celeba_dir "$HOME/scratch/Face_Dataset/celeba" \\
      --checkpoint_dir "$HOME/Face_Detection_In_Wild/weights" \\
      --backbone iresnet100 \\
      --batch_size 32 \\
      --epochs 25 \\
      --lr 0.1 \\
      --fp16
fi
EOF

chmod +x "$LAUNCHER"

# 3. Execute torchrun across all nodes in parallel using PBS pbsdsh
/opt/pbs/bin/pbsdsh -v -- "$LAUNCHER"
