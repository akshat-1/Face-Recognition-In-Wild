#!/bin/bash
#PBS -N Drive_Download
#PBS -q small20
#PBS -l select=1:ncpus=4
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o download_drive_live.log

# =====================================================================
# AQUA Cluster Google Drive Download Script for Face_Dataset -> /scratch
# Target: /scratch/na22b025/Face_Dataset
# NOTE: All execution must be submitted via qsub download_drive_aqua.cmd
# =====================================================================

cd $PBS_O_WORKDIR

echo "=========================================================="
echo "Starting Google Drive Download PBS Job on AQUA Compute Node"
echo "Target Scratch Directory: /scratch/na22b025/Face_Dataset"
echo "Date: $(date)"
echo "=========================================================="

# 1. Create target scratch directory
mkdir -p /scratch/na22b025/Face_Dataset

# 2. Activate Python environment or install gdown into user space
source ~/.venv/bin/activate || true
pip install gdown --quiet || true

# 3. Execute download autonomously on the compute node
python3 -m gdown --folder "https://drive.google.com/drive/folders/1bzwadTmTkp69kNkbdPNb-2tm7DKAvd7Q?usp=sharing" -O /scratch/na22b025/Face_Dataset --remaining-ok

echo "=========================================================="
echo "Download Job Completed at $(date)"
echo "=========================================================="
