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
# =====================================================================

cd $PBS_O_WORKDIR

echo "=========================================================="
echo "Starting Google Drive Download Job on AQUA Cluster"
echo "Target Scratch Directory: /scratch/na22b025/Face_Dataset"
echo "Date: $(date)"
echo "=========================================================="

# Create target scratch directory
mkdir -p /scratch/na22b025/Face_Dataset

# Activate Python environment
source ~/.venv/bin/activate || true

# Run high-speed multi-threaded download
python3 download_drive_to_aqua_scratch.py --dest /scratch/na22b025/Face_Dataset

echo "Download completed at $(date)"
