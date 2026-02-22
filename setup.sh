#!/bin/bash
set -e

echo "=== B300 Environment Setup ==="

# Check NVIDIA driver
nvidia-smi || { echo "ERROR: nvidia-smi failed"; exit 1; }

# Create conda environment
conda create -n b300 python=3.10 -y
source activate b300

# Install PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install requirements
pip install -r requirements.txt

# Verify NVLink
echo "=== NVLink Status ==="
nvidia-smi nvlink -s

# Verify all GPUs
echo "=== GPU List ==="
nvidia-smi -L

echo "=== Setup Complete ==="
