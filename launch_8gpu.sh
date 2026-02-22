#!/bin/bash
# Launch 7B training on single node (8x B300)

set -e

# Source NCCL settings
source scripts/nccl_env.sh

# Run training
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --master_addr=localhost \
    --master_port=29500 \
    scripts/train_7b.py \
    --data data/fineweb \
    --steps 100000

echo "Training complete!"
