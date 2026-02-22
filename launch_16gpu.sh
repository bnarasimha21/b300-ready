#!/bin/bash
# Launch 70B training on 2 nodes (16x B300)
# Run this script on BOTH nodes with appropriate NODE_RANK

set -e

# Configure these
MASTER_ADDR=${MASTER_ADDR:-"10.0.0.1"}  # IP of node 0
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}  # 0 for master, 1 for worker
NNODES=${NNODES:-2}

# Source NCCL settings
source scripts/nccl_env.sh

echo "Starting node $NODE_RANK of $NNODES"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Run training
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=8 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train_multinode.py \
    --data data/fineweb \
    --steps 10000 \
    --fp8

echo "Node $NODE_RANK complete!"
