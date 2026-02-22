#!/bin/bash
export NCCL_P2P_LEVEL=NVL
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_GID_INDEX=3
export NCCL_BUFFSIZE=8388608
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
echo "NCCL environment configured"
