#!/usr/bin/env python3
"""
Train 70B LLM across 2 nodes (16x B300) with FSDP - READY TO RUN

Usage:
    # Node 1 (master):
    MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
        --master_addr=10.0.0.1 --master_port=29500 \
        scripts/train_multinode.py
    
    # Node 2:
    MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
        --master_addr=10.0.0.1 --master_port=29500 \
        scripts/train_multinode.py
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np

# Transformer Engine for FP8
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    HAS_TE = True
except ImportError:
    HAS_TE = False

# HuggingFace
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


class TokenDataset(Dataset):
    """Simple token dataset from numpy files"""
    
    def __init__(self, data_path: str, seq_length: int = 4096):
        self.seq_length = seq_length
        data_path = Path(data_path)
        
        files = sorted(data_path.glob("*.npy"))
        if not files:
            # Create dummy data for testing
            print("No data found, creating dummy dataset...")
            data_path.mkdir(parents=True, exist_ok=True)
            dummy = np.random.randint(0, 32000, size=(10_000_000,), dtype=np.uint16)
            np.save(data_path / "dummy.npy", dummy)
            files = [data_path / "dummy.npy"]
        
        self.data = np.concatenate([np.load(f) for f in files])
        self.num_samples = len(self.data) // seq_length
        print(f"Dataset: {len(self.data)/1e6:.1f}M tokens, {self.num_samples} samples")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * self.seq_length
        tokens = self.data[start:start + self.seq_length + 1]
        x = torch.from_numpy(tokens[:-1].astype(np.int64))
        y = torch.from_numpy(tokens[1:].astype(np.int64))
        return x, y


def setup_distributed():
    """Initialize multi-node distributed training"""
    
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
    
    torch.cuda.set_device(local_rank)
    
    if rank == 0:
        print(f"Distributed: {world_size} GPUs across {world_size // 8} nodes")
    
    return rank, world_size, local_rank


def create_70b_model():
    """Create 70B parameter Llama model"""
    
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        max_position_embeddings=4096,
        torch_dtype=torch.bfloat16,
    )
    
    # Initialize on meta device (no memory)
    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    
    return model


def train(args):
    """Main training loop"""
    
    rank, world_size, local_rank = setup_distributed()
    device = f'cuda:{local_rank}'
    
    # Create model
    model = create_70b_model()
    
    # FSDP wrapping policy
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )
    
    # Mixed precision
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    # Wrap with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=bf16_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank,
        limit_all_gathers=True,
    )
    
    if rank == 0:
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"Model loaded. Memory per GPU: {mem_gb:.1f} GB")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # Dataset
    dataset = TokenDataset(args.data, seq_length=args.seq_length)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # FP8 recipe
    if HAS_TE and args.fp8:
        fp8_recipe = DelayedScaling(
            fp8_format=Format.HYBRID,
            amax_history_len=1024,
            amax_compute_algo="max"
        )
    
    # Training
    model.train()
    step = 0
    total_tokens = 0
    start_time = time.time()
    
    if rank == 0:
        print(f"\nStarting training for {args.steps} steps...")
        print("-" * 60)
    
    for epoch in range(1000):  # Large number, will break on steps
        sampler.set_epoch(epoch)
        
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            
            # Forward
            if HAS_TE and args.fp8:
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    outputs = model(input_ids=x, labels=y)
                    loss = outputs.loss
            else:
                outputs = model(input_ids=x, labels=y)
                loss = outputs.loss
            
            # Backward
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            step += 1
            total_tokens += args.batch_size * args.seq_length * world_size
            
            # Logging
            if step % 10 == 0 and rank == 0:
                elapsed = time.time() - start_time
                tps = total_tokens / elapsed
                mem = torch.cuda.memory_allocated() / 1e9
                
                print(f"Step {step:5d} | Loss: {loss.item():.4f} | "
                      f"Tokens/s: {tps:,.0f} | Mem: {mem:.1f} GB")
            
            # Checkpointing
            if step % args.save_interval == 0 and rank == 0:
                ckpt_dir = Path(args.checkpoint_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                
                # FSDP state dict
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                    state = model.state_dict()
                    torch.save(state, ckpt_dir / f"step_{step}.pt")
                print(f"Saved checkpoint at step {step}")
            
            if step >= args.steps:
                break
        
        if step >= args.steps:
            break
    
    # Final metrics
    if rank == 0:
        elapsed = time.time() - start_time
        print("-" * 60)
        print(f"Training complete!")
        print(f"Total steps: {step}")
        print(f"Total tokens: {total_tokens / 1e9:.2f}B")
        print(f"Average tokens/sec: {total_tokens / elapsed:,.0f}")
        print(f"Time: {elapsed / 3600:.2f} hours")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/fineweb")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--fp8", action="store_true", default=True)
    parser.add_argument("--no-fp8", action="store_true")
    args = parser.parse_args()
    
    if args.no_fp8:
        args.fp8 = False
    
    train(args)
