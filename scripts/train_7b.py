#!/usr/bin/env python3
"""
Train 7B LLM on 8x B300 with FP8 - READY TO RUN

Usage:
    # Single node (8 GPUs)
    torchrun --nproc_per_node=8 scripts/train_7b.py
    
    # With custom config
    torchrun --nproc_per_node=8 scripts/train_7b.py --config configs/7b_config.yaml
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np

# Optional: transformer_engine for FP8
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    HAS_TE = True
except ImportError:
    HAS_TE = False
    print("WARNING: transformer_engine not found, FP8 disabled")

# Optional: wandb for logging
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    

@dataclass 
class TrainConfig:
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    max_steps: int = 100000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seq_length: int = 4096
    fp8: bool = True
    save_interval: int = 1000
    log_interval: int = 10
    data_path: str = "data/fineweb"
    checkpoint_dir: str = "checkpoints"


class TokenDataset(Dataset):
    """Memory-mapped token dataset"""
    
    def __init__(self, data_path: str, seq_length: int):
        self.seq_length = seq_length
        self.data_path = Path(data_path)
        
        # Load all .npy files
        self.files = sorted(self.data_path.glob("*.npy"))
        if not self.files:
            raise ValueError(f"No .npy files found in {data_path}")
        
        # Memory map files
        self.data = []
        self.offsets = [0]
        total_tokens = 0
        
        for f in self.files:
            arr = np.load(f, mmap_mode='r')
            self.data.append(arr)
            total_tokens += len(arr)
            self.offsets.append(total_tokens)
        
        self.total_tokens = total_tokens
        self.num_samples = total_tokens // seq_length
        print(f"Loaded {len(self.files)} files, {total_tokens/1e9:.2f}B tokens, {self.num_samples} samples")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * self.seq_length
        
        # Find which file
        file_idx = 0
        for i, offset in enumerate(self.offsets[1:], 1):
            if start < offset:
                file_idx = i - 1
                break
        
        local_start = start - self.offsets[file_idx]
        tokens = self.data[file_idx][local_start:local_start + self.seq_length + 1]
        
        # Handle edge case
        if len(tokens) < self.seq_length + 1:
            tokens = np.pad(tokens, (0, self.seq_length + 1 - len(tokens)))
        
        x = torch.from_numpy(tokens[:-1].astype(np.int64))
        y = torch.from_numpy(tokens[1:].astype(np.int64))
        return x, y


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
    
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def create_model(config: ModelConfig, device):
    """Create Llama-style model"""
    from transformers import LlamaConfig, LlamaForCausalLM
    
    hf_config = LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=config.rope_theta,
        torch_dtype=torch.bfloat16,
    )
    
    model = LlamaForCausalLM(hf_config)
    model = model.to(device=device, dtype=torch.bfloat16)
    return model


def get_lr(step: int, config: TrainConfig) -> float:
    """Cosine learning rate with warmup"""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def train(model_config: ModelConfig, train_config: TrainConfig):
    """Main training loop"""
    
    rank, world_size, local_rank = setup_distributed()
    device = f'cuda:{local_rank}'
    
    if rank == 0:
        print(f"Training 7B model on {world_size} GPUs")
        print(f"FP8 enabled: {train_config.fp8 and HAS_TE}")
    
    # Create model
    model = create_model(model_config, device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=train_config.weight_decay
    )
    
    # Dataset
    dataset = TokenDataset(train_config.data_path, train_config.seq_length)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.micro_batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # FP8 recipe
    if train_config.fp8 and HAS_TE:
        fp8_recipe = DelayedScaling(
            fp8_format=Format.HYBRID,
            amax_history_len=1024,
            amax_compute_algo="max"
        )
    
    # Checkpointing
    checkpoint_dir = Path(train_config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Wandb
    if rank == 0 and HAS_WANDB:
        wandb.init(project="b300-7b-pretrain", config={
            "model": model_config.__dict__,
            "training": train_config.__dict__
        })
    
    # Training loop
    model.train()
    step = 0
    total_tokens = 0
    start_time = time.time()
    
    while step < train_config.max_steps:
        sampler.set_epoch(step // len(dataloader))
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            
            # Update learning rate
            lr = get_lr(step, train_config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass
            if train_config.fp8 and HAS_TE:
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    outputs = model(input_ids=x, labels=y)
                    loss = outputs.loss
            else:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(input_ids=x, labels=y)
                    loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / train_config.gradient_accumulation_steps
            loss.backward()
            
            # Gradient step
            if (batch_idx + 1) % train_config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                
                # Tokens processed
                batch_tokens = train_config.micro_batch_size * train_config.seq_length * world_size
                total_tokens += batch_tokens * train_config.gradient_accumulation_steps
                
                # Logging
                if step % train_config.log_interval == 0 and rank == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = total_tokens / elapsed
                    
                    mem_allocated = torch.cuda.memory_allocated() / 1e9
                    mem_reserved = torch.cuda.memory_reserved() / 1e9
                    
                    print(f"Step {step} | Loss: {loss.item() * train_config.gradient_accumulation_steps:.4f} | "
                          f"LR: {lr:.2e} | Tokens/s: {tokens_per_sec:.0f} | "
                          f"Mem: {mem_allocated:.1f}/{mem_reserved:.1f} GB")
                    
                    if HAS_WANDB:
                        wandb.log({
                            "loss": loss.item() * train_config.gradient_accumulation_steps,
                            "learning_rate": lr,
                            "tokens_per_second": tokens_per_sec,
                            "gpu_memory_gb": mem_allocated,
                            "step": step
                        })
                
                # Checkpointing
                if step % train_config.save_interval == 0 and rank == 0:
                    ckpt_path = checkpoint_dir / f"step_{step}.pt"
                    torch.save({
                        "step": step,
                        "model": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }, ckpt_path)
                    print(f"Saved checkpoint: {ckpt_path}")
                
                if step >= train_config.max_steps:
                    break
    
    # Final save
    if rank == 0:
        final_path = checkpoint_dir / "final.pt"
        torch.save({
            "model": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        }, final_path)
        print(f"Training complete. Final model: {final_path}")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--data", type=str, default="data/fineweb")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--no-fp8", action="store_true")
    args = parser.parse_args()
    
    model_config = ModelConfig()
    train_config = TrainConfig(
        data_path=args.data,
        max_steps=args.steps,
        fp8=not args.no_fp8
    )
    
    train(model_config, train_config)
