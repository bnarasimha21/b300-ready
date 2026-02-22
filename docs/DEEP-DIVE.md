# NVIDIA B300 Part 2: Dell XE9780 System Configuration

> Extension of the B300 GPU deep dive video - focusing on the full server configuration

> **Ready-to-run code:** https://github.com/bnarasimha21/b300-ready


## System Architecture Overview

### Hardware Configuration

| Component | Specification |
|-----------|---------------|
| **Server** | Dell PowerEdge XE9780 |
| **GPUs** | 8x NVIDIA B300 (Blackwell architecture) |
| **GPU Memory** | 192GB HBM3e per GPU (1.5TB total) |
| **CPU** | Dual Intel Xeon Scalable (6th Gen, Granite Rapids) |
| **Intra-node** | NVLink 5.0 (1.8 TB/s bidirectional) |
| **Inter-node** | 800 Gbps RoCEv2 (RDMA over Converged Ethernet) |

### B300 GPU Specifications (per GPU)

- **FP8 Performance**: 2.25 PFLOPS
- **FP16/BF16**: 1.125 PFLOPS  
- **FP32**: 562.5 TFLOPS
- **Memory Bandwidth**: 8 TB/s HBM3e
- **TDP**: 700W (liquid cooled configuration)
- **Transistors**: 208 billion
- **Architecture**: Blackwell (5th gen Tensor Cores)

### 8-GPU System Aggregate Performance

| Metric | Per GPU | 8-GPU Total |
|--------|---------|-------------|
| FP8 | 2.25 PFLOPS | **18 PFLOPS** |
| FP16/BF16 | 1.125 PFLOPS | **9 PFLOPS** |
| HBM3e Memory | 192GB | **1.5TB** |
| Memory Bandwidth | 8 TB/s | **64 TB/s** |

---

## NVLink 5.0 Topology

### Intra-Node Communication

NVLink 5.0 provides direct GPU-to-GPU communication without CPU involvement:

```
┌─────────────────────────────────────────────────────────────┐
│                    NVLink 5.0 Mesh                          │
│                                                             │
│   GPU0 ←──────→ GPU1 ←──────→ GPU2 ←──────→ GPU3           │
│     ↑             ↑             ↑             ↑             │
│     │             │             │             │             │
│     ↓             ↓             ↓             ↓             │
│   GPU4 ←──────→ GPU5 ←──────→ GPU6 ←──────→ GPU7           │
│                                                             │
│   Each link: 450 GB/s bidirectional                         │
│   Total bisection bandwidth: 1.8 TB/s                       │
└─────────────────────────────────────────────────────────────┘
```

### NVLink Bandwidth Comparison

| Generation | Per Link | Links/GPU | Total/GPU |
|------------|----------|-----------|-----------|
| NVLink 3.0 (A100) | 50 GB/s | 12 | 600 GB/s |
| NVLink 4.0 (H100) | 50 GB/s | 18 | 900 GB/s |
| **NVLink 5.0 (B300)** | **50 GB/s** | **18** | **900 GB/s** |

*Note: B300 uses NVLink 5.0 with improved efficiency and lower latency*

---

## RoCEv2 Inter-Node Networking

### 800 Gbps Configuration

```
┌──────────────────┐     800G RoCEv2      ┌──────────────────┐
│   XE9780 Node 1  │◄────────────────────►│   XE9780 Node 2  │
│   8x B300 GPUs   │                      │   8x B300 GPUs   │
└──────────────────┘                      └──────────────────┘
        │                                         │
        │              Spine Switch               │
        └─────────────►   (800G)  ◄───────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ┌─────────┐       ┌─────────┐       ┌─────────┐
   │ Node 3  │       │ Node 4  │       │ Node N  │
   └─────────┘       └─────────┘       └─────────┘
```

### Network Interface Configuration

Each XE9780 typically includes:
- 2x NVIDIA ConnectX-7 (400G each) or
- 1x NVIDIA ConnectX-8 (800G)
- RDMA-capable for GPUDirect RDMA

---

## Software Stack Setup

### 1. Driver and CUDA Installation

```bash
# Check current driver
nvidia-smi

# Expected output for B300
# Driver Version: 560.xx or higher
# CUDA Version: 12.6+

# Verify all 8 GPUs visible
nvidia-smi -L
# GPU 0: NVIDIA B300 (UUID: GPU-xxxx)
# GPU 1: NVIDIA B300 (UUID: GPU-xxxx)
# ... (8 total)
```

### 2. NVLink Topology Verification

```bash
# Check NVLink status
nvidia-smi nvlink --status

# Check NVLink topology
nvidia-smi topo -m

# Expected output (simplified):
#         GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
# GPU0     X    NV18  NV18  NV18  NV18  NV18  NV18  NV18
# GPU1    NV18   X    NV18  NV18  NV18  NV18  NV18  NV18
# ...
# NV18 = NVLink with 18 links
```

### 3. NCCL Configuration

```bash
# Install NCCL
sudo apt-get install libnccl2 libnccl-dev

# Environment variables for optimal performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL  # Force NVLink
```

---

## Practical Examples

### Example 1: NVLink Bandwidth Test

```python
# nvlink_bandwidth_test.py
import torch
import time

def test_nvlink_bandwidth():
    """Test GPU-to-GPU bandwidth via NVLink"""
    
    # Allocate large tensors on different GPUs
    size_gb = 10  # 10 GB transfer
    num_elements = (size_gb * 1024**3) // 4  # float32
    
    gpu0 = torch.device('cuda:0')
    gpu1 = torch.device('cuda:1')
    
    # Create tensors
    src = torch.randn(num_elements, device=gpu0)
    dst = torch.empty(num_elements, device=gpu1)
    
    # Warmup
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    
    # Benchmark
    iterations = 10
    start = time.perf_counter()
    for _ in range(iterations):
        dst.copy_(src)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    bandwidth = (size_gb * iterations) / elapsed
    print(f"NVLink Bandwidth: {bandwidth:.2f} GB/s")
    print(f"Expected: ~400-450 GB/s (unidirectional)")

if __name__ == "__main__":
    test_nvlink_bandwidth()
```

Run:
```bash
python nvlink_bandwidth_test.py
```

### Example 2: 8-GPU All-Reduce Benchmark

```python
# all_reduce_benchmark.py
import torch
import torch.distributed as dist
import os
import time

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def benchmark_all_reduce(rank, world_size):
    setup(rank, world_size)
    
    # Test different tensor sizes
    sizes_mb = [1, 10, 100, 1000, 10000]
    
    for size_mb in sizes_mb:
        num_elements = (size_mb * 1024 * 1024) // 4
        tensor = torch.randn(num_elements, device=f'cuda:{rank}')
        
        # Warmup
        for _ in range(5):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        
        # Benchmark
        iterations = 20
        start = time.perf_counter()
        for _ in range(iterations):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        if rank == 0:
            # All-reduce moves 2*(n-1)/n * size data
            algo_bw = (2 * (world_size - 1) / world_size * size_mb * iterations) / elapsed / 1024
            print(f"Size: {size_mb:5} MB | Time: {elapsed/iterations*1000:.2f} ms | Algo BW: {algo_bw:.1f} GB/s")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 8
    torch.multiprocessing.spawn(
        benchmark_all_reduce,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
```

Run:
```bash
python all_reduce_benchmark.py

# Expected output:
# Size:     1 MB | Time: 0.05 ms | Algo BW: 35.0 GB/s
# Size:    10 MB | Time: 0.08 ms | Algo BW: 218.8 GB/s
# Size:   100 MB | Time: 0.25 ms | Algo BW: 700.0 GB/s
# Size:  1000 MB | Time: 2.20 ms | Algo BW: 795.5 GB/s
# Size: 10000 MB | Time: 22.0 ms | Algo BW: 795.5 GB/s
```

### Example 3: Multi-Node Training Setup

```python
# multi_node_train.py
# Run on each node with appropriate RANK and WORLD_SIZE

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup_multi_node():
    """Setup for multi-node training with RoCEv2"""
    
    # These come from SLURM or your job scheduler
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Master node address (first node)
    master_addr = os.environ.get('MASTER_ADDR', 'node1')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Initialize with NCCL (uses NVLink intra-node, RoCEv2 inter-node)
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def train_llm():
    rank, world_size, local_rank = setup_multi_node()
    
    # Example: Large model that benefits from 8+ GPUs
    # With 1.5TB memory (8 GPUs), you can train 70B+ models
    
    model = YourLargeModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # Training loop...
    
    dist.destroy_process_group()
```

Launch script for 2 nodes (16 GPUs total):
```bash
# On node 1 (master):
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr=node1 \
    --master_port=29500 \
    multi_node_train.py

# On node 2:
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=1 \
    --master_addr=node1 \
    --master_port=29500 \
    multi_node_train.py
```

### Example 4: Memory Capacity Test - Loading Large Models

```python
# large_model_test.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_70b_model():
    """
    With 8x B300 (1.5TB HBM3e), you can load 70B+ models in full precision
    or 400B+ models in FP8
    """
    
    # Option 1: Device map auto (distributes across GPUs)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-70b-hf",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    # Check memory usage per GPU
    for i in range(8):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"GPU {i}: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")
    
    return model

def memory_capacity_estimate():
    """Estimate what models fit in 1.5TB"""
    
    estimates = {
        "Llama-2-70B (BF16)": 140,  # ~140GB
        "Llama-2-70B (FP8)": 70,    # ~70GB
        "Llama-3-405B (FP8)": 405,  # ~405GB
        "GPT-4 scale (FP8)": 800,   # ~800GB (estimated)
        "Mixture of Experts 1T (FP8)": 1000,  # Fits with activation memory
    }
    
    total_memory_gb = 192 * 8  # 1536 GB
    
    print(f"Total HBM3e Memory: {total_memory_gb} GB\n")
    print("Model Capacity Estimates:")
    print("-" * 50)
    
    for model, size_gb in estimates.items():
        fits = "✓" if size_gb < total_memory_gb * 0.8 else "✗"  # 80% usable
        print(f"{fits} {model}: ~{size_gb} GB")

if __name__ == "__main__":
    memory_capacity_estimate()
```

### Example 5: FP8 Inference Benchmark

```python
# fp8_inference.py
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import time

def fp8_benchmark():
    """
    Benchmark FP8 inference using Transformer Engine
    B300 excels at FP8 with 2.25 PFLOPS per GPU
    """
    
    # FP8 recipe
    fp8_recipe = recipe.DelayedScaling(
        margin=0,
        fp8_format=recipe.Format.HYBRID,
        amax_history_len=1024,
        amax_compute_algo="max",
    )
    
    # Create a transformer layer with FP8
    batch_size = 32
    seq_len = 4096
    hidden_size = 8192
    num_heads = 64
    
    layer = te.TransformerLayer(
        hidden_size=hidden_size,
        ffn_hidden_size=hidden_size * 4,
        num_attention_heads=num_heads,
        layer_type="encoder",
    ).cuda()
    
    # Input tensor
    x = torch.randn(seq_len, batch_size, hidden_size, device='cuda')
    
    # Warmup
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        for _ in range(10):
            _ = layer(x)
    torch.cuda.synchronize()
    
    # Benchmark
    iterations = 100
    start = time.perf_counter()
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        for _ in range(iterations):
            _ = layer(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    tokens_per_sec = (batch_size * seq_len * iterations) / elapsed
    print(f"FP8 Throughput: {tokens_per_sec/1e6:.2f}M tokens/sec")
    print(f"Latency per forward: {elapsed/iterations*1000:.2f} ms")

if __name__ == "__main__":
    fp8_benchmark()
```

---

## Real-World Workloads

### What You Can Run on 8x B300 (1.5TB HBM3e)

| Workload | Configuration | Expected Performance |
|----------|---------------|---------------------|
| **Llama-3.1-405B Inference** | FP8, tensor parallel=8 | ~50 tokens/sec |
| **Llama-2-70B Fine-tuning** | BF16, FSDP | ~3000 tokens/sec |
| **Stable Diffusion XL** | Batch inference | ~200 images/min |
| **GPT-4 scale training** | FP8, 8-way TP | Feasible with multi-node |
| **Embedding model (1B)** | Full batch | ~1M embeddings/sec |

### Training Throughput Estimates

```
Model Size vs Training Speed (8x B300):

┌─────────────────────────────────────────────────────────┐
│ Model    │ Precision │ Tokens/sec │ Time for 1T tokens │
├──────────┼───────────┼────────────┼────────────────────┤
│ 7B       │ BF16      │ 180,000    │ ~64 days           │
│ 13B      │ BF16      │ 95,000     │ ~122 days          │
│ 70B      │ FP8       │ 25,000     │ ~463 days          │
│ 70B      │ BF16      │ 12,000     │ ~965 days          │
└─────────────────────────────────────────────────────────┘

* Multi-node scales nearly linearly for large models
* 2 nodes (16 GPUs) roughly halves training time
```

---

## Monitoring and Profiling

### GPU Monitoring Script

```bash
#!/bin/bash
# monitor_gpus.sh

watch -n 1 'nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits | column -t -s,'
```

### NCCL Debug Logging

```bash
# Enable detailed NCCL logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Run your training script
python train.py 2>&1 | tee nccl_debug.log

# Check for NVLink usage
grep "NVLink" nccl_debug.log
```

### PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler('./log/profile'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()
```

---

## Cost and Power Analysis

### Power Consumption

| Component | Per Unit | Total (8 GPUs) |
|-----------|----------|----------------|
| B300 GPU | 700W | 5,600W |
| Intel Xeon (2x) | 350W | 700W |
| Memory, Storage, Cooling | ~500W | 500W |
| **Total System** | - | **~6,800W** |

### Performance per Watt

- **FP8**: 18 PFLOPS / 6.8 kW = **2.65 TFLOPS/W**
- **Comparison**: H100 system = ~1.5 TFLOPS/W
- **Improvement**: ~77% better efficiency

---

## Quick Reference Commands

```bash
# System health check
nvidia-smi -a

# NVLink status
nvidia-smi nvlink -s

# Topology matrix
nvidia-smi topo -m

# GPU reset (if needed)
sudo nvidia-smi -r

# Check PCIe bandwidth
nvidia-smi -q -d PCIE

# Monitor power
nvidia-smi dmon -s p

# DCGM metrics (if installed)
dcgmi diag -r 3
```

---

## Next Steps

When your XE9780 is available:

1. **Verify hardware**: Run `nvidia-smi -L` to confirm all 8 B300s
2. **Test NVLink**: Run the bandwidth test script
3. **Benchmark all-reduce**: Verify NCCL performance
4. **Load a large model**: Try Llama-70B to test memory
5. **Profile workloads**: Use PyTorch profiler for optimization

---

*Created: 2026-02-22*
*Configuration: Dell PowerEdge XE9780 + 8x NVIDIA B300 + 800G RoCEv2*

---

## Advanced Use Case 1: Pre-training a 7B LLM from Scratch

### Overview

Train a GPT-style 7B parameter model from scratch using NeMo Framework with FP8 precision on 8x B300.

### Why This Configuration Works

| Requirement | 8x B300 Capability |
|-------------|-------------------|
| Model weights (7B, FP8) | ~7GB (fits easily) |
| Optimizer states | ~56GB (Adam) |
| Activations (batch=32, seq=4096) | ~200GB |
| Total per-GPU | ~33GB (192GB available) |
| **Result** | Large batch sizes possible |

### Environment Setup

```bash
# Create conda environment
conda create -n nemo python=3.10 -y
conda activate nemo

# Install NeMo Framework
pip install nemo_toolkit[all]
pip install transformer_engine
pip install megatron-core

# Verify TransformerEngine FP8 support
python -c "import transformer_engine; print(transformer_engine.__version__)"
```

### Model Configuration

```yaml
# config/7b_fp8_config.yaml
trainer:
  devices: 8
  num_nodes: 1
  accelerator: gpu
  precision: bf16-mixed
  max_steps: 100000
  val_check_interval: 1000
  
model:
  micro_batch_size: 4
  global_batch_size: 256
  tensor_model_parallel_size: 2
  pipeline_model_parallel_size: 1
  
  encoder_seq_length: 4096
  max_position_embeddings: 4096
  
  num_layers: 32
  hidden_size: 4096
  ffn_hidden_size: 11008
  num_attention_heads: 32
  num_query_groups: 8  # GQA
  
  # FP8 Configuration
  fp8: true
  fp8_e4m3: true
  fp8_hybrid: true
  fp8_amax_history_len: 1024
  fp8_amax_compute_algo: max
  
  tokenizer:
    library: huggingface
    type: meta-llama/Llama-2-7b-hf
    
  data:
    data_impl: mmap
    splits_string: "99,1,0"
    seq_length: 4096
    
  optim:
    name: fused_adam
    lr: 3e-4
    betas: [0.9, 0.95]
    weight_decay: 0.1
```

### Data Preparation

```python
# prepare_data.py
"""Prepare FineWeb dataset for pre-training"""

from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from pathlib import Path

def prepare_fineweb():
    # Load FineWeb-Edu (high quality web text)
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",  # 10B tokens sample
        split="train",
        streaming=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    output_dir = Path("data/fineweb")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process in chunks
    buffer = []
    file_idx = 0
    tokens_per_file = 100_000_000  # 100M tokens per file
    
    for example in dataset:
        tokens = tokenizer.encode(example["text"])
        buffer.extend(tokens)
        
        if len(buffer) >= tokens_per_file:
            # Save as memory-mapped file
            arr = np.array(buffer[:tokens_per_file], dtype=np.uint16)
            np.save(output_dir / f"train_{file_idx:04d}.npy", arr)
            buffer = buffer[tokens_per_file:]
            file_idx += 1
            print(f"Saved file {file_idx}, {file_idx * tokens_per_file / 1e9:.1f}B tokens")

if __name__ == "__main__":
    prepare_fineweb()
```

### Training Script

```python
# train_7b.py
"""Pre-train 7B model with FP8 on 8x B300"""

import nemo.collections.nlp as nemo_nlp
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf import DictConfig, OmegaConf
import torch

@hydra_runner(config_path="config", config_name="7b_fp8_config")
def main(cfg: DictConfig):
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Initialize trainer with FP8
    trainer = nemo_nlp.trainers.MegatronTrainerBuilder(cfg).create_trainer()
    
    # Initialize model
    model = nemo_nlp.models.megatron_gpt_model.MegatronGPTModel(cfg.model, trainer)
    
    # Log GPU memory
    for i in range(8):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        logging.info(f"GPU {i}: {allocated:.1f} GB allocated")
    
    # Start training
    trainer.fit(model)
    
    # Save final checkpoint
    trainer.save_checkpoint("checkpoints/7b_final.ckpt")

if __name__ == "__main__":
    main()
```

### Launch Training

```bash
#!/bin/bash
# launch_training.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=INFO

# 8 GPUs, tensor parallel = 2, data parallel = 4
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --master_addr=localhost \
    --master_port=29500 \
    train_7b.py \
    trainer.devices=8 \
    model.tensor_model_parallel_size=2 \
    model.global_batch_size=256
```

### Expected Performance

| Metric | Value |
|--------|-------|
| Throughput | ~45,000 tokens/sec |
| Time per 1B tokens | ~6.2 hours |
| Time for 100B tokens | ~26 days |
| GPU Memory per device | ~80GB (of 192GB) |
| Power consumption | ~5,600W |

### Monitoring

```python
# monitor_training.py
import wandb

wandb.init(project="b300-7b-pretrain")

# Log during training
wandb.log({
    "loss": loss.item(),
    "learning_rate": scheduler.get_last_lr()[0],
    "tokens_per_second": tokens_processed / elapsed_time,
    "gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
})
```

---

## Advanced Use Case 2: Multi-Node Training with NVLink + RoCEv2

### Overview

Scale training across 2 XE9780 nodes (16x B300 total) using NVLink within nodes and 800G RoCEv2 between nodes.

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                         NODE 1 (XE9780)                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    NVLink 5.0 Mesh                       │  │
│  │   GPU0 ── GPU1 ── GPU2 ── GPU3                          │  │
│  │     │       │       │       │                            │  │
│  │   GPU4 ── GPU5 ── GPU6 ── GPU7                          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                            │                                   │
│                    ConnectX-8 (800G)                          │
└────────────────────────────┼───────────────────────────────────┘
                             │
                      800G RoCEv2 Fabric
                             │
┌────────────────────────────┼───────────────────────────────────┐
│                    ConnectX-8 (800G)                          │
│                            │                                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    NVLink 5.0 Mesh                       │  │
│  │   GPU8 ── GPU9 ── GPU10 ── GPU11                        │  │
│  │     │       │        │        │                          │  │
│  │   GPU12 ── GPU13 ── GPU14 ── GPU15                      │  │
│  └─────────────────────────────────────────────────────────┘  │
│                         NODE 2 (XE9780)                        │
└────────────────────────────────────────────────────────────────┘
```

### Parallelism Strategy

For 70B model on 16 GPUs:

| Parallelism | Dimension | Reasoning |
|-------------|-----------|-----------|
| Tensor Parallel | 8 (within node) | Uses fast NVLink |
| Pipeline Parallel | 1 | Avoid bubble overhead |
| Data Parallel | 2 (across nodes) | Uses RoCEv2 |

### Network Configuration

```bash
# verify_rdma.sh - Run on each node

# Check RDMA devices
rdma link show

# Check RoCEv2 mode
cat /sys/class/infiniband/*/ports/*/gid_attrs/types/*

# Test RDMA bandwidth
ib_write_bw -d mlx5_0 -F --report_gbits

# Expected: ~780 Gbps (of 800G theoretical)
```

### NCCL Configuration for Multi-Node

```bash
# nccl_env.sh

# Force NVLink for intra-node
export NCCL_P2P_LEVEL=NVL

# Enable GPUDirect RDMA for inter-node
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_GID_INDEX=3  # RoCEv2

# Optimal settings
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_BUFFSIZE=8388608  # 8MB buffers

# Debug
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
```

### Multi-Node Training Script

```python
# train_multinode.py
"""70B model training across 2 nodes (16x B300)"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import transformer_engine.pytorch as te

def setup_distributed():
    """Initialize distributed training with NCCL"""
    
    # From SLURM or launch script
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Set device before init
    torch.cuda.set_device(local_rank)
    
    # Initialize with NCCL
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    
    return rank, world_size, local_rank

def create_model_70b():
    """Create 70B model with FSDP sharding"""
    
    from transformers import LlamaConfig, LlamaForCausalLM
    
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        max_position_embeddings=4096,
    )
    
    # Initialize with meta device (no memory yet)
    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    
    return model

def main():
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print(f"Training on {world_size} GPUs across {world_size // 8} nodes")
    
    # Create model
    model = create_model_70b()
    
    # Wrap with FSDP
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={LlamaDecoderLayer},
    )
    
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=local_rank,
        sharding_strategy="FULL_SHARD",
        mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
    )
    
    # Log memory usage
    mem_allocated = torch.cuda.memory_allocated() / 1e9
    if rank == 0:
        print(f"Memory per GPU after FSDP init: {mem_allocated:.1f} GB")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    for step in range(1000):
        # Dummy data (replace with real dataloader)
        input_ids = torch.randint(0, 32000, (2, 4096), device=f"cuda:{local_rank}")
        
        # Forward with FP8
        with te.fp8_autocast(enabled=True):
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
        
        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if rank == 0 and step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

### SLURM Launch Script

```bash
#!/bin/bash
#SBATCH --job-name=b300-70b
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=48:00:00

# Load modules
module load cuda/12.4
module load nccl/2.20

# Set master
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# NCCL settings
source nccl_env.sh

# Launch
srun --ntasks-per-node=8 \
    torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_multinode.py
```

### Manual Launch (Without SLURM)

```bash
# On Node 1 (master):
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_multinode.py

# On Node 2:
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_multinode.py
```

### Scaling Efficiency Test

```python
# benchmark_scaling.py
"""Measure scaling efficiency across nodes"""

import torch
import torch.distributed as dist
import time

def benchmark_allreduce(size_mb, iterations=50):
    """Benchmark all-reduce at different scales"""
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    num_elements = (size_mb * 1024 * 1024) // 4
    tensor = torch.randn(num_elements, device=f"cuda:{rank % 8}")
    
    # Warmup
    for _ in range(10):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    if rank == 0:
        # Calculate bandwidth
        # All-reduce: 2 * (n-1) / n * size
        algobw = (2 * (world_size - 1) / world_size * size_mb * iterations) / elapsed / 1024
        print(f"Size: {size_mb:5} MB | Algo BW: {algobw:.1f} GB/s")
        
        # Expected:
        # Intra-node (8 GPU): ~750 GB/s (NVLink)
        # Inter-node (16 GPU): ~350 GB/s (limited by RoCEv2)

if __name__ == "__main__":
    setup_distributed()
    
    for size in [10, 100, 1000, 5000]:
        benchmark_allreduce(size)
```

### Expected Multi-Node Performance

| Configuration | Tokens/sec | Scaling Efficiency |
|--------------|------------|-------------------|
| 1 node (8 GPU) | 25,000 | 100% (baseline) |
| 2 nodes (16 GPU) | 47,000 | 94% |
| 4 nodes (32 GPU) | 90,000 | 90% |

### Troubleshooting

```bash
# Check NVLink is being used intra-node
nvidia-smi nvlink -s

# Check RDMA is working inter-node
ibstat

# Test inter-node connectivity
# On node 1:
ib_write_bw -d mlx5_0 --report_gbits

# On node 2:
ib_write_bw -d mlx5_0 10.0.0.1 --report_gbits

# NCCL debug for communication issues
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_FILE=/tmp/nccl_debug.%h.%p.log
```

---

*Updated: 2026-02-22*
*Added: Advanced Use Cases for 7B pre-training and multi-node training*
