# B300 Ready-to-Run Training Scripts

Complete training setup for Dell XE9780 + 8x NVIDIA B300. Everything is ready to run.

## Quick Start

### 1. Setup Environment

```bash
# Clone this repo
git clone https://github.com/bnarasimha21/b300-ready.git
cd b300-ready

# Run setup (creates conda env, installs deps)
./setup.sh

# Activate environment
conda activate b300
```

### 2. Verify Hardware

```bash
# Check all 8 GPUs
nvidia-smi -L

# Test NVLink bandwidth
python scripts/nvlink_bandwidth_test.py

# Test all-reduce performance
python scripts/allreduce_benchmark.py
```

### 3. Prepare Data

```bash
# Download and tokenize FineWeb (10B tokens)
python scripts/prepare_data.py --tokens 10

# Or use your own data (place .npy files in data/fineweb/)
```

### 4. Train

**Single Node (8x B300) - 7B Model:**
```bash
./launch_8gpu.sh
```

**Multi-Node (16x B300) - 70B Model:**
```bash
# On node 0 (master):
MASTER_ADDR=10.0.0.1 NODE_RANK=0 ./launch_16gpu.sh

# On node 1:
MASTER_ADDR=10.0.0.1 NODE_RANK=1 ./launch_16gpu.sh
```

## Directory Structure

```
b300-ready/
├── setup.sh                    # Environment setup
├── requirements.txt            # Python dependencies
├── launch_8gpu.sh              # Single-node launch
├── launch_16gpu.sh             # Multi-node launch
├── configs/
│   └── 7b_config.yaml          # Model & training config
├── scripts/
│   ├── nccl_env.sh             # NCCL environment vars
│   ├── nvlink_bandwidth_test.py # Test NVLink
│   ├── allreduce_benchmark.py  # Test collective ops
│   ├── prepare_data.py         # Data preparation
│   ├── train_7b.py             # 7B single-node training
│   └── train_multinode.py      # 70B multi-node training
├── data/
│   └── fineweb/                # Training data (.npy files)
└── checkpoints/                # Saved models
```

## Expected Performance

| Configuration | Model | Tokens/sec | 1T tokens |
|--------------|-------|------------|-----------|
| 8x B300 | 7B | ~45,000 | ~257 days |
| 8x B300 | 70B | ~12,000 | ~965 days |
| 16x B300 | 70B | ~22,000 | ~526 days |

## Monitoring

Training logs to stdout. For W&B:

```bash
pip install wandb
wandb login
# Training will auto-log to wandb
```

## Troubleshooting

**NVLink not detected:**
```bash
nvidia-smi nvlink -s  # Check status
nvidia-smi topo -m    # Check topology
```

**NCCL errors:**
```bash
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_FILE=/tmp/nccl.log
# Then check /tmp/nccl.log
```

**OOM errors:**
- Reduce `micro_batch_size` in config
- Reduce `seq_length`
- Enable gradient checkpointing

## Hardware Requirements

- 8x NVIDIA B300 (192GB each)
- NVLink 5.0 connectivity
- 800G RoCEv2 for multi-node
- 1TB+ NVMe storage
- CUDA 12.4+, cuDNN 9.x

## License

MIT
