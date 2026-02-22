# NVIDIA B300 + Dell XE9780: Complete Training Setup

Everything you need to train LLMs on Dell PowerEdge XE9780 + 8x NVIDIA B300.

## What's Included

| Content | Location |
|---------|----------|
| **Deep Dive Documentation** | [docs/DEEP-DIVE.md](docs/DEEP-DIVE.md) |
| **Ready-to-Run Code** | [scripts/](scripts/) |
| **Launch Scripts** | `launch_8gpu.sh`, `launch_16gpu.sh` |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/bnarasimha21/b300-ready.git
cd b300-ready

# 2. Setup environment
./setup.sh
conda activate b300

# 3. Verify hardware
python scripts/nvlink_bandwidth_test.py
python scripts/allreduce_benchmark.py

# 4. Prepare data (10B tokens)
python scripts/prepare_data.py --tokens 10

# 5. Train
./launch_8gpu.sh   # 7B model on 8 GPUs
# OR
./launch_16gpu.sh  # 70B model on 16 GPUs (2 nodes)
```

## Repository Structure

```
b300-ready/
├── docs/
│   └── DEEP-DIVE.md            # Full technical documentation
├── scripts/
│   ├── nvlink_bandwidth_test.py # Verify NVLink (ready to run)
│   ├── allreduce_benchmark.py   # Test 8-GPU collectives (ready to run)
│   ├── prepare_data.py          # Data prep (ready to run)
│   ├── train_7b.py              # 7B training (ready to run)
│   └── train_multinode.py       # 70B multi-node (ready to run)
├── configs/
│   └── 7b_config.yaml           # Training configuration
├── setup.sh                     # One-command setup
├── launch_8gpu.sh               # Single-node launch
├── launch_16gpu.sh              # Multi-node launch
└── requirements.txt             # Dependencies
```

## Hardware Configuration

| Component | Specification |
|-----------|---------------|
| Server | Dell PowerEdge XE9780 |
| GPUs | 8x NVIDIA B300 (192GB HBM3e each) |
| CPU | Intel Xeon (Granite Rapids) |
| Intra-node | NVLink 5.0 (1.8 TB/s) |
| Inter-node | 800 Gbps RoCEv2 |

## What You Can Train

| Model | GPUs | Tokens/sec | Time for 1T tokens |
|-------|------|------------|-------------------|
| 7B | 8 | ~45,000 | ~257 days |
| 70B | 8 | ~12,000 | ~965 days |
| 70B | 16 (2 nodes) | ~22,000 | ~526 days |

## Documentation

Full technical deep dive with architecture diagrams, code examples, and benchmarks:

**[Read the Deep Dive →](docs/DEEP-DIVE.md)**

Covers:
- System architecture & NVLink topology
- RoCEv2 inter-node networking
- Software stack setup
- 5 practical examples
- Advanced Use Case 1: Pre-training 7B from scratch
- Advanced Use Case 2: Multi-node 70B training
- Monitoring & profiling
- Power & cost analysis

## License

MIT
