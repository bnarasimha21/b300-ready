# NVIDIA B300 + Dell XE9780: Complete Training & Inference Setup

Everything you need to train and serve LLMs on Dell PowerEdge XE9780 + 8x NVIDIA B300.

## B300 Key Feature: NVFP4

B300 (Blackwell) introduces **FP4** precision - 2x throughput vs FP8:

| Precision | Performance (8x B300) | Best For |
|-----------|----------------------|----------|
| **FP4** | 36 PFLOPS | Inference |
| FP8 | 18 PFLOPS | Training + Inference |
| BF16 | 9 PFLOPS | Training |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/bnarasimha21/b300-ready.git
cd b300-ready

# 2. Setup
./setup.sh && conda activate b300

# 3. Verify hardware
python scripts/nvlink_bandwidth_test.py

# 4a. TRAINING (uses FP8/BF16)
python scripts/prepare_data.py --tokens 10
./launch_8gpu.sh

# 4b. INFERENCE (uses FP4)
python scripts/fp4_inference.py --mode compare
```

## Repository Structure

```
b300-ready/
├── docs/
│   └── DEEP-DIVE.md              # Full technical documentation
├── scripts/
│   ├── nvlink_bandwidth_test.py  # Verify NVLink
│   ├── allreduce_benchmark.py    # Test collectives
│   ├── prepare_data.py           # Data preparation
│   ├── train_7b.py               # 7B training (FP8/BF16)
│   ├── train_multinode.py        # 70B multi-node training
│   └── fp4_inference.py          # FP4 inference (NEW)
├── configs/
│   └── 7b_config.yaml
├── setup.sh
├── launch_8gpu.sh
├── launch_16gpu.sh
└── requirements.txt
```

## When to Use Which Precision

| Task | Script | Precision | Why |
|------|--------|-----------|-----|
| Training | `train_7b.py` | FP8/BF16 | Gradient stability |
| Inference | `fp4_inference.py` | FP4 | 2x throughput |

## Hardware Configuration

| Component | Specification |
|-----------|---------------|
| Server | Dell PowerEdge XE9780 |
| GPUs | 8x NVIDIA B300 (192GB HBM3e) |
| Intra-node | NVLink 5.0 (1.8 TB/s) |
| Inter-node | 800 Gbps RoCEv2 |

## Performance

| Workload | Precision | Throughput |
|----------|-----------|------------|
| 7B Training | FP8 | ~45K tokens/sec |
| 70B Training | FP8 | ~12K tokens/sec |
| 70B Inference | FP4 | ~8K tokens/sec (batched) |

## Documentation

**[Full Deep Dive →](docs/DEEP-DIVE.md)**

- System architecture
- NVLink/RoCEv2 networking
- FP4 inference pipeline
- Advanced training examples
- Monitoring & profiling

## License

MIT
