#!/usr/bin/env python3
"""Test NVLink bandwidth between GPUs - Ready to run"""

import torch
import time
import sys

def test_bandwidth(gpu_a: int, gpu_b: int, size_gb: float = 10.0):
    """Test bandwidth between two GPUs"""
    
    num_elements = int((size_gb * 1024**3) // 4)
    
    src = torch.randn(num_elements, device=f'cuda:{gpu_a}')
    dst = torch.empty(num_elements, device=f'cuda:{gpu_b}')
    
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
    return bandwidth

def main():
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    print("-" * 50)
    
    if num_gpus < 2:
        print("Need at least 2 GPUs for NVLink test")
        sys.exit(1)
    
    # Test all pairs
    for i in range(min(num_gpus, 8)):
        for j in range(i + 1, min(num_gpus, 8)):
            bw = test_bandwidth(i, j)
            print(f"GPU {i} -> GPU {j}: {bw:.1f} GB/s")
    
    print("-" * 50)
    print("Expected NVLink 5.0: 400-450 GB/s unidirectional")

if __name__ == "__main__":
    main()
