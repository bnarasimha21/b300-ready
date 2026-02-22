#!/usr/bin/env python3
"""8-GPU All-Reduce Benchmark - Ready to run"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def benchmark(rank, world_size):
    setup(rank, world_size)
    
    sizes_mb = [1, 10, 100, 1000, 5000]
    
    if rank == 0:
        print(f"\nAll-Reduce Benchmark ({world_size} GPUs)")
        print("-" * 60)
    
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
            avg_time_ms = (elapsed / iterations) * 1000
            algo_bw = (2 * (world_size - 1) / world_size * size_mb * iterations) / elapsed / 1024
            print(f"Size: {size_mb:5} MB | Time: {avg_time_ms:8.2f} ms | Algo BW: {algo_bw:6.1f} GB/s")
    
    dist.destroy_process_group()

def main():
    world_size = torch.cuda.device_count()
    print(f"Starting benchmark on {world_size} GPUs...")
    mp.spawn(benchmark, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
