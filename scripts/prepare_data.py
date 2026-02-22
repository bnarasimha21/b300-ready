#!/usr/bin/env python3
"""Prepare FineWeb dataset for pre-training - Ready to run"""

import os
import argparse
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def prepare_fineweb(output_dir: str, num_tokens_billions: float = 10.0):
    """Download and tokenize FineWeb dataset"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing {num_tokens_billions}B tokens...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset (streaming to avoid memory issues)
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True
    )
    
    buffer = []
    file_idx = 0
    tokens_per_file = 100_000_000  # 100M tokens per file
    total_tokens = 0
    target_tokens = int(num_tokens_billions * 1e9)
    
    pbar = tqdm(total=target_tokens, unit="tokens", unit_scale=True)
    
    for example in dataset:
        tokens = tokenizer.encode(example["text"], add_special_tokens=False)
        buffer.extend(tokens)
        
        while len(buffer) >= tokens_per_file:
            # Save chunk
            arr = np.array(buffer[:tokens_per_file], dtype=np.uint16)
            np.save(output_path / f"train_{file_idx:04d}.npy", arr)
            
            total_tokens += tokens_per_file
            pbar.update(tokens_per_file)
            
            buffer = buffer[tokens_per_file:]
            file_idx += 1
            
            if total_tokens >= target_tokens:
                break
        
        if total_tokens >= target_tokens:
            break
    
    # Save remaining
    if buffer and total_tokens < target_tokens:
        arr = np.array(buffer, dtype=np.uint16)
        np.save(output_path / f"train_{file_idx:04d}.npy", arr)
    
    pbar.close()
    print(f"Saved {file_idx + 1} files to {output_path}")
    print(f"Total tokens: {total_tokens / 1e9:.2f}B")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/fineweb", help="Output directory")
    parser.add_argument("--tokens", type=float, default=10.0, help="Billions of tokens")
    args = parser.parse_args()
    
    prepare_fineweb(args.output, args.tokens)
