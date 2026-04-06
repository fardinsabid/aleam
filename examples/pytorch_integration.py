#!/usr/bin/env python3
"""
PyTorch integration examples for Aleam.

This example demonstrates using Aleam's true random numbers
with PyTorch for generating tensors on both CPU and GPU.
"""

import torch
import aleam as al


def main():
    print("=" * 70)
    print("Aleam - PyTorch Integration Examples")
    print("=" * 70)
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   cuDNN version: {torch.backends.cudnn.version()}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Create generator
    gen = al.TorchGenerator(device=str(device))
    
    print("\n📊 Tensor Generation:")
    
    # Random uniform tensor
    t_uniform = gen.rand(3, 3)
    print(f"  rand(3,3) on {device}:\n{t_uniform}")
    
    # Random normal tensor
    t_normal = gen.randn(2, 4)
    print(f"\n  randn(2,4) on {device}:\n{t_normal}")
    
    # Random integer tensor
    t_int = gen.randint(0, 10, (3, 5))
    print(f"\n  randint(0,10,(3,5)) on {device}:\n{t_int}")
    
    print("\n🎯 Distribution Tensors:")
    
    # Uniform distribution
    t_uniform_dist = gen.uniform(0, 1, (5,))
    print(f"  uniform(0,1,5): {t_uniform_dist}")
    
    # Normal distribution
    t_normal_dist = gen.normal(0, 1, (5,))
    print(f"  normal(0,1,5): {t_normal_dist}")
    
    print("\n📈 Large Tensor Generation:")
    
    # Large tensor benchmark
    import time
    sizes = [(1000, 1000), (5000, 5000)]
    
    for size in sizes:
        total = size[0] * size[1]
        start = time.time()
        t = gen.randn(*size)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  randn{size}: {total/elapsed/1e6:.2f}M elements/sec")
    
    print("\n🔬 Statistical Properties:")
    
    # Generate large sample and verify statistics
    samples = gen.randn(100000)
    mean = samples.mean().item()
    std = samples.std().item()
    print(f"  randn(100000): mean={mean:.4f}, std={std:.4f}")
    print(f"  Expected: mean=0, std=1")
    
    print("\n🔀 Random Operations:")
    
    # Random choice from list
    choices = gen.choice(['a', 'b', 'c', 'd', 'e'], size=10)
    print(f"  choice(5 items, size=10): {choices}")
    
    # Random integer scalar
    rand_int = gen.randint_scalar(0, 100)
    print(f"  randint_scalar(0,100): {rand_int}")
    
    print("\n" + "=" * 70)
    print("✅ PyTorch integration demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()