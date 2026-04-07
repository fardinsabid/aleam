#!/usr/bin/env python3
"""
TensorFlow integration examples for Aleam (C++ Core).

This example demonstrates using Aleam's true random numbers
with TensorFlow for generating tensors on both CPU and GPU.

Note: The C++ core doesn't have built-in TFGenerator.
Use Aleam for true random seeds, then TensorFlow's native generators.
"""

import tensorflow as tf
import aleam as al
import numpy as np
import time


def main():
    print("=" * 70)
    print("Aleam - TensorFlow Integration Examples (C++ Core)")
    print("=" * 70)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\n GPUs available: {len(gpus)}")
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu}")
    
    # Create Aleam generator for true random seeds
    rng = al.Aleam()
    
    print("\n" + "=" * 70)
    print(" Method: True random seeds from Aleam + TensorFlow generation")
    print("=" * 70)
    
    print("\n Tensor Generation with True Random Seeds:")
    
    # Generate a true random seed
    true_seed = rng.random_uint64()
    print(f"\n  True random seed from Aleam: {true_seed}")
    
    # Set TensorFlow seed
    tf.random.set_seed(true_seed)
    
    # Random normal tensor
    t_normal = tf.random.normal((3, 3), mean=0.0, stddev=1.0)
    print(f"\n  tf.random.normal((3,3)) with Aleam seed:\n{t_normal.numpy()}")
    
    # Random uniform tensor
    t_uniform = tf.random.uniform((2, 4), minval=0, maxval=1)
    print(f"\n  tf.random.uniform((2,4)):\n{t_uniform.numpy()}")
    
    # Random integer tensor
    t_int = tf.random.uniform((3, 5), minval=0, maxval=10, dtype=tf.int32)
    print(f"\n  tf.random.uniform((3,5), 0, 10, int32):\n{t_int.numpy()}")
    
    # Truncated normal tensor
    t_trunc = tf.random.truncated_normal((3, 3), mean=0.0, stddev=1.0)
    print(f"\n  tf.random.truncated_normal((3,3)):\n{t_trunc.numpy()}")
    
    print("\n Large Tensor Generation with True Random Seeds:")
    
    # Large tensor benchmark
    sizes = [(1000, 1000), (5000, 5000)]
    
    for size in sizes:
        # Generate new true random seed for each run
        seed = rng.random_uint64()
        tf.random.set_seed(seed)
        
        total = size[0] * size[1]
        
        start = time.time()
        t = tf.random.normal(size)
        elapsed = time.time() - start
        
        print(f"  normal{size}: {total/elapsed/1e6:.2f}M elements/sec (seed: {seed & 0xFFFF:04x})")
    
    print("\n Statistical Properties with True Random Seeds:")
    
    # Generate large sample and verify statistics
    seed = rng.random_uint64()
    tf.random.set_seed(seed)
    
    samples = tf.random.normal((100000,))
    mean = tf.reduce_mean(samples).numpy()
    std = tf.math.reduce_std(samples).numpy()
    
    print(f"  True random seed used: {seed}")
    print(f"  normal(100000): mean={mean:.4f}, std={std:.4f}")
    print(f"  Expected: mean=0, std=1")
    
    # Uniform distribution test
    seed = rng.random_uint64()
    tf.random.set_seed(seed)
    
    uniform_samples = tf.random.uniform((100000,), minval=0, maxval=1)
    uniform_mean = tf.reduce_mean(uniform_samples).numpy()
    print(f"  uniform(100000): mean={uniform_mean:.4f}, expected=0.5")
    print(f"  Seed used: {seed}")
    
    print("\n Shuffling Tensors with True Random Seeds:")
    
    # Shuffle using TensorFlow with Aleam seed
    original = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    seed = rng.random_uint64()
    tf.random.set_seed(seed)
    
    # Get random permutation
    shuffled_indices = tf.random.shuffle(tf.range(tf.shape(original)[0]))
    shuffled = tf.gather(original, shuffled_indices)
    
    print(f"  Original: {original.numpy()}")
    print(f"  Shuffled: {shuffled.numpy()}")
    print(f"  Seed used: {seed}")
    
    # Shuffle 2D tensor (rows only)
    original_2d = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    
    seed = rng.random_uint64()
    tf.random.set_seed(seed)
    
    shuffled_indices_2d = tf.random.shuffle(tf.range(tf.shape(original_2d)[0]))
    shuffled_2d = tf.gather(original_2d, shuffled_indices_2d)
    
    print(f"\n  Original 2D:\n{original_2d.numpy()}")
    print(f"  Shuffled 2D (rows):\n{shuffled_2d.numpy()}")
    print(f"  Seed used: {seed}")
    
    print("\n GPU vs CPU Performance with True Random Seeds:")
    print("-" * 60)
    
    # Compare CPU and GPU performance if GPU available
    if gpus:
        sizes = [10000, 100000, 1000000, 10000000]
        
        print(f"\n  {'Size':<12} {'CPU Time (s)':<15} {'GPU Time (s)':<15} {'Speedup':<10}")
        print(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*10}")
        
        for n in sizes:
            # CPU
            with tf.device('/CPU:0'):
                seed = rng.random_uint64()
                tf.random.set_seed(seed)
                start = time.time()
                t_cpu = tf.random.normal((n,))
                elapsed_cpu = time.time() - start
            
            # GPU
            with tf.device('/GPU:0'):
                seed = rng.random_uint64()
                tf.random.set_seed(seed)
                start = time.time()
                t_gpu = tf.random.normal((n,))
                elapsed_gpu = time.time() - start
            
            speedup = elapsed_cpu / elapsed_gpu if elapsed_gpu > 0 else 0
            
            print(f"  {n:>12,} {elapsed_cpu:>15.4f} {elapsed_gpu:>15.4f} {speedup:>9.1f}x")
    
    print("\n" + "=" * 70)
    print(" TensorFlow integration demo complete")
    print("=" * 70)
    print("\n  Key takeaway: Use Aleam.random_uint64() for true random seeds,")
    print("  then use TensorFlow's native random functions with those seeds.")
    print("  This gives you true randomness at TensorFlow's native speed!")


if __name__ == "__main__":
    main()