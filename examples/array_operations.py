#!/usr/bin/env python3
"""
Array operations examples for Aleam (C++ Core).

This example demonstrates generating multi-dimensional arrays of
true random numbers including uniform, normal, and integer distributions.

Note: The C++ API returns numpy arrays directly, not nested lists.
"""

import aleam as al
import numpy as np


def main():
    print("=" * 70)
    print("Aleam - Array Operations Examples (C++ Core)")
    print("=" * 70)
    
    print("\n Uniform Random Array Generation:")
    
    # 1D array - using module-level function (returns numpy array)
    arr_1d = al.random_array(10)
    print(f"  al.random_array(10): {[f'{x:.4f}' for x in arr_1d]}")
    print(f"  Shape: {arr_1d.shape}, Type: {type(arr_1d)}")
    
    # 2D array
    arr_2d = al.random_array((3, 4))
    print(f"  al.random_array((3,4)):\n{arr_2d}")
    print(f"  Shape: {arr_2d.shape}")
    
    # 3D array
    arr_3d = al.random_array((2, 3, 2))
    print(f"  al.random_array((2,3,2)) shape: {arr_3d.shape}")
    print(f"  Sample value: {arr_3d[0,0,0]:.4f}")
    
    print("\n Normal Distribution Array Generation:")
    
    # 1D normal array
    norm_arr = al.randn_array(1000, mu=0, sigma=1)
    print(f"  al.randn_array(1000): mean={np.mean(norm_arr):.4f}, std={np.std(norm_arr):.4f}")
    print(f"  Shape: {norm_arr.shape}")
    
    # 2D normal array
    norm_2d = al.randn_array((5, 5), mu=0, sigma=1)
    print(f"  al.randn_array((5,5)):\n{norm_2d:.4f}")
    
    # Custom normal parameters
    norm_custom = al.randn_array(500, mu=10, sigma=2)
    print(f"  al.randn_array(500, mu=10, sigma=2): mean={np.mean(norm_custom):.4f}, std={np.std(norm_custom):.4f}")
    
    print("\n Integer Array Generation:")
    
    # 1D integer array
    int_arr = al.randint_array(10, low=0, high=10)
    print(f"  al.randint_array(10, 0, 10): {int_arr}")
    print(f"  Shape: {int_arr.shape}")
    
    # 2D integer array
    int_2d = al.randint_array((4, 5), low=0, high=100)
    print(f"  al.randint_array((4,5), 0, 100):\n{int_2d}")
    
    # Negative range integers
    int_neg = al.randint_array(10, low=-50, high=50)
    print(f"  al.randint_array(10, -50, 50): {int_neg}")
    
    print("\n Using Aleam instance methods:")
    
    # Create a generator instance
    rng = al.Aleam()
    
    # Instance methods also work (they call the same module-level functions)
    arr_instance = rng.random_array(10)
    print(f"  rng.random_array(10): {[f'{x:.4f}' for x in arr_instance]}")
    
    norm_instance = rng.randn_array(100, mu=0, sigma=1)
    print(f"  rng.randn_array(100): mean={np.mean(norm_instance):.4f}, std={np.std(norm_instance):.4f}")
    
    int_instance = rng.randint_array(20, low=1, high=10)
    print(f"  rng.randint_array(20, 1, 10): {int_instance}")
    
    print("\n Statistical Verification:")
    
    # Generate large sample to verify distribution
    sample_size = 100000
    
    uniform_sample = al.random_array(sample_size)
    print(f"  Uniform [0,1) over {sample_size:,} samples:")
    print(f"    Mean: {np.mean(uniform_sample):.4f} (expected 0.5000)")
    print(f"    Variance: {np.var(uniform_sample):.4f} (expected 0.08333)")
    
    normal_sample = al.randn_array(sample_size, mu=0, sigma=1)
    print(f"  Normal(0,1) over {sample_size:,} samples:")
    print(f"    Mean: {np.mean(normal_sample):.4f} (expected 0.0000)")
    print(f"    Std: {np.std(normal_sample):.4f} (expected 1.0000)")
    
    integer_sample = al.randint_array(sample_size, low=0, high=10)
    print(f"  Integer [0,10) over {sample_size:,} samples:")
    print(f"    Mean: {np.mean(integer_sample):.2f} (expected 4.50)")
    print(f"    Min: {np.min(integer_sample)}, Max: {np.max(integer_sample)}")
    
    print("\n Large Array Performance:")
    
    # Generate a large array and measure time
    import time
    size = (1000, 1000)
    total_elements = size[0] * size[1]
    
    start = time.time()
    large_arr = al.random_array(size)
    elapsed = time.time() - start
    
    print(f"  Generated {total_elements:,} random numbers in {elapsed:.2f} seconds")
    print(f"  Speed: {total_elements / elapsed / 1e6:.2f} million ops/sec")
    print(f"  Array shape: {large_arr.shape}")
    print(f"  Memory usage: {large_arr.nbytes / 1024 / 1024:.1f} MB")
    
    print("\n" + "=" * 70)
    print(" Array operations demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()