#!/usr/bin/env python3
"""
Array operations examples for Aleam.

This example demonstrates generating multi-dimensional arrays of
true random numbers including uniform, normal, and integer distributions.
"""

import aleam as al
import numpy as np


def main():
    print("=" * 70)
    print("Aleam - Array Operations Examples")
    print("=" * 70)
    
    rng = al.Aleam()
    
    print("\n📊 Random Array Generation:")
    
    # 1D array
    arr_1d = rng.random_array(10)
    print(f"  random_array(10): {[f'{x:.4f}' for x in arr_1d]}")
    
    # 2D array
    arr_2d = rng.random_array((3, 4))
    print(f"  random_array((3,4)):\n{np.array(arr_2d)}")
    
    # 3D array
    arr_3d = rng.random_array((2, 3, 2))
    print(f"  random_array((2,3,2)) shape: {np.array(arr_3d).shape}")
    
    print("\n📈 Normal Array Generation:")
    
    # Normal distribution array
    norm_arr = rng.randn_array((1000,), mu=0, sigma=1)
    print(f"  randn_array(1000): mean={np.mean(norm_arr):.4f}, std={np.std(norm_arr):.4f}")
    
    # 2D normal array
    norm_2d = rng.randn_array((5, 5), mu=0, sigma=1)
    print(f"  randn_array((5,5)):\n{np.array(norm_2d):.4f}")
    
    # Custom normal parameters
    norm_custom = rng.randn_array((500,), mu=10, sigma=2)
    print(f"  randn_array(500, mu=10, sigma=2): mean={np.mean(norm_custom):.4f}, std={np.std(norm_custom):.4f}")
    
    print("\n🔢 Integer Array Generation:")
    
    # Integer array
    int_arr = rng.randint_array((10,), low=0, high=10)
    print(f"  randint_array(10, 0, 10): {int_arr}")
    
    # 2D integer array
    int_2d = rng.randint_array((4, 5), low=0, high=100)
    print(f"  randint_array((4,5), 0, 100):\n{np.array(int_2d)}")
    
    # Negative range integers
    int_neg = rng.randint_array((10,), low=-50, high=50)
    print(f"  randint_array(10, -50, 50): {int_neg}")
    
    print("\n📊 Module-Level Array Functions:")
    
    # Using module-level functions directly
    arr_module = al.random_array((5, 5))
    print(f"  al.random_array((5,5)) shape: {arr_module.shape}")
    
    norm_module = al.randn_array((100,), mu=0, sigma=1)
    print(f"  al.randn_array(100): mean={np.mean(norm_module):.4f}, std={np.std(norm_module):.4f}")
    
    int_module = al.randint_array((20,), low=1, high=10)
    print(f"  al.randint_array(20, 1, 10): {int_module}")
    
    print("\n📐 Large Array Performance:")
    
    # Generate a large array and measure time
    import time
    size = (1000, 1000)
    total_elements = size[0] * size[1]
    
    start = time.time()
    large_arr = rng.random_array(size)
    elapsed = time.time() - start
    
    print(f"  Generated {total_elements:,} random numbers in {elapsed:.2f} seconds")
    print(f"  Speed: {total_elements / elapsed / 1e6:.2f} million ops/sec")
    
    print("\n" + "=" * 70)
    print("✅ Array operations demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()