#!/usr/bin/env python3
"""
Basic usage examples for Aleam (C++ Core).

This example demonstrates the fundamental random number generation
capabilities of Aleam including basic random numbers, integers,
sequences, and distributions.

Note: The C++ API uses numpy arrays for batch operations.
"""

import aleam as al
import numpy as np


def main():
    print("=" * 60)
    print("Aleam - Basic Usage Examples (C++ Core)")
    print("=" * 60)
    
    # Create a true random generator
    rng = al.Aleam()
    
    print("\n Basic Random Numbers:")
    print(f"  Random float: {rng.random():.6f}")
    print(f"  Random uint64: {rng.random_uint64()}")
    print(f"  Random int (1-100): {rng.randint(1, 100)}")
    print(f"  Random choice: {rng.choice(['AI', 'ML', 'DL', 'Aleam'])}")
    print(f"  Random uniform (5, 10): {rng.uniform(5, 10):.4f}")
    print(f"  Random normal (0,1): {rng.gauss():.4f}")
    
    print("\n Sampling Without Replacement:")
    print("  Note: sample() requires a list, not a range object")
    population = list(range(100))
    sample = rng.sample(population, 10)
    print(f"  Sample of 10 from 100: {sample}")
    
    print("\n Shuffling a List:")
    items = ['apple', 'banana', 'cherry', 'date', 'elderberry']
    print(f"  Original: {items}")
    rng.shuffle(items)
    print(f"  Shuffled: {items}")
    
    print("\n Random Bytes:")
    print(f"  8 random bytes: {rng.random_bytes(8).hex()}")
    print(f"  16 random bytes: {rng.random_bytes(16).hex()}")
    
    print("\n Batch Generation (100 numbers in one call):")
    batch = rng.random_batch(100)
    print(f"  Type: {type(batch)}")
    print(f"  Shape: {batch.shape}")
    print(f"  First 10: {[f'{x:.4f}' for x in batch[:10]]}")
    print(f"  Mean: {np.mean(batch):.6f}")
    print(f"  Expected mean: 0.5")
    
    print("\n Generator Statistics:")
    stats = rng.get_stats()
    print(f"  calls: {stats.calls}")
    print(f"  batch_size: {stats.batch_size}")
    print(f"  cache_hits: {stats.cache_hits}")
    print(f"  cache_misses: {stats.cache_misses}")
    print(f"  algorithm: {stats.algorithm}")
    print(f"  entropy_source: {stats.entropy_source}")
    print(f"  entropy_bits_per_call: {stats.entropy_bits_per_call}")
    
    print("\n Thread-Local Instance:")
    # Get thread-local instance for multi-threaded code
    tl_rng = al.get_thread_local_instance()
    print(f"  Thread-local random: {tl_rng.random():.6f}")
    print(f"  Same as default? {tl_rng is rng}")
    
    print("\n" + "=" * 60)
    print(" Basic usage complete")
    print("=" * 60)


if __name__ == "__main__":
    main()