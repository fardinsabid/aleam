#!/usr/bin/env python3
"""
Monte Carlo π estimation using Aleam true randomness (C++ Core).

This example demonstrates using Aleam's true random numbers for
Monte Carlo simulation to estimate the value of π.
The more random points used, the closer the estimate gets to π.
"""

import aleam as al
import math
import time


def estimate_pi(n_points, rng):
    """
    Estimate π using Monte Carlo method.
    
    The method: Generate random points in a square [-1, 1] x [-1, 1].
    Count how many fall inside the unit circle (x² + y² ≤ 1).
    π = 4 × (points inside circle) / (total points)
    
    Args:
        n_points: Number of random points to generate
        rng: Aleam random number generator
    
    Returns:
        float: Estimated value of π
    """
    inside = 0
    
    for _ in range(n_points):
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    
    return 4 * inside / n_points


def estimate_pi_batch(n_points, rng):
    """
    Estimate π using Monte Carlo method with batch generation.
    More efficient for large point counts.
    
    Args:
        n_points: Number of random points to generate
        rng: Aleam random number generator
    
    Returns:
        float: Estimated value of π
    """
    # Generate all x and y coordinates in batches for efficiency
    batch_size = 10000
    inside = 0
    points_processed = 0
    
    while points_processed < n_points:
        current_batch = min(batch_size, n_points - points_processed)
        
        # Generate x and y coordinates in batches
        x_batch = [rng.uniform(-1, 1) for _ in range(current_batch)]
        y_batch = [rng.uniform(-1, 1) for _ in range(current_batch)]
        
        # Count points inside circle
        for i in range(current_batch):
            if x_batch[i]*x_batch[i] + y_batch[i]*y_batch[i] <= 1:
                inside += 1
        
        points_processed += current_batch
    
    return 4 * inside / n_points


def main():
    print("=" * 70)
    print("Aleam - Monte Carlo π Estimation (C++ Core)")
    print("=" * 70)
    
    rng = al.Aleam()
    
    print("\n Estimating π using true random points in a square:")
    print("   π = 4 × (points inside circle) / (total points)")
    print("   The law of large numbers guarantees convergence with more points.")
    print("   True randomness ensures unbiased sampling.\n")
    
    sizes = [1000, 10000, 100000, 500000, 1000000]
    
    print("  Points     | π Estimate | Error      | Time (s) | Accuracy")
    print("  -----------|------------|------------|----------|-------------")
    
    for size in sizes:
        start = time.time()
        pi_est = estimate_pi(size, rng)
        elapsed = time.time() - start
        error = abs(pi_est - math.pi)
        
        # Visual indicator of accuracy
        if error < 0.001:
            accuracy = " Excellent"
        elif error < 0.01:
            accuracy = " Good"
        elif error < 0.1:
            accuracy = " Acceptable"
        else:
            accuracy = " Poor"
        
        print(f"  {size:>10,} | {pi_est:.8f} | {error:.8f} | {elapsed:>8.2f} | {accuracy}")
    
    # Batch version for large N
    print("\n" + "=" * 70)
    print(" Large Scale Monte Carlo (Batch Processing)")
    print("=" * 70)
    
    large_size = 10_000_000
    print(f"\n  Estimating π with {large_size:,} points using batch processing...")
    
    start = time.time()
    pi_est_batch = estimate_pi_batch(large_size, rng)
    elapsed = time.time() - start
    error = abs(pi_est_batch - math.pi)
    
    print(f"  π estimate: {pi_est_batch:.8f}")
    print(f"  Error: {error:.8f}")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Speed: {large_size / elapsed / 1e6:.2f} million points/sec")
    
    # Statistical analysis
    print("\n" + "=" * 70)
    print(" Statistical Analysis")
    print("=" * 70)
    
    # Run multiple trials to see variance
    n_trials = 10
    points_per_trial = 100000
    estimates = []
    
    print(f"\n  Running {n_trials} trials with {points_per_trial:,} points each...")
    
    for i in range(n_trials):
        pi_est = estimate_pi(points_per_trial, rng)
        estimates.append(pi_est)
    
    mean_est = sum(estimates) / len(estimates)
    std_est = (sum((x - mean_est) ** 2 for x in estimates) / len(estimates)) ** 0.5
    
    print(f"  Mean π estimate: {mean_est:.6f}")
    print(f"  Standard deviation: {std_est:.6f}")
    print(f"  Error of mean: {abs(mean_est - math.pi):.6f}")
    print(f"  Expected error ~ 1/√N = {1 / (points_per_trial ** 0.5):.6f}")
    
    # Convergence demonstration
    print("\n" + "=" * 70)
    print(" Convergence Demonstration")
    print("=" * 70)
    print("\n  As N increases, error should decrease like 1/√N")
    print("  This demonstrates the law of large numbers with true randomness.\n")
    
    convergence_sizes = [1000, 10000, 100000, 1000000]
    print("  N         | Error      | Expected 1/√N")
    print("  ----------|------------|---------------")
    
    for n in convergence_sizes:
        pi_est = estimate_pi(n, rng)
        error = abs(pi_est - math.pi)
        expected = 1 / (n ** 0.5)
        print(f"  {n:>9,} | {error:.6f} | {expected:.6f}")
    
    print("\n" + "=" * 70)
    print(" Monte Carlo estimation complete")
    print("=" * 70)
    print("\n  Key takeaways:")
    print("  1. True randomness ensures unbiased sampling")
    print("  2. Error decreases as 1/√N (law of large numbers)")
    print("  3. Batch processing improves performance for large N")
    print("  4. Monte Carlo methods work best with true randomness")


if __name__ == "__main__":
    main()