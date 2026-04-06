#!/usr/bin/env python3
"""
Monte Carlo π estimation using Aleam true randomness.

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


def main():
    print("=" * 70)
    print("Aleam - Monte Carlo π Estimation")
    print("=" * 70)
    
    rng = al.Aleam()
    
    print("\n🎯 Estimating π using true random points in a square:")
    print("   π = 4 × (points inside circle) / (total points)")
    print("   The law of large numbers guarantees convergence with more points.\n")
    
    sizes = [1000, 10000, 100000, 500000, 1000000]
    
    print("  Points     | π Estimate | Error      | Time (s)")
    print("  -----------|------------|------------|---------")
    
    for size in sizes:
        start = time.time()
        pi_est = estimate_pi(size, rng)
        elapsed = time.time() - start
        error = abs(pi_est - math.pi)
        
        # Visual indicator of accuracy
        if error < 0.001:
            accuracy = "★ Excellent"
        elif error < 0.01:
            accuracy = "✓ Good"
        elif error < 0.1:
            accuracy = "○ Acceptable"
        else:
            accuracy = "△ Poor"
        
        print(f"  {size:>10,} | {pi_est:.8f} | {error:.8f} | {elapsed:.2f}s  {accuracy}")
    
    print("\n" + "=" * 70)
    print("✅ Monte Carlo estimation complete")
    print("   As sample size increases, estimate converges to π")
    print("   True randomness ensures unbiased sampling")
    print("=" * 70)


if __name__ == "__main__":
    main()