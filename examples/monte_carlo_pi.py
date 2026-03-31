"""
Monte Carlo π estimation using Aleam true randomness.
"""

import aleam as al
import math


def estimate_pi(n_points, rng):
    """Estimate π using Monte Carlo method"""
    inside = 0
    
    for _ in range(n_points):
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    
    return 4 * inside / n_points


def main():
    print("=" * 70)
    print("ALEAM - Monte Carlo π Estimation")
    print("=" * 70)
    
    rng = al.Aleam()
    
    print("\n🎯 Estimating π using true random points in a square:")
    print("   π = 4 × (points inside circle) / (total points)")
    
    sizes = [1000, 10000, 100000, 500000, 1000000]
    
    print("\n  Points     | π Estimate | Error      | Time (s)")
    print("  -----------|------------|------------|---------")
    
    for size in sizes:
        import time
        start = time.time()
        pi_est = estimate_pi(size, rng)
        elapsed = time.time() - start
        error = abs(pi_est - math.pi)
        
        print(f"  {size:>10,} | {pi_est:.8f} | {error:.8f} | {elapsed:.2f}")
    
    print("\n" + "=" * 70)
    print("✅ Monte Carlo estimation complete")
    print("   As sample size increases, estimate converges to π")
    print("=" * 70)


if __name__ == "__main__":
    main()