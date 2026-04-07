#!/usr/bin/env python3
"""
Statistical distributions examples for Aleam (C++ Core).

This example demonstrates all the statistical distributions
available in Aleam including normal, exponential, beta, gamma,
Poisson, and many more.

Note: The C++ core provides distribution functions directly.
The distribution classes are not available in the C++ version.
"""

import aleam as al
import numpy as np


def main():
    print("=" * 70)
    print("Aleam - Statistical Distributions Examples (C++ Core)")
    print("=" * 70)
    
    rng = al.Aleam()
    
    print("\n Distribution Sampling (via module-level functions):")
    print("-" * 60)
    
    # Normal (Gaussian)
    normal_samples = [rng.gauss(0, 1) for _ in range(1000)]
    print(f"  Normal(0,1): mean={np.mean(normal_samples):.4f}, std={np.std(normal_samples):.4f}")
    
    # Uniform
    uniform_samples = [rng.uniform(5, 10) for _ in range(1000)]
    print(f"  Uniform(5,10): mean={np.mean(uniform_samples):.4f}, min={np.min(uniform_samples):.4f}, max={np.max(uniform_samples):.4f}")
    
    # Exponential
    exp_samples = [rng.exponential(1.0) for _ in range(1000)]
    print(f"  Exponential(rate=1.0): mean={np.mean(exp_samples):.4f} (expected 1.0)")
    
    # Beta
    beta_samples = [rng.beta(2, 5) for _ in range(1000)]
    print(f"  Beta(alpha=2, beta=5): mean={np.mean(beta_samples):.4f} (expected 0.2857)")
    
    # Gamma
    gamma_samples = [rng.gamma(3, 2) for _ in range(1000)]
    print(f"  Gamma(shape=3, scale=2): mean={np.mean(gamma_samples):.4f} (expected 6.0)")
    
    # Poisson
    poisson_samples = [rng.poisson(5) for _ in range(1000)]
    print(f"  Poisson(lam=5): mean={np.mean(poisson_samples):.4f} (expected 5.0)")
    
    # Laplace
    laplace_samples = [rng.laplace(0, 1) for _ in range(1000)]
    print(f"  Laplace(loc=0, scale=1): mean={np.mean(laplace_samples):.4f} (expected 0.0)")
    
    # Logistic
    logistic_samples = [rng.logistic(0, 1) for _ in range(1000)]
    print(f"  Logistic(loc=0, scale=1): mean={np.mean(logistic_samples):.4f} (expected 0.0)")
    
    # Log-Normal
    lognormal_samples = [rng.lognormal(0, 1) for _ in range(1000)]
    print(f"  LogNormal(mu=0, sigma=1): mean={np.mean(lognormal_samples):.4f} (expected 1.6487)")
    
    # Weibull
    weibull_samples = [rng.weibull(1.5, 1) for _ in range(1000)]
    print(f"  Weibull(shape=1.5, scale=1): mean={np.mean(weibull_samples):.4f}")
    
    # Pareto
    pareto_samples = [rng.pareto(2, 1) for _ in range(1000)]
    print(f"  Pareto(alpha=2, scale=1): mean={np.mean(pareto_samples):.4f} (expected 2.0)")
    
    # Chi-square
    chi2_samples = [rng.chi_square(5) for _ in range(1000)]
    print(f"  Chi-square(df=5): mean={np.mean(chi2_samples):.4f} (expected 5.0)")
    
    # Student's t
    t_samples = [rng.student_t(3) for _ in range(1000)]
    print(f"  Student's t(df=3): mean={np.mean(t_samples):.4f} (expected 0.0)")
    
    # F-distribution
    f_samples = [rng.f_distribution(5, 10) for _ in range(1000)]
    print(f"  F-distribution(df1=5, df2=10): mean={np.mean(f_samples):.4f} (expected 1.25)")
    
    # Dirichlet
    dirichlet_samples = [rng.dirichlet([1, 2, 3]) for _ in range(10)]
    print(f"  Dirichlet(alpha=[1,2,3]): first sample = {[f'{x:.4f}' for x in dirichlet_samples[0]]}")
    print(f"  Dirichlet sum: {sum(dirichlet_samples[0]):.4f} (expected 1.0)")
    
    print("\n Distribution Parameters (using functions directly):")
    print("-" * 60)
    print("  Note: Use module-level functions for distribution sampling.")
    print("  The distribution classes are not available in the C++ core.")
    
    print("\n  Examples of direct function calls:")
    print(f"    rng.gauss(0, 1)        -> {rng.gauss(0, 1):.4f}")
    print(f"    rng.uniform(0, 1)      -> {rng.uniform(0, 1):.4f}")
    print(f"    rng.exponential(2.0)   -> {rng.exponential(2.0):.4f}")
    print(f"    rng.beta(2, 5)         -> {rng.beta(2, 5):.4f}")
    print(f"    rng.gamma(3, 2)        -> {rng.gamma(3, 2):.4f}")
    print(f"    rng.poisson(5)         -> {rng.poisson(5)}")
    print(f"    rng.laplace(0, 1)      -> {rng.laplace(0, 1):.4f}")
    print(f"    rng.logistic(0, 1)     -> {rng.logistic(0, 1):.4f}")
    print(f"    rng.lognormal(0, 1)    -> {rng.lognormal(0, 1):.4f}")
    print(f"    rng.weibull(1.5, 1)    -> {rng.weibull(1.5, 1):.4f}")
    print(f"    rng.pareto(2, 1)       -> {rng.pareto(2, 1):.4f}")
    print(f"    rng.chi_square(5)      -> {rng.chi_square(5):.4f}")
    print(f"    rng.student_t(3)       -> {rng.student_t(3):.4f}")
    print(f"    rng.f_distribution(5, 10) -> {rng.f_distribution(5, 10):.4f}")
    print(f"    rng.dirichlet([1,2,3]) -> {[f'{x:.3f}' for x in rng.dirichlet([1,2,3])]}")
    
    print("\n Statistical Verification (Large Samples):")
    print("-" * 60)
    
    # Large sample verification
    n_samples = 10000
    
    normal_large = [rng.gauss(0, 1) for _ in range(n_samples)]
    print(f"  Normal(0,1) over {n_samples:,} samples:")
    print(f"    Mean: {np.mean(normal_large):.4f} (expected 0.0)")
    print(f"    Std: {np.std(normal_large):.4f} (expected 1.0)")
    
    uniform_large = [rng.uniform(0, 1) for _ in range(n_samples)]
    print(f"  Uniform(0,1) over {n_samples:,} samples:")
    print(f"    Mean: {np.mean(uniform_large):.4f} (expected 0.5)")
    print(f"    Var: {np.var(uniform_large):.4f} (expected 0.08333)")
    
    exp_large = [rng.exponential(1.0) for _ in range(n_samples)]
    print(f"  Exponential(1.0) over {n_samples:,} samples:")
    print(f"    Mean: {np.mean(exp_large):.4f} (expected 1.0)")
    
    poisson_large = [rng.poisson(5) for _ in range(n_samples)]
    print(f"  Poisson(5) over {n_samples:,} samples:")
    print(f"    Mean: {np.mean(poisson_large):.4f} (expected 5.0)")
    print(f"    Var: {np.var(poisson_large):.4f} (expected 5.0)")
    
    print("\n" + "=" * 70)
    print(" Distributions demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()