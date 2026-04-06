#!/usr/bin/env python3
"""
Statistical distributions examples for Aleam.

This example demonstrates all the statistical distributions
available in Aleam including normal, exponential, beta, gamma,
Poisson, and many more.
"""

import aleam as al
import numpy as np


def main():
    print("=" * 70)
    print("Aleam - Statistical Distributions Examples")
    print("=" * 70)
    
    rng = al.Aleam()
    
    print("\n📊 Distribution Sampling:")
    
    # Normal (Gaussian)
    normal_samples = [rng.gauss(0, 1) for _ in range(1000)]
    print(f"  Normal(0,1): mean={np.mean(normal_samples):.4f}, std={np.std(normal_samples):.4f}")
    
    # Uniform
    uniform_samples = [rng.uniform(5, 10) for _ in range(1000)]
    print(f"  Uniform(5,10): mean={np.mean(uniform_samples):.4f}, min={np.min(uniform_samples):.4f}, max={np.max(uniform_samples):.4f}")
    
    # Exponential
    exp_samples = [rng.exponential(rate=1.0) for _ in range(1000)]
    print(f"  Exponential(rate=1): mean={np.mean(exp_samples):.4f} (expected 1.0)")
    
    # Beta
    beta_samples = [rng.beta(alpha=2, beta=5) for _ in range(1000)]
    print(f"  Beta(alpha=2, beta=5): mean={np.mean(beta_samples):.4f} (expected 0.2857)")
    
    # Gamma
    gamma_samples = [rng.gamma(shape=3, scale=2) for _ in range(1000)]
    print(f"  Gamma(shape=3, scale=2): mean={np.mean(gamma_samples):.4f} (expected 6.0)")
    
    # Poisson
    poisson_samples = [rng.poisson(lam=5) for _ in range(1000)]
    print(f"  Poisson(lam=5): mean={np.mean(poisson_samples):.4f} (expected 5.0)")
    
    # Laplace
    laplace_samples = [rng.laplace(loc=0, scale=1) for _ in range(1000)]
    print(f"  Laplace(loc=0, scale=1): mean={np.mean(laplace_samples):.4f} (expected 0.0)")
    
    # Logistic
    logistic_samples = [rng.logistic(loc=0, scale=1) for _ in range(1000)]
    print(f"  Logistic(loc=0, scale=1): mean={np.mean(logistic_samples):.4f} (expected 0.0)")
    
    # Log-Normal
    lognormal_samples = [rng.lognormal(mu=0, sigma=1) for _ in range(1000)]
    print(f"  LogNormal(mu=0, sigma=1): mean={np.mean(lognormal_samples):.4f} (expected 1.6487)")
    
    # Weibull
    weibull_samples = [rng.weibull(shape=1.5, scale=1) for _ in range(1000)]
    print(f"  Weibull(shape=1.5, scale=1): mean={np.mean(weibull_samples):.4f}")
    
    # Pareto
    pareto_samples = [rng.pareto(alpha=2, scale=1) for _ in range(1000)]
    print(f"  Pareto(alpha=2, scale=1): mean={np.mean(pareto_samples):.4f} (expected 2.0)")
    
    # Chi-square
    chi2_samples = [rng.chi_square(df=5) for _ in range(1000)]
    print(f"  Chi-square(df=5): mean={np.mean(chi2_samples):.4f} (expected 5.0)")
    
    # Student's t
    t_samples = [rng.student_t(df=3) for _ in range(1000)]
    print(f"  Student's t(df=3): mean={np.mean(t_samples):.4f} (expected 0.0)")
    
    # F-distribution
    f_samples = [rng.f_distribution(df1=5, df2=10) for _ in range(1000)]
    print(f"  F-distribution(df1=5, df2=10): mean={np.mean(f_samples):.4f} (expected 1.25)")
    
    # Dirichlet
    dirichlet_samples = [rng.dirichlet([1, 2, 3]) for _ in range(10)]
    print(f"  Dirichlet(alpha=[1,2,3]): first sample = {[f'{x:.4f}' for x in dirichlet_samples[0]]}")
    
    print("\n📊 Distribution Class API:")
    
    # Using distribution classes
    from aleam.distributions import Normal, Uniform, Exponential, Beta, Gamma, Poisson
    
    # Normal distribution
    norm_dist = Normal(mu=0, sigma=1, rng=rng)
    norm_sample = norm_dist.sample()
    print(f"  Normal.sample(): {norm_sample:.4f}")
    print(f"  Normal.pdf(0): {norm_dist.pdf(0):.4f}")
    
    # Uniform distribution
    uniform_dist = Uniform(low=0, high=10, rng=rng)
    uniform_sample = uniform_dist.sample()
    print(f"  Uniform.sample(): {uniform_sample:.4f}")
    print(f"  Uniform.pdf(5): {uniform_dist.pdf(5):.4f}")
    
    # Exponential distribution
    exp_dist = Exponential(rate=2.0, rng=rng)
    exp_sample = exp_dist.sample()
    print(f"  Exponential.sample(): {exp_sample:.4f}")
    
    # Beta distribution
    beta_dist = Beta(alpha=2, beta=5, rng=rng)
    beta_sample = beta_dist.sample()
    print(f"  Beta.sample(): {beta_sample:.4f}")
    
    # Gamma distribution
    gamma_dist = Gamma(shape=3, scale=2, rng=rng)
    gamma_sample = gamma_dist.sample()
    print(f"  Gamma.sample(): {gamma_sample:.4f}")
    
    # Poisson distribution
    poisson_dist = Poisson(lam=5, rng=rng)
    poisson_sample = poisson_dist.sample()
    print(f"  Poisson.sample(): {poisson_sample}")
    
    print("\n" + "=" * 70)
    print("✅ Distributions demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()