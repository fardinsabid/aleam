"""
Statistical tests for Aleam randomness quality.
"""

import pytest
import math
from collections import Counter
from aleam import Aleam


class TestStatisticalQuality:
    """Rigorous statistical tests for randomness"""
    
    def setup_method(self):
        self.rng = Aleam()
    
    def test_uniformity_chi_square(self):
        """Chi-square test for uniformity"""
        n_samples = 10000
        bins = 20
        observed = [0] * bins
        
        for _ in range(n_samples):
            r = self.rng.random()
            idx = min(int(r * bins), bins - 1)
            observed[idx] += 1
        
        expected = n_samples / bins
        chi_square = sum((o - expected) ** 2 / expected for o in observed)
        critical = 30.14  # 95% confidence with 19 df
        
        assert chi_square < critical, f"Chi-square {chi_square} >= {critical}"
    
    def test_autocorrelation(self):
        """Test for zero autocorrelation"""
        n_samples = 10000
        samples = [self.rng.random() for _ in range(n_samples)]
        mean = sum(samples) / n_samples
        var = sum((x - mean) ** 2 for x in samples) / n_samples
        
        max_autocorr = 0
        for lag in range(1, 21):
            num = sum((samples[i] - mean) * (samples[i + lag] - mean) 
                      for i in range(n_samples - lag))
            denom = var * (n_samples - lag)
            autocorr = num / denom if denom != 0 else 0
            max_autocorr = max(max_autocorr, abs(autocorr))
        
        assert max_autocorr < 0.05, f"Max autocorrelation {max_autocorr} >= 0.05"
    
    def test_integer_distribution(self):
        """Test integer distribution is uniform"""
        n_samples = 10000
        counts = Counter()
        
        for _ in range(n_samples):
            counts[self.rng.randint(0, 9)] += 1
        
        expected = n_samples / 10
        chi_square = sum((c - expected) ** 2 / expected for c in counts.values())
        critical = 16.92  # 95% confidence with 9 df
        
        assert chi_square < critical, f"Integer chi-square {chi_square} >= {critical}"
    
    def test_mean_convergence(self):
        """Test that sample mean converges to 0.5"""
        n_samples = 100000
        samples = [self.rng.random() for _ in range(n_samples)]
        mean = sum(samples) / n_samples
        
        assert abs(mean - 0.5) < 0.01, f"Mean {mean} too far from 0.5"
    
    def test_variance_convergence(self):
        """Test that sample variance converges to 1/12 ≈ 0.08333"""
        n_samples = 100000
        samples = [self.rng.random() for _ in range(n_samples)]
        mean = sum(samples) / n_samples
        variance = sum((x - mean) ** 2 for x in samples) / n_samples
        
        assert abs(variance - 0.08333) < 0.005, f"Variance {variance} too far from 0.08333"
    
    def test_pi_estimation(self):
        """Monte Carlo π estimation"""
        n_samples = 100000
        inside = 0
        
        for _ in range(n_samples):
            x = self.rng.uniform(-1, 1)
            y = self.rng.uniform(-1, 1)
            if x*x + y*y <= 1:
                inside += 1
        
        pi_estimate = 4 * inside / n_samples
        error = abs(pi_estimate - math.pi)
        
        assert error < 0.02, f"π error {error} too large"
    
    def test_normal_distribution(self):
        """Test normal distribution properties"""
        n_samples = 10000
        samples = [self.rng.gauss(0, 1) for _ in range(n_samples)]
        mean = sum(samples) / n_samples
        variance = sum((x - mean) ** 2 for x in samples) / n_samples
        
        assert abs(mean) < 0.05, f"Normal mean {mean} too far from 0"
        assert abs(variance - 1) < 0.1, f"Normal variance {variance} too far from 1"
    
    def test_exponential_distribution(self):
        """Test exponential distribution properties"""
        n_samples = 10000
        rate = 2.0
        samples = [self.rng.exponential(rate) for _ in range(n_samples)]
        mean = sum(samples) / n_samples
        expected_mean = 1 / rate
        
        assert abs(mean - expected_mean) < 0.05, f"Exponential mean {mean} too far from {expected_mean}"
    
    def test_gamma_distribution(self):
        """Test gamma distribution properties"""
        n_samples = 10000
        shape, scale = 3.0, 2.0
        samples = [self.rng.gamma(shape, scale) for _ in range(n_samples)]
        mean = sum(samples) / n_samples
        expected_mean = shape * scale
        
        assert abs(mean - expected_mean) < 0.2, f"Gamma mean {mean} too far from {expected_mean}"
    
    def test_poisson_distribution(self):
        """Test Poisson distribution properties"""
        n_samples = 10000
        lam = 5.0
        samples = [self.rng.poisson(lam) for _ in range(n_samples)]
        mean = sum(samples) / n_samples
        variance = sum((x - mean) ** 2 for x in samples) / n_samples
        
        assert abs(mean - lam) < 0.2, f"Poisson mean {mean} too far from {lam}"
        assert abs(variance - lam) < 0.5, f"Poisson variance {variance} too far from {lam}"