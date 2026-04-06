"""
Unit tests for core Aleam functionality.
"""

import pytest
import math
import numpy as np
import aleam


class TestAleamCore:
    """Test core random generation functionality."""
    
    def test_random_range(self):
        """Test that random() returns values in [0, 1)."""
        for _ in range(1000):
            val = aleam.random()
            assert 0 <= val < 1
    
    def test_randint_range(self):
        """Test randint returns values within bounds."""
        a, b = 10, 20
        for _ in range(1000):
            val = aleam.randint(a, b)
            assert a <= val <= b
    
    def test_randint_inclusive(self):
        """Test that both endpoints are possible."""
        vals = set()
        for _ in range(1000):
            vals.add(aleam.randint(0, 5))
        assert len(vals) == 6
    
    def test_choice(self):
        """Test choice returns element from sequence."""
        seq = [1, 2, 3, 4, 5]
        for _ in range(100):
            val = aleam.choice(seq)
            assert val in seq
    
    def test_uniform_range(self):
        """Test uniform returns values within bounds."""
        low, high = 5.0, 10.0
        for _ in range(1000):
            val = aleam.uniform(low, high)
            assert low <= val <= high
    
    def test_gauss_shape(self):
        """Test gauss returns finite values."""
        for _ in range(1000):
            val = aleam.gauss(0, 1)
            assert isinstance(val, float)
            assert not math.isnan(val)
            assert not math.isinf(val)
    
    def test_sample_unique(self):
        """Test sample returns unique elements."""
        population = list(range(100))
        sample = aleam.sample(population, 50)
        assert len(sample) == 50
        assert len(set(sample)) == 50
    
    def test_shuffle(self):
        """Test shuffle modifies list."""
        lst = list(range(100))
        original = lst.copy()
        aleam.shuffle(lst)
        assert lst != original
        assert sorted(lst) == sorted(original)
    
    def test_random_bytes_length(self):
        """Test random_bytes returns correct length."""
        for n in [1, 8, 16, 32]:
            bytes_val = aleam.random_bytes(n)
            assert len(bytes_val) == n
    
    def test_seed_free_raises(self):
        """Test that seed_free raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            aleam.seed_free()


class TestAleamInstance:
    """Test instance methods of AleamCore."""
    
    def setup_method(self):
        self.rng = aleam.Aleam()
    
    def test_instance_random(self):
        for _ in range(100):
            val = self.rng.random()
            assert 0 <= val < 1
    
    def test_instance_randint(self):
        for _ in range(100):
            val = self.rng.randint(1, 100)
            assert 1 <= val <= 100
    
    def test_instance_gauss(self):
        for _ in range(100):
            val = self.rng.gauss(0, 1)
            assert isinstance(val, float)
    
    def test_instance_uniform(self):
        for _ in range(100):
            val = self.rng.uniform(5, 10)
            assert 5 <= val <= 10


class TestAleamDistributions:
    """Test distribution methods."""
    
    def test_exponential(self):
        samples = [aleam.exponential(1.0) for _ in range(1000)]
        assert all(s >= 0 for s in samples)
        mean = sum(samples) / len(samples)
        assert abs(mean - 1.0) < 0.1
    
    def test_beta(self):
        samples = [aleam.beta(2, 5) for _ in range(1000)]
        assert all(0 <= s <= 1 for s in samples)
        mean = sum(samples) / len(samples)
        expected_mean = 2 / 7
        assert abs(mean - expected_mean) < 0.05
    
    def test_gamma(self):
        samples = [aleam.gamma(2, 1) for _ in range(1000)]
        assert all(s >= 0 for s in samples)
        mean = sum(samples) / len(samples)
        assert abs(mean - 2) < 0.2
    
    def test_poisson(self):
        samples = [aleam.poisson(5) for _ in range(1000)]
        assert all(s >= 0 for s in samples)
        mean = sum(samples) / len(samples)
        assert abs(mean - 5) < 0.5
    
    def test_laplace(self):
        samples = [aleam.laplace(0, 1) for _ in range(1000)]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.11
    
    def test_logistic(self):
        samples = [aleam.logistic(0, 1) for _ in range(1000)]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.11
    
    def test_lognormal(self):
        samples = [aleam.lognormal(0, 1) for _ in range(1000)]
        assert all(s > 0 for s in samples)
    
    def test_weibull(self):
        samples = [aleam.weibull(1.5, 1) for _ in range(1000)]
        assert all(s >= 0 for s in samples)
    
    def test_pareto(self):
        samples = [aleam.pareto(2, 1) for _ in range(1000)]
        assert all(s >= 1 for s in samples)
    
    def test_chi_square(self):
        samples = [aleam.chi_square(5) for _ in range(1000)]
        assert all(s >= 0 for s in samples)
        mean = sum(samples) / len(samples)
        assert abs(mean - 5) < 0.5
    
    def test_student_t(self):
        samples = [aleam.student_t(3) for _ in range(1000)]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.21
    
    def test_f_distribution(self):
        samples = [aleam.f_distribution(5, 10) for _ in range(1000)]
        assert all(s >= 0 for s in samples)
    
    def test_dirichlet(self):
        alpha = [1, 2, 3]
        sample = aleam.dirichlet(alpha)
        assert len(sample) == 3
        assert all(s > 0 for s in sample)
        assert abs(sum(sample) - 1.0) < 1e-10


class TestAleamArrayOps:
    """Test array operations."""
    
    def test_random_array_1d(self):
        arr = aleam.random_array((100,))
        assert len(arr) == 100
        assert all(0 <= x < 1 for x in arr)
    
    def test_random_array_2d(self):
        arr = aleam.random_array((10, 10))
        assert arr.shape == (10, 10)
        assert all(0 <= x < 1 for x in arr.flatten())
    
    def test_randn_array(self):
        arr = aleam.randn_array((100,), 0, 1)
        assert len(arr) == 100
        mean = np.mean(arr)
        std = np.std(arr)
        assert abs(mean) < 0.21
        assert abs(std - 1) < 0.2
    
    def test_randint_array(self):
        arr = aleam.randint_array((50,), 0, 10)
        assert len(arr) == 50
        assert all(0 <= x <= 10 for x in arr)