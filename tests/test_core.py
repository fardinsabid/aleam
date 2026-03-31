"""
Unit tests for core Aleam functionality.
"""

import pytest
import math
from aleam import Aleam, AleamFast


class TestAleamCore:
    """Test core random generation"""
    
    def setup_method(self):
        self.rng = Aleam()
    
    def test_random_range(self):
        """Test that random() returns values in [0, 1)"""
        for _ in range(1000):
            val = self.rng.random()
            assert 0 <= val < 1
    
    def test_randint_range(self):
        """Test randint returns values within bounds"""
        a, b = 10, 20
        for _ in range(1000):
            val = self.rng.randint(a, b)
            assert a <= val <= b
    
    def test_randint_inclusive(self):
        """Test that both endpoints are possible"""
        vals = set()
        for _ in range(1000):
            vals.add(self.rng.randint(0, 5))
        assert len(vals) == 6
    
    def test_choice(self):
        """Test choice returns element from sequence"""
        seq = [1, 2, 3, 4, 5]
        val = self.rng.choice(seq)
        assert val in seq
    
    def test_uniform_range(self):
        """Test uniform returns values within bounds"""
        low, high = 5.0, 10.0
        for _ in range(1000):
            val = self.rng.uniform(low, high)
            assert low <= val <= high
    
    def test_gauss_shape(self):
        """Test gauss returns finite values"""
        for _ in range(1000):
            val = self.rng.gauss(0, 1)
            assert isinstance(val, float)
            assert not math.isnan(val)
            assert not math.isinf(val)
    
    def test_sample_unique(self):
        """Test sample returns unique elements"""
        population = list(range(100))
        sample = self.rng.sample(population, 50)
        assert len(sample) == 50
        assert len(set(sample)) == 50
    
    def test_shuffle(self):
        """Test shuffle modifies list"""
        lst = list(range(100))
        original = lst.copy()
        self.rng.shuffle(lst)
        assert lst != original
        assert sorted(lst) == sorted(original)
    
    def test_random_bytes_length(self):
        """Test random_bytes returns correct length"""
        for n in [1, 8, 16, 32, 64]:
            bytes_val = self.rng.random_bytes(n)
            assert len(bytes_val) == n
    
    def test_seed_raises(self):
        """Test that seed raises NotImplementedError"""
        with pytest.raises(NotImplementedError):
            self.rng.seed(42)
    
    def test_stateless(self):
        """Test that multiple calls are independent"""
        vals1 = [self.rng.random() for _ in range(100)]
        vals2 = [self.rng.random() for _ in range(100)]
        assert vals1 != vals2


class TestAleamFast:
    """Test fast version"""
    
    def setup_method(self):
        self.rng = AleamFast()
    
    def test_random_range(self):
        for _ in range(1000):
            assert 0 <= self.rng.random() < 1
    
    def test_randint(self):
        for _ in range(1000):
            assert 0 <= self.rng.randint(0, 10) <= 10


class TestAleamDistributions:
    """Test distribution methods"""
    
    def setup_method(self):
        self.rng = Aleam()
    
    def test_exponential(self):
        """Test exponential distribution"""
        samples = [self.rng.exponential(1.0) for _ in range(1000)]
        assert all(s >= 0 for s in samples)
        mean = sum(samples) / len(samples)
        assert abs(mean - 1.0) < 0.1  # Mean should be ~1/rate
    
    def test_beta(self):
        """Test beta distribution"""
        samples = [self.rng.beta(2, 5) for _ in range(1000)]
        assert all(0 <= s <= 1 for s in samples)
        mean = sum(samples) / len(samples)
        expected_mean = 2 / (2 + 5)
        assert abs(mean - expected_mean) < 0.05
    
    def test_gamma(self):
        """Test gamma distribution"""
        samples = [self.rng.gamma(2, 1) for _ in range(1000)]
        assert all(s >= 0 for s in samples)
        mean = sum(samples) / len(samples)
        assert abs(mean - 2) < 0.2  # Mean = shape * scale
    
    def test_poisson(self):
        """Test poisson distribution"""
        samples = [self.rng.poisson(5) for _ in range(1000)]
        assert all(s >= 0 for s in samples)
        mean = sum(samples) / len(samples)
        assert abs(mean - 5) < 0.5
    
    def test_laplace(self):
        """Test laplace distribution"""
        samples = [self.rng.laplace(0, 1) for _ in range(1000)]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.1
    
    def test_logistic(self):
        """Test logistic distribution"""
        samples = [self.rng.logistic(0, 1) for _ in range(1000)]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.1
    
    def test_lognormal(self):
        """Test lognormal distribution"""
        samples = [self.rng.lognormal(0, 1) for _ in range(1000)]
        assert all(s > 0 for s in samples)
        mean = sum(samples) / len(samples)
        assert mean > 0
    
    def test_weibull(self):
        """Test weibull distribution"""
        samples = [self.rng.weibull(1.5, 1) for _ in range(1000)]
        assert all(s >= 0 for s in samples)
    
    def test_pareto(self):
        """Test pareto distribution"""
        samples = [self.rng.pareto(2, 1) for _ in range(1000)]
        assert all(s >= 1 for s in samples)
    
    def test_chi_square(self):
        """Test chi-square distribution"""
        samples = [self.rng.chi_square(5) for _ in range(1000)]
        assert all(s >= 0 for s in samples)
        mean = sum(samples) / len(samples)
        assert abs(mean - 5) < 0.5  # Mean = df
    
    def test_student_t(self):
        """Test student's t distribution"""
        samples = [self.rng.student_t(3) for _ in range(10000)]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.2  # Wider tolerance for t-distribution
    
    def test_f_distribution(self):
        """Test F distribution"""
        samples = [self.rng.f_distribution(5, 10) for _ in range(1000)]
        assert all(s >= 0 for s in samples)
        mean = sum(samples) / len(samples)
        expected_mean = 10 / (10 - 2)  # For df2 > 2
        assert abs(mean - expected_mean) < 0.5
    
    def test_dirichlet(self):
        """Test dirichlet distribution"""
        alpha = [1, 2, 3]
        sample = self.rng.dirichlet(alpha)
        assert len(sample) == 3
        assert all(s > 0 for s in sample)
        assert abs(sum(sample) - 1.0) < 1e-10


class TestAleamArrayOps:
    """Test array operations"""
    
    def setup_method(self):
        self.rng = Aleam()
    
    def test_random_array_1d(self):
        arr = self.rng.random_array(100)
        assert len(arr) == 100
        assert all(0 <= x < 1 for x in arr)
    
    def test_random_array_2d(self):
        arr = self.rng.random_array((10, 10))
        assert len(arr) == 10
        assert len(arr[0]) == 10
    
    def test_randn_array(self):
        arr = self.rng.randn_array((100,), 0, 1)
        assert len(arr) == 100
        mean = sum(arr) / len(arr)
        assert abs(mean) < 0.2
    
    def test_randint_array(self):
        arr = self.rng.randint_array((50,), 0, 10)
        assert len(arr) == 50
        assert all(0 <= x <= 10 for x in arr)


class TestModuleFunctions:
    """Test module-level convenience functions"""
    
    def setup_method(self):
        from aleam import core
        # Reset default RNG to avoid interference
        core._default_rng = None
    
    def test_random_function(self):
        from aleam import random
        val = random()
        assert 0 <= val < 1
    
    def test_randint_function(self):
        from aleam import randint
        val = randint(1, 10)
        assert 1 <= val <= 10
    
    def test_choice_function(self):
        from aleam import choice
        val = choice([1, 2, 3])
        assert val in [1, 2, 3]
    
    def test_gauss_function(self):
        from aleam import gauss
        val = gauss(0, 1)
        assert isinstance(val, float)
    
    def test_shuffle_function(self):
        from aleam import shuffle
        lst = [1, 2, 3, 4, 5]
        original = lst.copy()
        shuffle(lst)
        assert lst != original
        assert sorted(lst) == sorted(original)