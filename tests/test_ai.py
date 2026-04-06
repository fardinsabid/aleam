"""
Unit tests for AI/ML features of Aleam.
"""

import pytest
import numpy as np
import aleam


class TestAIRandom:
    """Test AI-specific random utilities."""
    
    def setup_method(self):
        self.ai = aleam.AIRandom()
    
    def test_gradient_noise_shape(self):
        """Test gradient noise returns correct shape (flattened size)."""
        shape = 12  # 3 * 4 flattened
        noise = self.ai.gradient_noise(shape, scale=0.1)
        assert len(noise) == shape
        assert isinstance(noise, np.ndarray)
    
    def test_gradient_noise_stats(self):
        """Test gradient noise has correct statistical properties."""
        shape = 1000
        noise = self.ai.gradient_noise(shape, scale=0.1)
        # Wider tolerance for true randomness
        assert abs(np.mean(noise)) < 0.08
        assert abs(np.std(noise) - 0.1) < 0.08
    
    def test_latent_vector_normal(self):
        """Test latent vector with normal distribution."""
        dim = 128
        vec = self.ai.latent_vector(dim, "normal")
        assert len(vec) == dim
        assert isinstance(vec, np.ndarray)
        # Wider tolerance for true randomness
        assert abs(np.mean(vec)) < 0.3
    
    def test_latent_vector_uniform(self):
        """Test latent vector with uniform distribution."""
        dim = 128
        vec = self.ai.latent_vector(dim, "uniform")
        assert len(vec) == dim
        assert all(-1 <= v <= 1 for v in vec)
        assert abs(np.mean(vec)) < 0.2
    
    def test_latent_vector_invalid_distribution(self):
        """Test latent vector raises error for invalid distribution."""
        with pytest.raises(ValueError):
            self.ai.latent_vector(10, "invalid")
    
    def test_dropout_mask(self):
        """Test dropout mask generation."""
        size = 10000
        mask = self.ai.dropout_mask(size, 0.3)
        assert len(mask) == size
        assert all(v in (0, 1) for v in mask)
        # Use np.sum for numpy array (Python sum doesn't work well with uint8)
        active_pct = np.sum(mask) / size * 100
        assert 25 < active_pct < 35
    
    def test_dropout_mask_edge_cases(self):
        """Test dropout mask edge cases."""
        mask = self.ai.dropout_mask(100, 0)
        assert np.sum(mask) == 0
        mask = self.ai.dropout_mask(100, 1)
        assert np.sum(mask) == 100
    
    def test_augmentation_params(self):
        """Test augmentation parameters generation."""
        params = self.ai.augmentation_params()
        assert hasattr(params, 'rotation')
        assert hasattr(params, 'scale')
        assert hasattr(params, 'brightness')
        assert hasattr(params, 'contrast')
        assert hasattr(params, 'flip_horizontal')
        assert hasattr(params, 'flip_vertical')
        assert -30 <= params.rotation <= 30
        assert 0.8 <= params.scale <= 1.2
        assert 0.7 <= params.brightness <= 1.3
        assert 0.8 <= params.contrast <= 1.2
    
    def test_mini_batch(self):
        """Test mini-batch sampling."""
        dataset_size = 1000
        batch_size = 64
        batch = self.ai.mini_batch(dataset_size, batch_size)
        assert len(batch) == batch_size
        assert len(set(batch)) == batch_size
        assert all(0 <= idx < dataset_size for idx in batch)
    
    def test_mini_batch_invalid_size(self):
        """Test mini-batch raises error when batch_size > dataset_size."""
        with pytest.raises(ValueError):
            self.ai.mini_batch(100, 200)
    
    def test_exploration_noise(self):
        """Test exploration noise generation."""
        action_dim = 4
        noise = self.ai.exploration_noise(action_dim, 0.2)
        assert len(noise) == action_dim
        assert abs(np.mean(noise)) < 0.3
        assert np.std(noise) > 0.02


class TestGradientNoise:
    """Test gradient noise injection."""
    
    def setup_method(self):
        self.noise = aleam.GradientNoise(initial_scale=0.1, decay=0.99)
    
    def test_add_noise_shape(self):
        """Test noise addition (C++ flattens the array)."""
        gradients = np.ones((3, 3))
        result = self.noise.add_noise(gradients)
        # C++ add_noise returns flattened array
        assert len(result) == 9
        assert not np.array_equal(result, gradients.flatten())
    
    def test_add_noise_1d(self):
        """Test noise addition on 1D array."""
        gradients = np.ones(100)
        result = self.noise.add_noise(gradients)
        assert len(result) == 100
        assert not np.array_equal(result, gradients)
    
    def test_noise_decay(self):
        """Test noise decays over time."""
        gradients = np.zeros((5, 5))
        result1 = self.noise.add_noise(gradients)
        result2 = self.noise.add_noise(gradients)
        assert not np.array_equal(result1, result2)
    
    def test_noise_scale_decreases(self):
        """Test noise scale decreases with steps."""
        self.noise.reset()
        first_scale = self.noise.get_current_scale()
        
        self.noise.add_noise(np.zeros((1,)))
        assert self.noise.get_step() == 1
        
        current_scale = self.noise.get_current_scale()
        assert current_scale < first_scale
    
    def test_add_noise_2d(self):
        """Test 2D noise input (flattens to 1D)."""
        gradients = [[1.0, 2.0], [3.0, 4.0]]
        result = self.noise.add_noise(gradients)
        # C++ add_noise returns flattened array
        assert len(result) == 4
        assert isinstance(result, np.ndarray)
    
    def test_reset(self):
        """Test resetting step counter."""
        self.noise.add_noise(np.zeros((1,)))
        self.noise.add_noise(np.zeros((1,)))
        self.noise.reset()
        assert self.noise.get_step() == 0
    
    def test_get_current_scale(self):
        """Test getting current noise scale."""
        self.noise.reset()
        scale = self.noise.get_current_scale()
        assert scale == 0.1
        self.noise.add_noise(np.zeros((1,)))
        self.noise.add_noise(np.zeros((1,)))
        scale = self.noise.get_current_scale()
        expected = 0.1 * (0.99 ** 2)
        assert abs(scale - expected) < 1e-6


class TestLatentSampler:
    """Test latent space sampler."""
    
    def setup_method(self):
        self.sampler = aleam.LatentSampler(128, "normal")
    
    def test_sample_normal(self):
        """Test sampling from normal distribution."""
        samples = self.sampler.sample(10)
        assert len(samples) == 10
        assert len(samples[0]) == 128
        all_vals = [v for sample in samples for v in sample]
        assert abs(np.mean(all_vals)) < 0.2
        assert abs(np.std(all_vals) - 1) < 0.2
    
    def test_sample_uniform(self):
        """Test sampling from uniform distribution."""
        sampler = aleam.LatentSampler(128, "uniform")
        samples = sampler.sample(10)
        assert len(samples) == 10
        assert len(samples[0]) == 128
        all_vals = [v for sample in samples for v in sample]
        assert np.all(np.array(all_vals) >= -1)
        assert np.all(np.array(all_vals) <= 1)
        assert abs(np.mean(all_vals)) < 0.1
    
    def test_sample_one(self):
        """Test single sample."""
        vec = self.sampler.sample_one()
        assert len(vec) == 128
    
    def test_interpolate(self):
        """Test interpolation between two vectors."""
        z1 = np.ones(128)
        z2 = np.zeros(128)
        interp = self.sampler.interpolate(z1, z2, steps=5)
        assert len(interp) == 5
        assert len(interp[0]) == 128
        assert np.allclose(interp[0], z1)
        assert np.allclose(interp[-1], z2)
    
    def test_interpolate_linear(self):
        """Test linear interpolation values."""
        sampler = aleam.LatentSampler(3, "normal")
        z1 = np.array([1.0, 0.0, 0.0])
        z2 = np.array([0.0, 1.0, 0.0])
        interp = sampler.interpolate(z1, z2, steps=3)
        expected = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0]])
        assert np.allclose(interp, expected)
    
    def test_interpolate_same_vector(self):
        """Test interpolation with same start and end."""
        # Create a sampler with matching dimension
        sampler = aleam.LatentSampler(3, "normal")
        z1 = np.array([1.0, 2.0, 3.0])
        interp = sampler.interpolate(z1, z1, steps=3)
        for vec in interp:
            assert np.allclose(vec, z1)
    
    def test_get_latent_dim(self):
        """Test getting latent dimension."""
        assert self.sampler.get_latent_dim() == 128
    
    def test_get_distribution(self):
        """Test getting distribution type."""
        assert self.sampler.get_distribution() == "normal"
        sampler = aleam.LatentSampler(64, "uniform")
        assert sampler.get_distribution() == "uniform"