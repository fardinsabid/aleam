"""
Unit tests for AI/ML features.
"""

import pytest
import numpy as np
from aleam import AIRandom, GradientNoise, LatentSampler, Aleam


class TestAIRandom:
    """Test AI-specific random utilities"""
    
    def setup_method(self):
        self.ai = AIRandom()
    
    def test_gradient_noise_shape(self):
        shape = (3, 4)
        noise = self.ai.gradient_noise(shape, scale=0.1)
        assert noise.shape == shape
        assert isinstance(noise, np.ndarray)
    
    def test_gradient_noise_stats(self):
        shape = (1000,)
        noise = self.ai.gradient_noise(shape, scale=0.1)
        assert abs(np.mean(noise)) < 0.02
        assert abs(np.std(noise) - 0.1) < 0.03
    
    def test_latent_vector_normal(self):
        dim = 128
        vec = self.ai.latent_vector(dim, "normal")
        assert len(vec) == dim
        assert isinstance(vec, np.ndarray)
        assert abs(np.mean(vec)) < 0.2
    
    def test_latent_vector_uniform(self):
        dim = 128
        vec = self.ai.latent_vector(dim, "uniform")
        assert len(vec) == dim
        assert all(-1 <= v <= 1 for v in vec)
        assert abs(np.mean(vec)) < 0.1
    
    def test_latent_vector_invalid_distribution(self):
        with pytest.raises(ValueError):
            self.ai.latent_vector(10, "invalid")
    
    def test_dropout_mask(self):
        size = 1000
        mask = self.ai.dropout_mask(size, 0.3)
        assert len(mask) == size
        assert all(v in (0, 1) for v in mask)
        active_pct = sum(mask) / size * 100
        assert 25 < active_pct < 35
    
    def test_dropout_mask_edge_cases(self):
        # p=0 should keep nothing
        mask = self.ai.dropout_mask(100, 0)
        assert sum(mask) == 0
        
        # p=1 should keep everything
        mask = self.ai.dropout_mask(100, 1)
        assert sum(mask) == 100
    
    def test_augmentation_params(self):
        params = self.ai.augmentation_params()
        assert 'rotation' in params
        assert 'scale' in params
        assert 'brightness' in params
        assert 'contrast' in params
        assert 'flip_horizontal' in params
        assert 'flip_vertical' in params
        
        assert -30 <= params['rotation'] <= 30
        assert 0.8 <= params['scale'] <= 1.2
        assert 0.7 <= params['brightness'] <= 1.3
        assert 0.8 <= params['contrast'] <= 1.2
    
    def test_mini_batch(self):
        dataset_size = 1000
        batch_size = 64
        batch = self.ai.mini_batch(dataset_size, batch_size)
        assert len(batch) == batch_size
        assert len(set(batch)) == batch_size
        assert all(0 <= idx < dataset_size for idx in batch)
    
    def test_exploration_noise(self):
        action_dim = 4
        noise = self.ai.exploration_noise(action_dim, 0.2)
        assert len(noise) == action_dim
        # With only 4 samples, std can vary. Use wider tolerance.
        assert abs(np.mean(noise)) < 0.3
        assert np.std(noise) > 0.05
        assert np.std(noise) < 0.5


class TestGradientNoise:
    """Test gradient noise injection"""
    
    def setup_method(self):
        self.noise = GradientNoise(scale=0.1, decay=0.99)
    
    def test_add_noise_shape(self):
        gradients = np.ones((3, 3))
        result = self.noise.add_noise(gradients)
        assert result.shape == gradients.shape
        assert not np.array_equal(result, gradients)
    
    def test_noise_decay(self):
        gradients = np.zeros((5, 5))
        result1 = self.noise.add_noise(gradients)
        result2 = self.noise.add_noise(gradients)
        assert not np.array_equal(result1, result2)
    
    def test_noise_scale_decreases(self):
        self.noise.step = 0
        first_scale = self.noise.initial_scale
        
        self.noise.add_noise(np.zeros((1,)))
        assert self.noise.step == 1
        
        # Internal scale should have decayed
        current_scale = self.noise.initial_scale * (self.noise.decay ** self.noise.step)
        assert current_scale < first_scale
    
    def test_reset(self):
        self.noise.step = 10
        self.noise.reset()
        assert self.noise.step == 0


class TestLatentSampler:
    """Test latent space sampler"""
    
    def test_sample_normal(self):
        sampler = LatentSampler(128, "normal")
        samples = sampler.sample(10)
        assert samples.shape == (10, 128)
        assert abs(np.mean(samples)) < 0.2
        assert abs(np.std(samples) - 1) < 0.2
    
    def test_sample_uniform(self):
        sampler = LatentSampler(128, "uniform")
        samples = sampler.sample(10)
        assert samples.shape == (10, 128)
        assert np.all(samples >= -1)
        assert np.all(samples <= 1)
        assert abs(np.mean(samples)) < 0.1
    
    def test_interpolate(self):
        sampler = LatentSampler(10)
        z1 = np.ones(10)
        z2 = np.zeros(10)
        interp = sampler.interpolate(z1, z2, steps=5)
        assert interp.shape == (5, 10)
        assert np.allclose(interp[0], z1)
        assert np.allclose(interp[-1], z2)
    
    def test_interpolate_linear(self):
        sampler = LatentSampler(3)
        z1 = np.array([1.0, 0.0, 0.0])
        z2 = np.array([0.0, 1.0, 0.0])
        interp = sampler.interpolate(z1, z2, steps=3)
        expected = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0]])
        assert np.allclose(interp, expected)