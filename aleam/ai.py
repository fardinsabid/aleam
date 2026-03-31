"""
AI/ML specific randomness features for Aleam.
"""

import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from .core import Aleam, AleamBase


class AIRandom:
    """AI/ML specific random utilities using true randomness"""
    
    def __init__(self, rng: Optional[AleamBase] = None):
        self.rng = rng or Aleam()
    
    def gradient_noise(self, shape: Tuple[int, ...], scale: float = 0.1) -> np.ndarray:
        """
        Generate noise for gradient perturbation.
        
        Args:
            shape: Shape of gradient tensor
            scale: Standard deviation of noise
            
        Returns:
            Noise array of given shape with N(0, scale) distribution
        """
        noise = np.zeros(shape)
        for idx in np.ndindex(shape):
            noise[idx] = self.rng.gauss(0, scale)
        return noise
    
    def latent_vector(self, dim: int, distribution: str = "normal") -> np.ndarray:
        """
        Generate latent space vector.
        
        Args:
            dim: Dimension of latent space
            distribution: "normal" or "uniform"
            
        Returns:
            Latent vector of given dimension
        """
        if distribution == "normal":
            return np.array([self.rng.gauss() for _ in range(dim)])
        elif distribution == "uniform":
            return np.array([self.rng.uniform(-1, 1) for _ in range(dim)])
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    def dropout_mask(self, size: int, probability: float = 0.5) -> np.ndarray:
        """
        Generate dropout mask.
        
        Args:
            size: Number of neurons
            probability: Probability of KEEPING a neuron (0.5 = keep 50%)
            
        Returns:
            Binary mask array (1 = keep, 0 = drop)
        """
        return np.array([1 if self.rng.random() < probability else 0 for _ in range(size)])
    
    def augmentation_params(self) -> Dict[str, Any]:
        """
        Generate random data augmentation parameters.
        
        Returns:
            Dictionary with augmentation parameters
        """
        return {
            'rotation': self.rng.uniform(-30, 30),
            'scale': self.rng.uniform(0.8, 1.2),
            'brightness': self.rng.uniform(0.7, 1.3),
            'contrast': self.rng.uniform(0.8, 1.2),
            'flip_horizontal': self.rng.random() > 0.5,
            'flip_vertical': self.rng.random() > 0.5,
        }
    
    def mini_batch(self, dataset_size: int, batch_size: int) -> List[int]:
        """
        Sample mini-batch indices without replacement.
        
        Args:
            dataset_size: Total size of dataset
            batch_size: Desired batch size
            
        Returns:
            List of batch indices
        """
        dataset = list(range(dataset_size))
        return self.rng.sample(dataset, batch_size)
    
    def exploration_noise(self, action_dim: int, scale: float = 0.2) -> np.ndarray:
        """
        Generate exploration noise for reinforcement learning.
        
        Args:
            action_dim: Dimension of action space
            scale: Standard deviation of noise
            
        Returns:
            Noise vector for action perturbation
        """
        return np.array([self.rng.gauss(0, scale) for _ in range(action_dim)])


class GradientNoise:
    """
    Gradient noise injection for training with decay.
    
    Adds true random noise to gradients during training to improve generalization.
    """
    
    def __init__(self, scale: float = 0.01, decay: float = 0.99):
        """
        Args:
            scale: Initial noise scale
            decay: Decay factor per step
        """
        self.initial_scale = scale
        self.decay = decay
        self.rng = Aleam()
        self.step = 0
    
    def add_noise(self, gradients: np.ndarray) -> np.ndarray:
        """
        Add true random noise to gradients.
        
        Args:
            gradients: Gradient array
            
        Returns:
            Gradients with added noise
        """
        current_scale = self.initial_scale * (self.decay ** self.step)
        noise = np.zeros_like(gradients)
        
        # Fill with Gaussian noise
        for idx in np.ndindex(gradients.shape):
            noise[idx] = self.rng.gauss(0, current_scale)
        
        self.step += 1
        return gradients + noise
    
    def reset(self) -> None:
        """Reset step counter"""
        self.step = 0


class LatentSampler:
    """
    Latent space sampler for generative models.
    
    Provides true random sampling from latent space with interpolation.
    """
    
    def __init__(self, latent_dim: int, distribution: str = "normal"):
        """
        Args:
            latent_dim: Dimension of latent space
            distribution: "normal" or "uniform"
        """
        self.latent_dim = latent_dim
        self.distribution = distribution
        self.rng = Aleam()
    
    def sample(self, n: int = 1) -> np.ndarray:
        """
        Sample n latent vectors.
        
        Args:
            n: Number of vectors to sample
            
        Returns:
            Array of shape (n, latent_dim)
        """
        samples = []
        for _ in range(n):
            if self.distribution == "normal":
                vec = np.array([self.rng.gauss() for _ in range(self.latent_dim)])
            elif self.distribution == "uniform":
                vec = np.array([self.rng.uniform(-1, 1) for _ in range(self.latent_dim)])
            else:
                raise ValueError(f"Unknown distribution: {self.distribution}")
            samples.append(vec)
        
        return np.array(samples)
    
    def interpolate(self, z1: np.ndarray, z2: np.ndarray, steps: int = 10) -> np.ndarray:
        """
        Interpolate between two latent vectors.
        
        Args:
            z1: First latent vector
            z2: Second latent vector
            steps: Number of interpolation steps
            
        Returns:
            Array of interpolated vectors
        """
        alphas = np.linspace(0, 1, steps)
        interpolations = []
        for alpha in alphas:
            interpolations.append((1 - alpha) * z1 + alpha * z2)
        return np.array(interpolations)