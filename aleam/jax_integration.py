"""
JAX integration for Aleam.
Provides true random tensors for JAX.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple, Union, List
from .core import Aleam, AleamBase


class JAXGenerator:
    """
    JAX-compatible random generator using true randomness.
    
    Example:
        >>> from aleam.jax_integration import JAXGenerator
        >>> gen = JAXGenerator()
        >>> key = gen.key()
        >>> tensor = jax.random.normal(key, (100, 100))
    """
    
    def __init__(self, rng: Optional[AleamBase] = None):
        self.rng = rng or Aleam()
        self._counter = 0
    
    def key(self) -> jax.random.KeyArray:
        """
        Generate a true random JAX key.
        
        Returns:
            JAX PRNG key seeded with true entropy
        """
        # Generate 128-bit true entropy as seed
        entropy = self.rng.randint(0, 2**31 - 1)
        self._counter += 1
        # Use counter to ensure uniqueness
        return jax.random.key(entropy + self._counter)
    
    def normal(self, shape: Tuple[int, ...], mean: float = 0.0, stddev: float = 1.0,
               dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
        """Generate true random normal tensor"""
        key = self.key()
        return jax.random.normal(key, shape, dtype) * stddev + mean
    
    def uniform(self, shape: Tuple[int, ...], minval: float = 0.0, maxval: float = 1.0,
                dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
        """Generate true random uniform tensor"""
        key = self.key()
        return jax.random.uniform(key, shape, dtype, minval, maxval)
    
    def randint(self, shape: Tuple[int, ...], minval: int, maxval: int,
                dtype: jnp.dtype = jnp.int32) -> jnp.ndarray:
        """Generate true random integer tensor"""
        key = self.key()
        return jax.random.randint(key, shape, minval, maxval, dtype)
    
    def bernoulli(self, shape: Tuple[int, ...], p: float = 0.5,
                  dtype: jnp.dtype = jnp.bool_) -> jnp.ndarray:
        """Generate true random Bernoulli tensor"""
        key = self.key()
        return jax.random.bernoulli(key, p, shape, dtype)
    
    def gamma(self, shape: Tuple[int, ...], alpha: float, scale: float = 1.0,
              dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
        """Generate true random Gamma tensor"""
        key = self.key()
        return jax.random.gamma(key, alpha, shape, dtype) * scale
    
    def poisson(self, shape: Tuple[int, ...], lam: float,
                dtype: jnp.dtype = jnp.int32) -> jnp.ndarray:
        """Generate true random Poisson tensor"""
        key = self.key()
        return jax.random.poisson(key, lam, shape, dtype)