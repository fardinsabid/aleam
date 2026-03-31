"""
TensorFlow integration for Aleam.
Provides true random tensors for TensorFlow/Keras.
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Union, List, Tuple
from .core import Aleam, AleamBase


class TFGenerator:
    """
    TensorFlow-compatible random generator using true randomness.
    
    Example:
        >>> from aleam.tensorflow_integration import TFGenerator
        >>> gen = TFGenerator()
        >>> tensor = tf.random.normal((100, 100), generator=gen)
    """
    
    def __init__(self, rng: Optional[AleamBase] = None):
        """
        Args:
            rng: Optional Aleam instance
        """
        self.rng = rng or Aleam()
        self._seed = None  # No seed - true randomness
    
    def normal(self, shape: Tuple[int, ...], mean: float = 0.0, stddev: float = 1.0,
               dtype: tf.DType = tf.float32, name: Optional[str] = None) -> tf.Tensor:
        """
        Generate tensor of true random values from N(mean, stddev).
        
        Args:
            shape: Tensor dimensions
            mean: Mean of distribution
            stddev: Standard deviation
            dtype: Output data type
            name: Tensor name
        
        Returns:
            Tensor of shape shape with values from N(mean, stddev)
        """
        total = 1
        for dim in shape:
            total *= dim
        
        # Generate random values
        values = [self.rng.gauss(mean, stddev) for _ in range(total)]
        tensor = tf.constant(values, dtype=dtype, shape=shape, name=name)
        return tensor
    
    def uniform(self, shape: Tuple[int, ...], minval: float = 0.0, maxval: float = 1.0,
                dtype: tf.DType = tf.float32, name: Optional[str] = None) -> tf.Tensor:
        """
        Generate tensor of true random values from Uniform(minval, maxval).
        
        Args:
            shape: Tensor dimensions
            minval: Lower bound
            maxval: Upper bound
            dtype: Output data type
            name: Tensor name
        
        Returns:
            Tensor of shape shape with values from Uniform(minval, maxval)
        """
        total = 1
        for dim in shape:
            total *= dim
        
        # Generate random values
        values = [self.rng.uniform(minval, maxval) for _ in range(total)]
        tensor = tf.constant(values, dtype=dtype, shape=shape, name=name)
        return tensor
    
    def truncated_normal(self, shape: Tuple[int, ...], mean: float = 0.0, stddev: float = 1.0,
                         dtype: tf.DType = tf.float32, name: Optional[str] = None) -> tf.Tensor:
        """
        Generate tensor of true random truncated normal values.
        
        Values are from N(mean, stddev) clipped to [mean - 2*stddev, mean + 2*stddev].
        
        Args:
            shape: Tensor dimensions
            mean: Mean of distribution
            stddev: Standard deviation
            dtype: Output data type
            name: Tensor name
        
        Returns:
            Tensor of shape shape with truncated normal values
        """
        total = 1
        for dim in shape:
            total *= dim
        
        # Generate random values with rejection sampling
        values = []
        lower = mean - 2 * stddev
        upper = mean + 2 * stddev
        
        while len(values) < total:
            x = self.rng.gauss(mean, stddev)
            if lower <= x <= upper:
                values.append(x)
        
        tensor = tf.constant(values, dtype=dtype, shape=shape, name=name)
        return tensor
    
    def random(self) -> float:
        """Generate single random float (compatibility with Python random)"""
        return self.rng.random()
    
    def randint(self, shape: Tuple[int, ...], minval: int, maxval: int,
                dtype: tf.DType = tf.int32, name: Optional[str] = None) -> tf.Tensor:
        """
        Generate tensor of true random integers in [minval, maxval).
        
        Args:
            shape: Tensor dimensions
            minval: Lower bound (inclusive)
            maxval: Upper bound (exclusive)
            dtype: Output data type
            name: Tensor name
        
        Returns:
            Tensor of shape shape with random integers
        """
        total = 1
        for dim in shape:
            total *= dim
        
        # Generate random values
        values = [self.rng.randint(minval, maxval - 1) for _ in range(total)]
        tensor = tf.constant(values, dtype=dtype, shape=shape, name=name)
        return tensor
    
    def shuffle(self, tensor: tf.Tensor, seed: Optional[int] = None) -> tf.Tensor:
        """
        Shuffle tensor along first dimension using true randomness.
        
        Args:
            tensor: Input tensor
            seed: Ignored (for API compatibility)
        
        Returns:
            Shuffled tensor
        """
        # Convert to list and shuffle
        if tf.rank(tensor) == 0:
            return tensor
        
        # Get first dimension size
        size = tf.shape(tensor)[0].numpy() if hasattr(tensor, 'numpy') else tensor.shape[0]
        
        # Generate random permutation
        indices = list(range(size))
        self.rng.shuffle(indices)
        
        # Gather shuffled tensor
        return tf.gather(tensor, indices)
    
    def set_seed(self, seed: int):
        """
        Raise error explaining that Aleam doesn't use seeds.
        
        This method exists for compatibility with TensorFlow's random API.
        """
        raise NotImplementedError(
            "TFGenerator uses true randomness and does not support seeding. "
            "Each call is independent and stateless."
        )
    
    def make_seeds(self, count: int) -> List[int]:
        """
        Generate true random seeds for child generators.
        
        Args:
            count: Number of seeds to generate
        
        Returns:
            List of true random integers
        """
        return [self.rng.randint(0, 2**31 - 1) for _ in range(count)]


def tf_random_normal(shape: Tuple[int, ...], mean: float = 0.0, stddev: float = 1.0,
                     dtype: tf.DType = tf.float32, seed: Optional[int] = None,
                     name: Optional[str] = None) -> tf.Tensor:
    """
    Generate true random tensor from N(mean, stddev) using Aleam.
    
    Drop-in replacement for tf.random.normal with true randomness.
    
    Example:
        >>> from Aleam.tensorflow_integration import tf_random_normal
        >>> tensor = tf_random_normal((100, 100))
    """
    gen = TFGenerator()
    return gen.normal(shape, mean, stddev, dtype, name)


def tf_random_uniform(shape: Tuple[int, ...], minval: float = 0.0, maxval: float = 1.0,
                      dtype: tf.DType = tf.float32, seed: Optional[int] = None,
                      name: Optional[str] = None) -> tf.Tensor:
    """
    Generate true random tensor from Uniform(minval, maxval) using Aleam.
    
    Drop-in replacement for tf.random.uniform with true randomness.
    
    Example:
        >>> from aleam.tensorflow_integration import tf_random_uniform
        >>> tensor = tf_random_uniform((100, 100))
    """
    gen = TFGenerator()
    return gen.uniform(shape, minval, maxval, dtype, name)


def tf_random_truncated_normal(shape: Tuple[int, ...], mean: float = 0.0, stddev: float = 1.0,
                               dtype: tf.DType = tf.float32, seed: Optional[int] = None,
                               name: Optional[str] = None) -> tf.Tensor:
    """
    Generate true random truncated normal tensor using Aleam.
    
    Drop-in replacement for tf.random.truncated_normal with true randomness.
    
    Example:
        >>> from aleam.tensorflow_integration import tf_random_truncated_normal
        >>> tensor = tf_random_truncated_normal((100, 100))
    """
    gen = TFGenerator()
    return gen.truncated_normal(shape, mean, stddev, dtype, name)