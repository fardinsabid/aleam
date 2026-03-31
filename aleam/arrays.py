"""
Array operations for Aleam.
Provides NumPy-style random array generation.
"""

import numpy as np
from typing import Tuple, Optional, Union, List
from .core import Aleam, AleamBase


def random_array(
    shape: Tuple[int, ...],
    rng: Optional[AleamBase] = None,
    dtype: str = 'float64'
) -> np.ndarray:
    """
    Generate array of true random floats in [0, 1).
    
    Args:
        shape: Shape of output array
        rng: Optional Aleam instance
        dtype: NumPy dtype for output
    
    Returns:
        NumPy array of random floats
    """
    rng = rng or Aleam()
    size = np.prod(shape)
    result = np.zeros(size, dtype=dtype)
    
    for i in range(size):
        result[i] = rng.random()
    
    return result.reshape(shape)


def randn_array(
    shape: Tuple[int, ...],
    mu: float = 0.0,
    sigma: float = 1.0,
    rng: Optional[AleamBase] = None,
    dtype: str = 'float64'
) -> np.ndarray:
    """
    Generate array of true random normally distributed values.
    
    Args:
        shape: Shape of output array
        mu: Mean of distribution
        sigma: Standard deviation
        rng: Optional Aleam instance
        dtype: NumPy dtype for output
    
    Returns:
        NumPy array of normally distributed values
    """
    rng = rng or Aleam()
    size = np.prod(shape)
    result = np.zeros(size, dtype=dtype)
    
    for i in range(size):
        result[i] = rng.gauss(mu, sigma)
    
    return result.reshape(shape)


def randint_array(
    shape: Tuple[int, ...],
    low: int,
    high: Optional[int] = None,
    rng: Optional[AleamBase] = None,
    dtype: str = 'int64'
) -> np.ndarray:
    """
    Generate array of true random integers.
    
    Args:
        shape: Shape of output array
        low: Lowest integer (inclusive)
        high: Highest integer (exclusive) if provided
        rng: Optional Aleam instance
        dtype: NumPy dtype for output
    
    Returns:
        NumPy array of random integers
    """
    rng = rng or Aleam()
    
    if high is None:
        # Single argument: randint_array(shape, high)
        high = low
        low = 0
    
    size = np.prod(shape)
    result = np.zeros(size, dtype=dtype)
    
    for i in range(size):
        result[i] = rng.randint(low, high - 1)
    
    return result.reshape(shape)


def choice_array(
    a: Union[int, List],
    size: Optional[Tuple[int, ...]] = None,
    replace: bool = True,
    p: Optional[List[float]] = None,
    rng: Optional[AleamBase] = None
) -> np.ndarray:
    """
    Generate random samples from a given array with proper weighted sampling.
    """
    rng = rng or Aleam()
    
    # Convert integer to range
    if isinstance(a, int):
        population = list(range(a))
    else:
        population = list(a)
    
    total = len(population)
    
    # Handle probabilities
    if p is not None:
        if len(p) != total:
            raise ValueError("Length of p must match population size")
        # Normalize probabilities to sum to 1
        p_sum = sum(p)
        if abs(p_sum - 1.0) > 1e-10:
            p = [x / p_sum for x in p]
        # Build cumulative distribution
        cumsum = []
        running = 0.0
        for prob in p:
            running += prob
            cumsum.append(running)
    
    # Generate samples
    if size is None:
        if replace:
            if p is not None:
                u = rng.random()
                # Binary search in cumulative distribution
                idx = 0
                while idx < total and u > cumsum[idx]:
                    idx += 1
                return population[min(idx, total - 1)]
            else:
                return population[rng.randint(0, total - 1)]
        else:
            return rng.choice(population)
    else:
        sample_size = np.prod(size)
        result = []
        
        if replace:
            for _ in range(sample_size):
                if p is not None:
                    u = rng.random()
                    idx = 0
                    while idx < total and u > cumsum[idx]:
                        idx += 1
                    result.append(population[min(idx, total - 1)])
                else:
                    result.append(population[rng.randint(0, total - 1)])
        else:
            # Sample without replacement
            if sample_size > total:
                raise ValueError("Sample size cannot exceed population size when replace=False")
            shuffled = population.copy()
            for i in range(sample_size):
                j = rng.randint(i, total - 1)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
                result.append(shuffled[i])
        
        return np.array(result).reshape(size)