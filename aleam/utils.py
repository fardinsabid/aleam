"""
Utility functions for Aleam.
"""

from typing import List, Tuple, Optional
import numpy as np

from .core import Aleam, AleamBase


def random_array(shape: Tuple[int, ...], rng: Optional[AleamBase] = None) -> np.ndarray:
    """
    Generate array of true random floats in [0, 1).
    
    Args:
        shape: Shape of output array
        rng: Optional Aleam instance
    
    Returns:
        NumPy array of random floats
    """
    rng = rng or Aleam()
    result = np.zeros(shape)
    for idx in np.ndindex(shape):
        result[idx] = rng.random()
    return result


def normal_array(shape: Tuple[int, ...], mu: float = 0.0, sigma: float = 1.0,
                 rng: Optional[AleamBase] = None) -> np.ndarray:
    """
    Generate array of true random normally distributed values.
    
    Args:
        shape: Shape of output array
        mu: Mean of distribution
        sigma: Standard deviation
        rng: Optional Aleam instance
    
    Returns:
        NumPy array of normally distributed values
    """
    rng = rng or Aleam()
    result = np.zeros(shape)
    for idx in np.ndindex(shape):
        result[idx] = rng.gauss(mu, sigma)
    return result


def seed_free() -> None:
    """
    Raise error explaining that Aleam doesn't use seeds.
    
    This function exists to make it clear that Aleam is stateless
    and does not support seeding like traditional PRNGs.
    """
    raise NotImplementedError(
        "Aleam uses true randomness and does not support seeding. "
        "Each call is independent and stateless. "
        "If you need reproducible results, use Python's random module instead."
    )