"""
PyMC integration for Aleam.
Provides true random sampling for Bayesian models.
"""

import pymc as pm
import numpy as np
import aesara.tensor as at
from typing import Optional, Union, Any
from .core import Aleam, AleamBase


class PyMCGenerator:
    """
    PyMC-compatible random generator using true randomness.
    
    Example:
        >>> from aleam.pymc_integration import PyMCGenerator
        >>> gen = PyMCGenerator()
        >>> with pm.Model():
        ...     x = pm.Normal('x', mu=0, sigma=1, generator=gen)
    """
    
    def __init__(self, rng: Optional[AleamBase] = None):
        self.rng = rng or Aleam()
        self._seed = None
    
    def normal(self, size=None) -> np.ndarray:
        """Generate true random normal samples"""
        if size is None:
            return self.rng.gauss()
        else:
            return np.array([self.rng.gauss() for _ in range(size)])
    
    def uniform(self, low=0.0, high=1.0, size=None) -> np.ndarray:
        """Generate true random uniform samples"""
        if size is None:
            return self.rng.uniform(low, high)
        else:
            return np.array([self.rng.uniform(low, high) for _ in range(size)])
    
    def exponential(self, rate=1.0, size=None) -> np.ndarray:
        """Generate true random exponential samples"""
        if size is None:
            return self.rng.exponential(rate)
        else:
            return np.array([self.rng.exponential(rate) for _ in range(size)])
    
    def gamma(self, alpha, beta=1.0, size=None) -> np.ndarray:
        """Generate true random gamma samples"""
        scale = 1.0 / beta
        if size is None:
            return self.rng.gamma(alpha, scale)
        else:
            return np.array([self.rng.gamma(alpha, scale) for _ in range(size)])
    
    def beta(self, alpha, beta, size=None) -> np.ndarray:
        """Generate true random beta samples"""
        if size is None:
            return self.rng.beta(alpha, beta)
        else:
            return np.array([self.rng.beta(alpha, beta) for _ in range(size)])
    
    def poisson(self, lam, size=None) -> np.ndarray:
        """Generate true random poisson samples"""
        if size is None:
            return self.rng.poisson(lam)
        else:
            return np.array([self.rng.poisson(lam) for _ in range(size)])