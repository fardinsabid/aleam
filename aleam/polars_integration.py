"""
Polars integration for Aleam.
Provides true random data for Polars DataFrames.
"""

import polars as pl
import numpy as np
from typing import Optional, Union, List, Any
from .core import Aleam, AleamBase


class PolarsGenerator:
    """
    Polars-compatible random generator using true randomness.
    
    Example:
        >>> from aleam.polars_integration import PolarsGenerator
        >>> gen = PolarsGenerator()
        >>> df = gen.dataframe(100, columns=['a', 'b', 'c'])
    """
    
    def __init__(self, rng: Optional[AleamBase] = None):
        self.rng = rng or Aleam()
    
    def series(self, n: int, distribution: str = "uniform",
               params: dict = None, name: str = None) -> pl.Series:
        """Generate true random Polars Series"""
        params = params or {}
        
        if distribution == "uniform":
            low = params.get('low', 0.0)
            high = params.get('high', 1.0)
            data = [self.rng.uniform(low, high) for _ in range(n)]
        elif distribution == "normal":
            mu = params.get('mu', 0.0)
            sigma = params.get('sigma', 1.0)
            data = [self.rng.gauss(mu, sigma) for _ in range(n)]
        elif distribution == "exponential":
            rate = params.get('rate', 1.0)
            data = [self.rng.exponential(rate) for _ in range(n)]
        elif distribution == "poisson":
            lam = params.get('lam', 1.0)
            data = [self.rng.poisson(lam) for _ in range(n)]
        elif distribution == "choice":
            choices = params.get('choices', [0, 1])
            data = [self.rng.choice(choices) for _ in range(n)]
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        return pl.Series(name, data)
    
    def dataframe(self, n: int, columns: List[str],
                  distributions: Optional[dict] = None) -> pl.DataFrame:
        """Generate true random Polars DataFrame"""
        if distributions is None:
            distributions = {col: ('uniform', {}) for col in columns}
        
        data = {}
        for col in columns:
            if col in distributions:
                dist, params = distributions[col]
            else:
                dist, params = 'uniform', {}
            
            data[col] = self.series(n, dist, params, name=col)
        
        return pl.DataFrame(data)
    
    def shuffle(self, df: pl.DataFrame) -> pl.DataFrame:
        """Shuffle DataFrame rows with true randomness"""
        indices = list(range(df.height))
        self.rng.shuffle(indices)
        return df[indices]