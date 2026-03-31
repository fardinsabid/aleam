"""
Pandas integration for Aleam.
Provides true random data for DataFrames and Series.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Any
from .core import Aleam, AleamBase


class PandasGenerator:
    """
    Pandas-compatible random generator using true randomness.
    
    Example:
        >>> from aleam.pandas_integration import PandasGenerator
        >>> gen = PandasGenerator()
        >>> df = gen.dataframe(100, columns=['a', 'b', 'c'])
    """
    
    def __init__(self, rng: Optional[AleamBase] = None):
        self.rng = rng or Aleam()
    
    def series(self, n: int, distribution: str = "uniform",
               params: dict = None, name: str = None) -> pd.Series:
        """
        Generate true random pandas Series.
        
        Args:
            n: Number of elements
            distribution: 'uniform', 'normal', 'exponential', 'poisson', 'binomial'
            params: Distribution parameters
            name: Series name
        
        Returns:
            Pandas Series with true random values
        """
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
        elif distribution == "binomial":
            n_trials = params.get('n', 10)
            p = params.get('p', 0.5)
            data = [sum(1 for _ in range(n_trials) if self.rng.random() < p) for _ in range(n)]
        elif distribution == "choice":
            choices = params.get('choices', [0, 1])
            data = [self.rng.choice(choices) for _ in range(n)]
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        return pd.Series(data, name=name)
    
    def dataframe(self, n: int, columns: List[str],
                  distributions: Optional[dict] = None) -> pd.DataFrame:
        """
        Generate true random pandas DataFrame.
        
        Args:
            n: Number of rows
            columns: List of column names
            distributions: Dict mapping column names to (distribution, params)
        
        Returns:
            Pandas DataFrame with true random values
        """
        if distributions is None:
            distributions = {col: ('uniform', {}) for col in columns}
        
        data = {}
        for col in columns:
            if col in distributions:
                dist, params = distributions[col]
            else:
                dist, params = 'uniform', {}
            
            data[col] = self.series(n, dist, params, name=col)
        
        return pd.DataFrame(data)
    
    def choice(self, series: pd.Series, size: Optional[int] = None,
               replace: bool = True) -> Union[Any, pd.Series]:
        """Sample from pandas Series with true randomness"""
        if size is None:
            return self.rng.choice(series.tolist())
        else:
            return pd.Series([self.rng.choice(series.tolist()) for _ in range(size)])
    
    def shuffle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shuffle DataFrame rows with true randomness"""
        indices = list(range(len(df)))
        self.rng.shuffle(indices)
        return df.iloc[indices].reset_index(drop=True)