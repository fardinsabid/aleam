"""
Aleam - True Randomness for AI and Machine Learning

A Python library providing true randomness using the proven equation:
Ψ(t) = BLAKE2s( (Φ × Ξ(t)) ⊕ τ(t) )
"""

__version__ = "1.0.2"
__author__ = "Fardin Sabid"
__license__ = "MIT"

from .core import Aleam, AleamFast, AleamOptimized
from .ai import AIRandom, GradientNoise, LatentSampler
from .sources import EntropySource, SystemEntropy, HardwareEntropy
from .distributions import (
    Normal, Uniform, Exponential, Beta, Gamma, Poisson, Laplace
)
from .arrays import random_array, randn_array, randint_array, choice_array
from .utils import seed_free

# Import module-level functions from core
from .core import (
    random,
    randint,
    choice,
    uniform,
    gauss,
    shuffle,
    sample,
    random_bytes,
    exponential,
    beta,
    gamma,
    poisson,
    laplace,
    logistic,
    lognormal,
    weibull,
    pareto,
    chi_square,
    student_t,
    f_distribution,
    dirichlet,
)

__all__ = [
    "Aleam",
    "AleamFast",
    "AleamOptimized",
    "AIRandom",
    "GradientNoise",
    "LatentSampler",
    "EntropySource",
    "SystemEntropy",
    "HardwareEntropy",
    "Normal",
    "Uniform",
    "Exponential",
    "Beta",
    "Gamma",
    "Poisson",
    "Laplace",
    "random_array",
    "randn_array",
    "randint_array",
    "choice_array",
    "seed_free",
    # Module-level functions
    "random",
    "randint",
    "choice",
    "uniform",
    "gauss",
    "shuffle",
    "sample",
    "random_bytes",
    "exponential",
    "beta",
    "gamma",
    "poisson",
    "laplace",
    "logistic",
    "lognormal",
    "weibull",
    "pareto",
    "chi_square",
    "student_t",
    "f_distribution",
    "dirichlet",
]


def __getattr__(name):
    """Lazy load optional integrations"""
    
    # ==================== CUDA ====================
    if name == 'CUDAGenerator':
        try:
            from .cuda_integration import CUDAGenerator
            globals()['CUDAGenerator'] = CUDAGenerator
            return CUDAGenerator
        except ImportError:
            raise ImportError("CUDA support requires cupy or torch with CUDA")
    
    # ==================== PyTorch ====================
    if name in ['TorchGenerator', 'torch_rand', 'torch_randn', 'torch_randint']:
        try:
            from .torch_integration import TorchGenerator, torch_rand, torch_randn, torch_randint
            globals()['TorchGenerator'] = TorchGenerator
            globals()['torch_rand'] = torch_rand
            globals()['torch_randn'] = torch_randn
            globals()['torch_randint'] = torch_randint
            return globals()[name]
        except ImportError:
            raise ImportError(f"PyTorch not installed. Install with: pip install torch")
    
    # ==================== TensorFlow ====================
    if name in ['TFGenerator', 'tf_random_normal', 'tf_random_uniform', 'tf_random_truncated_normal']:
        try:
            from .tensorflow_integration import TFGenerator, tf_random_normal, tf_random_uniform, tf_random_truncated_normal
            globals()['TFGenerator'] = TFGenerator
            globals()['tf_random_normal'] = tf_random_normal
            globals()['tf_random_uniform'] = tf_random_uniform
            globals()['tf_random_truncated_normal'] = tf_random_truncated_normal
            return globals()[name]
        except ImportError:
            raise ImportError(f"TensorFlow not installed. Install with: pip install tensorflow")
    
    # ==================== JAX ====================
    if name == 'JAXGenerator':
        try:
            from .jax_integration import JAXGenerator
            globals()['JAXGenerator'] = JAXGenerator
            return JAXGenerator
        except ImportError:
            raise ImportError(f"JAX not installed. Install with: pip install jax jaxlib")
    
    # ==================== CuPy ====================
    if name == 'CuPyGenerator':
        try:
            from .cupy_integration import CuPyGenerator
            globals()['CuPyGenerator'] = CuPyGenerator
            return CuPyGenerator
        except ImportError:
            raise ImportError(f"CuPy not installed. Install with: pip install cupy")
    
    # ==================== Dask ====================
    if name == 'DaskGenerator':
        try:
            from .dask_integration import DaskGenerator
            globals()['DaskGenerator'] = DaskGenerator
            return DaskGenerator
        except ImportError:
            raise ImportError(f"Dask not installed. Install with: pip install dask")
    
    # ==================== Pandas ====================
    if name == 'PandasGenerator':
        try:
            from .pandas_integration import PandasGenerator
            globals()['PandasGenerator'] = PandasGenerator
            return PandasGenerator
        except ImportError:
            raise ImportError(f"Pandas not installed. Install with: pip install pandas")
    
    # ==================== Polars ====================
    if name == 'PolarsGenerator':
        try:
            from .polars_integration import PolarsGenerator
            globals()['PolarsGenerator'] = PolarsGenerator
            return PolarsGenerator
        except ImportError:
            raise ImportError(f"Polars not installed. Install with: pip install polars")
    
    # ==================== Xarray ====================
    if name == 'XarrayGenerator':
        try:
            from .xarray_integration import XarrayGenerator
            globals()['XarrayGenerator'] = XarrayGenerator
            return XarrayGenerator
        except ImportError:
            raise ImportError(f"Xarray not installed. Install with: pip install xarray")
    
    # ==================== PyMC ====================
    if name == 'PyMCGenerator':
        try:
            from .pymc_integration import PyMCGenerator
            globals()['PyMCGenerator'] = PyMCGenerator
            return PyMCGenerator
        except ImportError:
            raise ImportError(f"PyMC not installed. Install with: pip install pymc")
    
    raise AttributeError(f"module 'aleam' has no attribute '{name}'")