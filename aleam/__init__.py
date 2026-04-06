"""
Aleam - True Randomness for AI and Machine Learning
"""

__version__ = "1.0.3"
__author__ = "Fardin Sabid"
__license__ = "MIT"
__algorithm__ = "Ψ(t) = BLAKE2s( (Φ × Ξ(t)) ⊕ τ(t) )"

from . import _c_core

# Core Classes
AleamCore = _c_core.AleamCore
Aleam = AleamCore

# Module-Level Convenience Functions
random = _c_core.random
random_uint64 = _c_core.random_uint64
randint = _c_core.randint
choice = _c_core.choice
uniform = _c_core.uniform
gauss = _c_core.gauss
normalvariate = _c_core.gauss
shuffle = _c_core.shuffle
sample = _c_core.sample
random_bytes = _c_core.random_bytes

# Array functions
random_array = _c_core.random_array
randn_array = _c_core.randn_array
randint_array = _c_core.randint_array

# Distribution functions
exponential = _c_core.exponential
beta = _c_core.beta
gamma = _c_core.gamma
poisson = _c_core.poisson
laplace = _c_core.laplace
logistic = _c_core.logistic
lognormal = _c_core.lognormal
weibull = _c_core.weibull
pareto = _c_core.pareto
chi_square = _c_core.chi_square
student_t = _c_core.student_t
f_distribution = _c_core.f_distribution
dirichlet = _c_core.dirichlet

# AI/ML Classes
AIRandom = _c_core.AIRandom
GradientNoise = _c_core.GradientNoise
LatentSampler = _c_core.LatentSampler

# ============================================================================
# Framework Integrations (Lazy Load for Optional Dependencies)
# ============================================================================

def __getattr__(name):
    """Lazy load optional framework integrations."""
    
    if name == 'TorchGenerator':
        try:
            from ._c_core import TorchGenerator
            return TorchGenerator
        except ImportError:
            raise ImportError("PyTorch integration not available. Install with: pip install torch")
    
    if name == 'TFGenerator':
        try:
            from ._c_core import TFGenerator
            return TFGenerator
        except ImportError:
            raise ImportError("TensorFlow integration not available. Install with: pip install tensorflow")
    
    if name == 'JAXGenerator':
        try:
            from ._c_core import JAXGenerator
            return JAXGenerator
        except ImportError:
            raise ImportError("JAX integration not available. Install with: pip install jax jaxlib")
    
    if name == 'CuPyGenerator':
        try:
            from ._c_core import CuPyGenerator
            return CuPyGenerator
        except ImportError:
            raise ImportError("CuPy integration not available. Install with: pip install cupy")
    
    if name == 'PandasGenerator':
        try:
            from ._c_core import PandasGenerator
            return PandasGenerator
        except ImportError:
            raise ImportError("Pandas integration not available. Install with: pip install pandas")
    
    if name == 'PolarsGenerator':
        try:
            from ._c_core import PolarsGenerator
            return PolarsGenerator
        except ImportError:
            raise ImportError("Polars integration not available. Install with: pip install polars")
    
    if name == 'XarrayGenerator':
        try:
            from ._c_core import XarrayGenerator
            return XarrayGenerator
        except ImportError:
            raise ImportError("Xarray integration not available. Install with: pip install xarray")
    
    if name == 'PyMCGenerator':
        try:
            from ._c_core import PyMCGenerator
            return PyMCGenerator
        except ImportError:
            raise ImportError("PyMC integration not available. Install with: pip install pymc")
    
    if name == 'DaskGenerator':
        try:
            from ._c_core import DaskGenerator
            return DaskGenerator
        except ImportError:
            raise ImportError("Dask integration not available. Install with: pip install dask")
    
    if name == 'CUDAGenerator':
        try:
            from ._c_core import CuPyGenerator
            return CuPyGenerator
        except ImportError:
            raise ImportError("CUDA support requires cupy or torch with CUDA")
    
    raise AttributeError(f"module 'aleam' has no attribute '{name}'")


def seed_free():
    raise NotImplementedError(
        "Aleam uses true randomness and does not support seeding. "
        "Each call is independent and stateless."
    )


__all__ = [
    "__version__", "__author__", "__license__", "__algorithm__",
    "Aleam", "AleamCore",
    "random", "random_uint64", "randint", "choice", "uniform",
    "gauss", "normalvariate", "shuffle", "sample", "random_bytes",
    "random_array", "randn_array", "randint_array",
    "exponential", "beta", "gamma", "poisson", "laplace", "logistic",
    "lognormal", "weibull", "pareto", "chi_square", "student_t",
    "f_distribution", "dirichlet",
    "AIRandom", "GradientNoise", "LatentSampler",
    "TorchGenerator", "TFGenerator", "JAXGenerator", "CuPyGenerator",
    "PandasGenerator", "PolarsGenerator", "XarrayGenerator",
    "PyMCGenerator", "DaskGenerator", "CUDAGenerator",
    "seed_free",
]