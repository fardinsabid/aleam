"""
Statistical distributions using true randomness (Aleam).
"""

import math
from typing import Optional

from .core import Aleam, AleamBase


class Normal:
    """
    Normal (Gaussian) distribution using true randomness.
    
    N(μ, σ²) where μ is mean and σ is standard deviation.
    """
    
    def __init__(self, mu: float = 0.0, sigma: float = 1.0, rng: Optional[AleamBase] = None):
        self.mu = mu
        self.sigma = sigma
        self.rng = rng or Aleam()
    
    def sample(self) -> float:
        """Sample from normal distribution"""
        return self.rng.gauss(self.mu, self.sigma)
    
    def pdf(self, x: float) -> float:
        """Probability density function"""
        return (1 / (self.sigma * math.sqrt(2 * math.pi))) * \
               math.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)


class Uniform:
    """
    Uniform distribution using true randomness.
    
    U(a, b) where a is lower bound, b is upper bound.
    """
    
    def __init__(self, low: float = 0.0, high: float = 1.0, rng: Optional[AleamBase] = None):
        self.low = low
        self.high = high
        self.rng = rng or Aleam()
    
    def sample(self) -> float:
        """Sample from uniform distribution"""
        return self.rng.uniform(self.low, self.high)
    
    def pdf(self, x: float) -> float:
        """Probability density function"""
        if self.low <= x <= self.high:
            return 1 / (self.high - self.low)
        return 0.0


class Exponential:
    """
    Exponential distribution using true randomness.
    
    Exp(λ) where λ is rate parameter (λ > 0).
    PDF: f(x) = λe^{-λx} for x ≥ 0
    """
    
    def __init__(self, rate: float = 1.0, rng: Optional[AleamBase] = None):
        if rate <= 0:
            raise ValueError("Rate must be > 0")
        self.rate = rate
        self.rng = rng or Aleam()
    
    def sample(self) -> float:
        """Sample from exponential distribution using inverse transform"""
        u = self.rng.random()
        return -math.log(1 - u) / self.rate
    
    def pdf(self, x: float) -> float:
        """Probability density function"""
        if x >= 0:
            return self.rate * math.exp(-self.rate * x)
        return 0.0


class Beta:
    """
    Beta distribution using true randomness.
    
    Beta(α, β) where α, β > 0.
    Support: x ∈ [0, 1]
    """
    
    def __init__(self, alpha: float, beta: float, rng: Optional[AleamBase] = None):
        if alpha <= 0 or beta <= 0:
            raise ValueError("Alpha and beta must be > 0")
        self.alpha = alpha
        self.beta = beta
        self.rng = rng or Aleam()
    
    def sample(self) -> float:
        """
        Sample from Beta distribution using Gamma method.
        If X ~ Gamma(α, 1) and Y ~ Gamma(β, 1), then X/(X+Y) ~ Beta(α, β)
        """
        x = self.rng.gamma(self.alpha, 1)
        y = self.rng.gamma(self.beta, 1)
        return x / (x + y)
    
    def pdf(self, x: float) -> float:
        """Probability density function"""
        if 0 <= x <= 1:
            from scipy.special import beta as beta_func
            return (x ** (self.alpha - 1) * (1 - x) ** (self.beta - 1)) / beta_func(self.alpha, self.beta)
        return 0.0


class Gamma:
    """
    Gamma distribution using true randomness.
    
    Gamma(shape, scale) where shape > 0, scale > 0.
    PDF: f(x) = x^{shape-1} e^{-x/scale} / (Γ(shape) * scale^{shape})
    """
    
    def __init__(self, shape: float, scale: float = 1.0, rng: Optional[AleamBase] = None):
        if shape <= 0 or scale <= 0:
            raise ValueError("Shape and scale must be > 0")
        self.shape = shape
        self.scale = scale
        self.rng = rng or Aleam()
    
    def sample(self) -> float:
        """
        Sample from Gamma distribution using Marsaglia & Tsang method.
        Efficient for shape > 1.
        """
        if self.shape < 1:
            # For shape < 1, use acceptance-rejection with Exponential
            d = self.shape + 1
            while True:
                x = self.rng.exponential()
                u = self.rng.random()
                if u <= math.exp(-x) * x ** (self.shape - 1) / (d ** (self.shape - 1) * math.exp(-d)):
                    return x * self.scale
        else:
            # Marsaglia & Tsang method for shape >= 1
            d = self.shape - 1 / 3
            c = 1 / math.sqrt(9 * d)
            while True:
                v = self.rng.gauss()
                x = d * (1 + c * v) ** 3
                if x <= 0:
                    continue
                u = self.rng.random()
                if u < 1 - 0.0331 * v ** 4:
                    return x * self.scale
                if math.log(u) < 0.5 * v ** 2 + d * (1 - x + math.log(x)):
                    return x * self.scale
    
    def pdf(self, x: float) -> float:
        """Probability density function"""
        if x > 0:
            from scipy.special import gamma
            return (x ** (self.shape - 1) * math.exp(-x / self.scale)) / (gamma(self.shape) * self.scale ** self.shape)
        return 0.0


class Poisson:
    """
    Poisson distribution using true randomness.
    
    Poisson(λ) where λ > 0 (mean and variance).
    """
    
    def __init__(self, lam: float = 1.0, rng: Optional[AleamBase] = None):
        if lam <= 0:
            raise ValueError("Lambda must be > 0")
        self.lam = lam
        self.rng = rng or Aleam()
    
    def sample(self) -> int:
        """
        Sample from Poisson distribution.
        Uses Knuth's algorithm for small λ, normal approximation for large λ.
        """
        if self.lam < 10:
            # Knuth's algorithm (efficient for small lambda)
            L = math.exp(-self.lam)
            k = 0
            p = 1.0
            while p > L:
                p *= self.rng.random()
                k += 1
            return k - 1
        else:
            # Normal approximation with continuity correction for large lambda
            x = self.rng.gauss(self.lam, math.sqrt(self.lam))
            return max(0, int(x + 0.5))
    
    def pdf(self, k: int) -> float:
        """Probability mass function"""
        if k >= 0:
            return math.exp(-self.lam) * (self.lam ** k) / math.factorial(k)
        return 0.0


class Laplace:
    """
    Laplace (double exponential) distribution using true randomness.
    
    Laplace(loc, scale) where scale > 0.
    PDF: f(x) = (1/(2*scale)) * exp(-|x-loc|/scale)
    """
    
    def __init__(self, loc: float = 0.0, scale: float = 1.0, rng: Optional[AleamBase] = None):
        if scale <= 0:
            raise ValueError("Scale must be > 0")
        self.loc = loc
        self.scale = scale
        self.rng = rng or Aleam()
    
    def sample(self) -> float:
        """
        Sample from Laplace distribution.
        Uses inverse transform: F^{-1}(u) = loc - scale * sign(u-0.5) * ln(1 - 2|u-0.5|)
        """
        u = self.rng.random() - 0.5
        return self.loc - self.scale * math.copysign(1, u) * math.log(1 - 2 * abs(u))
    
    def pdf(self, x: float) -> float:
        """Probability density function"""
        return (1 / (2 * self.scale)) * math.exp(-abs(x - self.loc) / self.scale)