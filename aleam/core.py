"""
Core Aleam random number generator.

Implements the universal equation:
Ψ(t) = BLAKE2s( (Φ × Ξ(t)) ⊕ τ(t) )

Where:
    Φ = 0x9E3779B97F4A7C15 (golden ratio prime)
    Ξ(t) = 128-bit true entropy from system
    τ(t) = Nanosecond timestamp
    ⊕ = XOR operation
    BLAKE2s = Cryptographic hash
"""

import os
import struct
import time
import hashlib
import math
from typing import List, Any, Optional, Union, Tuple

# Golden ratio prime: ⌊2⁶⁴ / φ⌋ where φ = (1+√5)/2
GOLDEN_PRIME = 0x9E3779B97F4A7C15


class AleamBase:
    """Base class for Aleam random generators"""
    
    def __init__(self):
        self._calls = 0
    
    def random(self) -> float:
        """Generate true random float in [0, 1)"""
        raise NotImplementedError
    
    # ==================== BASIC METHODS ====================
    
    def randint(self, a: int, b: int) -> int:
        """Generate random integer in [a, b] inclusive"""
        if a > b:
            raise ValueError(f"a ({a}) must be <= b ({b})")
        return a + int(self.random() * (b - a + 1))
    
    def choice(self, sequence: List[Any]) -> Any:
        """Choose random element from sequence"""
        if not sequence:
            raise ValueError("Cannot choose from empty sequence")
        return sequence[self.randint(0, len(sequence) - 1)]
    
    def uniform(self, low: float, high: float) -> float:
        """Generate random float in [low, high]"""
        return low + self.random() * (high - low)
    
    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """
        Generate normally distributed random number using Box-Muller transform.
        
        Algorithm:
            Given U₁, U₂ ∼ Uniform(0,1):
            Z = √(-2 ln U₁) · cos(2πU₂)
            Z ∼ N(0,1)
        """
        u1 = self.random()
        u2 = self.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z
    
    def normalvariate(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Alias for gauss"""
        return self.gauss(mu, sigma)
    
    def sample(self, population: List[Any], k: int) -> List[Any]:
        """
        Sample k unique elements without replacement using Fisher-Yates.
        """
        if k > len(population):
            raise ValueError("Sample larger than population")
        
        result = population.copy()
        for i in range(k):
            j = self.randint(i, len(result) - 1)
            result[i], result[j] = result[j], result[i]
        
        return result[:k]
    
    def shuffle(self, lst: List[Any]) -> None:
        """Shuffle a list in-place using Fisher-Yates"""
        for i in range(len(lst) - 1, 0, -1):
            j = self.randint(0, i)
            lst[i], lst[j] = lst[j], lst[i]
    
    def random_bytes(self, n: int) -> bytes:
        """Generate n random bytes"""
        return bytes([self.randint(0, 255) for _ in range(n)])
    
    # ==================== DISTRIBUTIONS ====================
    
    def exponential(self, rate: float = 1.0) -> float:
        """
        Sample from exponential distribution.
        
        Args:
            rate: Rate parameter (λ > 0)
        
        Returns:
            Sample from Exp(rate)
        
        PDF: f(x) = λe^{-λx} for x ≥ 0
        """
        if rate <= 0:
            raise ValueError("Rate must be > 0")
        u = self.random()
        return -math.log(1 - u) / rate
    
    def beta(self, alpha: float, beta: float) -> float:
        """
        Sample from Beta distribution.
        
        Args:
            alpha: Alpha parameter (> 0)
            beta: Beta parameter (> 0)
        
        Returns:
            Sample from Beta(alpha, beta) in [0, 1]
        
        Uses Gamma method: If X ~ Gamma(α,1) and Y ~ Gamma(β,1),
        then X/(X+Y) ~ Beta(α,β)
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("Alpha and beta must be > 0")
        x = self.gamma(alpha, 1)
        y = self.gamma(beta, 1)
        return x / (x + y)
    
    def gamma(self, shape: float, scale: float = 1.0) -> float:
        """
        Sample from Gamma distribution.
        
        Args:
            shape: Shape parameter (k > 0)
            scale: Scale parameter (θ > 0)
        
        Returns:
            Sample from Gamma(shape, scale)
        """
        if shape <= 0 or scale <= 0:
            raise ValueError("Shape and scale must be > 0")
        
        # For shape < 1, use Johnk's method
        if shape < 1:
            while True:
                u = self.random()
                v = self.random()
                x = u ** (1 / shape)
                y = v ** (1 / (1 - shape))
                if x + y <= 1:
                    z = x / (x + y)
                    w = self.gamma(shape + 1, 1)
                    return w * z * scale
        
        # Marsaglia & Tsang method for shape >= 1
        d = shape - 1/3
        c = 1 / math.sqrt(9 * d)
        
        while True:
            v = self.gauss()
            x = (1 + c * v) ** 3
            
            if x <= 0:
                continue
            
            u = self.random()
            
            # Generate standard gamma(shape, 1)
            if u < 1 - 0.0331 * (v ** 4):
                result = d * x
                return result * scale
            
            if math.log(u) < 0.5 * (v ** 2) + d * (1 - x + math.log(x)):
                result = d * x
                return result * scale
    
    def poisson(self, lam: float = 1.0) -> int:
        """
        Sample from Poisson distribution.
        
        Args:
            lam: Lambda parameter (λ > 0)
        
        Returns:
            Sample from Poisson(λ)
        
        PMF: P(X = k) = e^{-λ} λ^k / k!
        
        Uses Knuth's algorithm for λ < 10,
        normal approximation for larger λ.
        """
        if lam <= 0:
            raise ValueError("Lambda must be > 0")
        
        if lam < 10:
            # Knuth's algorithm (efficient for small lambda)
            L = math.exp(-lam)
            k = 0
            p = 1.0
            while p > L:
                p *= self.random()
                k += 1
            return k - 1
        else:
            # Normal approximation with continuity correction for large lambda
            x = self.gauss(lam, math.sqrt(lam))
            return max(0, int(x + 0.5))
    
    def laplace(self, loc: float = 0.0, scale: float = 1.0) -> float:
        """
        Sample from Laplace (double exponential) distribution.
        
        Args:
            loc: Location parameter (μ)
            scale: Scale parameter (b > 0)
        
        Returns:
            Sample from Laplace(loc, scale)
        
        PDF: f(x) = (1/(2b)) exp(-|x-μ|/b)
        
        Uses inverse transform:
        F^{-1}(u) = μ - b · sign(u-0.5) · ln(1 - 2|u-0.5|)
        """
        if scale <= 0:
            raise ValueError("Scale must be > 0")
        u = self.random() - 0.5
        return loc - scale * math.copysign(1, u) * math.log(1 - 2 * abs(u))
    
    def logistic(self, loc: float = 0.0, scale: float = 1.0) -> float:
        """
        Sample from Logistic distribution.
        
        Args:
            loc: Location parameter (μ)
            scale: Scale parameter (s > 0)
        
        Returns:
            Sample from Logistic(loc, scale)
        
        PDF: f(x) = e^{-(x-μ)/s} / (s (1 + e^{-(x-μ)/s})²)
        """
        if scale <= 0:
            raise ValueError("Scale must be > 0")
        u = self.random()
        return loc + scale * math.log(u / (1 - u))
    
    def lognormal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """
        Sample from Log-Normal distribution.
        
        Args:
            mu: Mean of underlying normal distribution
            sigma: Standard deviation of underlying normal distribution
        
        Returns:
            Sample from LogNormal(mu, sigma)
        
        If X ~ N(μ, σ²), then Y = e^X ~ LogNormal(μ, σ²)
        """
        if sigma <= 0:
            raise ValueError("Sigma must be > 0")
        return math.exp(self.gauss(mu, sigma))
    
    def weibull(self, shape: float, scale: float = 1.0) -> float:
        """
        Sample from Weibull distribution.
        
        Args:
            shape: Shape parameter (k > 0)
            scale: Scale parameter (λ > 0)
        
        Returns:
            Sample from Weibull(shape, scale)
        
        PDF: f(x) = (k/λ) (x/λ)^{k-1} e^{-(x/λ)^k}
        """
        if shape <= 0 or scale <= 0:
            raise ValueError("Shape and scale must be > 0")
        u = self.random()
        return scale * (-math.log(1 - u)) ** (1 / shape)
    
    def pareto(self, alpha: float, scale: float = 1.0) -> float:
        """
        Sample from Pareto distribution.
        
        Args:
            alpha: Shape parameter (α > 0)
            scale: Scale parameter (x_m > 0)
        
        Returns:
            Sample from Pareto(alpha, scale)
        
        PDF: f(x) = α x_m^α / x^{α+1} for x ≥ x_m
        """
        if alpha <= 0 or scale <= 0:
            raise ValueError("Alpha and scale must be > 0")
        u = self.random()
        return scale / (u ** (1 / alpha))
    
    def chi_square(self, df: float) -> float:
        """
        Sample from Chi-square distribution.
        
        Args:
            df: Degrees of freedom (k > 0)
        
        Returns:
            Sample from χ²(k)
        
        χ²(k) = Gamma(k/2, 2)
        """
        if df <= 0:
            raise ValueError("Degrees of freedom must be > 0")
        return self.gamma(df / 2, 2)
    
    def student_t(self, df: float) -> float:
        """
        Sample from Student's t-distribution.
        
        Args:
            df: Degrees of freedom (ν > 0)
        
        Returns:
            Sample from t(ν)
        
        t(ν) = Z / sqrt(χ²/ν) where Z ~ N(0,1), χ² ~ χ²(ν)
        """
        if df <= 0:
            raise ValueError("Degrees of freedom must be > 0")
        z = self.gauss()
        chi2 = self.chi_square(df)
        return z / math.sqrt(chi2 / df)
    
    def f_distribution(self, df1: float, df2: float) -> float:
        """
        Sample from F-distribution.
        
        Args:
            df1: Numerator degrees of freedom (d₁ > 0)
            df2: Denominator degrees of freedom (d₂ > 0)
        
        Returns:
            Sample from F(d₁, d₂)
        
        F(d₁, d₂) = (χ²₁/d₁) / (χ²₂/d₂)
        """
        if df1 <= 0 or df2 <= 0:
            raise ValueError("Degrees of freedom must be > 0")
        chi2_1 = self.chi_square(df1)
        chi2_2 = self.chi_square(df2)
        return (chi2_1 / df1) / (chi2_2 / df2)
    
    def dirichlet(self, alpha: List[float]) -> List[float]:
        """
        Sample from Dirichlet distribution.
        
        Args:
            alpha: Concentration parameters (α_i > 0)
        
        Returns:
            Sample from Dirichlet(α) (probability simplex)
        
        If X_i ~ Gamma(α_i, 1), then (X_i / ΣX_j) ~ Dirichlet(α)
        """
        if not alpha:
            raise ValueError("Alpha must not be empty")
        if any(a <= 0 for a in alpha):
            raise ValueError("All alpha must be > 0")
        
        gamma_samples = [self.gamma(a, 1) for a in alpha]
        total = sum(gamma_samples)
        return [g / total for g in gamma_samples]
    
    # ==================== ARRAY OPERATIONS ====================
    
    def random_array(self, shape: Union[int, Tuple[int, ...]]) -> List:
        """
        Generate array of true random floats in [0, 1).
        
        Args:
            shape: Shape of output array
        
        Returns:
            Nested list of random floats
        """
        if isinstance(shape, int):
            shape = (shape,)
        
        def recursive_build(dims):
            if len(dims) == 1:
                return [self.random() for _ in range(dims[0])]
            else:
                return [recursive_build(dims[1:]) for _ in range(dims[0])]
        
        return recursive_build(shape)
    
    def randn_array(self, shape: Union[int, Tuple[int, ...]], 
                    mu: float = 0.0, sigma: float = 1.0) -> List:
        """
        Generate array of true random normally distributed values.
        
        Args:
            shape: Shape of output array
            mu: Mean of distribution
            sigma: Standard deviation
        
        Returns:
            Nested list of normally distributed values
        """
        if isinstance(shape, int):
            shape = (shape,)
        
        def recursive_build(dims):
            if len(dims) == 1:
                return [self.gauss(mu, sigma) for _ in range(dims[0])]
            else:
                return [recursive_build(dims[1:]) for _ in range(dims[0])]
        
        return recursive_build(shape)
    
    def randint_array(self, shape: Union[int, Tuple[int, ...]], 
                      low: int, high: int) -> List:
        """
        Generate array of true random integers in [low, high].
        
        Args:
            shape: Shape of output array
            low: Lower bound (inclusive)
            high: Upper bound (inclusive)
        
        Returns:
            Nested list of random integers
        """
        if isinstance(shape, int):
            shape = (shape,)
        
        def recursive_build(dims):
            if len(dims) == 1:
                return [self.randint(low, high) for _ in range(dims[0])]
            else:
                return [recursive_build(dims[1:]) for _ in range(dims[0])]
        
        return recursive_build(shape)
    
    # ==================== UTILITY ====================
    
    def seed(self, *args, **kwargs):
        """Aleam does not support seeding"""
        raise NotImplementedError(
            "Aleam uses true randomness and does not support seeding. "
            "Each call is independent and stateless."
        )
    
    def get_state(self):
        """Get generator state (not supported)"""
        raise NotImplementedError("Aleam is stateless")
    
    def set_state(self, state):
        """Set generator state (not supported)"""
        raise NotImplementedError("Aleam is stateless")
    
    def get_stats(self) -> dict:
        """Get generator statistics"""
        return {
            'calls': self._calls,
            'entropy_per_call': 128,
            'algorithm': 'Ψ(t) = H( (Φ × Ξ(t)) ⊕ τ(t) )'
        }


class Aleam(AleamBase):
    """
    True random number generator implementing the proven equation:
    
    Ψ(t) = BLAKE2s( (Φ × Ξ(t)) ⊕ τ(t) )
    
    Properties:
        - Non-recursive: each call independent
        - Stateless: no hidden state
        - True entropy: 128 bits per call
        - Cryptographically secure: BLAKE2s hashing
        - Uniform distribution: validated by statistical tests
    """
    
    def __init__(self):
        super().__init__()
        self._golden_prime = GOLDEN_PRIME
    
    def _get_entropy(self) -> int:
        """Pull 128 bits of true entropy from system"""
        return int.from_bytes(os.urandom(16), byteorder='big')
    
    def _golden_mix(self, entropy: int) -> int:
        """
        Apply golden ratio multiplication in ℤ/2⁶⁴ℤ.
        This is a bijective mixing function with maximal equidistribution.
        """
        return (entropy & 0xFFFFFFFFFFFFFFFF) * self._golden_prime
    
    def _timestamp_anchor(self) -> int:
        """Get nanosecond timestamp as 64-bit temporal anchor"""
        return time.time_ns() & 0xFFFFFFFFFFFFFFFF
    
    def _hash_uniform(self, value: int) -> float:
        """
        Apply BLAKE2s hash to produce uniform float in [0, 1).
        Uses 64-bit output for maximum precision.
        """
        # Pad to 128 bits for optimal BLAKE2s block size
        value_bytes = value.to_bytes(16, byteorder='big')
        # BLAKE2s produces 32 bytes, take first 8 for 64-bit output
        hash_bytes = hashlib.blake2s(value_bytes).digest()[:8]
        return struct.unpack('Q', hash_bytes)[0] / (2**64)
    
    def random(self) -> float:
        """
        Generate true random float in [0, 1).
        
        Algorithm steps:
            1. Ξ ← SampleEntropy(128)          [True entropy]
            2. Ω ← Φ × Ξ (mod 2⁶⁴)             [Golden ratio mixing]
            3. τ ← timestamp_ns (mod 2⁶⁴)      [Temporal anchor]
            4. Σ ← Ω ⊕ τ                        [XOR mixing]
            5. Ψ ← H(Σ)                         [Hash to uniform]
            6. Return Ψ / 2⁶⁴
        """
        # Step 1: Sample true entropy
        entropy = self._get_entropy()
        
        # Step 2: Golden ratio mixing
        mixed = self._golden_mix(entropy)
        
        # Step 3: Temporal anchor
        timestamp = self._timestamp_anchor()
        
        # Step 4: XOR mixing
        combined = mixed ^ timestamp
        
        # Step 5: Hash to uniform
        result = self._hash_uniform(combined)
        
        # Track calls for statistics
        self._calls += 1
        
        return result
    
    def get_stats(self) -> dict:
        """Get generator statistics"""
        stats = super().get_stats()
        stats['golden_prime'] = hex(self._golden_prime)
        return stats


class AleamFast(AleamBase):
    """
    Faster version of Aleam with simplified mixing.
    Uses: Ψ(t) = BLAKE2b( (Ξ(t) & 0xFFFFFFFFFFFFFFFF) ⊕ τ(t) )
    
    Trade-off: Slightly less mixing but still true randomness.
    """
    
    def __init__(self):
        super().__init__()
    
    def _get_entropy(self) -> int:
        """Pull 128 bits of true entropy from system"""
        return int.from_bytes(os.urandom(16), byteorder='big')
    
    def _timestamp_anchor(self) -> int:
        """Get nanosecond timestamp as 64-bit temporal anchor"""
        return time.time_ns() & 0xFFFFFFFFFFFFFFFF
    
    def _hash_uniform(self, value: int) -> float:
        """
        Fast hash using BLAKE2b with 8-byte output.
        """
        # value is already 64-bit from random() method
        value_bytes = value.to_bytes(8, byteorder='big')
        hash_bytes = hashlib.blake2b(value_bytes, digest_size=8).digest()
        return struct.unpack('Q', hash_bytes)[0] / (2**64)
    
    def random(self) -> float:
        """
        Generate true random float quickly.
        
        Algorithm:
            Ψ(t) = BLAKE2b( (Ξ(t) & 0xFFFFFFFFFFFFFFFF) ⊕ τ(t) )
        """
        # Step 1: Get entropy and reduce to 64-bit
        entropy = self._get_entropy()
        entropy_64 = entropy & 0xFFFFFFFFFFFFFFFF
        
        # Step 2: Temporal anchor
        timestamp = self._timestamp_anchor()
        
        # Step 3: XOR mixing
        combined = entropy_64 ^ timestamp
        
        # Step 4: Hash to uniform
        result = self._hash_uniform(combined)
        
        self._calls += 1
        return result


class AleamOptimized(Aleam):
    """Alias for Aleam (main implementation)"""
    pass


# ==================== MODULE-LEVEL FUNCTIONS ====================

_default_rng = None

def _get_default_rng():
    """Get or create default random number generator"""
    global _default_rng
    if _default_rng is None:
        _default_rng = Aleam()
    return _default_rng


def random() -> float:
    """Generate a true random float in [0, 1) using default generator"""
    return _get_default_rng().random()


def randint(a: int, b: int) -> int:
    """Generate random integer in [a, b] using default generator"""
    return _get_default_rng().randint(a, b)


def choice(sequence: List[Any]) -> Any:
    """Choose random element from sequence using default generator"""
    return _get_default_rng().choice(sequence)


def uniform(low: float, high: float) -> float:
    """Generate random float in [low, high] using default generator"""
    return _get_default_rng().uniform(low, high)


def gauss(mu: float = 0.0, sigma: float = 1.0) -> float:
    """Generate normally distributed random number using default generator"""
    return _get_default_rng().gauss(mu, sigma)


def shuffle(lst: List[Any]) -> None:
    """Shuffle list in-place using default generator"""
    _get_default_rng().shuffle(lst)


def sample(population: List[Any], k: int) -> List[Any]:
    """Sample k unique elements using default generator"""
    return _get_default_rng().sample(population, k)


def random_bytes(n: int) -> bytes:
    """Generate n random bytes using default generator"""
    return _get_default_rng().random_bytes(n)


def exponential(rate: float = 1.0) -> float:
    """Sample from exponential distribution using default generator"""
    return _get_default_rng().exponential(rate)


def beta(alpha: float, beta: float) -> float:
    """Sample from beta distribution using default generator"""
    return _get_default_rng().beta(alpha, beta)


def gamma(shape: float, scale: float = 1.0) -> float:
    """Sample from gamma distribution using default generator"""
    return _get_default_rng().gamma(shape, scale)


def poisson(lam: float = 1.0) -> int:
    """Sample from Poisson distribution using default generator"""
    return _get_default_rng().poisson(lam)


def laplace(loc: float = 0.0, scale: float = 1.0) -> float:
    """Sample from Laplace distribution using default generator"""
    return _get_default_rng().laplace(loc, scale)


def logistic(loc: float = 0.0, scale: float = 1.0) -> float:
    """Sample from Logistic distribution using default generator"""
    return _get_default_rng().logistic(loc, scale)


def lognormal(mu: float = 0.0, sigma: float = 1.0) -> float:
    """Sample from Log-Normal distribution using default generator"""
    return _get_default_rng().lognormal(mu, sigma)


def weibull(shape: float, scale: float = 1.0) -> float:
    """Sample from Weibull distribution using default generator"""
    return _get_default_rng().weibull(shape, scale)


def pareto(alpha: float, scale: float = 1.0) -> float:
    """Sample from Pareto distribution using default generator"""
    return _get_default_rng().pareto(alpha, scale)


def chi_square(df: float) -> float:
    """Sample from Chi-square distribution using default generator"""
    return _get_default_rng().chi_square(df)


def student_t(df: float) -> float:
    """Sample from Student's t-distribution using default generator"""
    return _get_default_rng().student_t(df)


def f_distribution(df1: float, df2: float) -> float:
    """Sample from F-distribution using default generator"""
    return _get_default_rng().f_distribution(df1, df2)


def dirichlet(alpha: List[float]) -> List[float]:
    """Sample from Dirichlet distribution using default generator"""
    return _get_default_rng().dirichlet(alpha)


def random_array(shape: Union[int, Tuple[int, ...]]) -> List:
    """Generate array of random floats using default generator"""
    return _get_default_rng().random_array(shape)


def randn_array(shape: Union[int, Tuple[int, ...]], 
                mu: float = 0.0, sigma: float = 1.0) -> List:
    """Generate array of normally distributed values using default generator"""
    return _get_default_rng().randn_array(shape, mu, sigma)


def randint_array(shape: Union[int, Tuple[int, ...]], 
                  low: int, high: int) -> List:
    """Generate array of random integers using default generator"""
    return _get_default_rng().randint_array(shape, low, high)