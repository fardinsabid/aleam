# Aleam Documentation

**True randomness. No recursion. No state. Just entropy.**

*A cryptographically secure random number generator built for AI.*

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core API Reference](#core-api-reference)
5. [Distributions API](#distributions-api)
6. [AI/ML Features](#aiml-features)
7. [Array Operations](#array-operations)
8. [Framework Integrations](#framework-integrations)
   - [PyTorch](#pytorch-integration)
   - [TensorFlow](#tensorflow-integration)
   - [JAX](#jax-integration)
   - [CuPy](#cupy-integration)
   - [Pandas](#pandas-integration)
   - [NumPy](#numpy-integration)
9. [CUDA Acceleration](#cuda-acceleration)
10. [Performance Benchmarks](#performance-benchmarks)
11. [Mathematical Foundation](#mathematical-foundation)
12. [FAQ](#faq)
13. [License](#license)

---

## Introduction

Aleam is a Python library that provides **true random number generation** using the proven equation:

```
Ψ(t) = BLAKE2s( (Φ × Ξ(t)) ⊕ τ(t) )
```

Where:
- **Φ** = Golden ratio prime (`0x9E3779B97F4A7C15`)
- **Ξ(t)** = 128-bit true entropy from system
- **τ(t)** = Nanosecond timestamp
- **⊕** = XOR operation
- **BLAKE2s** = Cryptographic hash

### Why Aleam?

Unlike traditional pseudo-random generators (Mersenne Twister, PCG, `random.random`), Aleam is:

| Feature | Aleam | Traditional PRNGs |
|---------|----------|-------------------|
| **Randomness Type** | True randomness | Pseudo-randomness |
| **Recursive** | ❌ No | ✅ Yes |
| **Stateful** | ❌ No | ✅ Yes |
| **Seeding Required** | ❌ No | ✅ Yes |
| **Periodic** | ❌ No | ✅ Yes |
| **Cryptographically Secure** | ✅ Yes | ❌ No |
| **Entropy Guarantee** | 128 bits/call | None |

---

## Installation

### Quick Install

```bash
pip install aleam
```

### From Source

```bash
git clone https://github.com/fardinsabid/aleam.git
cd aleam
pip install -e .
```

### Optional Dependencies

```bash
# PyTorch integration
pip install torch

# TensorFlow integration
pip install tensorflow

# JAX integration
pip install jax jaxlib

# CuPy GPU acceleration
pip install cupy-cuda12x

# Data science integration
pip install pandas polars xarray

# Bayesian inference
pip install pymc
```

---

## Quick Start

```python
import aleam as al

# Create a true random generator
rng = al.Aleam()

# Basic randomness
x = rng.random()              # Random float in [0, 1)
y = rng.randint(1, 100)       # Random integer between 1 and 100
z = rng.choice(['AI', 'ML'])  # Random choice from list

# Statistical distributions
normal = rng.gauss(0, 1)      # Gaussian distribution
uniform = rng.uniform(5, 10)  # Uniform distribution
exponential = rng.exponential(rate=1.0)  # Exponential distribution

# AI/ML features
noise = rng.gauss(0, 0.1)               # Gaussian noise for gradients
batch = rng.sample(range(10000), 64)    # Mini-batch sampling

# Array operations
arr = rng.random_array((100, 100))      # 2D array of random floats
```

---

## Core API Reference

### Generator Classes

| Class | Description |
|-------|-------------|
| `Aleam` | Main random generator (optimized, recommended) |
| `AleamFast` | Faster version with simplified mixing |
| `AleamOptimized` | Alias for `Aleam` |

### Core Methods

| Method | Description | Example |
|--------|-------------|---------|
| `random()` | Random float in [0, 1) | `rng.random()` |
| `randint(a, b)` | Random integer in [a, b] | `rng.randint(1, 100)` |
| `choice(seq)` | Random element from sequence | `rng.choice(['a', 'b', 'c'])` |
| `uniform(low, high)` | Random float in [low, high] | `rng.uniform(5.0, 10.0)` |
| `gauss(mu, sigma)` | Normally distributed value | `rng.gauss(0, 1)` |
| `normalvariate(mu, sigma)` | Alias for `gauss` | `rng.normalvariate(0, 1)` |
| `sample(pop, k)` | Sample k unique elements | `rng.sample(range(100), 10)` |
| `shuffle(lst)` | Shuffle list in-place | `rng.shuffle(my_list)` |
| `random_bytes(n)` | Generate n random bytes | `rng.random_bytes(16)` |

---

## Distributions API

Aleam provides a comprehensive set of statistical distributions:

| Distribution | Method | Parameters |
|--------------|--------|------------|
| **Normal** | `gauss(mu, sigma)` | μ (mean), σ (std dev) |
| **Uniform** | `uniform(low, high)` | a (low), b (high) |
| **Exponential** | `exponential(rate)` | λ (rate) |
| **Beta** | `beta(alpha, beta)` | α, β |
| **Gamma** | `gamma(shape, scale)` | k (shape), θ (scale) |
| **Poisson** | `poisson(lam)` | λ (mean) |
| **Laplace** | `laplace(loc, scale)` | μ (loc), b (scale) |
| **Logistic** | `logistic(loc, scale)` | μ (loc), s (scale) |
| **Log-Normal** | `lognormal(mu, sigma)` | μ, σ (of underlying normal) |
| **Weibull** | `weibull(shape, scale)` | k (shape), λ (scale) |
| **Pareto** | `pareto(alpha, scale)` | α (shape), x_m (scale) |
| **Chi-square** | `chi_square(df)` | k (degrees of freedom) |
| **Student's t** | `student_t(df)` | ν (degrees of freedom) |
| **F-distribution** | `f_distribution(df1, df2)` | d₁, d₂ (degrees of freedom) |
| **Dirichlet** | `dirichlet(alpha)` | α (concentration parameters) |

### Examples

```python
import aleam as al

rng = al.Aleam()

# Exponential distribution (waiting times)
wait_time = rng.exponential(rate=0.5)

# Beta distribution (probabilities)
probability = rng.beta(alpha=2, beta=5)

# Gamma distribution (waiting times for multiple events)
total_time = rng.gamma(shape=3, scale=2)

# Poisson distribution (event counts)
events = rng.poisson(lam=5)

# Laplace distribution (double exponential)
noise = rng.laplace(loc=0, scale=1)

# Dirichlet distribution (probability simplex)
probs = rng.dirichlet([1, 2, 3])  # Returns [p1, p2, p3] summing to 1
```

---

## AI/ML Features

### `AIRandom` Class

AI-specific random utilities.

```python
from aleam import AIRandom

ai = AIRandom()

# Gradient noise for training
noise = ai.gradient_noise(shape=(3, 3), scale=0.1)

# Latent space vector
latent = ai.latent_vector(dim=512, distribution="normal")

# Dropout mask
mask = ai.dropout_mask(size=100, probability=0.3)

# Data augmentation parameters
params = ai.augmentation_params()

# Mini-batch sampling
batch = ai.mini_batch(dataset_size=10000, batch_size=64)

# RL exploration noise
exploration = ai.exploration_noise(action_dim=4, scale=0.2)
```

### `GradientNoise` Class

Gradient noise injection with decay.

```python
from aleam import GradientNoise

noise = GradientNoise(scale=0.01, decay=0.99)

# During training loop
for step in range(steps):
    gradients = compute_gradients()
    noisy_gradients = noise.add_noise(gradients)
    optimizer.step(noisy_gradients)
```

### `LatentSampler` Class

Latent space sampling for generative models.

```python
from aleam import LatentSampler

sampler = LatentSampler(latent_dim=128, distribution="normal")

# Sample vectors
vectors = sampler.sample(n=10)

# Interpolate between two vectors
z1 = sampler.sample(1)[0]
z2 = sampler.sample(1)[0]
interpolated = sampler.interpolate(z1, z2, steps=5)
```

---

## Array Operations

### Module-Level Functions

| Function | Description |
|----------|-------------|
| `random_array(shape)` | Generate array of random floats in [0, 1) |
| `randn_array(shape, mu, sigma)` | Generate array of normally distributed values |
| `randint_array(shape, low, high)` | Generate array of random integers |
| `choice_array(a, size, replace, p)` | Sample from array with/without replacement |

### Examples

```python
import aleam as al

# 1D array of random floats
arr_1d = al.random_array(100)

# 2D array of random floats
arr_2d = al.random_array((10, 10))

# Normal distribution array
norm_arr = al.randn_array((1000,), mu=0, sigma=1)

# Integer array
int_arr = al.randint_array((50,), low=0, high=10)

# Sample from list with weights
fruits = ['apple', 'banana', 'cherry']
weights = [0.5, 0.3, 0.2]
choices = al.choice_array(fruits, size=(100,), p=weights)

# Sample without replacement
unique = al.choice_array(10, size=(5,), replace=False)
```

---

## Framework Integrations

### PyTorch Integration

```python
import torch
import aleam as al

# Create generator
gen = al.TorchGenerator(device='cuda' if torch.cuda.is_available() else 'cpu')

# Generate tensors
tensor = gen.randn(100, 100)              # N(0,1)
tensor = gen.rand(100, 100)               # Uniform(0,1)
tensor = gen.randint(0, 10, (100, 100))   # Integers
tensor = gen.normal(0, 1, (100,))         # Custom normal
tensor = gen.uniform(0, 1, (100,))        # Custom uniform

# Use with existing PyTorch models
model = MyModel()
with torch.no_grad():
    for param in model.parameters():
        param.data = gen.randn_like(param)
```

### TensorFlow Integration

```python
import tensorflow as tf
import aleam as al

# Create generator
gen = al.TFGenerator()

# Generate tensors
tensor = gen.normal((100, 100), mean=0, stddev=1)
tensor = gen.uniform((100, 100), minval=0, maxval=1)
tensor = gen.randint((100, 100), minval=0, maxval=10)

# Shuffle tensors
shuffled = gen.shuffle(my_tensor)

# Use in TensorFlow models
with tf.device('/GPU:0'):
    noise = gen.normal((batch_size, latent_dim))
```

### JAX Integration

```python
import jax
import aleam as al

# Create generator
gen = al.JAXGenerator()

# Generate random key with true entropy
key = gen.key()

# Use with JAX random functions
tensor = jax.random.normal(key, (100, 100))
tensor = jax.random.uniform(key, (100, 100))

# Direct generation
tensor = gen.normal((100, 100), mean=0, stddev=1)
tensor = gen.uniform((100, 100), minval=0, maxval=1)
tensor = gen.randint((100, 100), minval=0, maxval=10)
```

### CuPy Integration

```python
import cupy as cp
import aleam as al

# Create generator
gen = al.CuPyGenerator()

# Generate arrays directly on GPU
arr = gen.random((1000, 1000), dtype='float32')
arr = gen.randn((1000, 1000), mu=0, sigma=1)
arr = gen.randint((1000, 1000), low=0, high=10)

# Use with CuPy operations
result = cp.dot(arr, arr.T)
```

### Pandas Integration

```python
import pandas as pd
import aleam as al

# Create generator
gen = al.PandasGenerator()

# Generate Series
series = gen.series(100, distribution="normal", params={'mu': 0, 'sigma': 1})

# Generate DataFrame
df = gen.dataframe(100, columns=['a', 'b', 'c'],
                   distributions={
                       'a': ('normal', {'mu': 0, 'sigma': 1}),
                       'b': ('uniform', {'low': 0, 'high': 10}),
                       'c': ('poisson', {'lam': 5})
                   })

# Sample from Series
sample = gen.choice(series, size=10)

# Shuffle DataFrame
shuffled = gen.shuffle(df)
```

### NumPy Integration

Aleam provides NumPy-style array operations out of the box:

```python
import aleam as al
import numpy as np

# Generate arrays
arr = al.random_array((100, 100))      # Like np.random.rand
arr = al.randn_array((100, 100))       # Like np.random.randn
arr = al.randint_array((100,), 0, 10)  # Like np.random.randint

# Convert to NumPy array when needed
np_arr = np.array(arr)
```

---

## CUDA Acceleration

Aleam provides GPU acceleration through multiple backends:

### Unified CUDA Generator

```python
import aleam as al

# Create CUDA generator (auto-detects available backends)
cuda_gen = al.CUDAGenerator()

# CuPy backend
cupy_arr = cuda_gen.cupy_random((10000, 10000))

# PyTorch CUDA backend
torch_tensor = cuda_gen.torch_randn(10000, 10000, device='cuda')

# TensorFlow GPU backend
tf_tensor = cuda_gen.tf_random_normal((10000, 10000))

# JAX GPU backend
jax_arr = cuda_gen.jax_randn((10000, 10000))
```

### Automatic GPU Detection

Aleam automatically selects the best available GPU backend:

```python
import aleam as al

# Auto-detects PyTorch CUDA > CuPy > TensorFlow GPU > CPU
cuda_gen = al.CUDAGenerator()
tensor = cuda_gen.torch_randn(10000, 10000)  # Uses GPU if available
```

### Performance Comparison

| Method | Speed (elements/sec) |
|--------|---------------------|
| CPU (Python) | ~250,000 |
| CPU (NumPy) | ~5,000,000 |
| CuPy GPU | ~50,000,000 |
| PyTorch CUDA | ~100,000,000 |
| TensorFlow GPU | ~80,000,000 |
| JAX GPU | ~90,000,000 |

---

## Performance Benchmarks

### Core Operations

| Operation | Aleam | Python random | Speed Ratio |
|-----------|----------|---------------|-------------|
| `random()` | 270,000 ops/sec | 10,000,000 ops/sec | ~37x slower |
| `randint()` | 255,000 ops/sec | 7,500,000 ops/sec | ~29x slower |
| `gauss()` | 129,000 ops/sec | 6,000,000 ops/sec | ~46x slower |

> Note: Aleam is ~37x slower than Python's random on CPU — this is expected for true randomness. The trade-off is genuine entropy and cryptographic security. On GPU, Aleam can achieve 100M+ ops/sec, making it faster than CPU pseudo-random!

### Distribution Performance

| Distribution | Speed (ops/sec) |
|--------------|-----------------|
| `random()` | 270,000 |
| `uniform()` | 253,000 |
| `exponential()` | 248,000 |
| `laplace()` | 236,000 |
| `gauss()` | 129,000 |
| `gamma()` | 79,000 |
| `poisson()` | 46,000 |
| `beta()` | 41,000 |

---

## Mathematical Foundation

### The Core Equation

```
Ψ(t) = BLAKE2s( (Φ × Ξ(t)) ⊕ τ(t) )
```

### Properties

| Property | Value | Proof |
|----------|-------|-------|
| Uniformity | χ² = 21.40 < 30.14 | Statistical validation |
| Independence | Max |r| = 0.0094 | Autocorrelation test |
| Entropy | ≥ 64 bits/output | Information theory |
| Non-recursive | No state | By construction |

### Statistical Validation

| Test | Result | Status |
|------|--------|--------|
| Mean | 0.499578 | ✓ |
| Variance | 0.083154 | ✓ |
| Chi-square | 21.40 | ✓ PASS |
| Autocorrelation | 0.0094 | ✓ EXCELLENT |
| π Estimation | 0.0105% error | ✓ EXCELLENT |
| Shannon Entropy | 0.9999 | ✓ NEAR-PERFECT |

For full mathematical derivation, see the [research paper](ALEAM_RESEARCH_PAPER.md).

---

## FAQ

### Why not use Python's random module?

Python's `random` uses a pseudo-random number generator (Mersenne Twister) which is:
- **Recursive** — each number depends on the previous
- **Deterministic** — same seed produces same sequence
- **Periodic** — eventually repeats

Aleam provides **true randomness** with no recursion, no state, and no seeds.

### Is Aleam cryptographically secure?

Yes. Aleam uses:
- True entropy from the operating system
- BLAKE2s cryptographic hash function
- No internal state to extract or predict

### Can I reproduce results?

No. By design, Aleam does not support seeding. If you need reproducible randomness, use Python's `random` module. Aleam is for applications where true unpredictability matters.

### How fast is Aleam?

Aleam generates ~270,000 random numbers per second on a typical CPU. This is sufficient for most AI/ML workloads including:
- Gradient noise injection
- Mini-batch sampling
- Reinforcement learning exploration
- Data augmentation

On GPU with CUDA, Aleam can achieve 100M+ random numbers per second — faster than CPU pseudo-random!

### Does Aleam work on GPU?

Yes! Aleam supports:
- PyTorch CUDA
- TensorFlow GPU
- JAX GPU
- CuPy
- Numba CUDA

### What distributions are available?

Aleam provides 15+ distributions:
- Normal, Uniform, Exponential, Beta, Gamma, Poisson
- Laplace, Logistic, Log-Normal, Weibull, Pareto
- Chi-square, Student's t, F-distribution, Dirichlet

### Can I use Aleam with NumPy?

Yes. Aleam provides array operations that return Python lists that can be converted to NumPy arrays, or you can use the NumPy-compatible functions `random_array()`, `randn_array()`, etc.

---

## License

MIT License. See [LICENSE](https://github.com/fardinsabid/aleam/blob/main/LICENSE) for details.

---

## Links

| | |
|---|---|
| 📦 PyPI | [pypi.org/project/aleam](https://pypi.org/project/aleam) |
| 🐛 Issues | [GitHub Issues](https://github.com/fardinsabid/aleam/issues) |
| 📖 Source | [GitHub Repository](https://github.com/fardinsabid/aleam) |
| 📄 Research Paper | [ALEAM_RESEARCH_PAPER.md](https://github.com/fardinsabid/aleam/blob/main/ALEAM_RESEARCH_PAPER.md) |

---

*"True randomness is not a bug—it's a feature. Nature doesn't compute recursively, and neither should AI."*

— Fardin Sabid, Creator of Aleam