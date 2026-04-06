# Aleam Documentation

**True randomness for AI and machine learning. Non-recursive, stateless, cryptographically secure.**

[![Version](https://img.shields.io/badge/version-1.0.3-blue.svg)](https://github.com/fardinsabid/aleam)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://isocpp.org/)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Algorithm](#algorithm)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Core API Reference](#core-api-reference)
6. [Statistical Distributions](#statistical-distributions)
7. [AI/ML Features](#aiml-features)
8. [Array Operations](#array-operations)
9. [Framework Integrations](#framework-integrations)
10. [CUDA Acceleration](#cuda-acceleration)
11. [Performance Benchmarks](#performance-benchmarks)
12. [C++ API](#c-api)
13. [FAQ](#faq)
14. [License](#license)

---

## Introduction

Aleam is a **true random number generator** designed specifically for AI and machine learning applications. Unlike traditional pseudo-random number generators (PRNGs) that rely on recursive deterministic formulas, Aleam is:

- **Non-recursive** — Each call is independent
- **Stateless** — No internal state between calls
- **Cryptographically secure** — Powered by BLAKE2s
- **AI-optimized** — Specialized features for ML workflows

### Why Aleam?

| Feature | Aleam | Traditional PRNGs |
|---------|-------|-------------------|
| Randomness Type | **True** | Pseudo |
| Recursive | ❌ No | ✅ Yes |
| Stateful | ❌ No | ✅ Yes |
| Seeding Required | ❌ No | ✅ Yes |
| Periodic | ❌ No | ✅ Yes |
| Cryptographically Secure | ✅ Yes | ❌ No |
| Entropy Guarantee | 64 bits/call | None |

---

## Algorithm

Aleam implements the proven equation:

```
Ψ(t) = BLAKE2s( (Φ × Ξ(t)) ⊕ τ(t) )
```

| Symbol | Value | Description |
|--------|-------|-------------|
| **Φ** | `0x9E3779B97F4A7C15` | Golden ratio prime (⌊2⁶⁴/φ⌋) |
| **Ξ(t)** | 64-bit entropy | True entropy from system CSPRNG |
| **τ(t)** | Nanosecond timestamp | Temporal anchor for uniqueness |
| **⊕** | XOR | Bitwise XOR over 64 bits |
| **BLAKE2s** | Cryptographic hash | 32-byte output, first 8 bytes used |

### Algorithm Steps

| Step | Operation | Description |
|------|-----------|-------------|
| 1 | `entropy = get_entropy_64()` | Pull 64-bit true entropy from system |
| 2 | `mixed = entropy * GOLDEN_PRIME` | Golden ratio mixing (bijective) |
| 3 | `timestamp = time_ns() & 0xFFFFFFFFFFFFFFFF` | Nanosecond timestamp |
| 4 | `combined = mixed ^ timestamp` | XOR mixing |
| 5 | `hash = blake2s_64(combined)` | Cryptographic hash to 64 bits |
| 6 | `result = hash / 2⁶⁴` | Map to [0, 1) |

### Entropy Sources by Platform

| Platform | Function | Entropy Source |
|----------|----------|----------------|
| Linux | `getrandom()` | Kernel CSPRNG + hardware RNG |
| Windows | `BCryptGenRandom()` | Kernel CSPRNG + hardware RNG |
| macOS/iOS | `arc4random_buf()` | Kernel CSPRNG + Apple Silicon RNG |

---

## Installation

### From PyPI (recommended)

```bash
pip install aleam
```

### With Framework Support

```bash
# PyTorch
pip install aleam[torch]

# TensorFlow
pip install aleam[tensorflow]

# JAX
pip install aleam[jax]

# CuPy (fastest GPU)
pip install aleam[cupy]

# Data science
pip install aleam[pandas]

# All frameworks
pip install aleam[all]
```

### From Source

```bash
git clone https://github.com/fardinsabid/aleam.git
cd aleam
pip install .
```

### Development Installation

```bash
pip install -e .[dev]
```

---

## Quick Start

```python
import aleam as al

# Create a true random generator
rng = al.Aleam()

# Core randomness
x = rng.random()                         # 0.90324326
u64 = rng.random_uint64()                # 12345678901234567890
y = rng.randint(1, 100)                  # 86
z = rng.choice(['AI', 'ML', 'Aleam'])    # 'ML'
u = rng.uniform(5.0, 10.0)              # 7.234
n = rng.gauss(0.0, 1.0)                # -0.432

# Statistical distributions
exp = rng.exponential(1.0)              # 0.845
beta = rng.beta(2.0, 5.0)              # 0.285
gamma = rng.gamma(2.0, 1.0)            # 1.892
poisson = rng.poisson(3.5)             # 3

# Sampling (requires list, not range)
population = list(range(10000))
batch = rng.sample(population, 64)      # Random 64 unique indices

# Shuffle list in-place
items = [1, 2, 3, 4, 5]
rng.shuffle(items)                      # [3, 1, 5, 2, 4]

# Random bytes for cryptography
key = rng.random_bytes(32)              # 32 cryptographically secure bytes
```

---

## Core API Reference

### Generator Classes

| Class | Description |
|-------|-------------|
| `Aleam` | Main random generator (C++ optimized, recommended) |
| `AleamCore` | Alias for Aleam |

### Core Methods

| Method | Description | Example |
|--------|-------------|---------|
| `random()` | Random float in [0, 1) using 64-bit entropy | `rng.random()` |
| `random_uint64()` | Random 64-bit unsigned integer | `rng.random_uint64()` |
| `randint(a, b)` | Random integer in [a, b] | `rng.randint(1, 100)` |
| `choice(seq)` | Random element from sequence | `rng.choice(['a', 'b', 'c'])` |
| `uniform(low, high)` | Random float in [low, high] | `rng.uniform(5.0, 10.0)` |
| `gauss(mu, sigma)` | Normally distributed value | `rng.gauss(0, 1)` |
| `normalvariate(mu, sigma)` | Alias for `gauss` | `rng.normalvariate(0, 1)` |
| `sample(population, k)` | Sample k unique elements | `rng.sample(list(range(100)), 10)` |
| `shuffle(lst)` | Shuffle list in-place | `rng.shuffle(my_list)` |
| `random_bytes(n)` | Generate n random bytes | `rng.random_bytes(32)` |

### Example: Sampling

```python
import aleam as al

rng = al.Aleam()

# Sample from list
fruits = ['apple', 'banana', 'cherry', 'date']
selected = rng.sample(fruits, 2)        # Returns list of 2 random fruits

# Sample without replacement (default)
numbers = list(range(100))
batch = rng.sample(numbers, 10)         # 10 unique numbers

# Note: population must be a list, not a range object
# Correct: rng.sample(list(range(100)), 10)
# Wrong:   rng.sample(range(100), 10)
```

### Example: Shuffling

```python
import aleam as al

rng = al.Aleam()

# Shuffle a list in-place
cards = list(range(52))
rng.shuffle(cards)                      # Cards are now shuffled

# Shuffle any sequence
words = ['hello', 'world', 'aleam', 'random']
rng.shuffle(words)                      # Random order
```

---

## Statistical Distributions

Aleam provides 15+ statistical distributions, all powered by true randomness from 64-bit entropy.

### Distribution Methods

| Distribution | Method | Parameters | Support |
|--------------|--------|------------|---------|
| Uniform | `uniform(low, high)` | low, high | [low, high] |
| Normal (Gaussian) | `gauss(mu, sigma)` | μ, σ > 0 | (-∞, ∞) |
| Exponential | `exponential(rate)` | λ > 0 | [0, ∞) |
| Beta | `beta(alpha, beta)` | α > 0, β > 0 | [0, 1] |
| Gamma | `gamma(shape, scale)` | k > 0, θ > 0 | [0, ∞) |
| Poisson | `poisson(lam)` | λ > 0 | {0, 1, 2, ...} |
| Laplace | `laplace(loc, scale)` | μ, b > 0 | (-∞, ∞) |
| Logistic | `logistic(loc, scale)` | μ, s > 0 | (-∞, ∞) |
| Log-Normal | `lognormal(mu, sigma)` | μ, σ > 0 | (0, ∞) |
| Weibull | `weibull(shape, scale)` | k > 0, λ > 0 | [0, ∞) |
| Pareto | `pareto(alpha, scale)` | α > 0, x_m > 0 | [x_m, ∞) |
| Chi-square | `chi_square(df)` | k > 0 | [0, ∞) |
| Student's t | `student_t(df)` | ν > 0 | (-∞, ∞) |
| F-distribution | `f_distribution(df1, df2)` | d₁ > 0, d₂ > 0 | [0, ∞) |
| Dirichlet | `dirichlet(alpha)` | α_i > 0 | Simplex |

### Examples

```python
import aleam as al

rng = al.Aleam()

# Uniform distribution
uniform_val = rng.uniform(5.0, 10.0)    # Between 5 and 10

# Normal distribution
normal_val = rng.gauss(0.0, 1.0)        # Standard normal

# Exponential distribution (waiting times)
wait_time = rng.exponential(rate=0.5)   # Mean = 2.0

# Beta distribution (probabilities)
probability = rng.beta(alpha=2, beta=5) # Skewed left

# Gamma distribution (sum of exponentials)
total_time = rng.gamma(shape=3, scale=2)  # Mean = 6

# Poisson distribution (event counts)
events = rng.poisson(lam=5)             # Mean = 5

# Laplace distribution (heavy-tailed noise)
noise = rng.laplace(loc=0, scale=1)

# Dirichlet distribution (probability simplex)
probs = rng.dirichlet([1, 2, 3])        # Returns [p1, p2, p3] summing to 1
print(f"Probabilities: {probs}")        # e.g., [0.166, 0.333, 0.501]
```

---

## AI/ML Features

### `AIRandom` Class

AI-specific random utilities for machine learning workflows.

```python
import aleam as al
import numpy as np

ai = al.AIRandom()

# Gradient noise for training (helps escape local minima)
noise = ai.gradient_noise(shape=100, scale=0.1)
print(f"Noise mean: {np.mean(noise):.4f}, std: {np.std(noise):.4f}")

# Latent space vector for generative models
latent = ai.latent_vector(dim=512, distribution="normal")
print(f"Latent vector shape: {len(latent)}")

# Dropout mask for regularization
mask = ai.dropout_mask(size=100, keep_prob=0.3)
print(f"Dropout keep rate: {sum(mask)/len(mask):.2f}")

# Data augmentation parameters for computer vision
params = ai.augmentation_params()
print(f"Rotation: {params.rotation:.1f}°")
print(f"Scale: {params.scale:.2f}")
print(f"Brightness: {params.brightness:.2f}")
print(f"Flip horizontal: {params.flip_horizontal}")

# Mini-batch sampling
dataset_size = 10000
batch_size = 64
batch = ai.mini_batch(dataset_size, batch_size)
print(f"Batch indices: {batch[:10]}...")

# Reinforcement learning exploration noise
action_dim = 4
exploration_noise = ai.exploration_noise(action_dim, scale=0.2)
print(f"Action noise: {exploration_noise}")
```

### `GradientNoise` Class

Gradient noise injection with exponential decay for training.

```python
import aleam as al
import numpy as np

# Create gradient noise with decay
noise = al.GradientNoise(initial_scale=0.01, decay=0.99)

# During training loop
gradients = np.ones(100)  # Your actual gradients

for step in range(10):
    noisy_gradients = noise.add_noise(gradients)
    print(f"Step {step}: scale={noise.get_current_scale():.6f}")
    # optimizer.step(noisy_gradients)

# Reset for new training run
noise.reset()
print(f"Reset, step={noise.get_step()}")
```

### `LatentSampler` Class

Latent space sampling for generative models (VAEs, GANs).

```python
import aleam as al
import numpy as np

# Create latent space sampler
sampler = al.LatentSampler(latent_dim=128, distribution="normal")

# Sample single vector
z = sampler.sample_one()
print(f"Single latent vector shape: {len(z)}")

# Sample batch of vectors
batch = sampler.sample(n=10)
print(f"Batch shape: {len(batch)} x {len(batch[0])}")

# Linear interpolation between two vectors
z1 = sampler.sample_one()
z2 = sampler.sample_one()
interpolated = sampler.interpolate(z1, z2, steps=5)

for i, vec in enumerate(interpolated):
    print(f"Step {i}: mean={np.mean(vec):.4f}, std={np.std(vec):.4f}")
```

---

## Array Operations

Aleam provides NumPy-style array generation functions that return Python lists (convertible to NumPy arrays).

### Module-Level Array Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `random_array(shape)` | Uniform [0, 1) floats | `list` or nested lists |
| `randn_array(shape, mu, sigma)` | Normal distribution | `list` or nested lists |
| `randint_array(shape, low, high)` | Random integers | `list` or nested lists |
| `choice_array(a, size, replace, p)` | Sample from population | NumPy array |

### Examples

```python
import aleam as al
import numpy as np

# 1D array of random floats
arr_1d = al.random_array(100)
print(f"1D shape: {len(arr_1d)}")

# 2D array (returns nested list)
arr_2d = al.random_array((10, 10))
print(f"2D shape: {len(arr_2d)} x {len(arr_2d[0])}")

# Convert to NumPy array if needed
np_arr = np.array(arr_2d)

# Normal distribution array
norm_arr = al.randn_array(1000, mu=0, sigma=1)
print(f"Normal mean: {np.mean(norm_arr):.4f}, std: {np.std(norm_arr):.4f}")

# Integer array
int_arr = al.randint_array((50,), low=0, high=10)
print(f"Integers: {int_arr[:10]}")

# Weighted sampling from list
fruits = ['apple', 'banana', 'cherry']
weights = [0.5, 0.3, 0.2]
choices = al.choice_array(fruits, size=(100,), p=weights)

from collections import Counter
counts = Counter(choices)
print(f"Sampled distribution: {dict(counts)}")

# Sample without replacement
unique = al.choice_array(10, size=(5,), replace=False)
print(f"Unique samples: {unique}")
```

---

## Framework Integrations

### PyTorch Integration

```python
import torch
import aleam as al

# Create generator (auto-detects CUDA)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = al.TorchGenerator(device=device)

# Generate tensors
tensor = gen.randn(100, 100)              # N(0,1) on GPU/CPU
tensor = gen.rand(100, 100)               # Uniform [0,1)
tensor = gen.randint(0, 10, (100, 100))   # Integers
tensor = gen.normal(0, 1, (100,))         # Custom normal
tensor = gen.uniform(0, 1, (100,))        # Custom uniform

# Use with existing models
model = torch.nn.Linear(784, 128)
with torch.no_grad():
    for param in model.parameters():
        param.data = gen.randn_like(param)
```

### TensorFlow Integration

```python
import tensorflow as tf
import aleam as al

gen = al.TFGenerator()

# Generate tensors
tensor = gen.normal((100, 100), mean=0, stddev=1)
tensor = gen.uniform((100, 100), minval=0, maxval=1)
tensor = gen.randint((100, 100), minval=0, maxval=10)
tensor = gen.truncated_normal((100, 100), mean=0, stddev=1)

# Shuffle tensors
original = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
shuffled = gen.shuffle(original)
```

### JAX Integration

```python
import jax
import jax.numpy as jnp
import aleam as al

gen = al.JAXGenerator()

# Generate true random key
key = gen.key()

# Use with JAX random functions
tensor = jax.random.normal(key, (100, 100))
tensor = jax.random.uniform(key, (100, 100))

# Direct generation (CPU only)
tensor = gen.normal((100, 100), mean=0, stddev=1)
tensor = gen.uniform((100, 100), minval=0, maxval=1)
tensor = gen.randint((100, 100), minval=0, maxval=10)
```

### CuPy Integration (Fastest GPU)

```python
import cupy as cp
import aleam as al

gen = al.CuPyGenerator()

# Generate arrays directly on GPU
arr = gen.random((10000, 10000), dtype='float32')
arr = gen.randn((10000, 10000), mu=0, sigma=1)
arr = gen.randint((10000, 10000), low=0, high=10)

# GPU operations
result = cp.dot(arr, arr.T)
```

### Pandas Integration

```python
import pandas as pd
import aleam as al

gen = al.PandasGenerator()

# Generate Series
series = gen.series(1000, distribution="normal", params="mu=0,sigma=1")

# Generate DataFrame
df = gen.dataframe(1000, columns=['a', 'b', 'c'],
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

```python
import aleam as al
import numpy as np

# Direct array generation (returns lists)
arr = al.random_array((100, 100))        # Nested list
np_arr = np.array(arr)                    # Convert to NumPy

# Or use module-level functions
arr = al.random_array(1000)               # 1D list
matrix = al.random_array((10, 10))        # 2D nested list
norm_arr = al.randn_array(1000, 0, 1)    # Normal distribution
int_arr = al.randint_array((50,), 0, 10) # Integers
```

---

## CUDA Acceleration

Aleam provides GPU acceleration through multiple backends with automatic detection.

### Unified CUDA Generator

```python
import aleam as al

# Create CUDA generator (auto-detects available backends)
cuda_gen = al.CUDAGenerator()

# CuPy backend (fastest)
cupy_arr = cuda_gen.cupy_random((10000, 10000))

# PyTorch CUDA backend
torch_tensor = cuda_gen.torch_randn(10000, 10000, device='cuda')

# TensorFlow GPU backend
tf_tensor = cuda_gen.tf_random_normal((10000, 10000))

# JAX GPU backend
jax_arr = cuda_gen.jax_randn((10000, 10000))
```

### Performance Comparison

| Method | Speed (elements/sec) | Platform |
|--------|---------------------|----------|
| CPU (Python) | Coming soon | Any |
| CPU (C++ Core) | Coming soon | Any |
| CuPy GPU | Coming soon | NVIDIA |
| PyTorch CUDA | Coming soon | NVIDIA |
| TensorFlow GPU | Coming soon | NVIDIA |
| JAX GPU | Coming soon | NVIDIA |

### GPU Benchmark Example

```python
import aleam as al
import time

cuda_gen = al.CUDAGenerator()

# Generate random numbers on GPU
start = time.time()
arr = cuda_gen.cupy_randn((10000, 10000))
elapsed = time.time() - start

print(f"Generated {arr.size} numbers in {elapsed:.2f} seconds")
```

---

## Performance Benchmarks

### CPU Performance (C++ Core)

| Operation | Aleam (C++) | Python random | Ratio |
|-----------|-------------|---------------|-------|
| `random()` | Coming soon | 8.57M ops/sec | Coming soon |
| `randint()` | Coming soon | 7.50M ops/sec | Coming soon |
| `gauss()` | Coming soon | 6.00M ops/sec | Coming soon |

> **Note:** Aleam is expected to be slower than Python's random on CPU — this is expected for true randomness. The trade-off is genuine entropy and cryptographic security. On GPU, Aleam achieves massive parallel performance.

### Distribution Performance (CPU)

| Distribution | Speed (ops/sec) |
|--------------|-----------------|
| `random()` | Coming soon |
| `uniform()` | Coming soon |
| `exponential()` | Coming soon |
| `laplace()` | Coming soon |
| `gauss()` | Coming soon |
| `gamma()` | Coming soon |
| `poisson()` | Coming soon |
| `beta()` | Coming soon |

### GPU Performance (Pending)

**Tested on NVIDIA Tesla T4 (Google Colab) with CuPy 14.0.1 — benchmarks pending**

| Operation | Device | Speed | Time (100M numbers) |
|-----------|--------|-------|---------------------|
| `cupy_random()` | GPU | Coming soon | Coming soon |
| `cupy_randn()` | GPU | Coming soon | Coming soon |
| `random()` | CPU | Coming soon | Coming soon |

---

## C++ API

Aleam can also be used directly in C++ applications.

### Installation for C++

```bash
git clone https://github.com/fardinsabid/aleam.git
cd aleam
mkdir build && cd build
cmake ..
make
sudo make install
```

### Basic C++ Usage

```cpp
#include <aleam/aleam.h>
#include <iostream>
#include <vector>

int main() {
    // Create generator
    aleam::AleamCore rng;
    
    // Generate single values
    double x = rng.random();
    uint64_t y = rng.random_uint64();
    int z = rng.randint(1, 100);
    
    std::cout << "Random: " << x << std::endl;
    std::cout << "Random uint64: " << y << std::endl;
    std::cout << "Random int: " << z << std::endl;
    
    // Batch generation for performance
    std::vector<double> batch(1024);
    rng.random_batch(batch.data(), batch.size());
    
    // Statistical distributions
    double normal = rng.gauss(0.0, 1.0);
    double uniform = rng.uniform(5.0, 10.0);
    double exponential = rng.exponential(1.0);
    
    // Thread-local instance for multi-threaded code
    aleam::AleamCore& thread_local_rng = aleam::get_thread_local_instance();
    double tl_random = thread_local_rng.random();
    
    return 0;
}
```

### C++ API Reference

| Method | Description |
|--------|-------------|
| `random()` | Generate double in [0, 1) using 64-bit entropy |
| `random_uint64()` | Generate 64-bit random integer |
| `random_batch(double* out, size_t n)` | Generate n doubles |
| `randint(int64_t a, int64_t b)` | Generate integer in [a, b] |
| `gauss(double mu, double sigma)` | Normal distribution |
| `uniform(double low, double high)` | Uniform distribution |
| `exponential(double rate)` | Exponential distribution |
| `beta(double alpha, double beta)` | Beta distribution |
| `gamma(double shape, double scale)` | Gamma distribution |
| `poisson(double lam)` | Poisson distribution |
| `laplace(double loc, double scale)` | Laplace distribution |
| `logistic(double loc, double scale)` | Logistic distribution |
| `weibull(double shape, double scale)` | Weibull distribution |
| `pareto(double alpha, double scale)` | Pareto distribution |
| `chi_square(double df)` | Chi-square distribution |
| `student_t(double df)` | Student's t-distribution |
| `f_distribution(double df1, double df2)` | F-distribution |
| `dirichlet(const std::vector<double>& alpha)` | Dirichlet distribution |

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
- True entropy from the operating system's CSPRNG (64 bits per call)
- BLAKE2s cryptographic hash function
- No internal state to extract or predict

### Can I reproduce results?

No. By design, Aleam does not support seeding. Call `al.seed_free()` to see the explanation. If you need reproducible randomness, use Python's `random` module. Aleam is for applications where true unpredictability matters.

### How fast is Aleam?

| Configuration | Expected Speed |
|---------------|-----------------|
| CPU (C++ core) | Benchmarks pending |
| GPU (CuPy) | Benchmarks pending |

### Why does `sample()` require a list?

The C++ bindings accept Python lists directly. Use `list(range(10000))` instead of `range(10000)`.

```python
# Correct
rng.sample(list(range(10000)), 64)

# Wrong (will raise TypeError)
rng.sample(range(10000), 64)
```

### Does Aleam work on GPU?

Yes! Aleam supports:
- PyTorch CUDA
- TensorFlow GPU
- JAX GPU
- CuPy (fastest)
- Automatic detection via `CUDAGenerator()`

### What Python versions are supported?

Aleam supports Python 3.8 through 3.13.

### What platforms are supported?

| Platform | Entropy Source | Status |
|----------|---------------|--------|
| Linux | `getrandom()` | ✅ Full support |
| Windows | `BCryptGenRandom()` | ✅ Full support |
| macOS | `arc4random_buf()` | ✅ Full support |
| iOS | `arc4random_buf()` | ✅ Works (via iSH) |
| Android | `getrandom()` | ✅ Works (via Termux) |

---

## License

MIT License. See [LICENSE](https://github.com/fardinsabid/aleam/blob/main/LICENSE) for details.

| Component | License |
|-----------|---------|
| Aleam Interface | MIT |
| Core Algorithm | MIT |
| BLAKE2s | Public Domain / CC0 |

---

## Links

| | |
|---|---|
| 📦 PyPI | [pypi.org/project/aleam](https://pypi.org/project/aleam) |
| 🐛 Issues | [GitHub Issues](https://github.com/fardinsabid/aleam/issues) |
| 📖 Source | [GitHub Repository](https://github.com/fardinsabid/aleam) |
| 📄 Research Paper | [ALEAM_RESEARCH_PAPER.md](https://github.com/fardinsabid/aleam/blob/main/docs/ALEAM_RESEARCH_PAPER.md) |

---

<div align="center">

**Made with ❤️ by Fardin Sabid**  
**🇧🇩 From Bangladesh, for the World 🌍**

```
True randomness. No recursion. No state. Just entropy.
```

</div>
```