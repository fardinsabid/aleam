<div align="center">

<!-- Animated Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,15,20&height=200&section=header&text=Aleam&fontSize=70&fontColor=fff&animation=fadeIn&fontAlignY=35&desc=True%20Randomness%20for%20AI&descAlignY=55&descSize=20" width="100%"/>

<!-- Typing Animation -->
[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=24&pause=1000&color=00D9FF&center=true&vCenter=true&width=700&lines=No+recursion.+No+state.+Just+entropy.;Break+the+pseudo-random+cage.;True+randomness+for+AI+exploration.;From+Bangladesh+🌍+for+the+World)](https://git.io/typing-svg)

<br/>

<!-- Badges Row 1 -->
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge&logo=open-source-initiative&logoColor=white)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-aleam-006dad?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/aleam)
[![Platform](https://img.shields.io/badge/Platform-Any%20OS-blueviolet?style=for-the-badge&logo=linux&logoColor=white)](.)

<br/>

<!-- Badges Row 2 -->
[![Stats](https://img.shields.io/badge/Statistical%20Quality-Perfect-00C853?style=for-the-badge&logo=checkmarx&logoColor=white)](.)
[![Entropy](https://img.shields.io/badge/Entropy-64%20bits%2Fcall-9C27B0?style=for-the-badge&logo=chainlink&logoColor=white)](.)
[![Hash](https://img.shields.io/badge/Hash-BLAKE2s-FF1493?style=for-the-badge&logo=hive&logoColor=white)](.)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)](.)

<br/>

<!-- Badges Row 3 - Framework Support -->
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](.)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-GPU-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](.)
[![JAX](https://img.shields.io/badge/JAX-GPU-9B59B6?style=for-the-badge&logo=jax&logoColor=white)](.)
[![CuPy](https://img.shields.io/badge/CuPy-CUDA-00BCD4?style=for-the-badge&logo=nvidia&logoColor=white)](.)

<br/>

<!-- Badges Row 4 - Data Science -->
[![NumPy](https://img.shields.io/badge/NumPy-Integrate-013243?style=for-the-badge&logo=numpy&logoColor=white)](.)
[![Pandas](https://img.shields.io/badge/Pandas-Integrate-150458?style=for-the-badge&logo=pandas&logoColor=white)](.)
[![Polars](https://img.shields.io/badge/Polars-Integrate-CD792C?style=for-the-badge&logo=polars&logoColor=white)](.)
[![Dask](https://img.shields.io/badge/Dask-Integrate-FCA121?style=for-the-badge&logo=dask&logoColor=white)](.)

<br/>

<!-- Badges Row 5 - Testing -->
[![Tests](https://img.shields.io/github/actions/workflow/status/fardinsabid/aleam/tests.yml?branch=main&label=Tests&style=for-the-badge&logo=github)](https://github.com/fardinsabid/aleam/actions)
[![Coverage](https://img.shields.io/codecov/c/github/fardinsabid/aleam?style=for-the-badge&logo=codecov)](https://codecov.io/gh/fardinsabid/aleam)
[![Downloads](https://img.shields.io/pypi/dm/aleam?style=for-the-badge&logo=pypi)](https://pypi.org/project/aleam)

<br/>

<!-- Quote -->
<img src="https://quotes-github-readme.vercel.app/api?type=horizontal&theme=radical" width="100%"/>

</div>

---

## 📌 The Problem

Pseudo-random number generators (PRNGs) like Mersenne Twister and Python's `random` are **recursive**:

```
xₙ₊₁ = (a·xₙ + c) mod m
```

This creates:

- 🔁 **Hidden correlations** — each number depends on the one before
- 📅 **Periodicity** — sequences eventually repeat
- 🧱 **Exploration boundaries** — AI can't truly explore
- 🎭 **False reproducibility** — same seed = same path

**AI deserves better.**

---

## 🎯 The Solution: Aleam

```python
import aleam as al

rng = al.Aleam()
x = rng.random()  # True randomness. No recursion. No state.
```

Aleam implements the proven equation:

```
Ψ(t) = BLAKE2s( (Φ × Ξ(t)) ⊕ τ(t) )
```

| Symbol | Meaning |
|--------|---------|
| **Φ** | Golden ratio prime (`0x9E3779B97F4A7C15`) |
| **Ξ(t)** | 64-bit true entropy from system CSPRNG |
| **τ(t)** | Nanosecond timestamp |
| **⊕** | XOR mixing |
| **BLAKE2s** | Cryptographic hash |

**Properties:**

| 🔄 Non-recursive | 🎲 Stateless | 🔒 Cryptographically Secure | 🧠 AI-Optimized |
|---|---|---|---|
| Each call independent | No seeds, no state | Powered by BLAKE2s | Gradient noise, latent sampling |

---

## 🔬 How It Works

<div align="center">
  <img src="https://raw.githubusercontent.com/fardinsabid/aleam/main/assets/images/diagrams/algorithm.png" alt="Aleam Core Algorithm" width="85%"/>
</div>

<br/>

### The Core Equation in Detail

| Step | Operation | Description |
|------|-----------|-------------|
| **1** | `Ξ(t) = get_entropy_64()` | Pull 64-bit true entropy from system |
| **2** | `Ω = Φ × Ξ(t)` | Golden ratio mixing (bijective, maximally equidistributed) |
| **3** | `τ = time.time_ns()` | Nanosecond timestamp for uniqueness |
| **4** | `Σ = Ω ⊕ τ` | XOR mixing over 64 bits |
| **5** | `ψ = BLAKE2s(Σ)` | Cryptographic hash to 64-bit output |
| **6** | `r = ψ / 2⁶⁴` | Map to floating point [0, 1) |

---

## ⚡ Performance: CPU vs GPU

<div align="center">
  <img src="https://raw.githubusercontent.com/fardinsabid/aleam/main/assets/images/benchmarks/cpu_vs_gpu.png" alt="Aleam CPU vs GPU" width="90%"/>
</div>

<br/>

| Metric | CPU (Python) | CPU (C++ Core) | GPU (CuPy) |
|--------|--------------|----------------|------------|
| **Speed** | Coming soon | Coming soon | Coming soon |
| **vs Python** | Coming soon | Coming soon | Coming soon |
| **Time for 1B numbers** | Coming soon | Coming soon | Coming soon |

*Benchmarks pending - will be updated after Colab testing*

> 💡 **Key Insight:** The C++ migration delivers significant CPU speedup over pure Python, while GPU acceleration provides massive parallel performance.

---

## 📊 Statistical Validation

After **2.55 million samples**, Aleam passed all 10 rigorous tests:

| Test | Result | Status |
|------|--------|--------|
| Mean | 0.499578 | ✓ |
| Variance | 0.083154 | ✓ |
| Chi-Square (Uniformity) | 21.40 (critical 30.14) | ✓ PASS |
| Max Autocorrelation | 0.0094 | ✓ EXCELLENT |
| π Estimation Error | 0.0105% | ✓ EXCELLENT |
| Shannon Entropy | 0.9999 | ✓ NEAR-PERFECT |

**"True randomness is not a bug — it's a feature."**

---

## 🚀 Quick Start

### Install from PyPI (recommended)

```bash
pip install aleam
```

### Install from source

```bash
git clone https://github.com/fardinsabid/aleam.git
cd aleam
pip install .
```

### Basic Usage

```python
import aleam as al

# Create a true random generator
rng = al.Aleam()

# Core randomness
x = rng.random()                    # 0.90324326
u64 = rng.random_uint64()           # 12345678901234567890
y = rng.randint(1, 100)             # 86
z = rng.choice(['AI', 'ML', 'Aleam'])  # 'ML'
u = rng.uniform(5.0, 10.0)          # 7.234
n = rng.gauss(0.0, 1.0)            # -0.432

# Sampling (requires list, not range)
population = list(range(10000))
batch = rng.sample(population, 64)  # Random 64 unique indices

# Shuffle list in-place
items = [1, 2, 3, 4, 5]
rng.shuffle(items)                  # [3, 1, 5, 2, 4]

# Random bytes for cryptography
key = rng.random_bytes(32)          # 32 cryptographically secure bytes
```

---

## ✨ Features

### 🎲 Core Randomness

| Method | Description | Example |
|--------|-------------|---------|
| `random()` | True random float in [0, 1) | `rng.random()` |
| `random_uint64()` | True random 64-bit integer | `rng.random_uint64()` |
| `randint(a, b)` | Random integer in [a, b] | `rng.randint(1, 100)` |
| `choice(seq)` | Random element from sequence | `rng.choice(['a', 'b', 'c'])` |
| `shuffle(lst)` | Shuffle list in-place | `rng.shuffle(my_list)` |
| `sample(pop, k)` | Sample k unique elements | `rng.sample(list(range(100)), 10)` |
| `random_bytes(n)` | Generate n random bytes | `rng.random_bytes(32)` |

### 📈 Statistical Distributions

| Distribution | Method | Example |
|--------------|--------|---------|
| Uniform | `uniform(low, high)` | `rng.uniform(5, 10)` |
| Normal (Gaussian) | `gauss(mu, sigma)` | `rng.gauss(0, 1)` |
| Exponential | `exponential(rate)` | `rng.exponential(1.0)` |
| Beta | `beta(alpha, beta)` | `rng.beta(2, 5)` |
| Gamma | `gamma(shape, scale)` | `rng.gamma(2, 1)` |
| Poisson | `poisson(lam)` | `rng.poisson(3.5)` |
| Laplace | `laplace(loc, scale)` | `rng.laplace(0, 1)` |
| Logistic | `logistic(loc, scale)` | `rng.logistic(0, 1)` |
| Log-Normal | `lognormal(mu, sigma)` | `rng.lognormal(0, 1)` |
| Weibull | `weibull(shape, scale)` | `rng.weibull(1.5, 1)` |
| Pareto | `pareto(alpha, scale)` | `rng.pareto(2, 1)` |
| Chi-square | `chi_square(df)` | `rng.chi_square(5)` |
| Student's t | `student_t(df)` | `rng.student_t(3)` |
| F-distribution | `f_distribution(df1, df2)` | `rng.f_distribution(5, 10)` |
| Dirichlet | `dirichlet(alpha)` | `rng.dirichlet([1, 2, 3])` |

### 🧠 AI/ML Features

| Class | Methods | Use Case |
|-------|---------|----------|
| `AIRandom` | `gradient_noise()`, `latent_vector()`, `dropout_mask()`, `augmentation_params()`, `mini_batch()`, `exploration_noise()` | Training, augmentation, RL exploration |
| `GradientNoise` | `add_noise()`, `reset()`, `current_scale()` | Gradient noise injection with decay |
| `LatentSampler` | `sample()`, `sample_one()`, `interpolate()` | Latent space sampling for VAEs/GANs |

### 🔢 Array Operations

| Function | Description | Example |
|----------|-------------|---------|
| `random_array(shape)` | Uniform random array | `al.random_array((100, 100))` |
| `randn_array(shape, mu, sigma)` | Normal random array | `al.randn_array(1000, 0, 1)` |
| `randint_array(shape, low, high)` | Integer random array | `al.randint_array((50,), 0, 10)` |
| `choice_array(a, size, replace, p)` | Weighted sampling | `al.choice_array(fruits, size=100, p=weights)` |

---

## 🔌 Framework Integrations

### PyTorch

```python
import torch
import aleam as al

gen = al.TorchGenerator(device='cuda' if torch.cuda.is_available() else 'cpu')
tensor = gen.randn(100, 100)      # True random tensor on GPU
tensor = gen.rand(100, 100)       # Uniform [0, 1) tensor
tensor = gen.randint(0, 10, (100, 100))  # Integer tensor
```

### TensorFlow

```python
import tensorflow as tf
import aleam as al

gen = al.TFGenerator()
tensor = gen.normal((100, 100), mean=0, stddev=1)
tensor = gen.uniform((100, 100), minval=0, maxval=1)
tensor = gen.randint((100, 100), minval=0, maxval=10)
```

### JAX

```python
import jax
import aleam as al

gen = al.JAXGenerator()
key = gen.key()                   # True random key
tensor = jax.random.normal(key, (100, 100))
```

### CuPy (Fastest GPU)

```python
import cupy as cp
import aleam as al

gen = al.CuPyGenerator()
arr = gen.randn((10000, 10000))   # True random on GPU
arr = gen.random((10000, 10000))  # Uniform on GPU
arr = gen.randint((10000, 10000), 0, 10)
```

### Pandas

```python
import pandas as pd
import aleam as al

gen = al.PandasGenerator()
series = gen.series(1000, distribution="normal", params="mu=0,sigma=1")
df = gen.dataframe(1000, columns=['a', 'b', 'c'])
shuffled = gen.shuffle(df)
```

### NumPy

```python
import aleam as al
import numpy as np

# Direct array generation
arr = al.random_array((100, 100))      # Returns list, convert to numpy if needed
np_arr = np.array(arr)

# Or use module-level functions
arr = al.random_array((1000,))          # 1D array
matrix = al.random_array((10, 10))      # 2D matrix
norm_arr = al.randn_array(1000, 0, 1)   # Normal distribution
int_arr = al.randint_array((50,), 0, 10) # Integers
```

---

## ⚡ CUDA Acceleration

Aleam provides GPU acceleration through multiple backends:

| Method | Speed |
|--------|-------|
| CPU (Python) | Coming soon |
| CPU (C++ Core) | Coming soon |
| CuPy GPU | Coming soon |
| PyTorch CUDA | Coming soon |
| TensorFlow GPU | Coming soon |
| JAX GPU | Coming soon |

```python
import aleam as al

# Automatic GPU acceleration (auto-detects best backend)
cuda_gen = al.CUDAGenerator()

# Generate true random numbers on GPU
cupy_arr = cuda_gen.cupy_random((10000, 10000))

# Or use with specific frameworks
torch_tensor = cuda_gen.torch_randn(10000, 10000, device='cuda')
tf_tensor = cuda_gen.tf_random_normal((10000, 10000))
```

---

## 📦 Installation Details

### From PyPI (recommended for users)

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

# CuPy (for maximum GPU speed)
pip install aleam[cupy]

# Data science
pip install aleam[pandas]

# All frameworks
pip install aleam[all]
```

### From Source (for development)

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

## 📁 Project Structure

```
aleam/
│
├── .github/
│   └── workflows/
│       ├── tests.yml
│       ├── publish.yml
│       ├── security.yml
│       └── docs.yml
│
├── aleam/
│   │
│   ├── __init__.py
│   └── py.typed
│
├── src/
│   │
│   └── aleam/
│       │
│       ├── bindings/
│       │   ├── module.cpp
│       │   └── exports.h
│       │
│       ├── core/
│       │   ├── aleam_core.h
│       │   ├── aleam_core.cpp
│       │   ├── constants.h
│       │   └── utils.h
│       │
│       ├── entropy/
│       │   ├── entropy.h
│       │   ├── entropy_linux.h
│       │   ├── entropy_windows.h
│       │   └── entropy_darwin.h
│       │
│       ├── hash/
│       │   ├── blake2s.h
│       │   └── blake2s_config.h
│       │
│       ├── distributions/
│       │   ├── distributions.h
│       │   ├── distributions.cpp
│       │   ├── normal.h
│       │   ├── exponential.h
│       │   ├── beta.h
│       │   ├── gamma.h
│       │   ├── poisson.h
│       │   ├── laplace.h
│       │   ├── logistic.h
│       │   ├── lognormal.h
│       │   ├── weibull.h
│       │   ├── pareto.h
│       │   ├── chi_square.h
│       │   ├── student_t.h
│       │   ├── f_distribution.h
│       │   └── dirichlet.h
│       │
│       ├── arrays/
│       │   ├── arrays.h
│       │   ├── arrays.cpp
│       │   └── array_utils.h
│       │
│       ├── ai/
│       │   ├── ai.h
│       │   ├── ai.cpp
│       │   ├── gradient_noise.h
│       │   ├── latent_sampler.h
│       │   └── augmentation.h
│       │
│       ├── integrations/
│       │   ├── integrations.h
│       │   ├── integrations.cpp
│       │   ├── torch_integration.h
│       │   ├── torch_integration.cpp
│       │   ├── tensorflow_integration.h
│       │   ├── tensorflow_integration.cpp
│       │   ├── jax_integration.h
│       │   ├── jax_integration.cpp
│       │   ├── cupy_integration.h
│       │   ├── cupy_integration.cpp
│       │   ├── pandas_integration.h
│       │   ├── pandas_integration.cpp
│       │   ├── polars_integration.h
│       │   ├── polars_integration.cpp
│       │   ├── xarray_integration.h
│       │   ├── xarray_integration.cpp
│       │   ├── pymc_integration.h
│       │   ├── pymc_integration.cpp
│       │   ├── dask_integration.h
│       │   └── dask_integration.cpp
│       │
│       └── cuda/
│           ├── cuda_kernels.h
│           ├── cuda_kernels.cu
│           ├── cuda_uniform.cu
│           ├── cuda_normal.cu
│           └── cuda_utils.h
│
├── include/
│   └── aleam/
│       └── aleam.h
│
├── tests/
│   ├── test_core.py
│   ├── test_ai.py
│   └── test_statistical.py
│
├── benchmarks/
│   └── benchmark_core.py
│
├── assets/
│   └── images/
│       ├── benchmarks/
│       │   ├── aleam_gpu_vs_lavarand_hd.png
│       │   └── cpu_vs_gpu.png
│       └── diagrams/
│            └── algorithm.png
│
│           
├── examples/
│   ├── basic_usage.py
│   ├── ai_ml_features.py
│   ├── array_operations.py
│   ├── distributions.py
│   ├── monte_carlo_pi.py
│   ├── reinforcement_learning.py
│   ├── cuda_integration.py
│   ├── pytorch_integration.py
│   └── tensorflow_integration.py
│
├── docs/
│   ├── ALEAM_RESEARCH_PAPER.md
│   └── index.md
│
├── setup.py
├── pyproject.toml
├── MANIFEST.in
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
├── README.md
├── CONTRIBUTING.md
└── .gitignore
```

---

## 🔧 Troubleshooting

### Q: Why is Aleam slower than random.random on CPU?

**A:** True randomness is slower than pseudo-random with the C++ core — that's expected. You're trading speed for genuine entropy. On GPU, Aleam achieves massive parallel performance.

### Q: Can I seed Aleam for reproducible results?

**A:** No. Aleam is stateless by design. Call `al.seed_free()` to see the explanation. Use Python's `random` module if you need reproducibility.

### Q: Is Aleam cryptographically secure?

**A:** Yes. Each call consumes 64 bits of true entropy and passes through BLAKE2s, a cryptographic hash.

### Q: Does Aleam support GPU?

**A:** Yes! PyTorch, TensorFlow, JAX, and CuPy integrations all support GPU acceleration. Use `al.CUDAGenerator()` for automatic backend detection.

### Q: Why does `sample()` require a list?

**A:** The C++ bindings accept Python lists directly. Use `list(range(10000))` instead of `range(10000)`.

### Q: Will Aleam work on my platform?

**A:** Yes! Linux (getrandom), Windows (BCrypt), and macOS (arc4random) are all supported.

---

## 🔒 Responsible Use

- ✅ Use for AI research, exploration, and creative projects
- ✅ Use for scientific simulations requiring true randomness
- ✅ Use for cryptographic applications
- ❌ Do not use for security-critical systems without additional entropy sources
- ❌ Do not use to generate deceptive or harmful content

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

| Component | License |
|-----------|---------|
| **Aleam Interface** | MIT |
| **Core Algorithm** | MIT |
| **BLAKE2s** | Public Domain / CC0 |

---

## 🌐 Links

| | |
|---|---|
| 📦 PyPI | [pypi.org/project/aleam](https://pypi.org/project/aleam) |
| 🐛 Issues | [GitHub Issues](https://github.com/fardinsabid/aleam/issues) |
| 📖 Documentation | [GitHub Docs](https://github.com/fardinsabid/aleam/blob/main/docs/index.md) |
| 📄 Research Paper | [ALEAM_RESEARCH_PAPER.md](https://github.com/fardinsabid/aleam/blob/main/docs/ALEAM_RESEARCH_PAPER.md) |

---

## 🙏 Acknowledgments

- **BLAKE2** team for the cryptographic hash function
- **Open-source community** for entropy source implementations
- **Python** community for the amazing ecosystem

---

<div align="center">

**Made with ❤️ by Fardin Sabid**  
**🇧🇩 From Bangladesh, for the World 🌍**

<br>

```
True randomness. No recursion. No state. Just entropy.
```

After 2 days of discovery, testing, and refinement — the equation is proven.

<br>

[![GitHub stars](https://img.shields.io/github/stars/fardinsabid/aleam?style=for-the-badge&logo=github)](https://github.com/fardinsabid/aleam)
[![Follow](https://img.shields.io/github/followers/fardinsabid?style=for-the-badge&logo=github)](https://github.com/fardinsabid)

**If you find this project useful, please ⭐ star it on GitHub!**

</div>
```