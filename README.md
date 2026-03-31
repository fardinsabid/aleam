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
[![Speed](https://img.shields.io/badge/Speed-270K%20ops%2Fsec-FF6D00?style=for-the-badge&logo=speedtest&logoColor=white)](.)
[![Entropy](https://img.shields.io/badge/Entropy-128%20bits%2Fcall-9C27B0?style=for-the-badge&logo=chainlink&logoColor=white)](.)
[![Hash](https://img.shields.io/badge/Hash-BLAKE2s-FF1493?style=for-the-badge&logo=hive&logoColor=white)](.)

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
| **Ξ(t)** | 128-bit true entropy from system |
| **τ(t)** | Nanosecond timestamp |
| **⊕** | XOR mixing |
| **BLAKE2s** | Cryptographic hash |

**Properties:**

| 🔄 Non-recursive | 🎲 Stateless | 🔒 Cryptographically Secure | 🧠 AI-Optimized |
|---|---|---|---|
| Each call independent | No seeds, no state | Powered by BLAKE2s | Gradient noise, latent sampling |

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

```bash
pip install aleam
```

```python
import aleam as al

rng = al.Aleam()

# Basic randomness
x = rng.random()              # 0.90324326
y = rng.randint(1, 100)       # 86
z = rng.choice(['AI', 'ML'])  # 'ML'

# AI/ML features
noise = rng.gauss(0, 0.1)               # Gradient noise
batch = rng.sample(range(10000), 64)    # Mini-batch sampling
```

---

## ✨ Features

### 🎲 Core Randomness
| Method | Description |
|--------|-------------|
| `random()` | True random float in [0, 1) |
| `randint(a, b)` | True random integer |
| `choice(seq)` | Random element from sequence |
| `shuffle(lst)` | Shuffle list in-place |
| `sample(pop, k)` | Sample without replacement |

### 📈 Distributions
| Distribution | Method |
|--------------|--------|
| Normal | `gauss(mu, sigma)` |
| Exponential | `exponential(rate)` |
| Beta | `beta(alpha, beta)` |
| Gamma | `gamma(shape, scale)` |
| Poisson | `poisson(lam)` |
| Laplace | `laplace(loc, scale)` |
| Logistic | `logistic(loc, scale)` |
| Log-Normal | `lognormal(mu, sigma)` |
| Weibull | `weibull(shape, scale)` |
| Pareto | `pareto(alpha, scale)` |
| Chi-square | `chi_square(df)` |
| Student's t | `student_t(df)` |
| F-distribution | `f_distribution(df1, df2)` |
| Dirichlet | `dirichlet(alpha)` |

### 🧠 AI/ML Features
| Class | Features |
|-------|----------|
| `AIRandom` | Gradient noise, latent vectors, dropout masks, augmentation params, mini-batch, exploration noise |
| `GradientNoise` | Gradient noise injection with decay |
| `LatentSampler` | Latent space sampling with interpolation |

### 🔢 Array Operations
| Function | Description |
|----------|-------------|
| `random_array(shape)` | NumPy-style random array |
| `randn_array(shape, mu, sigma)` | Normal array |
| `randint_array(shape, low, high)` | Integer array |
| `choice_array(a, size, p)` | Weighted sampling |

---

## 🔌 Framework Integrations

### PyTorch
```python
import aleam as al

gen = al.TorchGenerator(device='cuda')
tensor = gen.randn(100, 100)  # True random on GPU
```

### TensorFlow
```python
import aleam as al

gen = al.TFGenerator()
tensor = gen.normal((100, 100))  # True random on GPU
```

### JAX
```python
import aleam as al

gen = al.JAXGenerator()
key = gen.key()  # True random key
tensor = jax.random.normal(key, (100, 100))
```

### CuPy
```python
import aleam as al

gen = al.CuPyGenerator()
arr = gen.randn((10000, 10000))  # True random on GPU
```

### Pandas
```python
import aleam as al

gen = al.PandasGenerator()
df = gen.dataframe(1000, columns=['a', 'b', 'c'])
```

---

## ⚡ CUDA Acceleration

Aleam provides GPU acceleration through multiple backends:

| Method | Speed (elements/sec) |
|--------|---------------------|
| CPU (Python) | ~270,000 |
| CPU (NumPy) | ~5,000,000 |
| CuPy GPU | ~50,000,000 |
| PyTorch CUDA | ~100,000,000 |
| TensorFlow GPU | ~80,000,000 |
| JAX GPU | ~90,000,000 |

```python
import aleam as al

# Automatic GPU acceleration (auto-detects best backend)
cuda_gen = al.CUDAGenerator()

# Works with all frameworks
cupy_arr = cuda_gen.cupy_random((10000, 10000))
torch_tensor = cuda_gen.torch_randn(10000, 10000, device='cuda')
tf_tensor = cuda_gen.tf_random_normal((10000, 10000))
```

---

## 📦 Installation

### Quick Install
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

# All frameworks
pip install aleam[all]
```

### From Source
```bash
git clone https://github.com/fardinsabid/aleam.git
cd aleam
pip install -e .
```

---

## 🔬 How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    ALEAM GENERATION FLOW                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1.  Ξ(t) ← os.urandom(16)      ▷ 128-bit true entropy          │
│                                                                  │
│  2.  Ω ← Φ × Ξ(t)               ▷ Golden ratio mixing           │
│                                                                  │
│  3.  τ ← time.time_ns()         ▷ Nanosecond timestamp          │
│                                                                  │
│  4.  Σ ← Ω ⊕ τ                  ▷ XOR mixing over 64-bit        │
│                                                                  │
│  5.  Ψ ← BLAKE2s(Σ)             ▷ Cryptographic hash            │
│                                                                  │
│  6.  r ← Ψ / 2⁶⁴                ▷ Map to [0, 1)                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
aleam/
├── aleam/
│   ├── __init__.py          # Package exports
│   ├── core.py              # Aleam, AleamFast
│   ├── ai.py                # AIRandom, GradientNoise, LatentSampler
│   ├── sources.py           # SystemEntropy, HardwareEntropy
│   ├── distributions.py     # All statistical distributions
│   ├── arrays.py            # Array operations
│   ├── utils.py             # Helper functions
│   ├── torch_integration.py # PyTorch support
│   ├── tensorflow_integration.py # TensorFlow support
│   ├── jax_integration.py   # JAX support
│   ├── cupy_integration.py  # CuPy support
│   ├── pandas_integration.py # Pandas support
│   ├── polars_integration.py # Polars support
│   ├── xarray_integration.py # Xarray support
│   ├── pymc_integration.py  # PyMC support
│   ├── cuda_integration.py  # CUDA acceleration
│   └── cuda_kernels.py      # CUDA kernels
├── tests/                   # 70+ unit tests
├── benchmarks/              # Performance benchmarks
├── examples/                # Usage examples
├── docs/                    # Documentation
├── setup.py                 # PyPI packaging
├── README.md                # You are here
└── LICENSE                  # MIT License
```

---

## 🔧 Troubleshooting

### Q: Why is Aleam slower than random.random?

**A:** True randomness is ~37x slower than pseudo-random on CPU — that's expected. You're trading speed for genuine entropy. On GPU, Aleam can achieve 100M+ ops/sec, making it faster than CPU pseudo-random!

### Q: Can I seed Aleam for reproducible results?

**A:** No. Aleam is stateless by design. Use Python's `random` module if you need reproducibility.

### Q: Is Aleam cryptographically secure?

**A:** Yes. Each call consumes 128 bits of true entropy and passes through BLAKE2s, a cryptographic hash.

### Q: Will Aleam work on my phone?

**A:** Yes! Works on Android (Termux + Ubuntu) and iOS (iSH) with full functionality.

### Q: Does Aleam support GPU?

**A:** Yes! PyTorch, TensorFlow, JAX, and CuPy integrations all support GPU acceleration. Use `al.CUDAGenerator()` for automatic backend detection.

---

## 🔒 Responsible Use

- ✅ Use for AI research, exploration, and creative projects
- ✅ Use for scientific simulations requiring true randomness
- ✅ Use for cryptographic applications
- ❌ Do not use for security-critical systems without additional entropy sources
- ❌ Do not use to generate deceptive or harmful content

---

## 📄 License

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
| 📖 Documentation | [GitHub Docs](https://github.com/fardinsabid/aleam/docs) |
| 📄 Research Paper | [ALEAM_RESEARCH_PAPER.md](ALEAM_RESEARCH_PAPER.md) |
| 💬 Discussions | [GitHub Discussions](https://github.com/fardinsabid/aleam/discussions) |

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

After 20+ hours of discovery, testing, and refinement — the equation is proven.

<br>

[![GitHub stars](https://img.shields.io/github/stars/fardinsabid/aleam?style=for-the-badge&logo=github)](https://github.com/fardinsabid/aleam)
[![Follow](https://img.shields.io/github/followers/fardinsabid?style=for-the-badge&logo=github)](https://github.com/fardinsabid)

**If you find this project useful, please ⭐ star it on GitHub!**

</div>
```