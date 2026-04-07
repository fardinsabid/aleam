# Aleam Roadmap

## Current Version: 1.0.3 (April 2026)

### ✅ Completed

- [x] C++ core implementation
- [x] Python bindings via pybind11
- [x] Platform entropy sources (Linux, Windows, macOS)
- [x] 15+ statistical distributions
- [x] AI/ML features (GradientNoise, LatentSampler, AIRandom)
- [x] CUDA GPU kernels
- [x] Framework integrations (PyTorch, TensorFlow, JAX, CuPy)
- [x] PyPI deployment with multi-platform wheels
- [x] Comprehensive documentation
- [x] Research paper

### 🚧 In Progress

| Feature | Status | Target |
|---------|--------|--------|
| Performance optimization | 80% | Q2 2026 |
| More statistical distributions | 50% | Q2 2026 |
| Additional framework integrations | 40% | Q3 2026 |

### 📅 Future Releases

#### v1.1.0 (Q3 2026)

- **Hardware Acceleration**
  - RDRAND instruction support for Intel/AMD CPUs
  - AES-NI acceleration for BLAKE2s
- **New Distributions**
  - Cauchy distribution
  - von Mises distribution
  - Negative binomial distribution
- **Performance**
  - SIMD optimizations for batch generation
  - Reduced memory footprint

#### v1.2.0 (Q4 2026)

- **Multi-GPU Support**
  - Distributed true randomness across multiple GPUs
  - Load balancing for large-scale deployments
- **New Framework Integrations**
  - Flax
  - Keras 3
  - PyTorch Lightning
- **WebAssembly Support**
  - Browser-based true randomness via Web Crypto API

#### v2.0.0 (2027)

- **Quantum Entropy Integration**
  - Integration with quantum random number generators (QRNGs)
  - Hybrid classical-quantum randomness
- **Formal Verification**
  - Cryptographic proofs of security properties
  - Formal verification of core algorithm
- **Distributed Generation**
  - Cluster-wide true randomness
  - Blockchain integration

## Feature Requests

We welcome feature requests! Please open an issue on GitHub with:

- Clear description of the feature
- Use case and motivation
- Proposed API (if applicable)

## Contribution Opportunities

| Area | Difficulty | Skills Needed |
|------|------------|---------------|
| New distributions | Medium | C++, Statistics |
| CUDA optimization | Hard | CUDA, GPU architecture |
| Framework integrations | Easy | Python, Framework knowledge |
| Documentation | Easy | Writing, Markdown |
| Performance benchmarks | Medium | Python, Data analysis |

---

**Last Updated:** April 2026
