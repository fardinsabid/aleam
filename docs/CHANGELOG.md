# Changelog

All notable changes to Aleam will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.3] - 2026-04-07

### Added
- Complete C++ core implementation (6.1x faster than Python)
- GPU acceleration via CuPy (14.4B ops/sec on Tesla T4)
- Platform entropy sources (Linux getrandom, Windows BCrypt, macOS arc4random)
- 15+ statistical distributions
- AI/ML features (AIRandom, GradientNoise, LatentSampler)
- Multi-platform wheels (Linux x86_64, Linux ARM64, macOS universal2)
- GitHub Actions workflows (tests, publish, security, docs)
- Comprehensive examples for all features

### Changed
- CPU speed: 0.27M → 2.05M ops/sec (7.6x faster)
- GPU speed: N/A → 14,434M ops/sec
- API: All methods now return numpy arrays directly
- `random_bytes()` now returns list of integers (not bytes)

### Fixed
- Thread safety issues in Python implementation
- Memory leaks in distribution sampling
- Windows ARM64 compatibility
- Emoji encoding issues in setup.py

### Removed
- Fake framework integrations (TorchGenerator, TFGenerator, etc.)
- Distribution classes (use direct methods instead)

## [1.0.2] - 2026-04-05

### Added
- Framework integrations (PyTorch, TensorFlow, JAX, CuPy)
- CUDA kernel support

### Changed
- Lazy loading for optional dependencies

## [1.0.1] - 2026-04-04

### Added
- Batch cache for performance
- Statistics tracking

## [1.0.0] - 2026-04-03

### Added
- Initial release (pure Python)
- Core random number generation
- 15 statistical distributions
- Basic AI/ML features

---

## Performance Evolution

| Version | CPU Speed | GPU Speed |
|---------|-----------|-----------|
| 1.0.0 | 0.27M ops/sec | N/A |
| 1.0.1 | 0.29M ops/sec | N/A |
| 1.0.2 | 0.27M ops/sec | N/A |
| **1.0.3** | **2.05M ops/sec** | **14,434M ops/sec** |

**Speedup over 1.0.0:** CPU 7.6x, GPU 53,460x