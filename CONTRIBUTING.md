# Contributing to Aleam

Thank you for your interest in contributing to Aleam! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How Can I Contribute?

### Reporting Bugs

Before submitting a bug report, please:

1. Check the [issue tracker](https://github.com/fardinsabid/aleam/issues) for similar issues
2. Ensure you're using the latest version
3. Test with a clean environment

**When reporting a bug, include:**

- Python version (`python --version`)
- Aleam version (`pip show aleam`)
- Operating system and architecture
- Minimal code example that reproduces the issue
- Expected vs actual behavior
- Full error message and stack trace

### Suggesting Enhancements

We welcome suggestions for:

- New statistical distributions
- Additional framework integrations
- Performance improvements
- Documentation improvements

**When suggesting an enhancement, include:**

- Clear description of the feature
- Use case and motivation
- Proposed API (if applicable)
- Potential implementation approach

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Install development dependencies**: `pip install -e .[dev]`
3. **Make your changes** following the coding standards
4. **Run tests**: `pytest tests/`
5. **Run benchmarks**: `python benchmarks/benchmark_core.py` (if performance-related)
6. **Update documentation** if needed
7. **Submit the pull request** with a clear description

## Development Setup

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.8+ | Main development |
| C++ Compiler | C++17 | Core compilation |
| CMake | 3.15+ | Build system |
| CUDA Toolkit | 11.0+ | GPU support (optional) |
| Git | Latest | Version control |

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/fardinsabid/aleam.git
cd aleam

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]
```

### Verify Installation

```bash
# Run tests
pytest tests/ -v

# Run benchmarks
python benchmarks/benchmark_core.py

# Test the installation
python -c "import aleam as al; print(al.random())"
```

## Project Structure

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

## Coding Standards

### Python Code

- **Formatting**: Black with line length 88
- **Linting**: Ruff (extends flake8, pycodestyle, isort)
- **Type hints**: Required for all public functions
- **Docstrings**: Google style

```python
def example_function(param1: int, param2: str = "default") -> float:
    """
    Brief description.
    
    Args:
        param1: Description of param1.
        param2: Description of param2 (default: "default").
    
    Returns:
        Description of return value.
    
    Raises:
        ValueError: When param1 is negative.
    
    Example:
        >>> example_function(10)
        42.0
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative")
    return 42.0
```

### C++ Code

- **Standard**: C++17
- **Formatting**: Consistent with existing code
- **Naming**:
  - Classes: `PascalCase`
  - Functions: `snake_case`
  - Variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
- **Comments**: Doxygen style for public APIs

```cpp
/**
 * @brief Brief description.
 * 
 * Detailed description.
 * 
 * @param param1 Description of param1.
 * @return Description of return value.
 */
double example_function(int64_t param1) {
    return 42.0;
}
```

### CUDA Code

- **Kernel naming**: `*_kernel` suffix
- **Grid/block configuration**: Calculate in host wrapper
- **Memory management**: RAII where possible
- **Error checking**: Use `CUDA_CHECK` macro

```cpp
__global__ void example_kernel(float* output, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 42.0f;
    }
}
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_core.py

# Run with coverage
pytest --cov=aleam tests/

# Run with verbose output
pytest -v tests/

# Run only slow tests
pytest -m slow tests/

# Run only GPU tests (requires CUDA)
pytest -m gpu tests/
```

### Writing Tests

```python
import aleam as al
import pytest

class TestExample:
    def test_random_range(self):
        """Test that random() returns values in [0, 1)."""
        rng = al.Aleam()
        for _ in range(1000):
            x = rng.random()
            assert 0.0 <= x < 1.0
    
    def test_randint_range(self):
        """Test that randint() returns values in [a, b]."""
        rng = al.Aleam()
        a, b = 5, 10
        for _ in range(1000):
            x = rng.randint(a, b)
            assert a <= x <= b
    
    def test_invalid_input(self):
        """Test that invalid inputs raise ValueError."""
        rng = al.Aleam()
        with pytest.raises(ValueError):
            rng.randint(10, 5)  # a > b
```

## Benchmarking

### Running Benchmarks

```bash
# Run all benchmarks
python benchmarks/benchmark_core.py

# Run with specific iterations
python benchmarks/benchmark_core.py --iterations 1000000

# Compare with baseline
python benchmarks/benchmark_core.py --compare
```

### Adding Benchmarks

```python
# benchmarks/benchmark_core.py

def benchmark_random():
    rng = al.Aleam()
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        x = rng.random()
    elapsed = time.perf_counter() - start
    return ITERATIONS / elapsed
```

## Documentation

### Building Documentation

```bash
# Install documentation dependencies
pip install mkdocs mkdocs-material mkdocstrings

# Serve locally
mkdocs serve

# Build static site
mkdocs build
```

### Docstring Format

Use Google-style docstrings with examples:

```python
def example_function(x: float) -> float:
    """
    Compute the square of a number.
    
    Args:
        x: Input number.
    
    Returns:
        Square of x.
    
    Example:
        >>> example_function(5)
        25
    """
    return x * x
```

## Pull Request Process

1. **Update the CHANGELOG.md** with your changes
2. **Update documentation** if changing public API
3. **Add tests** for new functionality
4. **Ensure all tests pass** locally
5. **Push to your fork** and submit a pull request
6. **Wait for CI** to complete successfully
7. **Address review comments** if any

### PR Title Format

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation update
- `perf:` - Performance improvement
- `test:` - Test addition/update
- `build:` - Build system change
- `ci:` - CI configuration change
- `refactor:` - Code refactoring

Example: `feat: add Cauchy distribution`

## CI/CD Pipeline

Aleam uses GitHub Actions for:

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `tests.yml` | Push, PR | Run tests on multiple Python versions |
| `publish.yml` | Release | Build and publish to PyPI |
| `security.yml` | Daily | Security vulnerability scan |
| `docs.yml` | Push to main | Deploy documentation |

## Adding New Distributions

To add a new statistical distribution:

1. **Create header**: `src/aleam/distributions/new_distribution.h`
2. **Implement class**: Template class with `operator()`
3. **Add to distributions.h**: Declare inline function
4. **Add to distributions.cpp**: Implement if complex
5. **Add to module.cpp**: Bind to Python
6. **Add to __init__.py**: Export to Python
7. **Add tests**: `tests/test_statistical.py`
8. **Update documentation**: `docs/index.md`

Example distribution template:

```cpp
// new_distribution.h
template<typename RealType = double>
class NewDistribution {
public:
    NewDistribution(RealType param, AleamCore& rng)
        : m_param(param), m_rng(rng) {}
    
    RealType operator()() {
        // Implementation using m_rng.random()
    }
    
private:
    RealType m_param;
    AleamCore& m_rng;
};
```

## Adding New Framework Integrations

To add a new framework integration:

1. **Create headers**: `src/aleam/integrations/new_integration.h`
2. **Implement class**: Inherit from `BaseGenerator`
3. **Add to integrations.cpp**: Implement methods
4. **Add to module.cpp**: Bind to Python
5. **Add to __init__.py**: Add lazy import
6. **Add to pyproject.toml**: Add optional dependency
7. **Add tests**: `tests/test_integrations.py`
8. **Update documentation**: `docs/index.md`

## Security Vulnerabilities

If you discover a security vulnerability, please **DO NOT** file a public issue.

**Contact:** contact.fardinsabid@gmail.com

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Getting Help

- **Documentation**: [GitHub Docs](https://github.com/fardinsabid/aleam/blob/main/docs/index.md)
- **Issues**: [GitHub Issues](https://github.com/fardinsabid/aleam/issues)
- **Research Paper**: [ALEAM_RESEARCH_PAPER.md](https://github.com/fardinsabid/aleam/blob/main/docs/ALEAM_RESEARCH_PAPER.md)

## Recognition

Contributors will be acknowledged in:

- **README.md** - Contributors section (upon request)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

<div align="center">

**Thank you for contributing to Aleam!**

Made with ❤️ by Fardin Sabid  
🇧🇩 From Bangladesh, for the World 🌍

</div>
```