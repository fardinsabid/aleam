# Installation Guide

## Quick Install (Recommended)

```bash
pip install aleam
```

## Detailed Installation

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.8+ | Required |
| pip | Latest | Required |
| C++ compiler | C++17 | For source builds only |
| CMake | 3.15+ | For source builds only |
| CUDA Toolkit | 11.0+ | Optional (GPU acceleration) |

### Platform-Specific Instructions

#### Ubuntu/Debian

```bash
# Install system dependencies (for source builds)
sudo apt update
sudo apt install -y build-essential cmake

# Install Aleam
pip install aleam

# Optional: Install CUDA support
pip install cupy-cuda12x
```

#### Arch Linux

```bash
# Install system dependencies (for source builds)
sudo pacman -S base-devel cmake

# Install Aleam
pip install aleam

# Optional: Install CUDA support
pip install cupy-cuda12x
```

#### macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies (for source builds)
brew install cmake

# Install Aleam
pip install aleam

# For Apple Silicon (M1/M2/M3), no special flags needed
```

#### Windows

```bash
# Install Visual Studio Build Tools (for source builds)
# Download from: https://visualstudio.microsoft.com/downloads/

# Install Aleam
pip install aleam

# Optional: Install CUDA support
pip install cupy-cuda12x
```

### With Framework Support

```bash
# PyTorch
pip install aleam[torch]

# TensorFlow
pip install aleam[tensorflow]

# JAX
pip install aleam[jax]

# CuPy (GPU acceleration)
pip install aleam[cupy]

# Data science (Pandas, Polars, etc.)
pip install aleam[pandas]

# All frameworks
pip install aleam[all]
```

### From Source

```bash
# Clone the repository
git clone https://github.com/fardinsabid/aleam.git
cd aleam

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install build dependencies
pip install pybind11 numpy setuptools wheel build

# Build and install
pip install .
```

### GPU Acceleration Setup

#### Option 1: CuPy (Recommended for speed)

```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x
```

#### Option 2: PyTorch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Option 3: TensorFlow

```bash
pip install tensorflow
```

### Verify Installation

```bash
# Test basic functionality
python -c "import aleam as al; print(al.random()); print(al.__version__)"

# Test GPU (if available)
python -c "
import cupy as cp
import aleam as al
seed = al.Aleam().random_uint64()
cp.random.seed(seed)
arr = cp.random.randn(1000, 1000)
print(f'GPU array shape: {arr.shape}')
"
```

### Troubleshooting

#### Error: `No module named 'aleam._c_core'`

**Solution:** Rebuild from source:

```bash
pip uninstall aleam -y
pip install --no-cache-dir aleam
```

#### Error: `CUDA not available`

**Solution:** Install CuPy:

```bash
pip install cupy-cuda12x
```

#### Error: `ImportError: cannot import name 'CuPyGenerator'`

**Solution:** Use the true seed pattern instead:

```python
# Wrong
cuda_gen = al.CuPyGenerator()

# Correct
import cupy as cp
seed = al.Aleam().random_uint64()
cp.random.seed(seed)
```

#### Error: `sample() requires list`

**Solution:** Convert range to list:

```python
# Wrong
rng.sample(range(100), 10)

# Correct
rng.sample(list(range(100)), 10)
```

### Docker Installation

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential cmake

RUN pip install aleam

CMD ["python", "-c", "import aleam as al; print(al.random())"]
```

### Uninstall

```bash
pip uninstall aleam
```

---

**For more help, see the [FAQ](index.md#faq) or open an [issue](https://github.com/fardinsabid/aleam/issues).**