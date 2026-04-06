#!/usr/bin/env python3
"""
Setup configuration for Aleam - True Randomness for AI and Machine Learning

This setup.py automatically detects CUDA and enables GPU kernels if available.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from setuptools import setup, find_packages, Extension

# ============================================================================
# Version
# ============================================================================
VERSION = "1.0.3"

# ============================================================================
# Read README
# ============================================================================
def read_readme():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "Aleam - True Randomness for AI and Machine Learning"

# ============================================================================
# CUDA Detection
# ============================================================================
def cuda_available():
    """Check if CUDA is available on the system."""
    
    # Check for nvcc
    try:
        result = subprocess.run(
            ["nvcc", "--version"], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass
    
    # Check for CUDA_PATH environment variable
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path and Path(cuda_path).exists():
        return True
    
    # Check for standard CUDA installation paths
    if platform.system() == "Linux":
        if Path("/usr/local/cuda/bin/nvcc").exists():
            return True
    elif platform.system() == "Windows":
        if Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/bin/nvcc.exe").exists():
            return True
    
    return False

# ============================================================================
# Collect CUDA sources if available
# ============================================================================
def get_extra_compile_args():
    """Get extra compile arguments based on platform and CUDA."""
    args = ["-std=c++17", "-O3", "-Wall", "-Wextra", "-fPIC"]
    
    if platform.system() == "Linux":
        args.append("-pthread")
    elif platform.system() == "Windows":
        args = ["/std:c++17", "/O2", "/EHsc"]
    
    return args

def get_sources():
    """Get C++ source files, including CUDA if available."""
    sources = [
        "src/aleam/core/aleam_core.cpp",
        "src/aleam/distributions/distributions.cpp",
        "src/aleam/arrays/arrays.cpp",
        "src/aleam/ai/ai.cpp",
        "src/aleam/integrations/integrations.cpp",
        "src/aleam/bindings/module.cpp",
    ]
    
    # Add CUDA sources if available
    if cuda_available():
        print("✅ CUDA detected - building with GPU kernel support")
        sources.extend([
            "src/aleam/cuda/cuda_kernels.cu",
            "src/aleam/cuda/cuda_uniform.cu",
            "src/aleam/cuda/cuda_normal.cu",
        ])
    else:
        print("💻 CUDA not detected - building without GPU kernels")
        print("   GPU acceleration still works via PyTorch/TensorFlow/JAX")
    
    return sources

def get_include_dirs():
    """Get include directories."""
    import pybind11
    return [
        "src/aleam",
        "src/aleam/core",
        "src/aleam/distributions",
        "src/aleam/arrays",
        "src/aleam/ai",
        "src/aleam/integrations",
        "src/aleam/bindings",
        "src/aleam/entropy",
        "src/aleam/hash",
        pybind11.get_include(),
    ]

def get_libraries():
    """Get libraries to link."""
    libraries = []
    
    # Windows needs bcrypt for entropy
    if platform.system() == "Windows":
        libraries.append("bcrypt")
    
    # CUDA libraries
    if cuda_available():
        if platform.system() == "Linux":
            libraries.append("cuda")
        elif platform.system() == "Windows":
            libraries.append("cudart")
    
    return libraries

def get_extra_link_args():
    """Get extra link arguments."""
    args = []
    
    if platform.system() == "Linux" and cuda_available():
        cuda_path = "/usr/local/cuda/lib64"
        if Path(cuda_path).exists():
            args.append(f"-L{cuda_path}")
    
    return args

# ============================================================================
# Extension module
# ============================================================================
ext_modules = [
    Extension(
        "aleam._c_core",
        sources=get_sources(),
        include_dirs=get_include_dirs(),
        libraries=get_libraries(),
        library_dirs=[],
        runtime_library_dirs=[],
        extra_compile_args=get_extra_compile_args(),
        extra_link_args=get_extra_link_args(),
    )
]

# ============================================================================
# Setup
# ============================================================================
setup(
    name="aleam",
    version=VERSION,
    author="Fardin Sabid",
    author_email="contact.fardinsabid@gmail.com",
    description="True randomness for AI and machine learning. Non-recursive, stateless, cryptographically secure random number generator.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/fardinsabid/aleam",
    project_urls={
        "Documentation": "https://github.com/fardinsabid/aleam/docs",
        "Source": "https://github.com/fardinsabid/aleam",
        "Issues": "https://github.com/fardinsabid/aleam/issues",
    },
    packages=find_packages(include=["aleam"]),
    include_package_data=True,
    package_data={"aleam": ["py.typed"]},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=["numpy>=1.24.0"],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "pytest-benchmark>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "ruff>=0.1",
            "build>=1.0",
            "twine>=4.0",
        ],
        "torch": ["torch>=2.0.0"],
        "tensorflow": ["tensorflow>=2.12.0"],
        "jax": ["jax>=0.4.0", "jaxlib>=0.4.0"],
        "cupy": ["cupy-cuda12x>=12.0.0"],
        "pandas": ["pandas>=1.5.0"],
        "polars": ["polars>=0.18.0"],
        "xarray": ["xarray>=2023.0.0"],
        "dask": ["dask>=2023.0.0"],
        "scipy": ["scipy>=1.9.0"],
        "pymc": ["pymc>=5.0.0"],
        "all": [
            "torch>=2.0.0",
            "tensorflow>=2.12.0",
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
            "cupy-cuda12x>=12.0.0",
            "pandas>=1.5.0",
            "polars>=0.18.0",
            "xarray>=2023.0.0",
            "dask>=2023.0.0",
            "scipy>=1.9.0",
            "pymc>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aleam=aleam.__main__:main",
        ],
    },
    ext_modules=ext_modules,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Security :: Cryptography",
    ],
)

# ============================================================================
# Print summary
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ALEAM - True Randomness for AI")
    print("=" * 60)
    print(f"\nVersion: {VERSION}")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"CUDA detected: {cuda_available()}")
    if cuda_available():
        print("  ✅ Building with GPU kernel support")
    else:
        print("  💻 Building without GPU kernels (GPU still works via PyTorch/TF/JAX)")
    print("\nFor documentation: https://github.com/fardinsabid/aleam")
    print("=" * 60)