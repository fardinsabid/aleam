"""
Setup configuration for Aleam.

Aleam - True randomness for AI and machine learning.
A cryptographically secure, non-recursive, stateless random number generator.
"""

import os
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

# Read long description from README
def read_readme():
    """Read README.md content"""
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    """Read requirements.txt content"""
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read version from package
def get_version():
    """Get version from package __init__.py"""
    version = {}
    with open("aleam/__init__.py", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("__version__"):
                exec(line, version)
                break
    return version.get("__version__", "1.0.1")

# Custom test command
class PyTest(TestCommand):
    """Custom test command for pytest"""
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)

# Package metadata
setup(
    # Basic metadata
    name="aleam",
    version=get_version(),
    author="Fardin Sabid",
    author_email="contact.fardinsabid@gmail.com",
    maintainer="Fardin Sabid",
    maintainer_email="contact.fardinsabid@gmail.com",
    description="True randomness for AI and machine learning. Non-recursive, stateless, cryptographically secure random number generator.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    license="MIT",
    
    # URLs
    url="https://github.com/fardinsabid/aleam",
    project_urls={
        "Documentation": "https://github.com/fardinsabid/aleam/docs",
        "Source": "https://github.com/fardinsabid/aleam",
        "Issue Tracker": "https://github.com/fardinsabid/aleam/issues",
        "Discussions": "https://github.com/fardinsabid/aleam/discussions",
        "Changelog": "https://github.com/fardinsabid/aleam/releases",
    },
    
    # Package configuration
    packages=find_packages(
        include=["aleam", "aleam.*"],
        exclude=["tests", "tests.*", "benchmarks", "benchmarks.*", "examples", "examples.*", "docs", "docs.*"]
    ),
    include_package_data=True,
    zip_safe=False,
    
    # Python version requirements
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=[
        "numpy>=2.4.4",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "pytest-xdist>=3.0",
            "pytest-benchmark>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "ruff>=0.1",
            "pre-commit>=3.0",
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
    
    # Entry points
    entry_points={
        "console_scripts": [
            "aleam=aleam.__main__:main",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Security :: Cryptography",
    ],
    
    # Keywords
    keywords=[
        "random", "randomness", "true-random", "cryptography",
        "entropy", "ai", "machine-learning", "deep-learning",
        "pytorch", "tensorflow", "jax", "cupy", "numpy", "pandas",
        "rng", "trng", "generator", "statistics", "distribution",
        "gradient-noise", "dropout", "latent-sampling",
    ],
    
    # Package data
    package_data={
        "aleam": [
            "py.typed",
            "*.pyi",
        ],
    },
    
    # Command line
    cmdclass={"test": PyTest},
    
    # Tests
    test_suite="tests",
    
    # Options
    options={
        "build_scripts": {
            "executable": "/usr/bin/env python3",
        },
    },
)

# Print installation info
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ALEAM - True Randomness for AI")
    print("=" * 60)
    print(f"\nVersion: {get_version()}")
    print(f"Python: {sys.version}")
    print("\nFor complete documentation: https://github.com/fardinsabid/aleam")
    print("=" * 60)