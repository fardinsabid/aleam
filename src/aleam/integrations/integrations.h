/**
 * @file integrations.h
 * @brief Framework integrations for Aleam - PyTorch, TensorFlow, JAX, CuPy, etc.
 * @license MIT
 * 
 * This file provides integration interfaces for popular ML frameworks.
 * These classes are designed to be exposed to Python via pybind11,
 * allowing seamless drop-in replacements for framework random generators.
 * 
 * Supported frameworks:
 * - PyTorch: TorchGenerator (replaces torch.Generator)
 * - TensorFlow: TFGenerator (replaces tf.random.Generator)
 * - JAX: JAXGenerator (provides true random keys)
 * - CuPy: CuPyGenerator (GPU arrays)
 * - Pandas: PandasGenerator (DataFrame/Series)
 * - Polars: PolarsGenerator (DataFrame/Series)
 * - Xarray: XarrayGenerator (DataArray/Dataset)
 * - PyMC: PyMCGenerator (Bayesian sampling)
 * - Dask: DaskGenerator (distributed arrays)
 */

#ifndef ALEAM_INTEGRATIONS_H
#define ALEAM_INTEGRATIONS_H

#include <cstdint>
#include <vector>
#include <string>
#include <functional>
#include <stdexcept>
#include <utility>
#include "../core/aleam_core.h"

namespace aleam {
namespace integrations {

/* ============================================================================
 * Base Generator Interface
 * ============================================================================ */

/**
 * @brief Base class for all framework integrations
 * 
 * Provides common functionality for all generators:
 * - Access to underlying AleamCore
 * - True random seed generation
 * - Batch generation utilities
 */
class BaseGenerator {
public:
    /**
     * @brief Construct a new BaseGenerator object
     * 
     * @param rng Pointer to AleamCore instance (uses thread-local if nullptr)
     */
    explicit BaseGenerator(AleamCore* rng = nullptr);
    
    /**
     * @brief Destroy the BaseGenerator object
     */
    virtual ~BaseGenerator() = default;
    
    /**
     * @brief Generate a true random float in [0, 1)
     * 
     * @return double Random float
     */
    double random();
    
    /**
     * @brief Generate a true random 64-bit integer
     * 
     * @return uint64_t Random integer
     */
    uint64_t random_uint64();
    
    /**
     * @brief Generate a batch of random floats
     * 
     * @param count Number of values to generate
     * @return std::vector<double> Batch of random floats
     */
    std::vector<double> random_batch(size_t count);
    
    /**
     * @brief Generate a true random seed (for compatibility with seeded PRNGs)
     * 
     * @return uint64_t True random seed
     */
    uint64_t true_seed();
    
    /**
     * @brief Generate multiple true random seeds
     * 
     * @param count Number of seeds to generate
     * @return std::vector<uint64_t> List of true random seeds
     */
    std::vector<uint64_t> true_seeds(size_t count);
    
protected:
    AleamCore* m_rng;           /**< Pointer to AleamCore instance */
    bool m_owns_rng;            /**< Whether we own the RNG */
    
    /**
     * @brief Get RNG instance (creates if needed)
     * 
     * @return AleamCore& Reference to RNG
     */
    AleamCore& get_rng();
};

/* ============================================================================
 * PyTorch Integration
 * ============================================================================ */

/**
 * @brief PyTorch-compatible random generator using true randomness
 * 
 * Drop-in replacement for torch.Generator with true entropy.
 * Usage: gen = TorchGenerator(); tensor = torch.randn(100, 100, generator=gen)
 */
class TorchGenerator : public BaseGenerator {
public:
    /**
     * @brief Construct a new TorchGenerator object
     * 
     * @param device Device name ("cpu", "cuda", etc.) - for Python binding
     * @param rng Pointer to AleamCore instance
     */
    explicit TorchGenerator(const std::string& device = "cpu", AleamCore* rng = nullptr);
    
    /**
     * @brief Get the device name
     * 
     * @return std::string Device name
     */
    const std::string& device() const { return m_device; }
    
    /**
     * @brief Set the device
     * 
     * @param device New device name
     */
    void set_device(const std::string& device) { m_device = device; }
    
    /**
     * @brief Generate random uniform tensor (returns data for Python to convert)
     * 
     * @param size Number of elements
     * @return std::vector<double> Random values
     */
    std::vector<double> rand(size_t size);
    
    /**
     * @brief Generate random normal tensor
     * 
     * @param size Number of elements
     * @return std::vector<double> Random normal values
     */
    std::vector<double> randn(size_t size);
    
    /**
     * @brief Generate random integers
     * 
     * @param low Lower bound (inclusive)
     * @param high Upper bound (exclusive)
     * @param size Number of elements
     * @return std::vector<int64_t> Random integers
     */
    std::vector<int64_t> randint(int64_t low, int64_t high, size_t size);
    
    /**
     * @brief Manual seed (throws error - Aleam doesn't support seeding)
     */
    void manual_seed(int64_t seed);
    
private:
    std::string m_device;       /**< PyTorch device name */
};

/* ============================================================================
 * TensorFlow Integration
 * ============================================================================ */

/**
 * @brief TensorFlow-compatible random generator using true randomness
 * 
 * Drop-in replacement for tf.random.Generator with true entropy.
 */
class TFGenerator : public BaseGenerator {
public:
    /**
     * @brief Construct a new TFGenerator object
     * 
     * @param rng Pointer to AleamCore instance
     */
    explicit TFGenerator(AleamCore* rng = nullptr);
    
    /**
     * @brief Generate random normal tensor
     * 
     * @param shape Shape as flattened size
     * @param mean Mean of distribution
     * @param stddev Standard deviation
     * @return std::vector<double> Random normal values
     */
    std::vector<double> normal(size_t shape, double mean = 0.0, double stddev = 1.0);
    
    /**
     * @brief Generate random uniform tensor
     * 
     * @param shape Shape as flattened size
     * @param minval Lower bound
     * @param maxval Upper bound
     * @return std::vector<double> Random uniform values
     */
    std::vector<double> uniform(size_t shape, double minval = 0.0, double maxval = 1.0);
    
    /**
     * @brief Generate random integers
     * 
     * @param shape Shape as flattened size
     * @param minval Lower bound (inclusive)
     * @param maxval Upper bound (exclusive)
     * @return std::vector<int64_t> Random integers
     */
    std::vector<int64_t> randint(size_t shape, int64_t minval, int64_t maxval);
    
    /**
     * @brief Generate truncated normal (clipped to [mean-2*stddev, mean+2*stddev])
     * 
     * @param shape Shape as flattened size
     * @param mean Mean of distribution
     * @param stddev Standard deviation
     * @return std::vector<double> Truncated normal values
     */
    std::vector<double> truncated_normal(size_t shape, double mean = 0.0, double stddev = 1.0);
};

/* ============================================================================
 * JAX Integration
 * ============================================================================ */

/**
 * @brief JAX-compatible random generator using true randomness
 * 
 * Provides true random keys for JAX's functional PRNG system.
 */
class JAXGenerator : public BaseGenerator {
public:
    /**
     * @brief Construct a new JAXGenerator object
     * 
     * @param rng Pointer to AleamCore instance
     */
    explicit JAXGenerator(AleamCore* rng = nullptr);
    
    /**
     * @brief Generate a true random JAX key (64-bit seed)
     * 
     * @return uint64_t True random seed for JAX key
     */
    uint64_t key();
    
    /**
     * @brief Generate multiple true random keys
     * 
     * @param count Number of keys to generate
     * @return std::vector<uint64_t> List of true random seeds
     */
    std::vector<uint64_t> keys(size_t count);
    
    /**
     * @brief Generate random normal values
     * 
     * @param shape Number of elements
     * @param mean Mean of distribution
     * @param stddev Standard deviation
     * @return std::vector<double> Random normal values
     */
    std::vector<double> normal(size_t shape, double mean = 0.0, double stddev = 1.0);
    
    /**
     * @brief Generate random uniform values
     * 
     * @param shape Number of elements
     * @param minval Lower bound
     * @param maxval Upper bound
     * @return std::vector<double> Random uniform values
     */
    std::vector<double> uniform(size_t shape, double minval = 0.0, double maxval = 1.0);
    
private:
    uint64_t m_counter;         /**< Counter for unique key generation */
};

/* ============================================================================
 * CuPy Integration
 * ============================================================================ */

/**
 * @brief CuPy-compatible random generator using true randomness
 * 
 * Provides true random arrays directly on GPU via CuPy.
 */
class CuPyGenerator : public BaseGenerator {
public:
    /**
     * @brief Construct a new CuPyGenerator object
     * 
     * @param rng Pointer to AleamCore instance
     */
    explicit CuPyGenerator(AleamCore* rng = nullptr);
    
    /**
     * @brief Generate random uniform array (returns data for Python to convert)
     * 
     * @param size Number of elements
     * @param dtype Data type name ("float32", "float64")
     * @return std::vector<double> Random values
     */
    std::vector<double> random(size_t size, const std::string& dtype = "float32");
    
    /**
     * @brief Generate random normal array
     * 
     * @param size Number of elements
     * @param mu Mean of distribution
     * @param sigma Standard deviation
     * @param dtype Data type name
     * @return std::vector<double> Random normal values
     */
    std::vector<double> randn(size_t size, double mu = 0.0, double sigma = 1.0, 
                               const std::string& dtype = "float32");
    
    /**
     * @brief Generate random integer array
     * 
     * @param size Number of elements
     * @param low Lower bound (inclusive)
     * @param high Upper bound (exclusive)
     * @return std::vector<int64_t> Random integers
     */
    std::vector<int64_t> randint(size_t size, int64_t low, int64_t high);
};

/* ============================================================================
 * Pandas Integration
 * ============================================================================ */

/**
 * @brief Pandas-compatible random generator using true randomness
 */
class PandasGenerator : public BaseGenerator {
public:
    /**
     * @brief Construct a new PandasGenerator object
     * 
     * @param rng Pointer to AleamCore instance
     */
    explicit PandasGenerator(AleamCore* rng = nullptr);
    
    /**
     * @brief Generate random series data
     * 
     * @param n Number of elements
     * @param distribution Distribution name ("uniform", "normal", "exponential", "poisson")
     * @param params Distribution parameters (encoded as string for Python)
     * @return std::vector<double> Random values
     */
    std::vector<double> series(size_t n, const std::string& distribution, 
                                const std::string& params);
    
    /**
     * @brief Shuffle indices (returns shuffled order for Python to apply)
     * 
     * @param n Number of elements
     * @return std::vector<size_t> Shuffled indices
     */
    std::vector<size_t> shuffle_indices(size_t n);
};

/* ============================================================================
 * Dask Integration
 * ============================================================================ */

/**
 * @brief Dask-compatible random generator using true randomness
 */
class DaskGenerator : public BaseGenerator {
public:
    /**
     * @brief Construct a new DaskGenerator object
     * 
     * @param rng Pointer to AleamCore instance
     */
    explicit DaskGenerator(AleamCore* rng = nullptr);
    
    /**
     * @brief Generate random values for a Dask block
     * 
     * @param block_shape Number of elements in block
     * @param distribution Distribution name
     * @return std::vector<double> Random values for block
     */
    std::vector<double> block_random(size_t block_shape, const std::string& distribution = "uniform");
};

}  // namespace integrations
}  // namespace aleam

#endif /* ALEAM_INTEGRATIONS_H */