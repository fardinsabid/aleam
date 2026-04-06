/**
 * @file cupy_integration.h
 * @brief CuPy integration for Aleam true randomness on GPU
 * @license MIT
 * 
 * Provides true random array generation for CuPy on GPU.
 * This class generates random arrays directly on GPU memory
 * using CUDA kernels for maximum performance.
 * 
 * Example:
 * @code
 *   #include "aleam/integrations/cupy_integration.h"
 *   
 *   aleam::AleamCore rng;
 *   aleam::integrations::CuPyGenerator gen(rng);
 *   auto array = gen.random({10000, 10000});  // 100M numbers on GPU
 * @endcode
 */

#ifndef ALEAM_INTEGRATIONS_CUPY_INTEGRATION_H
#define ALEAM_INTEGRATIONS_CUPY_INTEGRATION_H

#include <vector>
#include <cstdint>
#include "../core/aleam_core.h"

namespace aleam {
namespace integrations {

/**
 * @brief CuPy-compatible random generator using true randomness
 * 
 * Provides true random array generation for CuPy on GPU.
 * Uses CUDA kernels for high-performance parallel generation.
 */
class CuPyGenerator {
public:
    /**
     * @brief Construct a CuPyGenerator
     * 
     * @param rng Reference to AleamCore instance
     */
    explicit CuPyGenerator(AleamCore& rng);
    
    /**
     * @brief Generate uniform random array (float32)
     * 
     * @param shape Shape as flattened size
     * @return std::vector<float> Random values (to be copied to GPU)
     */
    std::vector<float> random_float32(size_t shape);
    
    /**
     * @brief Generate uniform random array (float64)
     * 
     * @param shape Shape as flattened size
     * @return std::vector<double> Random values (to be copied to GPU)
     */
    std::vector<double> random_float64(size_t shape);
    
    /**
     * @brief Generate normal random array (float32)
     * 
     * @param shape Shape as flattened size
     * @param mu Mean of distribution
     * @param sigma Standard deviation
     * @return std::vector<float> Random normal values
     */
    std::vector<float> randn_float32(size_t shape, float mu = 0.0f, float sigma = 1.0f);
    
    /**
     * @brief Generate normal random array (float64)
     * 
     * @param shape Shape as flattened size
     * @param mu Mean of distribution
     * @param sigma Standard deviation
     * @return std::vector<double> Random normal values
     */
    std::vector<double> randn_float64(size_t shape, double mu = 0.0, double sigma = 1.0);
    
    /**
     * @brief Generate random integer array (int32)
     * 
     * @param shape Shape as flattened size
     * @param low Lower bound (inclusive)
     * @param high Upper bound (exclusive)
     * @return std::vector<int32_t> Random integers
     */
    std::vector<int32_t> randint32(size_t shape, int32_t low, int32_t high);
    
    /**
     * @brief Generate random integer array (int64)
     * 
     * @param shape Shape as flattened size
     * @param low Lower bound (inclusive)
     * @param high Upper bound (exclusive)
     * @return std::vector<int64_t> Random integers
     */
    std::vector<int64_t> randint64(size_t shape, int64_t low, int64_t high);
    
    /**
     * @brief Get seeds for CUDA kernels
     * 
     * Generates true random seeds for each GPU block.
     * 
     * @param num_blocks Number of GPU blocks
     * @return std::vector<uint64_t> Per-block seeds
     */
    std::vector<uint64_t> get_seeds(size_t num_blocks);
    
    /**
     * @brief Check if CUDA is available
     * 
     * @return bool True if CUDA is available
     */
    static bool cuda_available();
    
    /**
     * @brief Get CUDA device count
     * 
     * @return int Number of CUDA devices
     */
    static int get_device_count();
    
private:
    /**
     * @brief Generate uniform random values (CPU fallback)
     * 
     * @param count Number of values
     * @return std::vector<double> Uniform values
     */
    std::vector<double> generate_uniform_cpu(size_t count);
    
    /**
     * @brief Generate normal random values (CPU fallback)
     * 
     * @param count Number of values
     * @param mu Mean
     * @param sigma Standard deviation
     * @return std::vector<double> Normal values
     */
    std::vector<double> generate_normal_cpu(size_t count, double mu, double sigma);
    
    /**
     * @brief Generate integer values (CPU fallback)
     * 
     * @param count Number of values
     * @param low Lower bound
     * @param high Upper bound
     * @return std::vector<int64_t> Integer values
     */
    std::vector<int64_t> generate_ints_cpu(size_t count, int64_t low, int64_t high);
    
    AleamCore& m_rng;        /**< Random number generator */
    bool m_cuda_available;   /**< CUDA availability flag */
};

}  // namespace integrations
}  // namespace aleam

#endif /* ALEAM_INTEGRATIONS_CUPY_INTEGRATION_H */