/**
 * @file dask_integration.h
 * @brief Dask integration for Aleam true randomness
 * @license MIT
 * 
 * Provides true random data generation for Dask distributed arrays.
 * Dask is a parallel computing library that scales from a single CPU to a cluster.
 * 
 * Example:
 * @code
 *   #include "aleam/integrations/dask_integration.h"
 *   
 *   aleam::AleamCore rng;
 *   aleam::integrations::DaskGenerator gen(rng);
 *   auto block = gen.block_random({1000, 1000});
 * @endcode
 */

#ifndef ALEAM_INTEGRATIONS_DASK_INTEGRATION_H
#define ALEAM_INTEGRATIONS_DASK_INTEGRATION_H

#include <vector>
#include <string>
#include <cstdint>
#include <functional>
#include "../core/aleam_core.h"

namespace aleam {
namespace integrations {

/**
 * @brief Dask-compatible random generator using true randomness
 * 
 * Provides true random data generation for Dask distributed arrays.
 * Generates data for individual blocks that can be composed into
 * a distributed Dask array.
 */
class DaskGenerator {
public:
    /**
     * @brief Construct a DaskGenerator
     * 
     * @param rng Reference to AleamCore instance
     */
    explicit DaskGenerator(AleamCore& rng);
    
    /**
     * @brief Generate a block of random data (uniform)
     * 
     * @param block_shape Shape of the block (flattened size)
     * @param chunks Chunk shape for Dask
     * @return std::vector<double> Block data (row-major order)
     */
    std::vector<double> block_uniform(const std::vector<size_t>& block_shape,
                                       const std::vector<size_t>& chunks);
    
    /**
     * @brief Generate a block of random normal data
     * 
     * @param block_shape Shape of the block
     * @param chunks Chunk shape for Dask
     * @param mu Mean of distribution
     * @param sigma Standard deviation
     * @return std::vector<double> Block data
     */
    std::vector<double> block_normal(const std::vector<size_t>& block_shape,
                                      const std::vector<size_t>& chunks,
                                      double mu = 0.0, double sigma = 1.0);
    
    /**
     * @brief Generate a block of random integer data
     * 
     * @param block_shape Shape of the block
     * @param chunks Chunk shape for Dask
     * @param low Lower bound (inclusive)
     * @param high Upper bound (exclusive)
     * @return std::vector<int64_t> Block data
     */
    std::vector<int64_t> block_randint(const std::vector<size_t>& block_shape,
                                        const std::vector<size_t>& chunks,
                                        int64_t low, int64_t high);
    
    /**
     * @brief Generate a block of random data with custom distribution
     * 
     * @param block_shape Shape of the block
     * @param chunks Chunk shape for Dask
     * @param sampler Function that generates random values
     * @return std::vector<double> Block data
     */
    std::vector<double> block_custom(const std::vector<size_t>& block_shape,
                                      const std::vector<size_t>& chunks,
                                      std::function<double()> sampler);
    
    /**
     * @brief Generate block seeds for deterministic reproducibility
     * 
     * @param block_id Block identifier
     * @param num_seeds Number of seeds needed
     * @return std::vector<uint64_t> Seeds for this block
     */
    std::vector<uint64_t> block_seeds(uint64_t block_id, size_t num_seeds);
    
    /**
     * @brief Calculate total elements from shape
     * 
     * @param shape Vector of dimensions
     * @return size_t Total number of elements
     */
    static size_t total_elements(const std::vector<size_t>& shape);
    
    /**
     * @brief Calculate number of blocks from overall shape and chunk size
     * 
     * @param shape Overall array shape
     * @param chunks Chunk shape
     * @return std::vector<size_t> Number of blocks per dimension
     */
    static std::vector<size_t> num_blocks(const std::vector<size_t>& shape,
                                           const std::vector<size_t>& chunks);
    
private:
    /**
     * @brief Generate uniform random values
     * 
     * @param count Number of values
     * @return std::vector<double> Uniform values in [0, 1)
     */
    std::vector<double> generate_uniform(size_t count);
    
    /**
     * @brief Generate normal random values
     * 
     * @param count Number of values
     * @param mu Mean
     * @param sigma Standard deviation
     * @return std::vector<double> Normal values
     */
    std::vector<double> generate_normal(size_t count, double mu, double sigma);
    
    /**
     * @brief Generate integer random values
     * 
     * @param count Number of values
     * @param low Lower bound
     * @param high Upper bound
     * @return std::vector<int64_t> Integer values
     */
    std::vector<int64_t> generate_ints(size_t count, int64_t low, int64_t high);
    
    AleamCore& m_rng;        /**< Random number generator */
};

}  // namespace integrations
}  // namespace aleam

#endif /* ALEAM_INTEGRATIONS_DASK_INTEGRATION_H */