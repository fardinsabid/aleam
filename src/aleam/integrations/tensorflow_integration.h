/**
 * @file tensorflow_integration.h
 * @brief TensorFlow integration for Aleam true randomness
 * @license MIT
 * 
 * Provides true random tensor generation for TensorFlow 2.x on both CPU and GPU.
 * 
 * Example:
 * @code
 *   #include "aleam/integrations/tensorflow_integration.h"
 *   
 *   aleam::AleamCore rng;
 *   aleam::integrations::TFGenerator gen(rng);
 *   auto tensor = gen.normal({100, 100});
 * @endcode
 */

#ifndef ALEAM_INTEGRATIONS_TENSORFLOW_INTEGRATION_H
#define ALEAM_INTEGRATIONS_TENSORFLOW_INTEGRATION_H

#include <vector>
#include <string>
#include "../core/aleam_core.h"

// Forward declare TensorFlow types (avoid including heavy headers)
namespace tensorflow {
    class Tensor;
    class Status;
    class Session;
}

namespace aleam {
namespace integrations {

/**
 * @brief TensorFlow-compatible random generator using true randomness
 * 
 * Provides true random tensor generation for TensorFlow.
 * Supports CPU and GPU devices.
 */
class TFGenerator {
public:
    /**
     * @brief Construct a TFGenerator
     * 
     * @param rng Reference to AleamCore instance
     * @param device Device to use ("cpu" or "gpu")
     */
    TFGenerator(AleamCore& rng, const std::string& device = "cpu");
    
    /**
     * @brief Generate normal random tensor
     * 
     * @param shape Tensor shape (flattened list of dimensions)
     * @param mean Mean of distribution
     * @param stddev Standard deviation
     * @return std::vector<float> Random values (to be converted to TF tensor)
     */
    std::vector<float> normal(const std::vector<int64_t>& shape, 
                               float mean = 0.0f, 
                               float stddev = 1.0f);
    
    /**
     * @brief Generate uniform random tensor
     * 
     * @param shape Tensor shape
     * @param minval Lower bound
     * @param maxval Upper bound
     * @return std::vector<float> Random values
     */
    std::vector<float> uniform(const std::vector<int64_t>& shape,
                                float minval = 0.0f,
                                float maxval = 1.0f);
    
    /**
     * @brief Generate truncated normal tensor (clipped to [mean-2*stddev, mean+2*stddev])
     * 
     * @param shape Tensor shape
     * @param mean Mean of distribution
     * @param stddev Standard deviation
     * @return std::vector<float> Truncated normal values
     */
    std::vector<float> truncated_normal(const std::vector<int64_t>& shape,
                                         float mean = 0.0f,
                                         float stddev = 1.0f);
    
    /**
     * @brief Generate random integer tensor
     * 
     * @param shape Tensor shape
     * @param minval Lower bound (inclusive)
     * @param maxval Upper bound (exclusive)
     * @return std::vector<int32_t> Random integers
     */
    std::vector<int32_t> randint(const std::vector<int64_t>& shape,
                                  int32_t minval,
                                  int32_t maxval);
    
    /**
     * @brief Generate random Bernoulli tensor (binary values)
     * 
     * @param shape Tensor shape
     * @param prob Probability of 1 (success)
     * @return std::vector<int32_t> Binary values (0 or 1)
     */
    std::vector<int32_t> bernoulli(const std::vector<int64_t>& shape, float prob = 0.5f);
    
    /**
     * @brief Shuffle the first dimension of a tensor
     * 
     * @param data Input data
     * @param dim0_size Size of first dimension
     * @param element_size Size of each element in bytes
     * @return std::vector<uint8_t> Shuffled data
     */
    std::vector<uint8_t> shuffle(const std::vector<uint8_t>& data,
                                  size_t dim0_size,
                                  size_t element_size);
    
    /**
     * @brief Get current device
     * 
     * @return std::string Device name
     */
    const std::string& device() const { return m_device; }
    
    /**
     * @brief Set device
     * 
     * @param device Device name ("cpu" or "gpu")
     */
    void set_device(const std::string& device);
    
    /**
     * @brief Generate a true random seed (for compatibility)
     * 
     * @return uint64_t True random seed
     */
    uint64_t make_seed();
    
    /**
     * @brief Generate multiple true random seeds
     * 
     * @param count Number of seeds
     * @return std::vector<uint64_t> Seeds
     */
    std::vector<uint64_t> make_seeds(size_t count);
    
private:
    /**
     * @brief Calculate total number of elements from shape
     * 
     * @param shape Tensor shape
     * @return size_t Total elements
     */
    size_t total_elements(const std::vector<int64_t>& shape) const;
    
    /**
     * @brief Generate standard normal random numbers using Box-Muller
     * 
     * @param count Number of values
     * @return std::vector<float> Standard normal values
     */
    std::vector<float> generate_standard_normal(size_t count);
    
    /**
     * @brief Generate uniform random numbers
     * 
     * @param count Number of values
     * @param minval Minimum value
     * @param maxval Maximum value
     * @return std::vector<float> Uniform values
     */
    std::vector<float> generate_uniform(size_t count, float minval, float maxval);
    
    /**
     * @brief Generate integers
     * 
     * @param count Number of values
     * @param minval Minimum value
     * @param maxval Maximum value
     * @return std::vector<int32_t> Integer values
     */
    std::vector<int32_t> generate_ints(size_t count, int32_t minval, int32_t maxval);
    
    AleamCore& m_rng;        /**< Random number generator */
    std::string m_device;    /**< Device name */
    bool m_gpu_available;    /**< GPU availability flag */
};

}  // namespace integrations
}  // namespace aleam

#endif /* ALEAM_INTEGRATIONS_TENSORFLOW_INTEGRATION_H */