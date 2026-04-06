/**
 * @file torch_integration.h
 * @brief PyTorch integration for Aleam true randomness
 * @license MIT
 * 
 * Provides true random tensor generation for PyTorch on both CPU and GPU.
 * This class can be used as a drop-in replacement for torch.Generator.
 * 
 * Example:
 * @code
 *   #include <torch/torch.h>
 *   #include "aleam/integrations/torch_integration.h"
 *   
 *   aleam::AleamCore rng;
 *   aleam::integrations::TorchGenerator gen(rng);
 *   auto tensor = gen.randn({100, 100});
 * @endcode
 */

#ifndef ALEAM_INTEGRATIONS_TORCH_INTEGRATION_H
#define ALEAM_INTEGRATIONS_TORCH_INTEGRATION_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include "../core/aleam_core.h"

namespace aleam {
namespace integrations {

/**
 * @brief PyTorch-compatible random generator using true randomness
 * 
 * Provides true random tensor generation for PyTorch.
 * Supports CPU and CUDA devices.
 */
class TorchGenerator {
public:
    /**
     * @brief Construct a TorchGenerator
     * 
     * @param rng Reference to AleamCore instance
     * @param device Device to use ("cpu" or "cuda")
     */
    TorchGenerator(AleamCore& rng, const std::string& device = "cpu");
    
    /**
     * @brief Generate uniform random tensor in [0, 1)
     * 
     * @param sizes Tensor dimensions
     * @return torch::Tensor Random tensor
     */
    torch::Tensor rand(const std::vector<int64_t>& sizes);
    
    /**
     * @brief Generate standard normal random tensor (mean=0, std=1)
     * 
     * @param sizes Tensor dimensions
     * @return torch::Tensor Random normal tensor
     */
    torch::Tensor randn(const std::vector<int64_t>& sizes);
    
    /**
     * @brief Generate uniform random tensor in [low, high)
     * 
     * @param low Lower bound (inclusive)
     * @param high Upper bound (exclusive)
     * @param sizes Tensor dimensions
     * @return torch::Tensor Random uniform tensor
     */
    torch::Tensor uniform(double low, double high, const std::vector<int64_t>& sizes);
    
    /**
     * @brief Generate normal random tensor with given mean and std
     * 
     * @param mean Mean of distribution
     * @param std Standard deviation
     * @param sizes Tensor dimensions
     * @return torch::Tensor Random normal tensor
     */
    torch::Tensor normal(double mean, double std, const std::vector<int64_t>& sizes);
    
    /**
     * @brief Generate random integer tensor in [low, high)
     * 
     * @param low Lower bound (inclusive)
     * @param high Upper bound (exclusive)
     * @param sizes Tensor dimensions
     * @return torch::Tensor Random integer tensor
     */
    torch::Tensor randint(int64_t low, int64_t high, const std::vector<int64_t>& sizes);
    
    /**
     * @brief Generate random values from a categorical distribution
     * 
     * @param probs Probability tensor (sums to 1)
     * @param num_samples Number of samples
     * @return torch::Tensor Sample indices
     */
    torch::Tensor multinomial(const torch::Tensor& probs, int64_t num_samples);
    
    /**
     * @brief Get current device
     * 
     * @return std::string Device name
     */
    const std::string& device() const { return m_device; }
    
    /**
     * @brief Set device
     * 
     * @param device Device name ("cpu" or "cuda")
     */
    void set_device(const std::string& device);
    
    /**
     * @brief Check if CUDA is available
     * 
     * @return bool True if CUDA is available
     */
    static bool cuda_available();
    
private:
    /**
     * @brief Generate random float values (CPU)
     * 
     * @param size Number of values
     * @return std::vector<float> Random values
     */
    std::vector<float> generate_floats(size_t size);
    
    /**
     * @brief Generate random double values (CPU)
     * 
     * @param size Number of values
     * @return std::vector<double> Random values
     */
    std::vector<double> generate_doubles(size_t size);
    
    /**
     * @brief Generate random integer values (CPU)
     * 
     * @param low Lower bound
     * @param high Upper bound
     * @param size Number of values
     * @return std::vector<int64_t> Random integers
     */
    std::vector<int64_t> generate_ints(int64_t low, int64_t high, size_t size);
    
    AleamCore& m_rng;        /**< Random number generator */
    std::string m_device;    /**< PyTorch device */
    bool m_cuda_available;   /**< CUDA availability flag */
};

}  // namespace integrations
}  // namespace aleam

#endif /* ALEAM_INTEGRATIONS_TORCH_INTEGRATION_H */