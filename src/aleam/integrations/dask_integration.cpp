/**
 * @file dask_integration.cpp
 * @brief Dask integration implementation for Aleam
 * @license MIT
 */

#include "dask_integration.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace aleam {
namespace integrations {

DaskGenerator::DaskGenerator(AleamCore& rng)
    : m_rng(rng) {
}

size_t DaskGenerator::total_elements(const std::vector<size_t>& shape) {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), 1ULL,
                           std::multiplies<size_t>());
}

std::vector<size_t> DaskGenerator::num_blocks(const std::vector<size_t>& shape,
                                                const std::vector<size_t>& chunks) {
    std::vector<size_t> blocks(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        blocks[i] = (shape[i] + chunks[i] - 1) / chunks[i];
    }
    return blocks;
}

std::vector<double> DaskGenerator::generate_uniform(size_t count) {
    std::vector<double> result(count);
    m_rng.random_batch(result.data(), count);
    return result;
}

std::vector<double> DaskGenerator::generate_normal(size_t count, double mu, double sigma) {
    std::vector<double> result(count);
    
    for (size_t i = 0; i < count; ++i) {
        double u1 = m_rng.random();
        double u2 = m_rng.random();
        double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        result[i] = mu + sigma * z;
    }
    
    return result;
}

std::vector<int64_t> DaskGenerator::generate_ints(size_t count, int64_t low, int64_t high) {
    std::vector<int64_t> result(count);
    int64_t range = high - low;
    
    for (size_t i = 0; i < count; ++i) {
        result[i] = low + static_cast<int64_t>(m_rng.random() * range);
        if (result[i] >= high) result[i] = high - 1;
    }
    
    return result;
}

std::vector<double> DaskGenerator::block_uniform(const std::vector<size_t>& block_shape,
                                                   const std::vector<size_t>& /*chunks*/) {
    size_t total = total_elements(block_shape);
    return generate_uniform(total);
}

std::vector<double> DaskGenerator::block_normal(const std::vector<size_t>& block_shape,
                                                  const std::vector<size_t>& /*chunks*/,
                                                  double mu, double sigma) {
    size_t total = total_elements(block_shape);
    return generate_normal(total, mu, sigma);
}

std::vector<int64_t> DaskGenerator::block_randint(const std::vector<size_t>& block_shape,
                                                    const std::vector<size_t>& /*chunks*/,
                                                    int64_t low, int64_t high) {
    size_t total = total_elements(block_shape);
    return generate_ints(total, low, high);
}

std::vector<double> DaskGenerator::block_custom(const std::vector<size_t>& block_shape,
                                                  const std::vector<size_t>& /*chunks*/,
                                                  std::function<double()> sampler) {
    size_t total = total_elements(block_shape);
    std::vector<double> result(total);
    
    for (size_t i = 0; i < total; ++i) {
        result[i] = sampler();
    }
    
    return result;
}

std::vector<uint64_t> DaskGenerator::block_seeds(uint64_t block_id, size_t num_seeds) {
    std::vector<uint64_t> seeds(num_seeds);
    
    // Mix block_id into each seed for uniqueness
    for (size_t i = 0; i < num_seeds; ++i) {
        uint64_t entropy = m_rng.random_uint64();
        seeds[i] = entropy ^ (block_id * 0x9E3779B97F4A7C15ULL) ^ (i * 0xBF58476D1CE4E5B9ULL);
    }
    
    return seeds;
}

}  // namespace integrations
}  // namespace aleam