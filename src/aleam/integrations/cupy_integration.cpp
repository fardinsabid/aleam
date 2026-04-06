/**
 * @file cupy_integration.cpp
 * @brief CuPy integration implementation for Aleam
 * @license MIT
 */

#include "cupy_integration.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace aleam {
namespace integrations {

// Golden ratio prime constant
constexpr uint64_t GOLDEN_PRIME = 0x9E3779B97F4A7C15ULL;

CuPyGenerator::CuPyGenerator(AleamCore& rng)
    : m_rng(rng)
    , m_cuda_available(false) {
    // Check CUDA availability (simplified - actual check would use CUDA runtime)
    // For now, assume CUDA is not available for CPU fallback
    m_cuda_available = false;
}

bool CuPyGenerator::cuda_available() {
    // This would check CUDA runtime in production
    // For now, return false to use CPU fallback
    return false;
}

int CuPyGenerator::get_device_count() {
    if (!cuda_available()) return 0;
    // Would return actual CUDA device count
    return 0;
}

std::vector<double> CuPyGenerator::generate_uniform_cpu(size_t count) {
    std::vector<double> result(count);
    m_rng.random_batch(result.data(), count);
    return result;
}

std::vector<double> CuPyGenerator::generate_normal_cpu(size_t count, double mu, double sigma) {
    std::vector<double> result(count);
    
    for (size_t i = 0; i < count; ++i) {
        // Box-Muller transform
        double u1 = m_rng.random();
        double u2 = m_rng.random();
        double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        result[i] = mu + sigma * z;
    }
    
    return result;
}

std::vector<int64_t> CuPyGenerator::generate_ints_cpu(size_t count, int64_t low, int64_t high) {
    std::vector<int64_t> result(count);
    int64_t range = high - low;
    
    for (size_t i = 0; i < count; ++i) {
        result[i] = low + static_cast<int64_t>(m_rng.random() * range);
        if (result[i] >= high) result[i] = high - 1;
    }
    
    return result;
}

std::vector<float> CuPyGenerator::random_float32(size_t shape) {
    auto uniform = generate_uniform_cpu(shape);
    std::vector<float> result(shape);
    for (size_t i = 0; i < shape; ++i) {
        result[i] = static_cast<float>(uniform[i]);
    }
    return result;
}

std::vector<double> CuPyGenerator::random_float64(size_t shape) {
    return generate_uniform_cpu(shape);
}

std::vector<float> CuPyGenerator::randn_float32(size_t shape, float mu, float sigma) {
    auto normal = generate_normal_cpu(shape, mu, sigma);
    std::vector<float> result(shape);
    for (size_t i = 0; i < shape; ++i) {
        result[i] = static_cast<float>(normal[i]);
    }
    return result;
}

std::vector<double> CuPyGenerator::randn_float64(size_t shape, double mu, double sigma) {
    return generate_normal_cpu(shape, mu, sigma);
}

std::vector<int32_t> CuPyGenerator::randint32(size_t shape, int32_t low, int32_t high) {
    auto ints = generate_ints_cpu(shape, low, high);
    std::vector<int32_t> result(shape);
    for (size_t i = 0; i < shape; ++i) {
        result[i] = static_cast<int32_t>(ints[i]);
    }
    return result;
}

std::vector<int64_t> CuPyGenerator::randint64(size_t shape, int64_t low, int64_t high) {
    return generate_ints_cpu(shape, low, high);
}

std::vector<uint64_t> CuPyGenerator::get_seeds(size_t num_blocks) {
    std::vector<uint64_t> seeds(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        seeds[i] = m_rng.random_uint64();
    }
    return seeds;
}

}  // namespace integrations
}  // namespace aleam