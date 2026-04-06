/**
 * @file tensorflow_integration.cpp
 * @brief TensorFlow integration implementation for Aleam
 * @license MIT
 */

#include "tensorflow_integration.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>

namespace aleam {
namespace integrations {

TFGenerator::TFGenerator(AleamCore& rng, const std::string& device)
    : m_rng(rng)
    , m_device(device)
    , m_gpu_available(false) {
    
    // Check GPU availability (simplified - actual TF check would be more complex)
    // For now, assume GPU is available if device is "gpu"
    m_gpu_available = (device == "gpu");
    
    set_device(device);
}

void TFGenerator::set_device(const std::string& device) {
    if (device == "gpu" && !m_gpu_available) {
        m_device = "cpu";
    } else {
        m_device = device;
    }
}

size_t TFGenerator::total_elements(const std::vector<int64_t>& shape) const {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), 1ULL,
                           std::multiplies<size_t>());
}

uint64_t TFGenerator::make_seed() {
    return m_rng.random_uint64();
}

std::vector<uint64_t> TFGenerator::make_seeds(size_t count) {
    std::vector<uint64_t> seeds(count);
    for (size_t i = 0; i < count; ++i) {
        seeds[i] = m_rng.random_uint64();
    }
    return seeds;
}

std::vector<float> TFGenerator::generate_standard_normal(size_t count) {
    std::vector<float> result(count);
    
    for (size_t i = 0; i < count; ++i) {
        // Box-Muller transform
        double u1 = m_rng.random();
        double u2 = m_rng.random();
        double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        result[i] = static_cast<float>(z);
    }
    
    return result;
}

std::vector<float> TFGenerator::generate_uniform(size_t count, float minval, float maxval) {
    std::vector<float> result(count);
    float range = maxval - minval;
    
    for (size_t i = 0; i < count; ++i) {
        result[i] = minval + static_cast<float>(m_rng.random()) * range;
    }
    
    return result;
}

std::vector<int32_t> TFGenerator::generate_ints(size_t count, int32_t minval, int32_t maxval) {
    std::vector<int32_t> result(count);
    int32_t range = maxval - minval;
    
    for (size_t i = 0; i < count; ++i) {
        result[i] = minval + static_cast<int32_t>(m_rng.random() * range);
        if (result[i] >= maxval) result[i] = maxval - 1;
    }
    
    return result;
}

std::vector<float> TFGenerator::normal(const std::vector<int64_t>& shape,
                                        float mean, float stddev) {
    size_t total = total_elements(shape);
    auto values = generate_standard_normal(total);
    
    // Scale and shift
    for (auto& v : values) {
        v = mean + v * stddev;
    }
    
    return values;
}

std::vector<float> TFGenerator::uniform(const std::vector<int64_t>& shape,
                                         float minval, float maxval) {
    size_t total = total_elements(shape);
    return generate_uniform(total, minval, maxval);
}

std::vector<float> TFGenerator::truncated_normal(const std::vector<int64_t>& shape,
                                                  float mean, float stddev) {
    size_t total = total_elements(shape);
    std::vector<float> result;
    result.reserve(total);
    
    float lower = mean - 2.0f * stddev;
    float upper = mean + 2.0f * stddev;
    
    while (result.size() < total) {
        float value = mean + generate_standard_normal(1)[0] * stddev;
        if (value >= lower && value <= upper) {
            result.push_back(value);
        }
    }
    
    return result;
}

std::vector<int32_t> TFGenerator::randint(const std::vector<int64_t>& shape,
                                           int32_t minval, int32_t maxval) {
    size_t total = total_elements(shape);
    return generate_ints(total, minval, maxval);
}

std::vector<int32_t> TFGenerator::bernoulli(const std::vector<int64_t>& shape, float prob) {
    size_t total = total_elements(shape);
    std::vector<int32_t> result(total);
    
    for (size_t i = 0; i < total; ++i) {
        result[i] = (m_rng.random() < prob) ? 1 : 0;
    }
    
    return result;
}

std::vector<uint8_t> TFGenerator::shuffle(const std::vector<uint8_t>& data,
                                           size_t dim0_size,
                                           size_t element_size) {
    if (dim0_size <= 1 || data.empty()) {
        return data;
    }
    
    size_t elements_per_row = data.size() / dim0_size;
    std::vector<size_t> indices(dim0_size);
    std::iota(indices.begin(), indices.end(), 0);
    
    // Fisher-Yates shuffle
    for (size_t i = dim0_size - 1; i > 0; --i) {
        size_t j = static_cast<size_t>(m_rng.random() * (i + 1));
        if (j > i) j = i;
        std::swap(indices[i], indices[j]);
    }
    
    // Reorder data
    std::vector<uint8_t> result(data.size());
    for (size_t i = 0; i < dim0_size; ++i) {
        size_t src_offset = indices[i] * elements_per_row;
        size_t dst_offset = i * elements_per_row;
        std::memcpy(result.data() + dst_offset,
                    data.data() + src_offset,
                    elements_per_row);
    }
    
    return result;
}

}  // namespace integrations
}  // namespace aleam