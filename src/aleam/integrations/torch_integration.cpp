/**
 * @file torch_integration.cpp
 * @brief PyTorch integration implementation for Aleam
 * @license MIT
 */

#include "torch_integration.h"
#include <cmath>
#include <algorithm>

namespace aleam {
namespace integrations {

TorchGenerator::TorchGenerator(AleamCore& rng, const std::string& device)
    : m_rng(rng)
    , m_device(device)
    , m_cuda_available(false) {
    
#ifdef TORCH_CUDA_AVAILABLE
    m_cuda_available = torch::cuda::is_available();
#endif
    
    set_device(device);
}

void TorchGenerator::set_device(const std::string& device) {
    if (device == "cuda" && !m_cuda_available) {
        m_device = "cpu";
    } else {
        m_device = device;
    }
}

bool TorchGenerator::cuda_available() {
#ifdef TORCH_CUDA_AVAILABLE
    return torch::cuda::is_available();
#else
    return false;
#endif
}

std::vector<float> TorchGenerator::generate_floats(size_t size) {
    std::vector<float> result(size);
    for (size_t i = 0; i < size; ++i) {
        result[i] = static_cast<float>(m_rng.random());
    }
    return result;
}

std::vector<double> TorchGenerator::generate_doubles(size_t size) {
    std::vector<double> result(size);
    m_rng.random_batch(result.data(), size);
    return result;
}

std::vector<int64_t> TorchGenerator::generate_ints(int64_t low, int64_t high, size_t size) {
    std::vector<int64_t> result(size);
    int64_t range = high - low;
    for (size_t i = 0; i < size; ++i) {
        result[i] = low + static_cast<int64_t>(m_rng.random() * range);
        if (result[i] >= high) result[i] = high - 1;
    }
    return result;
}

torch::Tensor TorchGenerator::rand(const std::vector<int64_t>& sizes) {
    size_t total = 1;
    for (int64_t s : sizes) total *= s;
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(m_device);
    torch::Tensor result = torch::empty(sizes, options);
    
    if (m_device == "cpu") {
        auto data = generate_floats(total);
        std::memcpy(result.data_ptr<float>(), data.data(), total * sizeof(float));
    } else {
#ifdef TORCH_CUDA_AVAILABLE
        auto cpu_tensor = torch::from_blob(generate_floats(total).data(), {static_cast<int64_t>(total)}, torch::kFloat32);
        result = cpu_tensor.to(torch::kCUDA).reshape(sizes);
#else
        throw std::runtime_error("CUDA not available");
#endif
    }
    
    return result;
}

torch::Tensor TorchGenerator::randn(const std::vector<int64_t>& sizes) {
    size_t total = 1;
    for (int64_t s : sizes) total *= s;
    
    std::vector<double> data(total);
    for (size_t i = 0; i < total; ++i) {
        // Box-Muller transform for normal distribution
        double u1 = m_rng.random();
        double u2 = m_rng.random();
        data[i] = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    }
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(m_device);
    torch::Tensor result = torch::empty(sizes, options);
    
    if (m_device == "cpu") {
        std::vector<float> float_data(data.begin(), data.end());
        std::memcpy(result.data_ptr<float>(), float_data.data(), total * sizeof(float));
    } else {
#ifdef TORCH_CUDA_AVAILABLE
        auto cpu_tensor = torch::from_blob(data.data(), {static_cast<int64_t>(total)}, torch::kFloat64);
        result = cpu_tensor.to(torch::kFloat32).to(torch::kCUDA).reshape(sizes);
#else
        throw std::runtime_error("CUDA not available");
#endif
    }
    
    return result;
}

torch::Tensor TorchGenerator::uniform(double low, double high, const std::vector<int64_t>& sizes) {
    auto tensor = rand(sizes);
    return tensor * (high - low) + low;
}

torch::Tensor TorchGenerator::normal(double mean, double std, const std::vector<int64_t>& sizes) {
    auto tensor = randn(sizes);
    return tensor * std + mean;
}

torch::Tensor TorchGenerator::randint(int64_t low, int64_t high, const std::vector<int64_t>& sizes) {
    size_t total = 1;
    for (int64_t s : sizes) total *= s;
    
    auto data = generate_ints(low, high, total);
    
    auto options = torch::TensorOptions().dtype(torch::kLong).device(m_device);
    torch::Tensor result = torch::empty(sizes, options);
    
    if (m_device == "cpu") {
        std::memcpy(result.data_ptr<int64_t>(), data.data(), total * sizeof(int64_t));
    } else {
#ifdef TORCH_CUDA_AVAILABLE
        auto cpu_tensor = torch::from_blob(data.data(), {static_cast<int64_t>(total)}, torch::kLong);
        result = cpu_tensor.to(torch::kCUDA).reshape(sizes);
#else
        throw std::runtime_error("CUDA not available");
#endif
    }
    
    return result;
}

torch::Tensor TorchGenerator::multinomial(const torch::Tensor& probs, int64_t num_samples) {
    auto options = torch::TensorOptions().dtype(torch::kLong).device(m_device);
    torch::Tensor result = torch::empty({num_samples}, options);
    
    // CPU implementation
    if (m_device == "cpu") {
        auto probs_cpu = probs.cpu();
        float* probs_ptr = probs_cpu.data_ptr<float>();
        int64_t num_classes = probs_cpu.numel();
        
        // Build cumulative distribution
        std::vector<double> cumsum(num_classes);
        double running = 0.0;
        for (int64_t i = 0; i < num_classes; ++i) {
            running += probs_ptr[i];
            cumsum[i] = running;
        }
        
        // Normalize
        for (auto& c : cumsum) {
            c /= running;
        }
        
        // Sample
        for (int64_t i = 0; i < num_samples; ++i) {
            double u = m_rng.random();
            int64_t idx = 0;
            while (idx < num_classes && u > cumsum[idx]) {
                ++idx;
            }
            if (idx >= num_classes) idx = num_classes - 1;
            result[i] = idx;
        }
    } else {
#ifdef TORCH_CUDA_AVAILABLE
        // For GPU, move to CPU for sampling (simplified)
        auto probs_cpu = probs.cpu();
        auto result_cpu = multinomial(probs_cpu, num_samples);
        result = result_cpu.to(torch::kCUDA);
#else
        throw std::runtime_error("CUDA not available");
#endif
    }
    
    return result;
}

}  // namespace integrations
}  // namespace aleam