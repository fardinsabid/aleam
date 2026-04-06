/**
 * @file xarray_integration.cpp
 * @brief Xarray integration implementation for Aleam
 * @license MIT
 */

#include "xarray_integration.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <sstream>

namespace aleam {
namespace integrations {

XarrayGenerator::XarrayGenerator(AleamCore& rng)
    : m_rng(rng) {
}

std::unordered_map<std::string, double> XarrayGenerator::parse_params(const std::string& params) {
    std::unordered_map<std::string, double> result;
    if (params.empty()) return result;
    
    std::istringstream iss(params);
    std::string token;
    
    while (std::getline(iss, token, ',')) {
        size_t eq_pos = token.find('=');
        if (eq_pos != std::string::npos) {
            std::string key = token.substr(0, eq_pos);
            double value = std::stod(token.substr(eq_pos + 1));
            result[key] = value;
        }
    }
    
    return result;
}

size_t XarrayGenerator::total_elements(const std::vector<size_t>& shape) {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), 1ULL,
                           std::multiplies<size_t>());
}

std::vector<double> XarrayGenerator::generate_uniform(size_t n, double low, double high) {
    std::vector<double> result(n);
    double range = high - low;
    
    for (size_t i = 0; i < n; ++i) {
        result[i] = low + m_rng.random() * range;
    }
    
    return result;
}

std::vector<double> XarrayGenerator::generate_normal(size_t n, double mu, double sigma) {
    std::vector<double> result(n);
    
    for (size_t i = 0; i < n; ++i) {
        double u1 = m_rng.random();
        double u2 = m_rng.random();
        double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        result[i] = mu + sigma * z;
    }
    
    return result;
}

std::vector<double> XarrayGenerator::dataarray(const std::vector<size_t>& shape,
                                                 const std::vector<std::string>& /*dims*/,
                                                 const std::string& distribution,
                                                 const std::string& params) {
    size_t total = total_elements(shape);
    auto parsed = parse_params(params);
    
    if (distribution == "uniform") {
        double low = parsed.count("low") ? parsed["low"] : 0.0;
        double high = parsed.count("high") ? parsed["high"] : 1.0;
        return generate_uniform(total, low, high);
    } else if (distribution == "normal") {
        double mu = parsed.count("mu") ? parsed["mu"] : 0.0;
        double sigma = parsed.count("sigma") ? parsed["sigma"] : 1.0;
        return generate_normal(total, mu, sigma);
    } else {
        // Default to uniform
        return generate_uniform(total, 0.0, 1.0);
    }
}

std::vector<int64_t> XarrayGenerator::dataarray_int(const std::vector<size_t>& shape,
                                                      const std::vector<std::string>& /*dims*/,
                                                      int64_t low, int64_t high) {
    size_t total = total_elements(shape);
    std::vector<int64_t> result(total);
    int64_t range = high - low;
    
    for (size_t i = 0; i < total; ++i) {
        result[i] = low + static_cast<int64_t>(m_rng.random() * range);
        if (result[i] >= high) result[i] = high - 1;
    }
    
    return result;
}

std::vector<uint8_t> XarrayGenerator::dataarray_bool(const std::vector<size_t>& shape,
                                                       const std::vector<std::string>& /*dims*/,
                                                       double p) {
    size_t total = total_elements(shape);
    std::vector<uint8_t> result(total);
    
    for (size_t i = 0; i < total; ++i) {
        result[i] = (m_rng.random() < p) ? 1 : 0;
    }
    
    return result;
}

std::vector<double> XarrayGenerator::coordinates(size_t size,
                                                   const std::string& coord_type,
                                                   const std::string& params) {
    auto parsed = parse_params(params);
    
    if (coord_type == "linear") {
        double start = parsed.count("start") ? parsed["start"] : 0.0;
        double stop = parsed.count("stop") ? parsed["stop"] : static_cast<double>(size);
        std::vector<double> result(size);
        for (size_t i = 0; i < size; ++i) {
            result[i] = start + (stop - start) * i / (size - 1);
        }
        return result;
    } else if (coord_type == "uniform") {
        double low = parsed.count("low") ? parsed["low"] : 0.0;
        double high = parsed.count("high") ? parsed["high"] : 1.0;
        return generate_uniform(size, low, high);
    } else if (coord_type == "normal") {
        double mu = parsed.count("mu") ? parsed["mu"] : 0.0;
        double sigma = parsed.count("sigma") ? parsed["sigma"] : 1.0;
        return generate_normal(size, mu, sigma);
    } else {
        // Default to linear
        std::vector<double> result(size);
        for (size_t i = 0; i < size; ++i) {
            result[i] = static_cast<double>(i);
        }
        return result;
    }
}

}  // namespace integrations
}  // namespace aleam