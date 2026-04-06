/**
 * @file polars_integration.cpp
 * @brief Polars integration implementation for Aleam
 * @license MIT
 */

#include "polars_integration.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <sstream>

namespace aleam {
namespace integrations {

PolarsGenerator::PolarsGenerator(AleamCore& rng)
    : m_rng(rng) {
}

std::unordered_map<std::string, double> PolarsGenerator::parse_params(const std::string& params) {
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

std::vector<double> PolarsGenerator::generate_uniform(size_t n, double low, double high) {
    std::vector<double> result(n);
    double range = high - low;
    
    for (size_t i = 0; i < n; ++i) {
        result[i] = low + m_rng.random() * range;
    }
    
    return result;
}

std::vector<double> PolarsGenerator::generate_normal(size_t n, double mu, double sigma) {
    std::vector<double> result(n);
    
    for (size_t i = 0; i < n; ++i) {
        double u1 = m_rng.random();
        double u2 = m_rng.random();
        double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        result[i] = mu + sigma * z;
    }
    
    return result;
}

std::vector<double> PolarsGenerator::generate_exponential(size_t n, double rate) {
    std::vector<double> result(n);
    
    for (size_t i = 0; i < n; ++i) {
        result[i] = -std::log(1.0 - m_rng.random()) / rate;
    }
    
    return result;
}

std::vector<double> PolarsGenerator::series_f64(size_t n, const std::string& distribution, 
                                                  const std::string& params) {
    auto parsed = parse_params(params);
    
    if (distribution == "uniform") {
        double low = parsed.count("low") ? parsed["low"] : 0.0;
        double high = parsed.count("high") ? parsed["high"] : 1.0;
        return generate_uniform(n, low, high);
    } else if (distribution == "normal") {
        double mu = parsed.count("mu") ? parsed["mu"] : 0.0;
        double sigma = parsed.count("sigma") ? parsed["sigma"] : 1.0;
        return generate_normal(n, mu, sigma);
    } else if (distribution == "exponential") {
        double rate = parsed.count("rate") ? parsed["rate"] : 1.0;
        return generate_exponential(n, rate);
    } else {
        // Default to uniform
        return generate_uniform(n, 0.0, 1.0);
    }
}

std::vector<int64_t> PolarsGenerator::series_i64(size_t n, int64_t low, int64_t high) {
    std::vector<int64_t> result(n);
    int64_t range = high - low;
    
    for (size_t i = 0; i < n; ++i) {
        result[i] = low + static_cast<int64_t>(m_rng.random() * range);
        if (result[i] >= high) result[i] = high - 1;
    }
    
    return result;
}

std::vector<bool> PolarsGenerator::series_bool(size_t n, double p) {
    std::vector<bool> result(n);
    
    for (size_t i = 0; i < n; ++i) {
        result[i] = (m_rng.random() < p);
    }
    
    return result;
}

std::vector<size_t> PolarsGenerator::shuffle_indices(size_t n) {
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    
    // Fisher-Yates shuffle
    for (size_t i = n - 1; i > 0; --i) {
        size_t j = static_cast<size_t>(m_rng.random() * (i + 1));
        if (j > i) j = i;
        std::swap(indices[i], indices[j]);
    }
    
    return indices;
}

std::vector<std::vector<double>> PolarsGenerator::dataframe_f64(size_t rows, size_t columns,
                                                                  const std::string& distribution) {
    std::vector<std::vector<double>> result(rows, std::vector<double>(columns));
    
    for (size_t j = 0; j < columns; ++j) {
        auto col_data = series_f64(rows, distribution);
        for (size_t i = 0; i < rows; ++i) {
            result[i][j] = col_data[i];
        }
    }
    
    return result;
}

std::vector<std::vector<int64_t>> PolarsGenerator::dataframe_i64(size_t rows, size_t columns,
                                                                   int64_t low, int64_t high) {
    std::vector<std::vector<int64_t>> result(rows, std::vector<int64_t>(columns));
    
    for (size_t j = 0; j < columns; ++j) {
        auto col_data = series_i64(rows, low, high);
        for (size_t i = 0; i < rows; ++i) {
            result[i][j] = col_data[i];
        }
    }
    
    return result;
}

}  // namespace integrations
}  // namespace aleam