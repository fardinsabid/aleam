/**
 * @file pandas_integration.cpp
 * @brief Pandas integration implementation for Aleam
 * @license MIT
 */

#include "pandas_integration.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <sstream>

namespace aleam {
namespace integrations {

PandasGenerator::PandasGenerator(AleamCore& rng)
    : m_rng(rng) {
}

std::unordered_map<std::string, double> PandasGenerator::parse_params(const std::string& params) {
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

std::vector<double> PandasGenerator::generate_uniform(size_t n, double low, double high) {
    std::vector<double> result(n);
    double range = high - low;
    
    for (size_t i = 0; i < n; ++i) {
        result[i] = low + m_rng.random() * range;
    }
    
    return result;
}

std::vector<double> PandasGenerator::generate_normal(size_t n, double mu, double sigma) {
    std::vector<double> result(n);
    
    for (size_t i = 0; i < n; ++i) {
        double u1 = m_rng.random();
        double u2 = m_rng.random();
        double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        result[i] = mu + sigma * z;
    }
    
    return result;
}

std::vector<double> PandasGenerator::generate_exponential(size_t n, double rate) {
    std::vector<double> result(n);
    
    for (size_t i = 0; i < n; ++i) {
        result[i] = -std::log(1.0 - m_rng.random()) / rate;
    }
    
    return result;
}

std::vector<int64_t> PandasGenerator::generate_poisson(size_t n, double lambda) {
    std::vector<int64_t> result(n);
    
    for (size_t i = 0; i < n; ++i) {
        if (lambda < 10.0) {
            // Knuth's algorithm
            double L = std::exp(-lambda);
            int64_t k = 0;
            double p = 1.0;
            while (p > L) {
                p *= m_rng.random();
                ++k;
            }
            result[i] = k - 1;
        } else {
            // Normal approximation
            double u1 = m_rng.random();
            double u2 = m_rng.random();
            double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
            double x = lambda + std::sqrt(lambda) * z;
            result[i] = static_cast<int64_t>(std::max(0.0, std::round(x)));
        }
    }
    
    return result;
}

std::vector<int64_t> PandasGenerator::generate_binomial(size_t n, int trials, double p) {
    std::vector<int64_t> result(n);
    
    for (size_t i = 0; i < n; ++i) {
        int64_t successes = 0;
        for (int j = 0; j < trials; ++j) {
            if (m_rng.random() < p) {
                ++successes;
            }
        }
        result[i] = successes;
    }
    
    return result;
}

std::vector<double> PandasGenerator::series(size_t n, const std::string& distribution, const std::string& params) {
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

std::vector<int64_t> PandasGenerator::series_int(size_t n, int64_t low, int64_t high) {
    std::vector<int64_t> result(n);
    int64_t range = high - low;
    
    for (size_t i = 0; i < n; ++i) {
        result[i] = low + static_cast<int64_t>(m_rng.random() * range);
        if (result[i] >= high) result[i] = high - 1;
    }
    
    return result;
}

std::vector<bool> PandasGenerator::series_bool(size_t n, double p) {
    std::vector<bool> result(n);
    
    for (size_t i = 0; i < n; ++i) {
        result[i] = (m_rng.random() < p);
    }
    
    return result;
}

std::vector<size_t> PandasGenerator::series_categorical(size_t n,
                                                          const std::vector<double>& categories,
                                                          const std::vector<double>& probabilities) {
    std::vector<size_t> result(n);
    size_t num_cats = categories.size();
    
    std::vector<double> cumsum(num_cats);
    double total = 0.0;
    
    if (probabilities.empty()) {
        // Uniform probabilities
        for (size_t i = 0; i < num_cats; ++i) {
            cumsum[i] = static_cast<double>(i + 1) / num_cats;
        }
    } else {
        // Use provided probabilities
        for (size_t i = 0; i < num_cats; ++i) {
            total += probabilities[i];
            cumsum[i] = total;
        }
        // Normalize
        for (auto& c : cumsum) {
            c /= total;
        }
    }
    
    for (size_t i = 0; i < n; ++i) {
        double u = m_rng.random();
        size_t idx = 0;
        while (idx < num_cats && u > cumsum[idx]) {
            ++idx;
        }
        if (idx >= num_cats) idx = num_cats - 1;
        result[i] = idx;
    }
    
    return result;
}

std::vector<size_t> PandasGenerator::shuffle_indices(size_t n) {
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

std::vector<std::vector<double>> PandasGenerator::dataframe(size_t rows, size_t columns,
                                                              const std::string& distribution) {
    std::vector<std::vector<double>> result(rows, std::vector<double>(columns));
    
    for (size_t i = 0; i < rows; ++i) {
        auto col_data = series(columns, distribution);
        result[i] = col_data;
    }
    
    return result;
}

std::vector<std::vector<double>> PandasGenerator::dataframe_mixed(
    size_t rows,
    const std::vector<std::tuple<std::string, std::string, std::string>>& col_specs) {
    
    size_t columns = col_specs.size();
    std::vector<std::vector<double>> result(rows, std::vector<double>(columns));
    
    for (size_t j = 0; j < columns; ++j) {
        const auto& spec = col_specs[j];
        const std::string& name = std::get<0>(spec);
        const std::string& dist = std::get<1>(spec);
        const std::string& params = std::get<2>(spec);
        
        auto col_data = series(rows, dist, params);
        for (size_t i = 0; i < rows; ++i) {
            result[i][j] = col_data[i];
        }
    }
    
    return result;
}

}  // namespace integrations
}  // namespace aleam