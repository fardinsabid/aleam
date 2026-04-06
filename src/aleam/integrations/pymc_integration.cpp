/**
 * @file pymc_integration.cpp
 * @brief PyMC integration implementation for Aleam
 * @license MIT
 */

#include "pymc_integration.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace aleam {
namespace integrations {

PyMCGenerator::PyMCGenerator(AleamCore& rng)
    : m_rng(rng) {
}

double PyMCGenerator::standard_normal() {
    double u1 = m_rng.random();
    double u2 = m_rng.random();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

double PyMCGenerator::sample_gamma(double shape, double scale) {
    if (shape < 1.0) {
        // Johnk's method for shape < 1
        while (true) {
            double u = m_rng.random();
            double v = m_rng.random();
            double x = std::pow(u, 1.0 / shape);
            double y = std::pow(v, 1.0 / (1.0 - shape));
            if (x + y <= 1.0) {
                double z = x / (x + y);
                double w = sample_gamma(shape + 1.0, 1.0);
                return w * z * scale;
            }
        }
    }
    
    // Marsaglia & Tsang method for shape >= 1
    double d = shape - 1.0 / 3.0;
    double c = 1.0 / std::sqrt(9.0 * d);
    
    while (true) {
        double v = standard_normal();
        double x = (1.0 + c * v);
        if (x <= 0.0) continue;
        x = x * x * x;
        double u = m_rng.random();
        if (u < 1.0 - 0.0331 * (v * v * v * v)) {
            return d * x * scale;
        }
        if (std::log(u) < 0.5 * (v * v) + d * (1.0 - x + std::log(x))) {
            return d * x * scale;
        }
    }
}

int64_t PyMCGenerator::sample_poisson(double lambda) {
    if (lambda < 10.0) {
        // Knuth's algorithm
        double L = std::exp(-lambda);
        int64_t k = 0;
        double p = 1.0;
        while (p > L) {
            p *= m_rng.random();
            ++k;
        }
        return k - 1;
    } else {
        // Normal approximation
        double x = lambda + std::sqrt(lambda) * standard_normal();
        return static_cast<int64_t>(std::max(0.0, std::round(x)));
    }
}

std::vector<double> PyMCGenerator::normal(size_t n, double mu, double sigma) {
    std::vector<double> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = mu + sigma * standard_normal();
    }
    return result;
}

std::vector<double> PyMCGenerator::uniform(size_t n, double lower, double upper) {
    std::vector<double> result(n);
    double range = upper - lower;
    for (size_t i = 0; i < n; ++i) {
        result[i] = lower + m_rng.random() * range;
    }
    return result;
}

std::vector<double> PyMCGenerator::exponential(size_t n, double rate) {
    std::vector<double> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = -std::log(1.0 - m_rng.random()) / rate;
    }
    return result;
}

std::vector<double> PyMCGenerator::gamma(size_t n, double alpha, double beta) {
    std::vector<double> result(n);
    double scale = 1.0 / beta;
    for (size_t i = 0; i < n; ++i) {
        result[i] = sample_gamma(alpha, scale);
    }
    return result;
}

std::vector<double> PyMCGenerator::beta(size_t n, double alpha, double beta) {
    std::vector<double> result(n);
    for (size_t i = 0; i < n; ++i) {
        double x = sample_gamma(alpha, 1.0);
        double y = sample_gamma(beta, 1.0);
        result[i] = x / (x + y);
    }
    return result;
}

std::vector<int64_t> PyMCGenerator::poisson(size_t n, double mu) {
    std::vector<int64_t> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = sample_poisson(mu);
    }
    return result;
}

std::vector<int64_t> PyMCGenerator::binomial(size_t n, int64_t trials, double p) {
    std::vector<int64_t> result(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t successes = 0;
        for (int64_t j = 0; j < trials; ++j) {
            if (m_rng.random() < p) {
                ++successes;
            }
        }
        result[i] = successes;
    }
    return result;
}

std::vector<int64_t> PyMCGenerator::bernoulli(size_t n, double p) {
    std::vector<int64_t> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = (m_rng.random() < p) ? 1 : 0;
    }
    return result;
}

std::vector<size_t> PyMCGenerator::categorical(size_t n, const std::vector<double>& probabilities) {
    std::vector<size_t> result(n);
    size_t k = probabilities.size();
    
    // Build cumulative distribution
    std::vector<double> cumsum(k);
    double total = 0.0;
    for (size_t i = 0; i < k; ++i) {
        total += probabilities[i];
        cumsum[i] = total;
    }
    // Normalize
    for (auto& c : cumsum) {
        c /= total;
    }
    
    for (size_t i = 0; i < n; ++i) {
        double u = m_rng.random();
        size_t idx = 0;
        while (idx < k && u > cumsum[idx]) {
            ++idx;
        }
        if (idx >= k) idx = k - 1;
        result[i] = idx;
    }
    
    return result;
}

std::vector<std::vector<double>> PyMCGenerator::dirichlet(size_t n, const std::vector<double>& alpha) {
    std::vector<std::vector<double>> result(n, std::vector<double>(alpha.size()));
    
    for (size_t i = 0; i < n; ++i) {
        double total = 0.0;
        for (size_t j = 0; j < alpha.size(); ++j) {
            result[i][j] = sample_gamma(alpha[j], 1.0);
            total += result[i][j];
        }
        for (size_t j = 0; j < alpha.size(); ++j) {
            result[i][j] /= total;
        }
    }
    
    return result;
}

std::vector<double> PyMCGenerator::wald(size_t n, double mu, double lam) {
    std::vector<double> result(n);
    
    for (size_t i = 0; i < n; ++i) {
        double nu = standard_normal();
        double y = nu * nu;
        double x = mu + (mu * mu * y) / (2.0 * lam) - 
                   (mu / (2.0 * lam)) * std::sqrt(4.0 * mu * lam * y + mu * mu * y * y);
        
        double u = m_rng.random();
        if (u <= mu / (mu + x)) {
            result[i] = x;
        } else {
            result[i] = mu * mu / x;
        }
    }
    
    return result;
}

}  // namespace integrations
}  // namespace aleam