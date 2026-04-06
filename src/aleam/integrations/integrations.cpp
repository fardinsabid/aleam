/**
 * @file integrations.cpp
 * @brief Implementation of framework integrations for Aleam
 * @license MIT
 * 
 * This file implements the integration classes for various ML frameworks.
 */

#include "integrations.h"
#include "../distributions/distributions.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace aleam {
namespace integrations {

/* ============================================================================
 * BaseGenerator Implementation
 * ============================================================================ */

BaseGenerator::BaseGenerator(AleamCore* rng)
    : m_rng(rng)
    , m_owns_rng(false) {
    if (m_rng == nullptr) {
        m_rng = &get_thread_local_instance();
        m_owns_rng = false;
    }
}

AleamCore& BaseGenerator::get_rng() {
    return *m_rng;
}

double BaseGenerator::random() {
    return get_rng().random();
}

uint64_t BaseGenerator::random_uint64() {
    return get_rng().random_uint64();
}

std::vector<double> BaseGenerator::random_batch(size_t count) {
    std::vector<double> result(count);
    get_rng().random_batch(result.data(), count);
    return result;
}

uint64_t BaseGenerator::true_seed() {
    return get_rng().random_uint64();
}

std::vector<uint64_t> BaseGenerator::true_seeds(size_t count) {
    std::vector<uint64_t> result(count);
    for (size_t i = 0; i < count; i++) {
        result[i] = get_rng().random_uint64();
    }
    return result;
}

/* ============================================================================
 * TorchGenerator Implementation
 * ============================================================================ */

TorchGenerator::TorchGenerator(const std::string& device, AleamCore* rng)
    : BaseGenerator(rng)
    , m_device(device) {
}

std::vector<double> TorchGenerator::rand(size_t size) {
    return random_batch(size);
}

std::vector<double> TorchGenerator::randn(size_t size) {
    std::vector<double> result(size);
    AleamCore& rng = get_rng();
    
    for (size_t i = 0; i < size; i++) {
        result[i] = distributions::normal(rng, 0.0, 1.0);
    }
    
    return result;
}

std::vector<int64_t> TorchGenerator::randint(int64_t low, int64_t high, size_t size) {
    if (low >= high) {
        throw std::invalid_argument("low must be less than high");
    }
    
    std::vector<int64_t> result(size);
    AleamCore& rng = get_rng();
    int64_t range = high - low;
    
    for (size_t i = 0; i < size; i++) {
        result[i] = low + static_cast<int64_t>(rng.random() * range);
        if (result[i] >= high) result[i] = high - 1;
    }
    
    return result;
}

void TorchGenerator::manual_seed(int64_t /*seed*/) {
    throw std::runtime_error(
        "TorchGenerator uses true randomness and does not support seeding. "
        "Each call is independent and stateless."
    );
}

/* ============================================================================
 * TFGenerator Implementation
 * ============================================================================ */

TFGenerator::TFGenerator(AleamCore* rng)
    : BaseGenerator(rng) {
}

std::vector<double> TFGenerator::normal(size_t shape, double mean, double stddev) {
    if (stddev <= 0.0) {
        throw std::invalid_argument("stddev must be > 0");
    }
    
    std::vector<double> result(shape);
    AleamCore& rng = get_rng();
    
    for (size_t i = 0; i < shape; i++) {
        result[i] = distributions::normal(rng, mean, stddev);
    }
    
    return result;
}

std::vector<double> TFGenerator::uniform(size_t shape, double minval, double maxval) {
    if (minval >= maxval) {
        throw std::invalid_argument("minval must be less than maxval");
    }
    
    std::vector<double> result(shape);
    AleamCore& rng = get_rng();
    
    for (size_t i = 0; i < shape; i++) {
        result[i] = distributions::uniform(rng, minval, maxval);
    }
    
    return result;
}

std::vector<int64_t> TFGenerator::randint(size_t shape, int64_t minval, int64_t maxval) {
    if (minval >= maxval) {
        throw std::invalid_argument("minval must be less than maxval");
    }
    
    std::vector<int64_t> result(shape);
    AleamCore& rng = get_rng();
    int64_t range = maxval - minval;
    
    for (size_t i = 0; i < shape; i++) {
        result[i] = minval + static_cast<int64_t>(rng.random() * range);
        if (result[i] >= maxval) result[i] = maxval - 1;
    }
    
    return result;
}

std::vector<double> TFGenerator::truncated_normal(size_t shape, double mean, double stddev) {
    if (stddev <= 0.0) {
        throw std::invalid_argument("stddev must be > 0");
    }
    
    std::vector<double> result(shape);
    AleamCore& rng = get_rng();
    double lower = mean - 2.0 * stddev;
    double upper = mean + 2.0 * stddev;
    
    for (size_t i = 0; i < shape; i++) {
        double value;
        do {
            value = distributions::normal(rng, mean, stddev);
        } while (value < lower || value > upper);
        result[i] = value;
    }
    
    return result;
}

/* ============================================================================
 * JAXGenerator Implementation
 * ============================================================================ */

JAXGenerator::JAXGenerator(AleamCore* rng)
    : BaseGenerator(rng)
    , m_counter(0) {
}

uint64_t JAXGenerator::key() {
    /* Mix counter into entropy for uniqueness */
    uint64_t entropy = get_rng().random_uint64();
    m_counter++;
    return entropy ^ (m_counter * 0x9E3779B97F4A7C15ULL);
}

std::vector<uint64_t> JAXGenerator::keys(size_t count) {
    std::vector<uint64_t> result(count);
    for (size_t i = 0; i < count; i++) {
        result[i] = key();
    }
    return result;
}

std::vector<double> JAXGenerator::normal(size_t shape, double mean, double stddev) {
    if (stddev <= 0.0) {
        throw std::invalid_argument("stddev must be > 0");
    }
    
    std::vector<double> result(shape);
    AleamCore& rng = get_rng();
    
    for (size_t i = 0; i < shape; i++) {
        result[i] = distributions::normal(rng, mean, stddev);
    }
    
    return result;
}

std::vector<double> JAXGenerator::uniform(size_t shape, double minval, double maxval) {
    if (minval >= maxval) {
        throw std::invalid_argument("minval must be less than maxval");
    }
    
    std::vector<double> result(shape);
    AleamCore& rng = get_rng();
    
    for (size_t i = 0; i < shape; i++) {
        result[i] = distributions::uniform(rng, minval, maxval);
    }
    
    return result;
}

/* ============================================================================
 * CuPyGenerator Implementation
 * ============================================================================ */

CuPyGenerator::CuPyGenerator(AleamCore* rng)
    : BaseGenerator(rng) {
}

std::vector<double> CuPyGenerator::random(size_t size, const std::string& /*dtype*/) {
    return random_batch(size);
}

std::vector<double> CuPyGenerator::randn(size_t size, double mu, double sigma, const std::string& /*dtype*/) {
    std::vector<double> result(size);
    AleamCore& rng = get_rng();
    
    for (size_t i = 0; i < size; i++) {
        result[i] = distributions::normal(rng, mu, sigma);
    }
    
    return result;
}

std::vector<int64_t> CuPyGenerator::randint(size_t size, int64_t low, int64_t high) {
    if (low >= high) {
        throw std::invalid_argument("low must be less than high");
    }
    
    std::vector<int64_t> result(size);
    AleamCore& rng = get_rng();
    int64_t range = high - low;
    
    for (size_t i = 0; i < size; i++) {
        result[i] = low + static_cast<int64_t>(rng.random() * range);
        if (result[i] >= high) result[i] = high - 1;
    }
    
    return result;
}

/* ============================================================================
 * PandasGenerator Implementation
 * ============================================================================ */

PandasGenerator::PandasGenerator(AleamCore* rng)
    : BaseGenerator(rng) {
}

std::vector<double> PandasGenerator::series(size_t n, const std::string& distribution, 
                                              const std::string& params) {
    std::vector<double> result(n);
    AleamCore& rng = get_rng();
    
    /* Parse params string (format: "key=value,key=value") */
    /* For simplicity, use hardcoded defaults - Python will parse properly */
    
    if (distribution == "uniform") {
        double low = 0.0, high = 1.0;
        for (size_t i = 0; i < n; i++) {
            result[i] = distributions::uniform(rng, low, high);
        }
    } else if (distribution == "normal") {
        double mu = 0.0, sigma = 1.0;
        for (size_t i = 0; i < n; i++) {
            result[i] = distributions::normal(rng, mu, sigma);
        }
    } else if (distribution == "exponential") {
        double rate = 1.0;
        for (size_t i = 0; i < n; i++) {
            result[i] = distributions::exponential(rng, rate);
        }
    } else if (distribution == "poisson") {
        double lambda = 1.0;
        for (size_t i = 0; i < n; i++) {
            result[i] = static_cast<double>(distributions::poisson(rng, lambda));
        }
    } else {
        /* Default to uniform */
        for (size_t i = 0; i < n; i++) {
            result[i] = rng.random();
        }
    }
    
    return result;
}

std::vector<size_t> PandasGenerator::shuffle_indices(size_t n) {
    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; i++) {
        indices[i] = i;
    }
    
    /* Fisher-Yates shuffle */
    AleamCore& rng = get_rng();
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = static_cast<size_t>(rng.random() * (i + 1));
        if (j > i) j = i;
        std::swap(indices[i], indices[j]);
    }
    
    return indices;
}

/* ============================================================================
 * DaskGenerator Implementation
 * ============================================================================ */

DaskGenerator::DaskGenerator(AleamCore* rng)
    : BaseGenerator(rng) {
}

std::vector<double> DaskGenerator::block_random(size_t block_shape, const std::string& distribution) {
    std::vector<double> result(block_shape);
    AleamCore& rng = get_rng();
    
    if (distribution == "normal") {
        for (size_t i = 0; i < block_shape; i++) {
            result[i] = distributions::normal(rng, 0.0, 1.0);
        }
    } else if (distribution == "uniform") {
        for (size_t i = 0; i < block_shape; i++) {
            result[i] = rng.random();
        }
    } else {
        /* Default to uniform */
        for (size_t i = 0; i < block_shape; i++) {
            result[i] = rng.random();
        }
    }
    
    return result;
}

}  // namespace integrations
}  // namespace aleam