/**
 * @file aleam_core.cpp
 * @brief AleamCore class implementation - True random number generator core
 * @license MIT
 * 
 * This file implements the AleamCore class which provides true random
 * number generation using the Aleam algorithm.
 * 
 * The core algorithm:
 *     Ψ(t) = BLAKE2s( (Φ × Ξ(t)) ⊕ τ(t) )
 * 
 * Implementation notes:
 * - Uses platform-specific entropy via entropy.h
 * - Uses BLAKE2s for cryptographic hashing
 * - Supports batch generation for performance
 * - Thread-safe via thread-local instances
 */

#include "aleam_core.h"
#include "../hash/blake2s.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace aleam {

/* ============================================================================
 * Construction and Initialization
 * ============================================================================ */

AleamCore::AleamCore()
    : m_cache_pos(0)
    , m_batch_size(DEFAULT_BATCH_SIZE)
    , m_has_spare(false)
    , m_spare(0.0)
    , m_calls(0)
    , m_cache_hits(0)
    , m_cache_misses(0) {
    
    /* Pre-allocate cache vector to avoid reallocations */
    m_cache.reserve(m_batch_size);
    
    /* Initial cache refill */
    refill_cache();
}

/* ============================================================================
 * Core Random Generation
 * ============================================================================ */

uint64_t AleamCore::get_entropy() {
    return entropy_get_64();
}

uint64_t AleamCore::get_timestamp() {
    auto now = std::chrono::steady_clock::now();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    );
    return static_cast<uint64_t>(nanos.count());
}

uint64_t AleamCore::golden_mix(uint64_t entropy) {
    return entropy * GOLDEN_PRIME;
}

uint64_t AleamCore::blake2s_hash(uint64_t input) {
    return blake2s_64(input);
}

uint64_t AleamCore::generate_one() {
    /* Step 1: Get true entropy */
    uint64_t entropy = get_entropy();
    
    /* Step 2: Apply golden ratio mixing */
    uint64_t mixed = golden_mix(entropy);
    
    /* Step 3: Get nanosecond timestamp */
    uint64_t timestamp = get_timestamp();
    
    /* Step 4: XOR mixing */
    uint64_t combined = mixed ^ timestamp;
    
    /* Step 5: Hash to produce uniform output */
    return blake2s_hash(combined);
}

void AleamCore::generate_batch(uint64_t* output, size_t count) {
    if (count == 0) return;
    
    /* Fetch base entropy and timestamp once for the entire batch */
    uint64_t base_entropy = get_entropy();
    uint64_t base_timestamp = get_timestamp();
    
    /* Generate each value using index-based mixing */
    for (size_t i = 0; i < count; i++) {
        /* Mix index into entropy and timestamp for uniqueness */
        uint64_t entropy = base_entropy ^ (i * GOLDEN_PRIME);
        uint64_t timestamp = base_timestamp ^ (i * 0xBF58476D1CE4E5B9ULL);
        
        /* Golden ratio mixing */
        uint64_t mixed = golden_mix(entropy);
        
        /* XOR mixing */
        uint64_t combined = mixed ^ timestamp;
        
        /* Hash to produce output */
        output[i] = blake2s_hash(combined);
    }
}

/* ============================================================================
 * Public Random Generation Methods
 * ============================================================================ */

double AleamCore::random() {
    m_calls++;
    
    /* Check if we need to refill the cache */
    if (m_cache_pos >= m_cache.size()) {
        refill_cache();
    }
    
    /* Return cached value and advance position */
    m_cache_hits++;
    return m_cache[m_cache_pos++];
}

uint64_t AleamCore::random_uint64() {
    m_calls++;
    return generate_one();
}

void AleamCore::random_batch(double* output, size_t count) {
    if (count == 0) return;
    
    /* Generate uint64_t batch */
    std::vector<uint64_t> temp(count);
    generate_batch(temp.data(), count);
    
    /* Convert to double in [0, 1) */
    for (size_t i = 0; i < count; i++) {
        output[i] = static_cast<double>(temp[i]) / TWO_64;
    }
    
    m_calls += count;
}

void AleamCore::random_uint64_batch(uint64_t* output, size_t count) {
    if (count == 0) return;
    
    generate_batch(output, count);
    m_calls += count;
}

/* ============================================================================
 * Integer and Sequence Methods
 * ============================================================================ */

int64_t AleamCore::randint(int64_t a, int64_t b) {
    if (a > b) {
        throw std::invalid_argument("a must be <= b");
    }
    return a + static_cast<int64_t>(random() * (b - a + 1));
}

std::vector<uint8_t> AleamCore::random_bytes(size_t n) {
    std::vector<uint8_t> bytes(n);
    for (size_t i = 0; i < n; i++) {
        bytes[i] = static_cast<uint8_t>(randint(0, 255));
    }
    return bytes;
}

/* ============================================================================
 * Distribution Methods
 * ============================================================================ */

double AleamCore::uniform(double low, double high) {
    return low + random() * (high - low);
}

double AleamCore::box_muller() {
    if (m_has_spare) {
        m_has_spare = false;
        return m_spare;
    }
    
    double u1 = random();
    double u2 = random();
    double r = std::sqrt(-2.0 * std::log(u1));
    double theta = 2.0 * M_PI * u2;
    
    m_spare = r * std::sin(theta);
    m_has_spare = true;
    
    return r * std::cos(theta);
}

double AleamCore::gauss(double mu, double sigma) {
    if (sigma <= 0.0) {
        throw std::invalid_argument("sigma must be > 0");
    }
    return mu + sigma * box_muller();
}

double AleamCore::exponential(double rate) {
    if (rate <= 0.0) {
        throw std::invalid_argument("rate must be > 0");
    }
    return -std::log(1.0 - random()) / rate;
}

double AleamCore::beta(double alpha, double beta) {
    if (alpha <= 0.0 || beta <= 0.0) {
        throw std::invalid_argument("alpha and beta must be > 0");
    }
    double x = gamma(alpha, 1.0);
    double y = gamma(beta, 1.0);
    return x / (x + y);
}

double AleamCore::gamma(double shape, double scale) {
    if (shape <= 0.0 || scale <= 0.0) {
        throw std::invalid_argument("shape and scale must be > 0");
    }
    
    // For shape < 1, use Johnk's method
    if (shape < 1.0) {
        while (true) {
            double u = random();
            double v = random();
            double x = std::pow(u, 1.0 / shape);
            double y = std::pow(v, 1.0 / (1.0 - shape));
            if (x + y <= 1.0) {
                double z = x / (x + y);
                double w = gamma(shape + 1.0, 1.0);
                return w * z * scale;
            }
        }
    }
    
    // Marsaglia & Tsang method for shape >= 1
    double d = shape - 1.0 / 3.0;
    double c = 1.0 / std::sqrt(9.0 * d);
    
    while (true) {
        double v = gauss(0.0, 1.0);
        double x = (1.0 + c * v);
        
        if (x <= 0.0) continue;
        
        x = x * x * x;
        double u = random();
        
        if (u < 1.0 - 0.0331 * (v * v * v * v)) {
            return d * x * scale;
        }
        
        if (std::log(u) < 0.5 * (v * v) + d * (1.0 - x + std::log(x))) {
            return d * x * scale;
        }
    }
}

int AleamCore::poisson(double lam) {
    if (lam <= 0.0) {
        throw std::invalid_argument("lam must be > 0");
    }
    
    if (lam < 10.0) {
        double L = std::exp(-lam);
        int k = 0;
        double p = 1.0;
        while (p > L) {
            p *= random();
            k++;
        }
        return k - 1;
    } else {
        double x = gauss(lam, std::sqrt(lam));
        return static_cast<int>(std::max(0.0, std::round(x)));
    }
}

double AleamCore::laplace(double loc, double scale) {
    if (scale <= 0.0) {
        throw std::invalid_argument("scale must be > 0");
    }
    double u = random() - 0.5;
    double sign = (u > 0.0) ? 1.0 : -1.0;
    return loc - scale * sign * std::log(1.0 - 2.0 * std::abs(u));
}

double AleamCore::logistic(double loc, double scale) {
    if (scale <= 0.0) {
        throw std::invalid_argument("scale must be > 0");
    }
    double u = random();
    return loc + scale * std::log(u / (1.0 - u));
}

double AleamCore::lognormal(double mu, double sigma) {
    if (sigma <= 0.0) {
        throw std::invalid_argument("sigma must be > 0");
    }
    return std::exp(gauss(mu, sigma));
}

double AleamCore::weibull(double shape, double scale) {
    if (shape <= 0.0 || scale <= 0.0) {
        throw std::invalid_argument("shape and scale must be > 0");
    }
    double u = random();
    return scale * std::pow(-std::log(1.0 - u), 1.0 / shape);
}

double AleamCore::pareto(double alpha, double scale) {
    if (alpha <= 0.0 || scale <= 0.0) {
        throw std::invalid_argument("alpha and scale must be > 0");
    }
    double u = random();
    return scale / std::pow(u, 1.0 / alpha);
}

double AleamCore::chi_square(double df) {
    if (df <= 0.0) {
        throw std::invalid_argument("df must be > 0");
    }
    return gamma(df / 2.0, 2.0);
}

double AleamCore::student_t(double df) {
    if (df <= 0.0) {
        throw std::invalid_argument("df must be > 0");
    }
    double z = gauss(0.0, 1.0);
    double chi2 = chi_square(df);
    return z / std::sqrt(chi2 / df);
}

double AleamCore::f_distribution(double df1, double df2) {
    if (df1 <= 0.0 || df2 <= 0.0) {
        throw std::invalid_argument("df1 and df2 must be > 0");
    }
    double chi2_1 = chi_square(df1);
    double chi2_2 = chi_square(df2);
    return (chi2_1 / df1) / (chi2_2 / df2);
}

std::vector<double> AleamCore::dirichlet(const std::vector<double>& alpha) {
    if (alpha.empty()) {
        throw std::invalid_argument("alpha must not be empty");
    }
    
    std::vector<double> samples(alpha.size());
    double sum = 0.0;
    
    for (size_t i = 0; i < alpha.size(); i++) {
        if (alpha[i] <= 0.0) {
            throw std::invalid_argument("all alpha values must be > 0");
        }
        samples[i] = gamma(alpha[i], 1.0);
        sum += samples[i];
    }
    
    for (double& s : samples) {
        s /= sum;
    }
    
    return samples;
}

/* ============================================================================
 * Batch Cache Management
 * ============================================================================ */

void AleamCore::refill_cache() {
    /* Resize cache to batch size */
    m_cache.resize(m_batch_size);
    
    /* Generate new batch directly into cache */
    random_batch(m_cache.data(), m_batch_size);
    
    /* Reset position and increment miss counter */
    m_cache_pos = 0;
    m_cache_misses++;
}

void AleamCore::set_batch_size(size_t size) {
    /* Clamp to valid range */
    if (size < MIN_BATCH_SIZE) {
        size = MIN_BATCH_SIZE;
    } else if (size > MAX_BATCH_SIZE) {
        size = MAX_BATCH_SIZE;
    }
    
    /* Update batch size */
    m_batch_size = size;
    
    /* Clear and refill cache with new size */
    m_cache.clear();
    m_cache.reserve(m_batch_size);
    m_cache_pos = 0;
    refill_cache();
}

size_t AleamCore::get_batch_size() const {
    return m_batch_size;
}

void AleamCore::clear_cache() {
    m_cache.clear();
    m_cache_pos = 0;
    refill_cache();
}

/* ============================================================================
 * Statistics and Information
 * ============================================================================ */

AleamCore::Stats AleamCore::get_stats() const {
    Stats stats;
    stats.calls = m_calls;
    stats.batch_size = m_batch_size;
    stats.cache_hits = m_cache_hits;
    stats.cache_misses = m_cache_misses;
    stats.algorithm = ALGORITHM_NAME;
    stats.entropy_source = entropy_get_platform_name();
    stats.entropy_bits_per_call = ENTROPY_BITS_PER_CALL;
    return stats;
}

void AleamCore::reset_stats() {
    m_calls = 0;
    m_cache_hits = 0;
    m_cache_misses = 0;
}

/* ============================================================================
 * Thread-Local Instance
 * ============================================================================ */

AleamCore& get_thread_local_instance() {
    thread_local AleamCore instance;
    return instance;
}

}  // namespace aleam