/**
 * @file jax_integration.cpp
 * @brief JAX integration implementation for Aleam
 * @license MIT
 */

#include "jax_integration.h"
#include <cmath>
#include <algorithm>

namespace aleam {
namespace integrations {

// Golden ratio prime constant
constexpr uint64_t GOLDEN_PRIME = 0x9E3779B97F4A7C15ULL;

JAXGenerator::JAXGenerator(AleamCore& rng)
    : m_rng(rng)
    , m_counter(0) {
}

uint64_t JAXGenerator::splitmix64(uint64_t x) {
    x = x + GOLDEN_PRIME;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = x ^ (x >> 31);
    return x;
}

uint64_t JAXGenerator::golden_mix(uint64_t x) {
    return x * GOLDEN_PRIME;
}

uint64_t JAXGenerator::key() {
    // Get true entropy from system
    uint64_t entropy = m_rng.random_uint64();
    
    // Mix with counter for uniqueness
    m_counter++;
    uint64_t result = splitmix64(entropy ^ (m_counter * GOLDEN_PRIME));
    
    return result;
}

std::vector<uint64_t> JAXGenerator::keys(size_t count) {
    std::vector<uint64_t> result;
    result.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        result.push_back(key());
    }
    
    return result;
}

std::pair<uint64_t, uint64_t> JAXGenerator::key_pair() {
    uint64_t base = key();
    uint64_t k1 = splitmix64(base);
    uint64_t k2 = splitmix64(base ^ GOLDEN_PRIME);
    return {k1, k2};
}

std::pair<uint64_t, uint64_t> JAXGenerator::split(uint64_t parent_key) {
    uint64_t k1 = splitmix64(parent_key);
    uint64_t k2 = splitmix64(parent_key ^ GOLDEN_PRIME);
    return {k1, k2};
}

uint64_t JAXGenerator::fold_in(uint64_t base_key, uint64_t fold_value) {
    return splitmix64(base_key ^ (fold_value * GOLDEN_PRIME));
}

std::vector<double> JAXGenerator::normal(size_t count, uint64_t key) {
    std::vector<double> result(count);
    uint64_t state = key;
    
    for (size_t i = 0; i < count; ++i) {
        // Generate two uniform values using splitmix64
        state = splitmix64(state);
        double u1 = static_cast<double>(state) / 18446744073709551616.0;
        
        state = splitmix64(state);
        double u2 = static_cast<double>(state) / 18446744073709551616.0;
        
        // Box-Muller transform
        double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        result[i] = z;
    }
    
    return result;
}

std::vector<double> JAXGenerator::uniform(size_t count, uint64_t key,
                                           double minval, double maxval) {
    std::vector<double> result(count);
    uint64_t state = key;
    double range = maxval - minval;
    
    for (size_t i = 0; i < count; ++i) {
        state = splitmix64(state);
        double u = static_cast<double>(state) / 18446744073709551616.0;
        result[i] = minval + u * range;
    }
    
    return result;
}

}  // namespace integrations
}  // namespace aleam