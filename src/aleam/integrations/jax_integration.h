/**
 * @file jax_integration.h
 * @brief JAX integration for Aleam true randomness
 * @license MIT
 * 
 * Provides true random key generation for JAX's functional PRNG system.
 * JAX uses a pure functional approach with explicit PRNG keys.
 * This class provides true random keys seeded from system entropy.
 * 
 * Example:
 * @code
 *   #include "aleam/integrations/jax_integration.h"
 *   
 *   aleam::AleamCore rng;
 *   aleam::integrations::JAXGenerator gen(rng);
 *   uint64_t key = gen.key();
 *   // Use key with JAX's random functions
 * @endcode
 */

#ifndef ALEAM_INTEGRATIONS_JAX_INTEGRATION_H
#define ALEAM_INTEGRATIONS_JAX_INTEGRATION_H

#include <vector>
#include <cstdint>
#include "../core/aleam_core.h"

namespace aleam {
namespace integrations {

/**
 * @brief JAX-compatible random generator using true randomness
 * 
 * Provides true random keys for JAX's functional PRNG system.
 * Each key is a 64-bit value derived from true system entropy.
 */
class JAXGenerator {
public:
    /**
     * @brief Construct a JAXGenerator
     * 
     * @param rng Reference to AleamCore instance
     */
    explicit JAXGenerator(AleamCore& rng);
    
    /**
     * @brief Generate a true random JAX key (64-bit)
     * 
     * Returns a 64-bit value that can be used as a JAX PRNG key.
     * 
     * @return uint64_t True random key
     */
    uint64_t key();
    
    /**
     * @brief Generate multiple true random JAX keys
     * 
     * @param count Number of keys to generate
     * @return std::vector<uint64_t> List of true random keys
     */
    std::vector<uint64_t> keys(size_t count);
    
    /**
     * @brief Generate a key pair (for split operations)
     * 
     * Returns two independent keys derived from a single entropy source.
     * 
     * @return std::pair<uint64_t, uint64_t> Two independent keys
     */
    std::pair<uint64_t, uint64_t> key_pair();
    
    /**
     * @brief Split a key into two new keys (JAX-style)
     * 
     * @param parent_key Parent key to split
     * @return std::pair<uint64_t, uint64_t> Two child keys
     */
    std::pair<uint64_t, uint64_t> split(uint64_t parent_key);
    
    /**
     * @brief Fold a key with an integer (for generating sub-keys)
     * 
     * @param base_key Base key
     * @param fold_value Integer value to fold in
     * @return uint64_t New key
     */
    uint64_t fold_in(uint64_t base_key, uint64_t fold_value);
    
    /**
     * @brief Generate standard normal random values (for use with JAX)
     * 
     * @param count Number of values
     * @param key JAX PRNG key
     * @return std::vector<double> Standard normal values
     */
    std::vector<double> normal(size_t count, uint64_t key);
    
    /**
     * @brief Generate uniform random values (for use with JAX)
     * 
     * @param count Number of values
     * @param key JAX PRNG key
     * @param minval Minimum value
     * @param maxval Maximum value
     * @return std::vector<double> Uniform values
     */
    std::vector<double> uniform(size_t count, uint64_t key,
                                 double minval = 0.0, double maxval = 1.0);
    
    /**
     * @brief Get the counter (number of keys generated)
     * 
     * @return uint64_t Key counter
     */
    uint64_t counter() const { return m_counter; }
    
    /**
     * @brief Reset the key counter
     */
    void reset_counter() { m_counter = 0; }
    
private:
    /**
     * @brief Mix a 64-bit value using SplitMix64
     * 
     * @param x Input value
     * @return uint64_t Mixed value
     */
    uint64_t splitmix64(uint64_t x);
    
    /**
     * @brief Mix using golden ratio multiplication
     * 
     * @param x Input value
     * @return uint64_t Mixed value
     */
    uint64_t golden_mix(uint64_t x);
    
    AleamCore& m_rng;        /**< Random number generator */
    uint64_t m_counter;      /**< Key counter for uniqueness */
};

}  // namespace integrations
}  // namespace aleam

#endif /* ALEAM_INTEGRATIONS_JAX_INTEGRATION_H */