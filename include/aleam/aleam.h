/**
 * @file aleam.h
 * @brief Public C++ API for Aleam true random number generator
 * @license MIT
 * 
 * This is the public C++ header for using Aleam directly in C++ applications.
 * It provides the core random number generation functionality without
 * requiring Python bindings.
 * 
 * Example usage:
 * @code
 *   #include <aleam/aleam.h>
 *   
 *   aleam::AleamCore rng;
 *   double x = rng.random();
 *   uint64_t y = rng.random_uint64();
 *   
 *   // Batch generation
 *   std::vector<double> batch(1024);
 *   rng.random_batch(batch.data(), batch.size());
 * @endcode
 */

#ifndef ALEAM_ALEAM_H
#define ALEAM_ALEAM_H

/**
 * @mainpage Aleam - True Random Number Generator
 * 
 * Aleam implements the proven equation:
 * 
 *     Ψ(t) = BLAKE2s( (Φ × Ξ(t)) ⊕ τ(t) )
 * 
 * Where:
 * - Φ = Golden ratio prime (0x9E3779B97F4A7C15)
 * - Ξ(t) = True entropy from system CSPRNG
 * - τ(t) = Nanosecond timestamp
 * - ⊕ = XOR operation
 * - BLAKE2s = Cryptographic hash function
 * 
 * Features:
 * - Non-recursive: Each call independent
 * - Stateless: No internal state between calls
 * - Cryptographically secure
 * - Thread-safe (with thread-local instances)
 * - Batch generation for high performance
 */

#include <cstdint>
#include <vector>
#include <cstddef>

namespace aleam {

// Forward declaration of implementation class
class AleamCoreImpl;

/**
 * @brief AleamCore - Main random number generator class
 * 
 * This is the primary interface for generating true random numbers.
 * It is designed to be lightweight and efficient.
 */
class AleamCore {
public:
    /**
     * @brief Construct a new AleamCore object
     */
    AleamCore();
    
    /**
     * @brief Destroy the AleamCore object
     */
    ~AleamCore();
    
    // Disable copy (move is allowed)
    AleamCore(const AleamCore&) = delete;
    AleamCore& operator=(const AleamCore&) = delete;
    AleamCore(AleamCore&& other) noexcept;
    AleamCore& operator=(AleamCore&& other) noexcept;
    
    /* ========================================================================
     * Core Generation Methods
     * ======================================================================== */
    
    /**
     * @brief Generate a true random double in [0, 1)
     * 
     * @return double Random value in [0, 1)
     */
    double random();
    
    /**
     * @brief Generate a true random 64-bit unsigned integer
     * 
     * @return uint64_t Random value in [0, 2^64)
     */
    uint64_t random_uint64();
    
    /**
     * @brief Generate a batch of random doubles
     * 
     * @param output Pointer to output buffer (must be pre-allocated)
     * @param count Number of values to generate
     */
    void random_batch(double* output, size_t count);
    
    /**
     * @brief Generate a batch of random 64-bit integers
     * 
     * @param output Pointer to output buffer (must be pre-allocated)
     * @param count Number of values to generate
     */
    void random_uint64_batch(uint64_t* output, size_t count);
    
    /* ========================================================================
     * Configuration
     * ======================================================================== */
    
    /**
     * @brief Set the batch cache size
     * 
     * @param size Batch size (default 1024)
     */
    void set_batch_size(size_t size);
    
    /**
     * @brief Get the current batch cache size
     * 
     * @return size_t Current batch size
     */
    size_t get_batch_size() const;
    
    /**
     * @brief Clear the batch cache
     */
    void clear_cache();
    
    /* ========================================================================
     * Statistics
     * ======================================================================== */
    
    /**
     * @brief Get generation statistics
     */
    struct Stats {
        uint64_t calls;              /**< Total random() calls */
        size_t batch_size;           /**< Current batch size */
        size_t cache_hits;           /**< Cache hits */
        size_t cache_misses;         /**< Cache misses (refills) */
        const char* algorithm;       /**< Algorithm description */
        const char* entropy_source;  /**< Platform entropy source */
        int entropy_bits_per_call;   /**< Bits of entropy per call */
    };
    
    /**
     * @brief Get generator statistics
     * 
     * @return Stats Statistics structure
     */
    Stats get_stats() const;
    
    /**
     * @brief Reset statistics counters
     */
    void reset_stats();
    
private:
    AleamCoreImpl* m_impl;  /**< Pointer to implementation (PIMPL) */
};

/**
 * @brief Get thread-local AleamCore instance
 * 
 * Each thread gets its own instance with its own batch cache.
 * This is the recommended way to use AleamCore in multi-threaded code.
 * 
 * @return AleamCore& Reference to thread-local instance
 */
AleamCore& get_thread_local_instance();

/**
 * @brief Get Aleam version string
 * 
 * @return const char* Version string (e.g., "1.0.3")
 */
const char* get_version();

/**
 * @brief Get Aleam algorithm description
 * 
 * @return const char* Algorithm description
 */
const char* get_algorithm_description();

}  // namespace aleam

#endif /* ALEAM_ALEAM_H */