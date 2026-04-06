/**
 * @file aleam_core.h
 * @brief AleamCore class - True random number generator core
 * @license MIT
 * 
 * This file defines the AleamCore class which implements the core
 * true random number generation algorithm:
 * 
 *     Ψ(t) = BLAKE2s( (Φ × Ξ(t)) ⊕ τ(t) )
 * 
 * Where:
 *     Φ = Golden ratio prime (0x9E3779B97F4A7C15)
 *     Ξ(t) = 64-bit true entropy from system CSPRNG
 *     τ(t) = Nanosecond timestamp
 *     ⊕ = XOR operation
 *     BLAKE2s = Cryptographic hash function
 * 
 * Properties:
 *     - Non-recursive: Each call is independent
 *     - Stateless: No internal state between calls (except optional cache)
 *     - Cryptographically secure: Uses system entropy + BLAKE2s
 *     - Thread-safe: When using per-thread batch caching
 */

#ifndef ALEAM_CORE_H
#define ALEAM_CORE_H

#include <cstdint>
#include <vector>
#include <chrono>
#include "constants.h"
#include "../entropy/entropy.h"

namespace aleam {

/* ============================================================================
 * AleamCore Class Declaration
 * ============================================================================ */

/**
 * @brief Core true random number generator
 * 
 * This class implements the Aleam algorithm. It can be used in two modes:
 * 
 * 1. Direct mode: Each call fetches fresh entropy (slower but guaranteed)
 * 2. Batch mode: Uses cached values from a batch (faster for sequential calls)
 * 
 * The batch cache is thread-local when used with get_thread_local_instance(),
 * making it safe for concurrent use without locks.
 */
class AleamCore {
public:
    /* ========================================================================
     * Construction and Destruction
     * ======================================================================== */
    
    /**
     * @brief Construct a new Aleam Core object
     * 
     * Initializes the generator with default batch size of 1024.
     * No other initialization is needed as the generator is stateless.
     */
    AleamCore();
    
    /**
     * @brief Destroy the Aleam Core object
     */
    ~AleamCore() = default;
    
    /* Disable copy and move (no need to copy random generators) */
    AleamCore(const AleamCore&) = delete;
    AleamCore& operator=(const AleamCore&) = delete;
    AleamCore(AleamCore&&) = delete;
    AleamCore& operator=(AleamCore&&) = delete;
    
    /* ========================================================================
     * Core Random Generation Methods
     * ======================================================================== */
    
    /**
     * @brief Generate a true random double in [0, 1)
     * 
     * This method implements the full Aleam algorithm:
     * 1. Fetch 64 bits of true entropy from the system
     * 2. Get nanosecond timestamp
     * 3. Apply golden ratio multiplication
     * 4. XOR with timestamp
     * 5. Hash with BLAKE2s
     * 6. Convert to double in [0, 1)
     * 
     * @return double Random number in [0, 1)
     */
    double random();
    
    /**
     * @brief Generate a true random 64-bit unsigned integer
     * 
     * Same as random() but returns the raw 64-bit hash value
     * instead of converting to double.
     * 
     * @return uint64_t Random 64-bit value in [0, 2^64)
     */
    uint64_t random_uint64();
    
    /**
     * @brief Generate a batch of random doubles efficiently
     * 
     * This method generates a large batch of random numbers using
     * only 2 system calls (entropy + timestamp) for the entire batch.
     * 
     * @param output Pointer to output buffer (must be pre-allocated)
     * @param count Number of random numbers to generate
     */
    void random_batch(double* output, size_t count);
    
    /**
     * @brief Generate a batch of random 64-bit integers efficiently
     * 
     * @param output Pointer to output buffer (must be pre-allocated)
     * @param count Number of random numbers to generate
     */
    void random_uint64_batch(uint64_t* output, size_t count);
    
    /* ========================================================================
     * Integer and Sequence Methods
     * ======================================================================== */
    
    /**
     * @brief Generate a random integer in [a, b] inclusive
     * 
     * @param a Lower bound (inclusive)
     * @param b Upper bound (inclusive)
     * @return int64_t Random integer
     */
    int64_t randint(int64_t a, int64_t b);
    
    /**
     * @brief Generate random bytes
     * 
     * @param n Number of bytes to generate
     * @return std::vector<uint8_t> Random bytes
     */
    std::vector<uint8_t> random_bytes(size_t n);
    
    /* ========================================================================
     * Distribution Methods
     * ======================================================================== */
    
    /**
     * @brief Generate a random double in [low, high]
     */
    double uniform(double low, double high);
    
    /**
     * @brief Generate a normally distributed random number
     */
    double gauss(double mu = 0.0, double sigma = 1.0);
    
    /**
     * @brief Generate an exponentially distributed random number
     */
    double exponential(double rate = 1.0);
    
    /**
     * @brief Generate a Beta distributed random number
     */
    double beta(double alpha, double beta);
    
    /**
     * @brief Generate a Gamma distributed random number
     */
    double gamma(double shape, double scale = 1.0);
    
    /**
     * @brief Generate a Poisson distributed random integer
     */
    int poisson(double lam = 1.0);
    
    /**
     * @brief Generate a Laplace distributed random number
     */
    double laplace(double loc = 0.0, double scale = 1.0);
    
    /**
     * @brief Generate a Logistic distributed random number
     */
    double logistic(double loc = 0.0, double scale = 1.0);
    
    /**
     * @brief Generate a Log-Normal distributed random number
     */
    double lognormal(double mu = 0.0, double sigma = 1.0);
    
    /**
     * @brief Generate a Weibull distributed random number
     */
    double weibull(double shape, double scale = 1.0);
    
    /**
     * @brief Generate a Pareto distributed random number
     */
    double pareto(double alpha, double scale = 1.0);
    
    /**
     * @brief Generate a Chi-square distributed random number
     */
    double chi_square(double df);
    
    /**
     * @brief Generate a Student's t distributed random number
     */
    double student_t(double df);
    
    /**
     * @brief Generate an F-distributed random number
     */
    double f_distribution(double df1, double df2);
    
    /**
     * @brief Generate a Dirichlet distributed probability vector
     */
    std::vector<double> dirichlet(const std::vector<double>& alpha);
    
    /* ========================================================================
     * Batch Cache Management
     * ======================================================================== */
    
    /**
     * @brief Set the batch size for cached random numbers
     * 
     * When using the cached random() method, this determines how many
     * numbers are generated at once. Larger batches = fewer system calls
     * but more memory usage.
     * 
     * @param size New batch size (between MIN_BATCH_SIZE and MAX_BATCH_SIZE)
     */
    void set_batch_size(size_t size);
    
    /**
     * @brief Get the current batch size
     * 
     * @return size_t Current batch size
     */
    size_t get_batch_size() const;
    
    /**
     * @brief Clear the batch cache and force fresh entropy on next call
     * 
     * Useful when you need guaranteed fresh entropy immediately.
     */
    void clear_cache();
    
    /* ========================================================================
     * Statistics and Information
     * ======================================================================== */
    
    /**
     * @brief Get generator statistics
     * 
     * Returns a struct containing information about the generator.
     * 
     * @return struct Stats Statistics structure
     */
    struct Stats {
        uint64_t calls;              /**< Number of random() calls */
        size_t batch_size;           /**< Current batch size */
        size_t cache_hits;           /**< Number of cache hits */
        size_t cache_misses;         /**< Number of cache misses (refills) */
        const char* algorithm;       /**< Algorithm description */
        const char* entropy_source;  /**< Platform entropy source name */
        int entropy_bits_per_call;   /**< Bits of entropy per call */
    };
    
    Stats get_stats() const;
    
    /**
     * @brief Reset statistics counters
     */
    void reset_stats();
    
private:
    /* ========================================================================
     * Core Algorithm Steps
     * ======================================================================== */
    
    /**
     * @brief Get 64 bits of true entropy from system
     * 
     * @return uint64_t Random 64-bit value from system CSPRNG
     */
    uint64_t get_entropy();
    
    /**
     * @brief Get nanosecond timestamp as 64-bit value
     * 
     * @return uint64_t Nanosecond timestamp since epoch
     */
    uint64_t get_timestamp();
    
    /**
     * @brief Apply golden ratio multiplication
     * 
     * @param entropy Input entropy value
     * @return uint64_t Mixed value
     */
    uint64_t golden_mix(uint64_t entropy);
    
    /**
     * @brief Apply BLAKE2s hash to produce 64-bit output
     * 
     * @param input 64-bit input value
     * @return uint64_t 64-bit hash output
     */
    uint64_t blake2s_hash(uint64_t input);
    
    /**
     * @brief Generate a single random uint64_t using the core algorithm
     * 
     * @return uint64_t Random value
     */
    uint64_t generate_one();
    
    /**
     * @brief Generate a batch of random uint64_t values
     * 
     * @param output Output buffer
     * @param count Number of values
     */
    void generate_batch(uint64_t* output, size_t count);
    
    /**
     * @brief Refill the batch cache
     */
    void refill_cache();
    
    /**
     * @brief Box-Muller transform for Gaussian distribution
     * 
     * @return double Standard normal random value
     */
    double box_muller();
    
    /* ========================================================================
     * Member Variables
     * ======================================================================== */
    
    /* Batch cache for faster sequential calls */
    std::vector<double> m_cache;        /**< Cached random doubles */
    size_t m_cache_pos;                 /**< Current position in cache */
    size_t m_batch_size;                /**< Size of batch to generate */
    
    /* Box-Muller cache */
    bool m_has_spare;                   /**< Whether a spare normal value is cached */
    double m_spare;                     /**< Cached spare normal value */
    
    /* Statistics */
    uint64_t m_calls;                   /**< Total random() calls */
    size_t m_cache_hits;                /**< Cache hits */
    size_t m_cache_misses;              /**< Cache misses (refills) */
};

/* ============================================================================
 * Thread-Local Instance Helper
 * ============================================================================ */

/**
 * @brief Get thread-local AleamCore instance
 * 
 * Returns a reference to a thread-local AleamCore instance with its
 * own batch cache. This is the recommended way to use AleamCore in
 * multi-threaded applications.
 * 
 * @return AleamCore& Thread-local instance
 */
AleamCore& get_thread_local_instance();

}  // namespace aleam

#endif /* ALEAM_CORE_H */