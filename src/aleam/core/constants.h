/**
 * @file constants.h
 * @brief Mathematical constants used in Aleam core algorithm
 * @license MIT
 * 
 * This file defines the constants used in Aleam's core random number
 * generation equation:
 * 
 *     Ψ(t) = BLAKE2s( (Φ × Ξ(t)) ⊕ τ(t) )
 * 
 * Where:
 *     Φ = Golden ratio prime (floor(2^64 / φ))
 *     φ = (1 + √5) / 2 ≈ 1.6180339887498948482
 * 
 * The golden ratio prime is a 64-bit constant with excellent
 * equidistribution properties when used as a multiplier in modular arithmetic.
 */

#ifndef ALEAM_CONSTANTS_H
#define ALEAM_CONSTANTS_H

#include <cstdint>

namespace aleam {

/* ============================================================================
 * Golden Ratio Prime
 * ============================================================================ */

/**
 * @brief Golden ratio prime constant Φ = floor(2^64 / φ)
 * 
 * Where φ = (1 + √5)/2 ≈ 1.6180339887498948482
 * 
 * This constant has the following properties:
 * 
 * 1. It is odd, so multiplication modulo 2^64 is bijective
 * 2. The sequence {Φ × k mod 2^64} is maximally equidistributed in 1D
 * 3. It is the closest 64-bit approximation to the golden ratio
 * 4. Used in the XorShift family of generators as a good multiplier
 * 
 * Value: 0x9E3779B97F4A7C15
 * 
 * This is the same constant used in:
 * - TEA/XTEA block cipher (as delta constant)
 * - SplitMix64 random number generator
 * - Various hashing algorithms
 * 
 * The hex representation spells "9E3779B9" which is 2^32 / φ for 32-bit,
 * extended to 64 bits with "7F4A7C15" for the lower half.
 */
constexpr uint64_t GOLDEN_PRIME = 0x9E3779B97F4A7C15ULL;

/**
 * @brief 32-bit golden ratio prime for compatibility
 * 
 * This is the 32-bit version: floor(2^32 / φ) = 0x9E3779B9
 * Used when working with 32-bit values or for reference.
 */
constexpr uint32_t GOLDEN_PRIME_32 = 0x9E3779B9UL;

/* ============================================================================
 * BLAKE2s Parameters for Aleam
 * ============================================================================ */

/**
 * @brief Bits of entropy consumed per random() call
 * 
 * Aleam consumes 64 bits of true entropy per random() call.
 * This guarantees at least 64 bits of cryptographic security per output.
 */
constexpr int ENTROPY_BITS_PER_CALL = 64;

/**
 * @brief Bytes of entropy consumed per random() call
 */
constexpr int ENTROPY_BYTES_PER_CALL = 8;

/* ============================================================================
 * Batch Size Constants
 * ============================================================================ */

/**
 * @brief Default batch size for cached random numbers
 * 
 * When using batch mode, Aleam generates this many random numbers
 * at once using a single entropy fetch, dramatically improving
 * performance for sequential calls.
 */
constexpr size_t DEFAULT_BATCH_SIZE = 1024;

/**
 * @brief Maximum batch size (prevents memory exhaustion)
 */
constexpr size_t MAX_BATCH_SIZE = 1048576;  /* 1 million */

/**
 * @brief Minimum batch size (prevents excessive system calls)
 */
constexpr size_t MIN_BATCH_SIZE = 64;

/* ============================================================================
 * Aleam Algorithm Identifiers
 * ============================================================================ */

/**
 * @brief Algorithm version string
 */
constexpr const char* ALGORITHM_VERSION = "Ψ(t) = BLAKE2s( (Φ × Ξ(t)) ⊕ τ(t) )";

/**
 * @brief Algorithm name and version for get_stats()
 */
constexpr const char* ALGORITHM_NAME = "Aleam Core v2.0 (C++ Optimized)";

/* ============================================================================
 * Utility Constants
 * ============================================================================ */

/**
 * @brief 2^64 as a double (used for converting hash to float)
 * 
 * random() = hash_value / TWO_64
 * This maps the 64-bit hash uniformly to [0, 1)
 */
constexpr double TWO_64 = 18446744073709551616.0;

/**
 * @brief 2^64 as an unsigned 128-bit integer
 * 
 * Used for precise arithmetic when needed.
 */
constexpr __uint128_t TWO_64_U128 = (__uint128_t)1 << 64;

/**
 * @brief Maximum value of a 64-bit unsigned integer
 */
constexpr uint64_t UINT64_MAX_VALUE = 0xFFFFFFFFFFFFFFFFULL;

}  // namespace aleam

#endif /* ALEAM_CONSTANTS_H */