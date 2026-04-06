/**
 * @file blake2s_config.h
 * @brief BLAKE2s configuration - platform detection and algorithm parameters
 * @license Public Domain
 * 
 * This file configures BLAKE2s hash function parameters and platform-specific
 * optimizations for the Aleam true random number generator.
 */

#ifndef BLAKE2S_CONFIG_H
#define BLAKE2S_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * BLAKE2s Algorithm Constants
 * ============================================================================ */

/**
 * @def BLAKE2S_BLOCKBYTES
 * @brief Size of BLAKE2s block in bytes
 * 
 * BLAKE2s processes data in 64-byte (512-bit) blocks.
 * This matches the SHA-256 block size for compatibility.
 */
#define BLAKE2S_BLOCKBYTES    64

/**
 * @def BLAKE2S_OUTBYTES
 * @brief Default output size of BLAKE2s in bytes
 * 
 * Standard BLAKE2s produces 32 bytes (256 bits) of output.
 * Aleam uses only the first 8 bytes for 64-bit output.
 */
#define BLAKE2S_OUTBYTES      32

/**
 * @def BLAKE2S_KEYBYTES
 * @brief Maximum key size for keyed hashing in bytes
 * 
 * BLAKE2s supports keys up to 32 bytes for HMAC-like functionality.
 * Aleam does not use keyed hashing.
 */
#define BLAKE2S_KEYBYTES      32

/**
 * @def BLAKE2S_SALTBYTES
 * @brief Salt size in bytes
 * 
 * Optional salt for randomized hashing (not used by Aleam).
 */
#define BLAKE2S_SALTBYTES     8

/**
 * @def BLAKE2S_PERSONALBYTES
 * @brief Personalization string size in bytes
 * 
 * Optional personalization string (not used by Aleam).
 */
#define BLAKE2S_PERSONALBYTES 8

/* ============================================================================
 * Platform Detection for 64-bit Optimization
 * ============================================================================ */

/**
 * @def BLAKE2S_USE_64BIT
 * @brief Enable 64-bit optimization on supported platforms
 * 
 * When enabled, uses 64-bit registers for faster mixing operations.
 * Disabled on 32-bit platforms where 64-bit ops are emulated.
 */
#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)
    #define BLAKE2S_USE_64BIT 1
#else
    #define BLAKE2S_USE_64BIT 0
#endif

/* ============================================================================
 * Inline Function Macros for Performance
 * ============================================================================ */

/**
 * @def BLAKE2S_INLINE
 * @brief Force inline for small hash functions
 * 
 * Different compilers use different inline syntax.
 * This macro unifies them for maximum performance.
 */
#if defined(_MSC_VER)
    /* Microsoft Visual Studio */
    #define BLAKE2S_INLINE __forceinline static
#elif defined(__GNUC__) || defined(__clang__)
    /* GCC, Clang, and compatible */
    #define BLAKE2S_INLINE static inline __attribute__((always_inline))
#else
    /* Fallback to standard inline */
    #define BLAKE2S_INLINE static inline
#endif

/* ============================================================================
 * Rotate Right Macro
 * ============================================================================ */

/**
 * @def BLAKE2S_ROTR32
 * @brief Rotate 32-bit value right by n bits
 * 
 * BLAKE2s uses 32-bit rotations. This macro uses compiler intrinsics
 * when available for better performance.
 */
#if defined(_MSC_VER)
    /* Microsoft Visual Studio has _rotr intrinsic */
    #include <stdlib.h>
    #define BLAKE2S_ROTR32(x, n) _rotr(x, n)
#else
    /* Generic rotation for other compilers */
    #define BLAKE2S_ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#endif

#ifdef __cplusplus
}
#endif

#endif /* BLAKE2S_CONFIG_H */