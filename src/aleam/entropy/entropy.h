/**
 * @file entropy.h
 * @brief Platform-agnostic entropy source abstraction
 * @license MIT
 * 
 * This file provides a unified interface to platform-specific entropy sources.
 * It includes the appropriate platform header and defines a common API.
 * 
 * Supported platforms:
 * - Linux / Android: getrandom() system call
 * - Windows: BCryptGenRandom() API
 * - macOS / iOS / tvOS / watchOS: arc4random_buf()
 * 
 * Usage:
 *     uint64_t random_value = get_entropy_64();
 *     uint8_t buffer[32];
 *     get_entropy_bytes(buffer, sizeof(buffer));
 */

#ifndef ENTROPY_H
#define ENTROPY_H

#include <stdint.h>
#include <stddef.h>

/* ============================================================================
 * Platform Detection and Inclusion
 * ============================================================================ */

#ifdef __linux__
    /* Linux (including Android) - use getrandom() */
    #include "entropy_linux.h"
    
    #define ENTROPY_PLATFORM "Linux/Android"
    #define get_entropy_bytes(buf, len) get_entropy_bytes_linux(buf, len)
    #define get_entropy_64() get_entropy_64_linux()

#elif defined(_WIN32)
    /* Microsoft Windows - use BCryptGenRandom() */
    #include "entropy_windows.h"
    
    #define ENTROPY_PLATFORM "Windows"
    #define get_entropy_bytes(buf, len) get_entropy_bytes_windows(buf, len)
    #define get_entropy_64() get_entropy_64_windows()

#elif defined(__APPLE__)
    /* Apple platforms (macOS, iOS, etc.) - use arc4random_buf() */
    #include "entropy_darwin.h"
    
    #define ENTROPY_PLATFORM "Apple (macOS/iOS)"
    #define get_entropy_bytes(buf, len) get_entropy_bytes_darwin(buf, len)
    #define get_entropy_64() get_entropy_64_darwin()

#else
    /* Unknown or unsupported platform */
    #error "Unsupported platform. Aleam requires Linux, Windows, or macOS/iOS."
    
#endif

/* ============================================================================
 * Public Entropy API
 * ============================================================================ */

/**
 * @brief Get cryptographically secure random bytes
 * 
 * This is the main entropy function. It fills the provided buffer
 * with random bytes from the system's CSPRNG.
 * 
 * @param buf   Pointer to buffer to fill
 * @param len   Number of bytes to generate
 * 
 * @return 0 on success, -1 on error (note: Apple version always succeeds)
 */
static inline int entropy_get_bytes(void* buf, size_t len) {
    return get_entropy_bytes(buf, len);
}

/**
 * @brief Get a 64-bit cryptographically random value
 * 
 * Convenience function that returns a single 64-bit random value.
 * 
 * @return 64-bit random value
 */
static inline uint64_t entropy_get_64(void) {
    return get_entropy_64();
}

/**
 * @brief Get entropy platform name (for debugging)
 * 
 * Returns a string identifying the entropy source platform.
 * Useful for debugging and logging.
 * 
 * @return Pointer to platform name string
 */
static inline const char* entropy_get_platform_name(void) {
    return ENTROPY_PLATFORM;
}

#endif /* ENTROPY_H */