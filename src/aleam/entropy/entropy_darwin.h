/**
 * @file entropy_darwin.h
 * @brief macOS/iOS entropy source using arc4random_buf()
 * @license MIT
 * 
 * Provides true entropy for Apple platforms (macOS, iOS, tvOS, watchOS)
 * using the arc4random_buf() function.
 * 
 * Apple platforms use a CSPRNG that is seeded from hardware entropy sources
 * including:
 * - CPU cycle counters
 * - Boot time
 * - Interrupt timing
 * - Hardware RNG on Apple Silicon (ARMv8.5-A +)
 * 
 * The arc4random_buf() function is thread-safe, cryptographically secure,
 * and available on all Apple operating systems since OS X 10.7 and iOS 4.3.
 */

#ifndef ENTROPY_DARWIN_H
#define ENTROPY_DARWIN_H

#ifdef __APPLE__

#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Darwin/macOS/iOS Entropy Functions
 * ============================================================================ */

/**
 * @brief Get entropy bytes from Apple platform using arc4random_buf()
 * 
 * This function reads cryptographically secure random bytes using Apple's
 * arc4random_buf() function. The underlying CSPRNG is reseeded regularly
 * from the kernel entropy pool.
 * 
 * arc4random_buf() never fails and never blocks. It is the recommended
 * interface for generating random numbers on Apple platforms.
 * 
 * @param buf   Pointer to buffer to fill with random bytes
 * @param len   Number of bytes to read
 * 
 * @return 0 on success (always succeeds on Apple platforms)
 */
static inline int get_entropy_bytes_darwin(void* buf, size_t len) {
    /* arc4random_buf never fails on Apple platforms */
    arc4random_buf(buf, len);
    return 0;
}

/**
 * @brief Get a single 64-bit random value from Apple platform
 * 
 * Convenience wrapper for get_entropy_bytes_darwin() that returns
 * a uint64_t directly.
 * 
 * @return 64-bit cryptographically random value
 */
static inline uint64_t get_entropy_64_darwin(void) {
    uint64_t value;
    
    /* arc4random_buf never fails, so no fallback needed */
    arc4random_buf(&value, sizeof(value));
    
    return value;
}

/**
 * @brief Get a 32-bit random value (for older compatibility)
 * 
 * Some older Apple platforms may prefer arc4random() which returns
 * a 32-bit value. This function provides that interface.
 * 
 * @return 32-bit cryptographically random value
 */
static inline uint32_t get_entropy_32_darwin(void) {
    /* arc4random returns a 32-bit unsigned int */
    return arc4random();
}

#ifdef __cplusplus
}
#endif

#endif /* __APPLE__ */

#endif /* ENTROPY_DARWIN_H */