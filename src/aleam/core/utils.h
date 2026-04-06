/**
 * @file utils.h
 * @brief Utility functions for Aleam core
 * @license MIT
 * 
 * Helper functions for common operations used throughout Aleam.
 */

#ifndef ALEAM_CORE_UTILS_H
#define ALEAM_CORE_UTILS_H

#include <cstdint>
#include <chrono>
#include <cstring>

namespace aleam {
namespace utils {

/* ============================================================================
 * Endianness Conversion (if needed)
 * ============================================================================ */

/**
 * @brief Convert 64-bit value from host to little-endian
 * 
 * Most modern systems are little-endian (x86, x64, ARM).
 * This function is a no-op on little-endian systems.
 * 
 * @param value Input value in host byte order
 * @return uint64_t Value in little-endian byte order
 */
inline uint64_t host_to_le64(uint64_t value) {
    #if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        // Big-endian system - need to swap
        return __builtin_bswap64(value);
    #else
        // Little-endian system - no change needed
        return value;
    #endif
}

/**
 * @brief Convert 32-bit value from host to little-endian
 * 
 * @param value Input value in host byte order
 * @return uint32_t Value in little-endian byte order
 */
inline uint32_t host_to_le32(uint32_t value) {
    #if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        return __builtin_bswap32(value);
    #else
        return value;
    #endif
}

/* ============================================================================
 * Timing Utilities
 * ============================================================================ */

/**
 * @brief Get high-resolution timestamp in nanoseconds
 * 
 * Uses steady_clock which is monotonic and not affected
 * by system time changes.
 * 
 * @return uint64_t Nanosecond timestamp
 */
inline uint64_t get_nanoseconds() {
    auto now = std::chrono::steady_clock::now();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    );
    return static_cast<uint64_t>(nanos.count());
}

/* ============================================================================
 * Bit Manipulation
 * ============================================================================ */

/**
 * @brief Rotate left a 64-bit value
 * 
 * @param x Value to rotate
 * @param n Number of bits to rotate (0-63)
 * @return uint64_t Rotated value
 */
inline uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

/**
 * @brief Rotate right a 64-bit value
 * 
 * @param x Value to rotate
 * @param n Number of bits to rotate (0-63)
 * @return uint64_t Rotated value
 */
inline uint64_t rotr64(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

/**
 * @brief Rotate left a 32-bit value
 * 
 * @param x Value to rotate
 * @param n Number of bits to rotate (0-31)
 * @return uint32_t Rotated value
 */
inline uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

/**
 * @brief Rotate right a 32-bit value
 * 
 * @param x Value to rotate
 * @param n Number of bits to rotate (0-31)
 * @return uint32_t Rotated value
 */
inline uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

/* ============================================================================
 * Memory Utilities
 * ============================================================================ */

/**
 * @brief Secure zero a memory region
 * 
 * Prevents compiler optimization from removing the memset.
 * 
 * @param ptr Pointer to memory region
 * @param len Size in bytes
 */
inline void secure_zero(void* ptr, size_t len) {
    volatile char* vptr = static_cast<volatile char*>(ptr);
    for (size_t i = 0; i < len; i++) {
        vptr[i] = 0;
    }
}

/**
 * @brief Copy bytes with overlap safety
 * 
 * @param dest Destination buffer
 * @param src Source buffer
 * @param len Number of bytes to copy
 */
inline void safe_memcpy(void* dest, const void* src, size_t len) {
    if (dest == src || len == 0) return;
    memmove(dest, src, len);
}

/* ============================================================================
 * Random Helpers
 * ============================================================================ */

/**
 * @brief Convert a 64-bit value to a double in [0, 1)
 * 
 * @param value 64-bit unsigned integer
 * @return double Value in [0, 1)
 */
inline double uint64_to_double(uint64_t value) {
    constexpr double TWO_64 = 18446744073709551616.0;
    return static_cast<double>(value) / TWO_64;
}

/**
 * @brief Convert a 64-bit value to a float in [0, 1)
 * 
 * @param value 64-bit unsigned integer
 * @return float Value in [0, 1)
 */
inline float uint64_to_float(uint64_t value) {
    constexpr double TWO_64 = 18446744073709551616.0;
    return static_cast<float>(static_cast<double>(value) / TWO_64);
}

}  // namespace utils
}  // namespace aleam

#endif /* ALEAM_CORE_UTILS_H */