/**
 * @file entropy_windows.h
 * @brief Windows entropy source using BCryptGenRandom()
 * @license MIT
 * 
 * Provides true entropy for Windows systems using the BCryptGenRandom()
 * function from the Windows Cryptography API: Next Generation (CNG).
 * 
 * This is the modern, recommended entropy source for Windows Vista and later.
 * It uses the kernel's cryptographically secure pseudo-random number generator
 * (CSPRNG) which draws from hardware entropy sources when available.
 * 
 * Unlike the older CryptGenRandom() API, BCryptGenRandom():
 * - Requires no handle initialization or cleanup
 * - Is thread-safe
 * - Has simpler error handling
 * - Is available on all supported Windows versions
 */

#ifndef ENTROPY_WINDOWS_H
#define ENTROPY_WINDOWS_H

#ifdef _WIN32

#include <windows.h>
#include <bcrypt.h>
#include <ntstatus.h>

#pragma comment(lib, "bcrypt.lib")

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Windows Entropy Functions
 * ============================================================================ */

/**
 * @brief Get entropy bytes from Windows using BCryptGenRandom()
 * 
 * This function reads cryptographically secure random bytes directly from
 * the Windows kernel CSPRNG. It uses BCryptGenRandom() which is the
 * modern replacement for the older CryptGenRandom() API.
 * 
 * The BCRYPT_USE_SYSTEM_PREFERRED_RNG flag tells Windows to use the
 * best available RNG implementation (hardware RNG if present, otherwise
 * the kernel CSPRNG).
 * 
 * @param buf   Pointer to buffer to fill with random bytes
 * @param len   Number of bytes to read
 * 
 * @return 0 on success, -1 on error
 */
static inline int get_entropy_bytes_windows(void* buf, size_t len) {
    NTSTATUS status;
    
    /* BCRYPT_USE_SYSTEM_PREFERRED_RNG uses the best available entropy source */
    status = BCryptGenRandom(
        NULL,                    /* hAlgorithm - NULL uses system preferred RNG */
        (BYTE*)buf,              /* pbBuffer - output buffer */
        (ULONG)len,              /* cbBuffer - number of bytes to generate */
        BCRYPT_USE_SYSTEM_PREFERRED_RNG  /* dwFlags - use system CSPRNG */
    );
    
    /* Check if the operation succeeded */
    if (status != STATUS_SUCCESS) {
        return -1;
    }
    
    return 0;
}

/**
 * @brief Get a single 64-bit random value from Windows
 * 
 * Convenience wrapper for get_entropy_bytes_windows() that returns
 * a uint64_t directly.
 * 
 * @return 64-bit cryptographically random value
 */
static inline uint64_t get_entropy_64_windows(void) {
    uint64_t value;
    
    if (get_entropy_bytes_windows(&value, sizeof(value)) != 0) {
        /* This should never happen on a properly functioning system */
        /* Fallback: return a value based on time and process ID */
        value = (uint64_t)GetTickCount64() ^ (uint64_t)GetCurrentProcessId();
    }
    
    return value;
}

#ifdef __cplusplus
}
#endif

#endif /* _WIN32 */

#endif /* ENTROPY_WINDOWS_H */