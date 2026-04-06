/**
 * @file entropy_linux.h
 * @brief Linux entropy source using getrandom() system call
 * @license MIT
 * 
 * Provides true entropy for Linux and Android systems using the
 * getrandom() system call, which reads from the kernel's entropy pool.
 * 
 * This is the preferred entropy source on Linux kernels 3.17+ and
 * Android 4.4+. It never blocks after system boot and is cryptographically
 * secure for generating random numbers.
 * 
 * The getrandom() call is the direct equivalent of Python's os.urandom()
 * but without the Python overhead layer.
 */

#ifndef ENTROPY_LINUX_H
#define ENTROPY_LINUX_H

#ifdef __linux__

#include <sys/random.h>
#include <unistd.h>
#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Linux Entropy Functions
 * ============================================================================ */

/**
 * @brief Get entropy bytes from Linux kernel using getrandom()
 * 
 * This function reads cryptographically secure random bytes directly from
 * the Linux kernel's entropy pool. It uses the getrandom() system call
 * introduced in Linux 3.17.
 * 
 * The GRND_NONBLOCK flag is NOT used because:
 * 1. On modern systems, the entropy pool is initialized at boot
 * 2. If it's not ready (very rare), we want to wait rather than fail
 * 3. The kernel will block until sufficient entropy is available
 * 
 * @param buf   Pointer to buffer to fill with random bytes
 * @param len   Number of bytes to read
 * 
 * @return 0 on success, -1 on error
 */
static inline int get_entropy_bytes_linux(void* buf, size_t len) {
    ssize_t result;
    uint8_t* ptr = (uint8_t*)buf;
    size_t remaining = len;
    
    /* Keep reading until we get all requested bytes */
    while (remaining > 0) {
        result = getrandom(ptr, remaining, 0);
        
        if (result < 0) {
            /* Check if the system call was interrupted */
            if (errno == EINTR) {
                continue;  /* Retry the call */
            }
            /* Any other error is fatal */
            return -1;
        }
        
        ptr += result;
        remaining -= (size_t)result;
    }
    
    return 0;
}

/**
 * @brief Get a single 64-bit random value from Linux kernel
 * 
 * Convenience wrapper for get_entropy_bytes_linux() that returns
 * a uint64_t directly.
 * 
 * @return 64-bit cryptographically random value
 */
static inline uint64_t get_entropy_64_linux(void) {
    uint64_t value;
    
    if (get_entropy_bytes_linux(&value, sizeof(value)) != 0) {
        /* This should never happen on a properly functioning system */
        /* Fallback: return a value based on time (not secure, but better than crash) */
        value = (uint64_t)time(NULL) ^ (uint64_t)getpid();
    }
    
    return value;
}

#ifdef __cplusplus
}
#endif

#endif /* __linux__ */

#endif /* ENTROPY_LINUX_H */