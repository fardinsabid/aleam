/**
 * @file blake2s.h
 * @brief BLAKE2s cryptographic hash function implementation
 * @license Public Domain
 * 
 * BLAKE2s is a fast, secure cryptographic hash function designed for
 * 8- to 32-bit platforms. It is an improved version of BLAKE.
 * 
 * Aleam uses BLAKE2s to hash the entropy + timestamp mixture into
 * a uniformly distributed 64-bit output.
 * 
 * Reference: https://www.blake2.net/
 */

#ifndef BLAKE2S_H
#define BLAKE2S_H

#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include "blake2s_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * BLAKE2s Initialization Vector (IV)
 * ============================================================================ */

/**
 * @brief BLAKE2s initialization constants
 * 
 * These are the initial hash values derived from the fractional parts
 * of the square roots of the first 8 prime numbers.
 * Same as SHA-256 initialization vector.
 */
static const uint32_t blake2s_iv[8] = {
    0x6A09E667,  /* sqrt(2) fractional part */
    0xBB67AE85,  /* sqrt(3) fractional part */
    0x3C6EF372,  /* sqrt(5) fractional part */
    0xA54FF53A,  /* sqrt(7) fractional part */
    0x510E527F,  /* sqrt(11) fractional part */
    0x9B05688C,  /* sqrt(13) fractional part */
    0x1F83D9AB,  /* sqrt(17) fractional part */
    0x5BE0CD19   /* sqrt(19) fractional part */
};

/* ============================================================================
 * BLAKE2s Permutation Sigma Constants
 * ============================================================================ */

/**
 * @brief BLAKE2s message scheduling permutations
 * 
 * These 10 permutations define the order in which message words
 * are mixed in each of the 10 rounds.
 */
static const uint8_t blake2s_sigma[10][16] = {
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
    {14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3},
    {11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4},
    { 7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8},
    { 9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13},
    { 2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9},
    {12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11},
    {13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10},
    { 6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5},
    {10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0}
};

/* ============================================================================
 * BLAKE2s State Structure
 * ============================================================================ */

/**
 * @brief BLAKE2s hash state
 * 
 * Contains all internal state for incremental hashing.
 */
typedef struct {
    uint32_t h[8];              /**< Hash state (8 x 32-bit words) */
    uint32_t t[2];              /**< Message byte counter (64-bit) */
    uint32_t f[2];              /**< Finalization flags */
    uint8_t buf[BLAKE2S_BLOCKBYTES]; /**< Input buffer */
    size_t buflen;              /**< Current buffer length */
    size_t outlen;              /**< Desired output length in bytes */
} blake2s_state;

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * @brief BLAKE2s G function - core mixing operation
 * 
 * The G function mixes two 32-bit inputs into four 32-bit state variables.
 * This is the fundamental building block of the BLAKE2s compression function.
 * 
 * @param a First state variable (in/out)
 * @param b Second state variable (in/out)
 * @param c Third state variable (in/out)
 * @param d Fourth state variable (in/out)
 * @param x First message word to mix
 * @param y Second message word to mix
 */
BLAKE2S_INLINE void blake2s_g(uint32_t* a, uint32_t* b, uint32_t* c, uint32_t* d,
                               uint32_t x, uint32_t y) {
    *a = *a + *b + x;
    *d = BLAKE2S_ROTR32(*d ^ *a, 16);
    *c = *c + *d;
    *b = BLAKE2S_ROTR32(*b ^ *c, 12);
    *a = *a + *b + y;
    *d = BLAKE2S_ROTR32(*d ^ *a, 8);
    *c = *c + *d;
    *b = BLAKE2S_ROTR32(*b ^ *c, 7);
}

/* ============================================================================
 * Compression Function
 * ============================================================================ */

/**
 * @brief BLAKE2s compression function
 * 
 * Compresses a 64-byte block into the 32-byte hash state.
 * Performs 10 rounds of mixing operations.
 * 
 * @param S Pointer to BLAKE2s state
 * @param block 64-byte message block to compress
 */
static void blake2s_compress(blake2s_state* S, const uint8_t* block) {
    uint32_t v[16];  /* Working variables */
    uint32_t m[16];  /* Message words */
    size_t i;

    /* Copy state to working variables */
    for (i = 0; i < 8; i++) {
        v[i] = S->h[i];
    }
    
    /* Copy initialization vector to second half */
    for (i = 0; i < 8; i++) {
        v[i + 8] = blake2s_iv[i];
    }

    /* XOR counter and finalization flags */
    v[12] ^= S->t[0];
    v[13] ^= S->t[1];
    v[14] ^= S->f[0];
    v[15] ^= S->f[1];

    /* Unpack 64-byte block into 16 x 32-bit little-endian words */
    for (i = 0; i < 16; i++) {
        m[i] = (uint32_t)block[4 * i] |
               ((uint32_t)block[4 * i + 1] << 8) |
               ((uint32_t)block[4 * i + 2] << 16) |
               ((uint32_t)block[4 * i + 3] << 24);
    }

    /* 10 rounds of mixing */
    for (i = 0; i < 10; i++) {
        const uint8_t* s = blake2s_sigma[i];
        
        /* Column rounds */
        blake2s_g(&v[0], &v[4], &v[8],  &v[12], m[s[0]], m[s[1]]);
        blake2s_g(&v[1], &v[5], &v[9],  &v[13], m[s[2]], m[s[3]]);
        blake2s_g(&v[2], &v[6], &v[10], &v[14], m[s[4]], m[s[5]]);
        blake2s_g(&v[3], &v[7], &v[11], &v[15], m[s[6]], m[s[7]]);
        
        /* Diagonal rounds */
        blake2s_g(&v[0], &v[5], &v[10], &v[15], m[s[8]],  m[s[9]]);
        blake2s_g(&v[1], &v[6], &v[11], &v[12], m[s[10]], m[s[11]]);
        blake2s_g(&v[2], &v[7], &v[8],  &v[13], m[s[12]], m[s[13]]);
        blake2s_g(&v[3], &v[4], &v[9],  &v[14], m[s[14]], m[s[15]]);
    }

    /* Update hash state with mixing results */
    for (i = 0; i < 8; i++) {
        S->h[i] ^= v[i] ^ v[i + 8];
    }
}

/* ============================================================================
 * Public API Functions
 * ============================================================================ */

/**
 * @brief Initialize BLAKE2s hash state
 * 
 * @param S Pointer to uninitialized state structure
 * @param outlen Desired output length in bytes (1 to 32)
 */
static void blake2s_init(blake2s_state* S, size_t outlen) {
    size_t i;
    
    /* Zero out the state */
    memset(S, 0, sizeof(blake2s_state));
    
    /* Initialize hash with IV */
    for (i = 0; i < 8; i++) {
        S->h[i] = blake2s_iv[i];
    }
    
    /* XOR in parameter block: outlen | keylen | fanout | depth | etc. */
    S->h[0] ^= 0x01010000 ^ ((uint32_t)outlen << 24);
    S->outlen = outlen;
}

/**
 * @brief Update BLAKE2s hash with more data
 * 
 * @param S Pointer to initialized state
 * @param data Input data to hash
 * @param len Length of input data in bytes
 */
static void blake2s_update(blake2s_state* S, const uint8_t* data, size_t len) {
    size_t left = S->buflen;
    size_t fill = BLAKE2S_BLOCKBYTES - left;

    /* Update message byte counter (64-bit) */
    S->t[0] += (uint32_t)len;
    if (S->t[0] < len) {
        S->t[1]++;
    }

    /* If we have pending data + new data fills a block, process it */
    if (left && len >= fill) {
        memcpy(S->buf + left, data, fill);
        blake2s_compress(S, S->buf);
        data += fill;
        len -= fill;
        left = 0;
    }

    /* Process full blocks directly */
    while (len >= BLAKE2S_BLOCKBYTES) {
        blake2s_compress(S, data);
        data += BLAKE2S_BLOCKBYTES;
        len -= BLAKE2S_BLOCKBYTES;
    }

    /* Store remaining data in buffer */
    if (len > 0) {
        memcpy(S->buf + left, data, len);
        S->buflen = left + len;
    } else {
        S->buflen = 0;
    }
}

/**
 * @brief Finalize BLAKE2s hash and produce output
 * 
 * @param S Pointer to initialized state
 * @param out Output buffer for hash (must be at least outlen bytes)
 */
static void blake2s_final(blake2s_state* S, uint8_t* out) {
    size_t i;

    /* Set finalization flag */
    S->f[0] = 0xFFFFFFFF;

    /* Pad remaining data with zeros and compress */
    if (S->buflen > 0) {
        memset(S->buf + S->buflen, 0, BLAKE2S_BLOCKBYTES - S->buflen);
        blake2s_compress(S, S->buf);
    }

    /* Output hash (little-endian) */
    for (i = 0; i < 8 && i * 4 < S->outlen; i++) {
        out[i * 4]     = (uint8_t)(S->h[i]);
        out[i * 4 + 1] = (uint8_t)(S->h[i] >> 8);
        out[i * 4 + 2] = (uint8_t)(S->h[i] >> 16);
        out[i * 4 + 3] = (uint8_t)(S->h[i] >> 24);
    }
}

/* ============================================================================
 * One-shot Hash Functions for Aleam
 * ============================================================================ */

/**
 * @brief One-shot BLAKE2s hash (full output)
 * 
 * @param input Input data to hash
 * @param len Length of input in bytes
 * @param output Output buffer for hash
 * @param outlen Desired output length in bytes
 */
static void blake2s_hash(const uint8_t* input, size_t len, 
                          uint8_t* output, size_t outlen) {
    blake2s_state S;
    blake2s_init(&S, outlen);
    blake2s_update(&S, input, len);
    blake2s_final(&S, output);
}

/**
 * @brief Fast 64-bit BLAKE2s hash for Aleam core
 * 
 * This is the primary hash function used by Aleam.
 * It takes a 64-bit input and returns a 64-bit uniformly distributed hash.
 * 
 * @param input 64-bit value to hash
 * @return 64-bit uniformly distributed hash value
 */
BLAKE2S_INLINE uint64_t blake2s_64(uint64_t input) {
    uint8_t hash[8];
    uint8_t in[8];
    
    /* Convert 64-bit integer to little-endian bytes */
    in[0] = (uint8_t)(input);
    in[1] = (uint8_t)(input >> 8);
    in[2] = (uint8_t)(input >> 16);
    in[3] = (uint8_t)(input >> 24);
    in[4] = (uint8_t)(input >> 32);
    in[5] = (uint8_t)(input >> 40);
    in[6] = (uint8_t)(input >> 48);
    in[7] = (uint8_t)(input >> 56);
    
    /* Hash to 64 bits */
    blake2s_hash(in, 8, hash, 8);
    
    /* Convert back to 64-bit integer */
    return (uint64_t)hash[0] |
           ((uint64_t)hash[1] << 8) |
           ((uint64_t)hash[2] << 16) |
           ((uint64_t)hash[3] << 24) |
           ((uint64_t)hash[4] << 32) |
           ((uint64_t)hash[5] << 40) |
           ((uint64_t)hash[6] << 48) |
           ((uint64_t)hash[7] << 56);
}

#ifdef __cplusplus
}
#endif

#endif /* BLAKE2S_H */