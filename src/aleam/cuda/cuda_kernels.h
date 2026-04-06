/**
 * @file cuda_kernels.h
 * @brief CUDA kernel declarations for true random number generation on GPU
 * @license MIT
 * 
 * This file declares the CUDA kernels and provides host-side wrappers
 * for launching them. These functions are called from the Python bindings
 * to generate true random numbers directly on GPU.
 * 
 * Performance: Up to 100M ops/sec on NVIDIA Tesla T4
 */

#ifndef ALEAM_CUDA_KERNELS_H
#define ALEAM_CUDA_KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Kernel Declarations
 * ============================================================================ */

/**
 * @brief Generate uniform random floats in [0, 1) on GPU
 * 
 * @param output Output device array (float)
 * @param seeds Per-block true random seeds (device array)
 * @param total_elements Total number of elements to generate
 * @param elements_per_thread Number of elements per thread
 */
__global__ void uniform_float_kernel(
    float* output,
    const uint64_t* seeds,
    size_t total_elements,
    int elements_per_thread
);

/**
 * @brief Generate uniform random doubles in [0, 1) on GPU
 * 
 * @param output Output device array (double)
 * @param seeds Per-block true random seeds (device array)
 * @param total_elements Total number of elements to generate
 * @param elements_per_thread Number of elements per thread
 */
__global__ void uniform_double_kernel(
    double* output,
    const uint64_t* seeds,
    size_t total_elements,
    int elements_per_thread
);

/**
 * @brief Generate normal random floats on GPU
 * 
 * @param output Output device array (float)
 * @param mu Mean of distribution
 * @param sigma Standard deviation
 * @param seeds Per-block true random seeds (device array)
 * @param total_elements Total number of elements to generate
 * @param elements_per_thread Number of elements per thread
 */
__global__ void normal_float_kernel(
    float* output,
    float mu,
    float sigma,
    const uint64_t* seeds,
    size_t total_elements,
    int elements_per_thread
);

/**
 * @brief Generate normal random doubles on GPU
 * 
 * @param output Output device array (double)
 * @param mu Mean of distribution
 * @param sigma Standard deviation
 * @param seeds Per-block true random seeds (device array)
 * @param total_elements Total number of elements to generate
 * @param elements_per_thread Number of elements per thread
 */
__global__ void normal_double_kernel(
    double* output,
    double mu,
    double sigma,
    const uint64_t* seeds,
    size_t total_elements,
    int elements_per_thread
);

/**
 * @brief Generate random integers on GPU
 * 
 * @param output Output device array (int)
 * @param low Lower bound (inclusive)
 * @param high Upper bound (inclusive)
 * @param seeds Per-block true random seeds (device array)
 * @param total_elements Total number of elements to generate
 * @param elements_per_thread Number of elements per thread
 */
__global__ void randint_kernel(
    int* output,
    int low,
    int high,
    const uint64_t* seeds,
    size_t total_elements,
    int elements_per_thread
);

/* ============================================================================
 * Host-Side Launch Wrappers
 * ============================================================================ */

/**
 * @brief Launch parameters structure
 */
typedef struct {
    int grid_size;              /**< Number of blocks */
    int block_size;             /**< Threads per block (typically 256) */
    int elements_per_thread;    /**< Elements each thread generates */
} LaunchParams;

/**
 * @brief Calculate optimal launch parameters
 * 
 * @param total_elements Total number of elements to generate
 * @param block_size Desired block size (256 recommended)
 * @return LaunchParams Calculated launch parameters
 */
LaunchParams calculate_launch_params(size_t total_elements, int block_size);

/**
 * @brief Launch uniform float kernel with optimal parameters
 * 
 * @param output Output device array (float)
 * @param seeds Per-block true random seeds (device array)
 * @param total_elements Total number of elements
 * @param stream CUDA stream (0 for default)
 * @return cudaError_t CUDA error code
 */
cudaError_t launch_uniform_float(
    float* output,
    const uint64_t* seeds,
    size_t total_elements,
    cudaStream_t stream = 0
);

/**
 * @brief Launch uniform double kernel with optimal parameters
 * 
 * @param output Output device array (double)
 * @param seeds Per-block true random seeds (device array)
 * @param total_elements Total number of elements
 * @param stream CUDA stream (0 for default)
 * @return cudaError_t CUDA error code
 */
cudaError_t launch_uniform_double(
    double* output,
    const uint64_t* seeds,
    size_t total_elements,
    cudaStream_t stream = 0
);

/**
 * @brief Launch normal float kernel with optimal parameters
 * 
 * @param output Output device array (float)
 * @param mu Mean of distribution
 * @param sigma Standard deviation
 * @param seeds Per-block true random seeds (device array)
 * @param total_elements Total number of elements
 * @param stream CUDA stream (0 for default)
 * @return cudaError_t CUDA error code
 */
cudaError_t launch_normal_float(
    float* output,
    float mu,
    float sigma,
    const uint64_t* seeds,
    size_t total_elements,
    cudaStream_t stream = 0
);

/**
 * @brief Launch normal double kernel with optimal parameters
 * 
 * @param output Output device array (double)
 * @param mu Mean of distribution
 * @param sigma Standard deviation
 * @param seeds Per-block true random seeds (device array)
 * @param total_elements Total number of elements
 * @param stream CUDA stream (0 for default)
 * @return cudaError_t CUDA error code
 */
cudaError_t launch_normal_double(
    double* output,
    double mu,
    double sigma,
    const uint64_t* seeds,
    size_t total_elements,
    cudaStream_t stream = 0
);

/**
 * @brief Launch randint kernel with optimal parameters
 * 
 * @param output Output device array (int)
 * @param low Lower bound (inclusive)
 * @param high Upper bound (inclusive)
 * @param seeds Per-block true random seeds (device array)
 * @param total_elements Total number of elements
 * @param stream CUDA stream (0 for default)
 * @return cudaError_t CUDA error code
 */
cudaError_t launch_randint(
    int* output,
    int low,
    int high,
    const uint64_t* seeds,
    size_t total_elements,
    cudaStream_t stream = 0
);

/* ============================================================================
 * Device Management Helpers
 * ============================================================================ */

/**
 * @brief Get number of CUDA devices available
 * 
 * @return int Number of CUDA devices
 */
int get_cuda_device_count();

/**
 * @brief Get CUDA device name
 * 
 * @param device_id Device ID (0-based)
 * @param name Buffer to store device name
 * @param name_size Size of name buffer
 * @return cudaError_t CUDA error code
 */
cudaError_t get_cuda_device_name(int device_id, char* name, int name_size);

/**
 * @brief Check if CUDA is available
 * 
 * @return int 1 if CUDA available, 0 otherwise
 */
int is_cuda_available();

#ifdef __cplusplus
}
#endif

#endif /* ALEAM_CUDA_KERNELS_H */