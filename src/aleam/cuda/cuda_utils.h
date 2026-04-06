/**
 * @file cuda_utils.h
 * @brief CUDA utility functions for Aleam GPU acceleration
 * @license MIT
 * 
 * Provides helper functions for CUDA device management,
 * kernel launch configuration, and error handling.
 */

#ifndef ALEAM_CUDA_CUDA_UTILS_H
#define ALEAM_CUDA_CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include <string>

namespace aleam {
namespace cuda {

/* ============================================================================
 * Error Handling
 * ============================================================================ */

/**
 * @brief Check CUDA error and print message if any
 * 
 * @param code CUDA error code
 * @param file Source file name
 * @param line Line number
 * @return bool True if no error
 */
inline bool check_cuda_error(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        const char* error_str = cudaGetErrorString(code);
        printf("CUDA Error at %s:%d: %s\n", file, line, error_str);
        return false;
    }
    return true;
}

/**
 * @brief Macro for checking CUDA errors
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

/* ============================================================================
 * Device Management
 * ============================================================================ */

/**
 * @brief Get number of CUDA devices
 * 
 * @return int Number of devices (0 if none)
 */
inline int get_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

/**
 * @brief Get current device ID
 * 
 * @return int Current device ID (-1 if error)
 */
inline int get_current_device() {
    int device = -1;
    cudaGetDevice(&device);
    return device;
}

/**
 * @brief Set active CUDA device
 * 
 * @param device_id Device ID (0-based)
 * @return bool True if successful
 */
inline bool set_device(int device_id) {
    int count = get_device_count();
    if (device_id < 0 || device_id >= count) {
        return false;
    }
    cudaSetDevice(device_id);
    return true;
}

/**
 * @brief Get device name
 * 
 * @param device_id Device ID
 * @return std::string Device name (empty if error)
 */
inline std::string get_device_name(int device_id) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        return "";
    }
    return std::string(prop.name);
}

/**
 * @brief Get device memory info
 * 
 * @param device_id Device ID
 * @param free_mem Free memory in bytes
 * @param total_mem Total memory in bytes
 * @return bool True if successful
 */
inline bool get_device_memory_info(int device_id, size_t* free_mem, size_t* total_mem) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return false;
    
    err = cudaMemGetInfo(free_mem, total_mem);
    return (err == cudaSuccess);
}

/* ============================================================================
 * Launch Configuration
 * ============================================================================ */

/**
 * @brief Launch parameters structure
 */
struct LaunchParams {
    int grid_size;              /**< Number of blocks */
    int block_size;             /**< Threads per block (typically 256) */
    int elements_per_thread;    /**< Elements each thread generates */
};

/**
 * @brief Calculate optimal launch parameters
 * 
 * @param total_elements Total number of elements to generate
 * @param block_size Desired block size (256 recommended)
 * @return LaunchParams Calculated launch parameters
 */
inline LaunchParams calculate_launch_params(size_t total_elements, int block_size = 256) {
    LaunchParams params;
    params.block_size = block_size;
    
    int max_blocks = 65535;  // CUDA limit
    
    // Calculate threads needed
    size_t threads_needed = total_elements;
    int threads_available = max_blocks * block_size;
    
    if (threads_needed <= static_cast<size_t>(threads_available)) {
        // One element per thread
        params.grid_size = (total_elements + block_size - 1) / block_size;
        params.elements_per_thread = 1;
    } else {
        // Multiple elements per thread
        params.grid_size = max_blocks;
        params.elements_per_thread = (total_elements + max_blocks * block_size - 1) / 
                                      (max_blocks * block_size);
    }
    
    return params;
}

/* ============================================================================
 * Memory Management
 * ============================================================================ */

/**
 * @brief Allocate device memory
 * 
 * @param ptr Pointer to device pointer
 * @param size Size in bytes
 * @return bool True if successful
 */
template<typename T>
inline bool allocate_device(T** ptr, size_t size) {
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(ptr), size);
    return (err == cudaSuccess);
}

/**
 * @brief Free device memory
 * 
 * @param ptr Device pointer
 */
template<typename T>
inline void free_device(T* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

/**
 * @brief Copy data from host to device
 * 
 * @param dst Device destination
 * @param src Host source
 * @param size Size in bytes
 * @return bool True if successful
 */
inline bool copy_host_to_device(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    return (err == cudaSuccess);
}

/**
 * @brief Copy data from device to host
 * 
 * @param dst Host destination
 * @param src Device source
 * @param size Size in bytes
 * @return bool True if successful
 */
inline bool copy_device_to_host(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess);
}

/**
 * @brief Synchronize device
 * 
 * @return bool True if successful
 */
inline bool synchronize_device() {
    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess);
}

/**
 * @brief Reset device (clears all state)
 * 
 * @return bool True if successful
 */
inline bool reset_device() {
    cudaError_t err = cudaDeviceReset();
    return (err == cudaSuccess);
}

/* ============================================================================
 * Stream Management
 * ============================================================================ */

/**
 * @brief Create CUDA stream
 * 
 * @param stream Pointer to stream
 * @return bool True if successful
 */
inline bool create_stream(cudaStream_t* stream) {
    cudaError_t err = cudaStreamCreate(stream);
    return (err == cudaSuccess);
}

/**
 * @brief Destroy CUDA stream
 * 
 * @param stream Stream to destroy
 * @return bool True if successful
 */
inline bool destroy_stream(cudaStream_t stream) {
    cudaError_t err = cudaStreamDestroy(stream);
    return (err == cudaSuccess);
}

/**
 * @brief Synchronize stream
 * 
 * @param stream Stream to synchronize
 * @return bool True if successful
 */
inline bool synchronize_stream(cudaStream_t stream) {
    cudaError_t err = cudaStreamSynchronize(stream);
    return (err == cudaSuccess);
}

/* ============================================================================
 * Information
 * ============================================================================ */

/**
 * @brief Check if CUDA is available
 * 
 * @return bool True if CUDA is available
 */
inline bool is_cuda_available() {
    int count = get_device_count();
    return count > 0;
}

/**
 * @brief Get CUDA driver version
 * 
 * @return int Driver version (0 if error)
 */
inline int get_driver_version() {
    int version = 0;
    cudaDriverGetVersion(&version);
    return version;
}

/**
 * @brief Get CUDA runtime version
 * 
 * @return int Runtime version (0 if error)
 */
inline int get_runtime_version() {
    int version = 0;
    cudaRuntimeGetVersion(&version);
    return version;
}

}  // namespace cuda
}  // namespace aleam

#endif /* ALEAM_CUDA_CUDA_UTILS_H */