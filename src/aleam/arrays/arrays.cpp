/**
 * @file arrays.cpp
 * @brief Implementation of array operations for Aleam
 * @license MIT
 * 
 * This file implements the array generation functions for creating
 * multi-dimensional arrays of true random numbers.
 */

#include "arrays.h"
#include "../distributions/distributions.h"
#include <numeric>
#include <stdexcept>

namespace aleam {
namespace arrays {

/* ============================================================================
 * Core Array Generation Helpers
 * ============================================================================ */

/**
 * @brief Calculate total number of elements from shape
 * 
 * Computes the product of all dimensions in the shape vector.
 * 
 * @param shape Vector of dimensions
 * @return size_t Total number of elements
 */
size_t total_elements(const Shape& shape) {
    if (shape.empty()) {
        return 0;
    }
    
    size_t total = 1;
    for (size_t dim : shape) {
        total *= dim;
    }
    return total;
}

/**
 * @brief Flatten a multi-dimensional index to linear index
 * 
 * Converts indices like [row, col, depth] to a linear position
 * in a flattened array using row-major (C-style) ordering.
 * 
 * @param shape Shape of the array
 * @param indices Multi-dimensional index
 * @return size_t Linear index
 */
size_t flatten_index(const Shape& shape, const std::vector<size_t>& indices) {
    if (shape.size() != indices.size()) {
        throw std::invalid_argument("indices dimension must match shape dimension");
    }
    
    size_t linear = 0;
    size_t stride = 1;
    
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
        if (indices[i] >= shape[i]) {
            throw std::out_of_range("index out of bounds");
        }
        linear += indices[i] * stride;
        stride *= shape[i];
    }
    
    return linear;
}

/**
 * @brief Convert linear index to multi-dimensional indices
 * 
 * Converts a linear position in a flattened array back to
 * multi-dimensional indices using row-major (C-style) ordering.
 * 
 * @param shape Shape of the array
 * @param linear Linear index
 * @return std::vector<size_t> Multi-dimensional indices
 */
std::vector<size_t> unravel_index(const Shape& shape, size_t linear) {
    if (linear >= total_elements(shape)) {
        throw std::out_of_range("linear index out of bounds");
    }
    
    std::vector<size_t> indices(shape.size());
    size_t remaining = linear;
    
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
        size_t stride = 1;
        for (size_t j = 0; j < static_cast<size_t>(i); j++) {
            stride *= shape[j];
        }
        indices[i] = remaining / stride;
        remaining %= stride;
    }
    
    return indices;
}

/* ============================================================================
 * Uniform Distribution Arrays
 * ============================================================================ */

/**
 * @brief Generate 1D array of true random floats in [0, 1)
 * 
 * Uses batch generation for efficiency when size is large.
 * 
 * @param rng Reference to AleamCore instance
 * @param size Number of elements
 * @return std::vector<double> Array of random floats
 */
std::vector<double> random_array_1d(AleamCore& rng, size_t size) {
    if (size == 0) {
        return std::vector<double>();
    }
    
    std::vector<double> result(size);
    
    /* Use batch generation for efficiency */
    rng.random_batch(result.data(), size);
    
    return result;
}

/**
 * @brief Generate 2D array of true random floats in [0, 1)
 * 
 * @param rng Reference to AleamCore instance
 * @param rows Number of rows
 * @param cols Number of columns
 * @return std::vector<std::vector<double>> 2D array of random floats
 */
std::vector<std::vector<double>> random_array_2d(AleamCore& rng, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        return std::vector<std::vector<double>>();
    }
    
    std::vector<std::vector<double>> result(rows, std::vector<double>(cols));
    
    /* Generate each row using batch generation */
    for (size_t i = 0; i < rows; i++) {
        rng.random_batch(result[i].data(), cols);
    }
    
    return result;
}

/**
 * @brief Generate N-dimensional array of true random floats in [0, 1)
 * 
 * Returns a flattened array. The caller should use the shape parameter
 * to interpret the indexing.
 * 
 * @param rng Reference to AleamCore instance
 * @param shape Shape of the array
 * @return std::vector<double> Flattened array
 */
std::vector<double> random_array_nd(AleamCore& rng, const Shape& shape) {
    size_t total = total_elements(shape);
    return random_array_1d(rng, total);
}

/* ============================================================================
 * Normal Distribution Arrays
 * ============================================================================ */

/**
 * @brief Generate 1D array of true random normally distributed values
 * 
 * @param rng Reference to AleamCore instance
 * @param size Number of elements
 * @param mu Mean of distribution
 * @param sigma Standard deviation
 * @return std::vector<double> Array of normally distributed values
 */
std::vector<double> randn_array_1d(AleamCore& rng, size_t size, double mu, double sigma) {
    if (size == 0) {
        return std::vector<double>();
    }
    
    std::vector<double> result(size);
    
    for (size_t i = 0; i < size; i++) {
        result[i] = distributions::normal(rng, mu, sigma);
    }
    
    return result;
}

/**
 * @brief Generate 2D array of true random normally distributed values
 * 
 * @param rng Reference to AleamCore instance
 * @param rows Number of rows
 * @param cols Number of columns
 * @param mu Mean of distribution
 * @param sigma Standard deviation
 * @return std::vector<std::vector<double>> 2D array of normally distributed values
 */
std::vector<std::vector<double>> randn_array_2d(AleamCore& rng, size_t rows, size_t cols,
                                                  double mu, double sigma) {
    if (rows == 0 || cols == 0) {
        return std::vector<std::vector<double>>();
    }
    
    std::vector<std::vector<double>> result(rows, std::vector<double>(cols));
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i][j] = distributions::normal(rng, mu, sigma);
        }
    }
    
    return result;
}

/**
 * @brief Generate N-dimensional array of true random normally distributed values
 * 
 * Returns a flattened array. The caller should use the shape parameter
 * to interpret the indexing.
 * 
 * @param rng Reference to AleamCore instance
 * @param shape Shape of the array
 * @param mu Mean of distribution
 * @param sigma Standard deviation
 * @return std::vector<double> Flattened array
 */
std::vector<double> randn_array_nd(AleamCore& rng, const Shape& shape, double mu, double sigma) {
    size_t total = total_elements(shape);
    return randn_array_1d(rng, total, mu, sigma);
}

/* ============================================================================
 * Integer Arrays
 * ============================================================================ */

/**
 * @brief Generate 1D array of true random integers in [low, high]
 * 
 * @param rng Reference to AleamCore instance
 * @param size Number of elements
 * @param low Lower bound (inclusive)
 * @param high Upper bound (inclusive)
 * @return std::vector<int64_t> Array of random integers
 */
std::vector<int64_t> randint_array_1d(AleamCore& rng, size_t size, int64_t low, int64_t high) {
    if (size == 0) {
        return std::vector<int64_t>();
    }
    if (low > high) {
        throw std::invalid_argument("low must be <= high");
    }
    
    std::vector<int64_t> result(size);
    int64_t range = high - low + 1;
    
    for (size_t i = 0; i < size; i++) {
        result[i] = low + static_cast<int64_t>(rng.random() * range);
        /* Ensure value is within bounds (handle floating point edge cases) */
        if (result[i] > high) result[i] = high;
        if (result[i] < low) result[i] = low;
    }
    
    return result;
}

/**
 * @brief Generate 2D array of true random integers in [low, high]
 * 
 * @param rng Reference to AleamCore instance
 * @param rows Number of rows
 * @param cols Number of columns
 * @param low Lower bound (inclusive)
 * @param high Upper bound (inclusive)
 * @return std::vector<std::vector<int64_t>> 2D array of random integers
 */
std::vector<std::vector<int64_t>> randint_array_2d(AleamCore& rng, size_t rows, size_t cols,
                                                     int64_t low, int64_t high) {
    if (rows == 0 || cols == 0) {
        return std::vector<std::vector<int64_t>>();
    }
    if (low > high) {
        throw std::invalid_argument("low must be <= high");
    }
    
    std::vector<std::vector<int64_t>> result(rows, std::vector<int64_t>(cols));
    int64_t range = high - low + 1;
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i][j] = low + static_cast<int64_t>(rng.random() * range);
            if (result[i][j] > high) result[i][j] = high;
            if (result[i][j] < low) result[i][j] = low;
        }
    }
    
    return result;
}

/**
 * @brief Generate N-dimensional array of true random integers
 * 
 * Returns a flattened array. The caller should use the shape parameter
 * to interpret the indexing.
 * 
 * @param rng Reference to AleamCore instance
 * @param shape Shape of the array
 * @param low Lower bound (inclusive)
 * @param high Upper bound (inclusive)
 * @return std::vector<int64_t> Flattened array
 */
std::vector<int64_t> randint_array_nd(AleamCore& rng, const Shape& shape,
                                        int64_t low, int64_t high) {
    size_t total = total_elements(shape);
    return randint_array_1d(rng, total, low, high);
}

}  // namespace arrays
}  // namespace aleam