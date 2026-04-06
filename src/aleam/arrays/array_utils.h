/**
 * @file array_utils.h
 * @brief Utility functions for array operations in Aleam
 * @license MIT
 * 
 * Helper functions for multi-dimensional array manipulation,
 * shape handling, and index calculations.
 */

#ifndef ALEAM_ARRAYS_ARRAY_UTILS_H
#define ALEAM_ARRAYS_ARRAY_UTILS_H

#include <vector>
#include <cstddef>
#include <numeric>
#include <stdexcept>

namespace aleam {
namespace arrays {
namespace utils {

/* ============================================================================
 * Shape and Index Operations
 * ============================================================================ */

/**
 * @brief Calculate total number of elements from shape
 * 
 * Computes the product of all dimensions in the shape vector.
 * 
 * @param shape Vector of dimensions (each > 0)
 * @return size_t Total number of elements
 * @throws std::invalid_argument if shape is empty or contains zero
 */
inline size_t total_elements(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        throw std::invalid_argument("shape cannot be empty");
    }
    
    size_t total = 1;
    for (size_t dim : shape) {
        if (dim == 0) {
            throw std::invalid_argument("dimension cannot be zero");
        }
        total *= dim;
    }
    return total;
}

/**
 * @brief Calculate strides for row-major (C-style) ordering
 * 
 * Stride[i] = product of dimensions from i+1 to end
 * 
 * @param shape Vector of dimensions
 * @return std::vector<size_t> Strides for each dimension
 */
inline std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return std::vector<size_t>();
    }
    
    std::vector<size_t> strides(shape.size());
    size_t stride = 1;
    
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    
    return strides;
}

/**
 * @brief Convert multi-dimensional index to linear index
 * 
 * Uses row-major (C-style) ordering.
 * 
 * @param shape Vector of dimensions
 * @param indices Multi-dimensional index
 * @return size_t Linear index
 * @throws std::invalid_argument if dimensions mismatch
 * @throws std::out_of_range if index out of bounds
 */
inline size_t flatten_index(const std::vector<size_t>& shape, 
                             const std::vector<size_t>& indices) {
    if (shape.size() != indices.size()) {
        throw std::invalid_argument("shape and indices dimensions must match");
    }
    
    size_t linear = 0;
    size_t stride = 1;
    
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        if (indices[i] >= shape[i]) {
            throw std::out_of_range("index out of bounds");
        }
        linear += indices[i] * stride;
        stride *= shape[i];
    }
    
    return linear;
}

/**
 * @brief Convert linear index to multi-dimensional index
 * 
 * Uses row-major (C-style) ordering.
 * 
 * @param shape Vector of dimensions
 * @param linear Linear index
 * @return std::vector<size_t> Multi-dimensional index
 * @throws std::out_of_range if linear index out of bounds
 */
inline std::vector<size_t> unravel_index(const std::vector<size_t>& shape, 
                                          size_t linear) {
    size_t total = total_elements(shape);
    if (linear >= total) {
        throw std::out_of_range("linear index out of bounds");
    }
    
    std::vector<size_t> indices(shape.size());
    size_t remaining = linear;
    
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        size_t stride = 1;
        for (size_t j = 0; j < static_cast<size_t>(i); ++j) {
            stride *= shape[j];
        }
        indices[i] = remaining / stride;
        remaining %= stride;
    }
    
    return indices;
}

/**
 * @brief Check if two shapes are equal
 * 
 * @param shape1 First shape
 * @param shape2 Second shape
 * @return bool True if shapes are identical
 */
inline bool shape_equal(const std::vector<size_t>& shape1, 
                         const std::vector<size_t>& shape2) {
    if (shape1.size() != shape2.size()) {
        return false;
    }
    for (size_t i = 0; i < shape1.size(); ++i) {
        if (shape1[i] != shape2[i]) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Get the number of dimensions (rank) of a shape
 * 
 * @param shape Vector of dimensions
 * @return size_t Number of dimensions
 */
inline size_t rank(const std::vector<size_t>& shape) {
    return shape.size();
}

/**
 * @brief Check if shape is 1D
 * 
 * @param shape Vector of dimensions
 * @return bool True if shape has one dimension
 */
inline bool is_1d(const std::vector<size_t>& shape) {
    return shape.size() == 1;
}

/**
 * @brief Check if shape is 2D
 * 
 * @param shape Vector of dimensions
 * @return bool True if shape has two dimensions
 */
inline bool is_2d(const std::vector<size_t>& shape) {
    return shape.size() == 2;
}

/**
 * @brief Check if shape is 3D
 * 
 * @param shape Vector of dimensions
 * @return bool True if shape has three dimensions
 */
inline bool is_3d(const std::vector<size_t>& shape) {
    return shape.size() == 3;
}

/* ============================================================================
 * Vector Conversion Utilities
 * ============================================================================ */

/**
 * @brief Flatten a 2D vector to 1D
 * 
 * @tparam T Element type
 * @param vec 2D vector to flatten
 * @return std::vector<T> Flattened 1D vector
 */
template<typename T>
inline std::vector<T> flatten_2d(const std::vector<std::vector<T>>& vec) {
    if (vec.empty()) {
        return std::vector<T>();
    }
    
    size_t rows = vec.size();
    size_t cols = vec[0].size();
    std::vector<T> result(rows * cols);
    
    for (size_t i = 0; i < rows; ++i) {
        if (vec[i].size() != cols) {
            throw std::invalid_argument("jagged array detected");
        }
        std::copy(vec[i].begin(), vec[i].end(), result.begin() + i * cols);
    }
    
    return result;
}

/**
 * @brief Reshape a 1D vector to 2D
 * 
 * @tparam T Element type
 * @param vec 1D vector to reshape
 * @param rows Number of rows
 * @param cols Number of columns
 * @return std::vector<std::vector<T>> 2D vector
 */
template<typename T>
inline std::vector<std::vector<T>> reshape_2d(const std::vector<T>& vec, 
                                               size_t rows, size_t cols) {
    if (vec.size() != rows * cols) {
        throw std::invalid_argument("vector size does not match target shape");
    }
    
    std::vector<std::vector<T>> result(rows, std::vector<T>(cols));
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = vec[i * cols + j];
        }
    }
    
    return result;
}

/**
 * @brief Create a deep copy of a 2D vector
 * 
 * @tparam T Element type
 * @param vec Input 2D vector
 * @return std::vector<std::vector<T>> Deep copy
 */
template<typename T>
inline std::vector<std::vector<T>> deep_copy_2d(const std::vector<std::vector<T>>& vec) {
    std::vector<std::vector<T>> result;
    result.reserve(vec.size());
    
    for (const auto& row : vec) {
        result.push_back(row);
    }
    
    return result;
}

/* ============================================================================
 * Shape Validation
 * ============================================================================ */

/**
 * @brief Validate that a shape is positive in all dimensions
 * 
 * @param shape Vector of dimensions
 * @return bool True if all dimensions > 0
 */
inline bool is_valid_shape(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return false;
    }
    for (size_t dim : shape) {
        if (dim == 0) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Get a string representation of a shape
 * 
 * @param shape Vector of dimensions
 * @return std::string Shape string like "(10, 20, 30)"
 */
inline std::string shape_to_string(const std::vector<size_t>& shape) {
    std::string result = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        result += std::to_string(shape[i]);
        if (i < shape.size() - 1) {
            result += ", ";
        }
    }
    result += ")";
    return result;
}

}  // namespace utils
}  // namespace arrays
}  // namespace aleam

#endif /* ALEAM_ARRAYS_ARRAY_UTILS_H */