/**
 * @file arrays.h
 * @brief Array operations for Aleam - NumPy-style random array generation
 * @license MIT
 * 
 * This file provides functions for generating arrays of true random numbers
 * in various shapes and distributions. These functions are the C++ equivalents
 * of NumPy's random array functions, but using true randomness from AleamCore.
 * 
 * All functions return std::vector<double> or nested vectors for multi-dimensional
 * arrays. For Python bindings, these will be converted to NumPy arrays.
 * 
 * Available functions:
 * - random_array: Uniform [0, 1) floats
 * - randn_array: Normal (Gaussian) distribution
 * - randint_array: Uniform integers
 * - choice_array: Random sampling from a population
 */

#ifndef ALEAM_ARRAYS_H
#define ALEAM_ARRAYS_H

#include <vector>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include "../core/aleam_core.h"

namespace aleam {
namespace arrays {

/* ============================================================================
 * Type Definitions
 * ============================================================================ */

/**
 * @brief Shape of a multi-dimensional array
 * 
 * Represented as a vector of dimensions. For example:
 * - shape = {10}      -> 1D array with 10 elements
 * - shape = {5, 5}    -> 2D array (5x5 matrix)
 * - shape = {3, 4, 5} -> 3D array
 */
using Shape = std::vector<size_t>;

/* ============================================================================
 * Core Array Generation Helpers
 * ============================================================================ */

/**
 * @brief Calculate total number of elements from shape
 * 
 * @param shape Vector of dimensions
 * @return size_t Total number of elements (product of all dimensions)
 */
size_t total_elements(const Shape& shape);

/**
 * @brief Flatten a multi-dimensional index to linear index
 * 
 * @param shape Shape of the array
 * @param indices Multi-dimensional index
 * @return size_t Linear index
 */
size_t flatten_index(const Shape& shape, const std::vector<size_t>& indices);

/**
 * @brief Convert linear index to multi-dimensional indices
 * 
 * @param shape Shape of the array
 * @param linear Linear index
 * @return std::vector<size_t> Multi-dimensional indices
 */
std::vector<size_t> unravel_index(const Shape& shape, size_t linear);

/* ============================================================================
 * Uniform Distribution Arrays
 * ============================================================================ */

/**
 * @brief Generate 1D array of true random floats in [0, 1)
 * 
 * @param rng Reference to AleamCore instance
 * @param size Number of elements
 * @return std::vector<double> Array of random floats
 */
std::vector<double> random_array_1d(AleamCore& rng, size_t size);

/**
 * @brief Generate 2D array of true random floats in [0, 1)
 * 
 * @param rng Reference to AleamCore instance
 * @param rows Number of rows
 * @param cols Number of columns
 * @return std::vector<std::vector<double>> 2D array of random floats
 */
std::vector<std::vector<double>> random_array_2d(AleamCore& rng, size_t rows, size_t cols);

/**
 * @brief Generate N-dimensional array of true random floats in [0, 1)
 * 
 * @param rng Reference to AleamCore instance
 * @param shape Shape of the array
 * @return std::vector<double> Flattened array (use with shape for indexing)
 */
std::vector<double> random_array_nd(AleamCore& rng, const Shape& shape);

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
std::vector<double> randn_array_1d(AleamCore& rng, size_t size, double mu = 0.0, double sigma = 1.0);

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
                                                   double mu = 0.0, double sigma = 1.0);

/**
 * @brief Generate N-dimensional array of true random normally distributed values
 * 
 * @param rng Reference to AleamCore instance
 * @param shape Shape of the array
 * @param mu Mean of distribution
 * @param sigma Standard deviation
 * @return std::vector<double> Flattened array (use with shape for indexing)
 */
std::vector<double> randn_array_nd(AleamCore& rng, const Shape& shape,
                                     double mu = 0.0, double sigma = 1.0);

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
std::vector<int64_t> randint_array_1d(AleamCore& rng, size_t size, int64_t low, int64_t high);

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
                                                     int64_t low, int64_t high);

/**
 * @brief Generate N-dimensional array of true random integers in [low, high]
 * 
 * @param rng Reference to AleamCore instance
 * @param shape Shape of the array
 * @param low Lower bound (inclusive)
 * @param high Upper bound (inclusive)
 * @return std::vector<int64_t> Flattened array (use with shape for indexing)
 */
std::vector<int64_t> randint_array_nd(AleamCore& rng, const Shape& shape,
                                        int64_t low, int64_t high);

/* ============================================================================
 * Choice Arrays (Sampling)
 * ============================================================================ */

/**
 * @brief Sample random elements from a population with replacement
 * 
 * @tparam T Type of elements in population
 * @param rng Reference to AleamCore instance
 * @param population Vector of elements to sample from
 * @param size Number of samples to draw
 * @return std::vector<T> Vector of sampled elements
 */
template<typename T>
std::vector<T> choice_with_replacement(AleamCore& rng, const std::vector<T>& population, size_t size) {
    if (population.empty()) {
        throw std::invalid_argument("population cannot be empty");
    }
    
    std::vector<T> result;
    result.reserve(size);
    size_t n = population.size();
    
    for (size_t i = 0; i < size; i++) {
        size_t idx = static_cast<size_t>(rng.random() * n);
        if (idx >= n) idx = n - 1;  // Handle floating point edge case
        result.push_back(population[idx]);
    }
    
    return result;
}

/**
 * @brief Sample random elements from a population without replacement
 * 
 * Uses Fisher-Yates shuffle on a copy of the population.
 * 
 * @tparam T Type of elements in population
 * @param rng Reference to AleamCore instance
 * @param population Vector of elements to sample from
 * @param size Number of samples to draw (must be <= population.size())
 * @return std::vector<T> Vector of sampled elements
 */
template<typename T>
std::vector<T> choice_without_replacement(AleamCore& rng, const std::vector<T>& population, size_t size) {
    if (population.empty()) {
        throw std::invalid_argument("population cannot be empty");
    }
    if (size > population.size()) {
        throw std::invalid_argument("sample size cannot exceed population size when replace=false");
    }
    
    /* Copy population and shuffle the first 'size' elements */
    std::vector<T> result = population;
    
    for (size_t i = 0; i < size; i++) {
        size_t j = i + static_cast<size_t>(rng.random() * (result.size() - i));
        if (j >= result.size()) j = result.size() - 1;
        std::swap(result[i], result[j]);
    }
    
    /* Return first 'size' elements */
    result.resize(size);
    return result;
}

/**
 * @brief Sample random elements from a population with optional weights
 * 
 * @tparam T Type of elements in population
 * @param rng Reference to AleamCore instance
 * @param population Vector of elements to sample from
 * @param size Number of samples to draw
 * @param weights Probability weights for each element (must sum to 1)
 * @param replace Whether to sample with replacement
 * @return std::vector<T> Vector of sampled elements
 */
template<typename T>
std::vector<T> choice_weighted(AleamCore& rng, std::vector<T> population,
                                 size_t size, std::vector<double> weights,
                                 bool replace = true) {
    if (population.empty()) {
        throw std::invalid_argument("population cannot be empty");
    }
    if (weights.size() != population.size()) {
        throw std::invalid_argument("weights must have same size as population");
    }
    
    /* Build cumulative distribution */
    std::vector<double> cumsum;
    cumsum.reserve(weights.size());
    double running = 0.0;
    for (double w : weights) {
        running += w;
        cumsum.push_back(running);
    }
    
    /* Normalize if sum is not exactly 1 */
    if (std::abs(running - 1.0) > 1e-10) {
        for (double& c : cumsum) {
            c /= running;
        }
    }
    
    std::vector<T> result;
    result.reserve(size);
    
    for (size_t i = 0; i < size; i++) {
        double u = rng.random();
        
        /* Binary search in cumulative distribution */
        size_t idx = 0;
        while (idx < cumsum.size() && u > cumsum[idx]) {
            idx++;
        }
        if (idx >= population.size()) {
            idx = population.size() - 1;
        }
        
        result.push_back(population[idx]);
        
        /* If sampling without replacement, remove the chosen element */
        if (!replace) {
            population.erase(population.begin() + idx);
            weights.erase(weights.begin() + idx);
            /* Rebuild cumulative distribution */
            cumsum.clear();
            running = 0.0;
            for (double w : weights) {
                running += w;
                cumsum.push_back(running);
            }
            if (std::abs(running - 1.0) > 1e-10) {
                for (double& c : cumsum) {
                    c /= running;
                }
            }
        }
    }
    
    return result;
}

/* ============================================================================
 * Shuffle Array (In-Place)
 * ============================================================================ */

/**
 * @brief Shuffle a vector in-place using Fisher-Yates algorithm
 * 
 * @tparam T Type of elements in vector
 * @param rng Reference to AleamCore instance
 * @param vec Vector to shuffle
 */
template<typename T>
void shuffle(AleamCore& rng, std::vector<T>& vec) {
    size_t n = vec.size();
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = static_cast<size_t>(rng.random() * (i + 1));
        if (j > i) j = i;
        std::swap(vec[i], vec[j]);
    }
}

}  // namespace arrays
}  // namespace aleam

#endif /* ALEAM_ARRAYS_H */