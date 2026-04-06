/**
 * @file polars_integration.h
 * @brief Polars integration for Aleam true randomness
 * @license MIT
 * 
 * Provides true random data generation for Polars DataFrames.
 * Polars is a fast DataFrame library written in Rust.
 * 
 * Example:
 * @code
 *   #include "aleam/integrations/polars_integration.h"
 *   
 *   aleam::AleamCore rng;
 *   aleam::integrations::PolarsGenerator gen(rng);
 *   auto series = gen.series(100, "normal");
 *   auto indices = gen.shuffle_indices(1000);
 * @endcode
 */

#ifndef ALEAM_INTEGRATIONS_POLARS_INTEGRATION_H
#define ALEAM_INTEGRATIONS_POLARS_INTEGRATION_H

#include <vector>
#include <string>
#include <cstdint>
#include "../core/aleam_core.h"

namespace aleam {
namespace integrations {

/**
 * @brief Polars-compatible random generator using true randomness
 * 
 * Provides true random data generation for Polars DataFrames.
 * Returns data as vectors that can be easily converted to Polars Series.
 */
class PolarsGenerator {
public:
    /**
     * @brief Construct a PolarsGenerator
     * 
     * @param rng Reference to AleamCore instance
     */
    explicit PolarsGenerator(AleamCore& rng);
    
    /**
     * @brief Generate a random Series (float64)
     * 
     * @param n Number of elements
     * @param distribution Distribution name ("uniform", "normal", "exponential")
     * @param params Distribution parameters
     * @return std::vector<double> Random values
     */
    std::vector<double> series_f64(size_t n, 
                                    const std::string& distribution = "uniform",
                                    const std::string& params = "");
    
    /**
     * @brief Generate a random Series (int64)
     * 
     * @param n Number of elements
     * @param low Lower bound (inclusive)
     * @param high Upper bound (exclusive)
     * @return std::vector<int64_t> Random integers
     */
    std::vector<int64_t> series_i64(size_t n, int64_t low, int64_t high);
    
    /**
     * @brief Generate a random Series (boolean)
     * 
     * @param n Number of elements
     * @param p Probability of true
     * @return std::vector<bool> Random booleans
     */
    std::vector<bool> series_bool(size_t n, double p = 0.5);
    
    /**
     * @brief Generate shuffled indices for DataFrame rows
     * 
     * @param n Number of rows
     * @return std::vector<size_t> Shuffled indices
     */
    std::vector<size_t> shuffle_indices(size_t n);
    
    /**
     * @brief Generate random DataFrame (float64)
     * 
     * @param rows Number of rows
     * @param columns Number of columns
     * @param distribution Distribution name
     * @return std::vector<std::vector<double>> 2D data
     */
    std::vector<std::vector<double>> dataframe_f64(size_t rows, size_t columns,
                                                    const std::string& distribution = "uniform");
    
    /**
     * @brief Generate random integer DataFrame
     * 
     * @param rows Number of rows
     * @param columns Number of columns
     * @param low Lower bound
     * @param high Upper bound
     * @return std::vector<std::vector<int64_t>> 2D integer data
     */
    std::vector<std::vector<int64_t>> dataframe_i64(size_t rows, size_t columns,
                                                     int64_t low, int64_t high);
    
private:
    /**
     * @brief Parse parameters string
     * 
     * @param params Parameters string
     * @return std::unordered_map<std::string, double> Parsed parameters
     */
    std::unordered_map<std::string, double> parse_params(const std::string& params);
    
    /**
     * @brief Generate uniform distribution
     * 
     * @param n Number of elements
     * @param low Lower bound
     * @param high Upper bound
     * @return std::vector<double> Uniform values
     */
    std::vector<double> generate_uniform(size_t n, double low, double high);
    
    /**
     * @brief Generate normal distribution
     * 
     * @param n Number of elements
     * @param mu Mean
     * @param sigma Standard deviation
     * @return std::vector<double> Normal values
     */
    std::vector<double> generate_normal(size_t n, double mu, double sigma);
    
    /**
     * @brief Generate exponential distribution
     * 
     * @param n Number of elements
     * @param rate Rate parameter
     * @return std::vector<double> Exponential values
     */
    std::vector<double> generate_exponential(size_t n, double rate);
    
    AleamCore& m_rng;        /**< Random number generator */
};

}  // namespace integrations
}  // namespace aleam

#endif /* ALEAM_INTEGRATIONS_POLARS_INTEGRATION_H */