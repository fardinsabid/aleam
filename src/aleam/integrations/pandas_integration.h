/**
 * @file pandas_integration.h
 * @brief Pandas integration for Aleam true randomness
 * @license MIT
 * 
 * Provides true random data generation for Pandas DataFrames and Series.
 * 
 * Example:
 * @code
 *   #include "aleam/integrations/pandas_integration.h"
 *   
 *   aleam::AleamCore rng;
 *   aleam::integrations::PandasGenerator gen(rng);
 *   auto series = gen.series(100, "normal");
 *   auto indices = gen.shuffle_indices(1000);
 * @endcode
 */

#ifndef ALEAM_INTEGRATIONS_PANDAS_INTEGRATION_H
#define ALEAM_INTEGRATIONS_PANDAS_INTEGRATION_H

#include <vector>
#include <string>
#include <cstdint>
#include "../core/aleam_core.h"

namespace aleam {
namespace integrations {

/**
 * @brief Pandas-compatible random generator using true randomness
 * 
 * Provides true random data generation for Pandas DataFrames and Series.
 * Returns data as vectors that can be easily converted to Pandas objects.
 */
class PandasGenerator {
public:
    /**
     * @brief Construct a PandasGenerator
     * 
     * @param rng Reference to AleamCore instance
     */
    explicit PandasGenerator(AleamCore& rng);
    
    /**
     * @brief Generate a random Series
     * 
     * @param n Number of elements
     * @param distribution Distribution name ("uniform", "normal", "exponential", "poisson", "binomial", "choice")
     * @param params Distribution parameters (encoded as string for flexibility)
     * @return std::vector<double> Random values
     */
    std::vector<double> series(size_t n, 
                                const std::string& distribution = "uniform",
                                const std::string& params = "");
    
    /**
     * @brief Generate a random integer Series
     * 
     * @param n Number of elements
     * @param low Lower bound (inclusive)
     * @param high Upper bound (exclusive)
     * @return std::vector<int64_t> Random integers
     */
    std::vector<int64_t> series_int(size_t n, int64_t low, int64_t high);
    
    /**
     * @brief Generate a random boolean Series (Bernoulli)
     * 
     * @param n Number of elements
     * @param p Probability of True (0 to 1)
     * @return std::vector<bool> Random booleans
     */
    std::vector<bool> series_bool(size_t n, double p = 0.5);
    
    /**
     * @brief Generate a random categorical Series
     * 
     * @param n Number of elements
     * @param categories Vector of category values
     * @param probabilities Optional probabilities (must sum to 1)
     * @return std::vector<size_t> Category indices
     */
    std::vector<size_t> series_categorical(size_t n,
                                            const std::vector<double>& categories,
                                            const std::vector<double>& probabilities = {});
    
    /**
     * @brief Generate shuffled indices for DataFrame rows
     * 
     * @param n Number of rows
     * @return std::vector<size_t> Shuffled indices
     */
    std::vector<size_t> shuffle_indices(size_t n);
    
    /**
     * @brief Generate random DataFrame column data
     * 
     * @param rows Number of rows
     * @param columns Number of columns
     * @param distribution Distribution name
     * @return std::vector<std::vector<double>> 2D data (rows x columns)
     */
    std::vector<std::vector<double>> dataframe(size_t rows, size_t columns,
                                                const std::string& distribution = "uniform");
    
    /**
     * @brief Generate random DataFrame with different distributions per column
     * 
     * @param rows Number of rows
     * @param col_specs Vector of (name, distribution, params) tuples
     * @return std::vector<std::vector<double>> 2D data (rows x columns)
     */
    std::vector<std::vector<double>> dataframe_mixed(size_t rows,
                                                       const std::vector<std::tuple<std::string, std::string, std::string>>& col_specs);
    
private:
    /**
     * @brief Parse parameters string (format: "key=value,key=value")
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
    
    /**
     * @brief Generate Poisson distribution
     * 
     * @param n Number of elements
     * @param lambda Mean parameter
     * @return std::vector<int64_t> Poisson values
     */
    std::vector<int64_t> generate_poisson(size_t n, double lambda);
    
    /**
     * @brief Generate binomial distribution
     * 
     * @param n Number of elements
     * @param trials Number of trials
     * @param p Probability of success
     * @return std::vector<int64_t> Binomial values
     */
    std::vector<int64_t> generate_binomial(size_t n, int trials, double p);
    
    AleamCore& m_rng;        /**< Random number generator */
};

}  // namespace integrations
}  // namespace aleam

#endif /* ALEAM_INTEGRATIONS_PANDAS_INTEGRATION_H */