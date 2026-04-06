/**
 * @file xarray_integration.h
 * @brief Xarray integration for Aleam true randomness
 * @license MIT
 * 
 * Provides true random data generation for Xarray DataArrays and Datasets.
 * Xarray is a Python library for labeled multi-dimensional arrays.
 * 
 * Example:
 * @code
 *   #include "aleam/integrations/xarray_integration.h"
 *   
 *   aleam::AleamCore rng;
 *   aleam::integrations::XarrayGenerator gen(rng);
 *   auto data = gen.dataarray({100, 100}, {"x", "y"});
 * @endcode
 */

#ifndef ALEAM_INTEGRATIONS_XARRAY_INTEGRATION_H
#define ALEAM_INTEGRATIONS_XARRAY_INTEGRATION_H

#include <vector>
#include <string>
#include <cstdint>
#include "../core/aleam_core.h"

namespace aleam {
namespace integrations {

/**
 * @brief Xarray-compatible random generator using true randomness
 * 
 * Provides true random data generation for Xarray DataArrays and Datasets.
 * Returns data as vectors that can be easily converted to Xarray objects.
 */
class XarrayGenerator {
public:
    /**
     * @brief Construct an XarrayGenerator
     * 
     * @param rng Reference to AleamCore instance
     */
    explicit XarrayGenerator(AleamCore& rng);
    
    /**
     * @brief Generate a random DataArray (flattened)
     * 
     * @param shape Vector of dimensions
     * @param dims Dimension names
     * @param distribution Distribution name ("uniform", "normal")
     * @param params Distribution parameters
     * @return std::vector<double> Flattened data (row-major order)
     */
    std::vector<double> dataarray(const std::vector<size_t>& shape,
                                   const std::vector<std::string>& dims,
                                   const std::string& distribution = "uniform",
                                   const std::string& params = "");
    
    /**
     * @brief Generate a random DataArray with integer data
     * 
     * @param shape Vector of dimensions
     * @param dims Dimension names
     * @param low Lower bound (inclusive)
     * @param high Upper bound (exclusive)
     * @return std::vector<int64_t> Flattened integer data
     */
    std::vector<int64_t> dataarray_int(const std::vector<size_t>& shape,
                                        const std::vector<std::string>& dims,
                                        int64_t low, int64_t high);
    
    /**
     * @brief Generate a random DataArray with boolean data
     * 
     * @param shape Vector of dimensions
     * @param dims Dimension names
     * @param p Probability of true
     * @return std::vector<uint8_t> Flattened boolean data (0/1)
     */
    std::vector<uint8_t> dataarray_bool(const std::vector<size_t>& shape,
                                         const std::vector<std::string>& dims,
                                         double p = 0.5);
    
    /**
     * @brief Generate random coordinates for a dimension
     * 
     * @param size Number of coordinates
     * @param coord_type Type ("linear", "uniform", "normal")
     * @param params Parameters
     * @return std::vector<double> Coordinate values
     */
    std::vector<double> coordinates(size_t size,
                                     const std::string& coord_type = "linear",
                                     const std::string& params = "");
    
    /**
     * @brief Calculate total elements from shape
     * 
     * @param shape Vector of dimensions
     * @return size_t Total number of elements
     */
    static size_t total_elements(const std::vector<size_t>& shape);
    
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
    
    AleamCore& m_rng;        /**< Random number generator */
};

}  // namespace integrations
}  // namespace aleam

#endif /* ALEAM_INTEGRATIONS_XARRAY_INTEGRATION_H */