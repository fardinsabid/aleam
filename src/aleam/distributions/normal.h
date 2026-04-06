/**
 * @file normal.h
 * @brief Normal (Gaussian) distribution using Aleam true randomness
 * @license MIT
 * 
 * Implements the normal distribution using the Box-Muller transform.
 * The Box-Muller method converts two uniform random variables into
 * two independent standard normal variables.
 */

#ifndef ALEAM_DISTRIBUTIONS_NORMAL_H
#define ALEAM_DISTRIBUTIONS_NORMAL_H

#include <cmath>
#include <stdexcept>
#include "../core/aleam_core.h"

namespace aleam {
namespace distributions {

/**
 * @brief Normal (Gaussian) distribution class
 * 
 * Represents a normal distribution with mean μ and standard deviation σ.
 * 
 * Probability density function:
 *     f(x) = 1/(σ√(2π)) * exp(-(x-μ)²/(2σ²))
 * 
 * @tparam RealType Floating point type (float or double)
 */
template<typename RealType = double>
class NormalDistribution {
public:
    /**
     * @brief Construct a normal distribution
     * 
     * @param mu Mean (μ) of the distribution
     * @param sigma Standard deviation (σ) (> 0)
     * @param rng Reference to AleamCore instance
     */
    NormalDistribution(RealType mu = 0.0, RealType sigma = 1.0, AleamCore& rng)
        : m_mu(mu)
        , m_sigma(sigma)
        , m_rng(rng)
        , m_has_spare(false)
        , m_spare(0.0) {
        if (sigma <= 0.0) {
            throw std::invalid_argument("sigma must be > 0");
        }
    }
    
    /**
     * @brief Generate a random number from the distribution
     * 
     * Uses the Box-Muller transform which generates two normal
     * variates at once. The second is cached for next call.
     * 
     * @return RealType Random value from N(μ, σ²)
     */
    RealType operator()() {
        if (m_has_spare) {
            m_has_spare = false;
            return m_mu + m_sigma * m_spare;
        }
        
        // Generate two uniform random numbers
        RealType u1 = static_cast<RealType>(m_rng.random());
        RealType u2 = static_cast<RealType>(m_rng.random());
        
        // Box-Muller transform
        RealType r = std::sqrt(-2.0 * std::log(u1));
        RealType theta = 2.0 * M_PI * u2;
        
        RealType z0 = r * std::cos(theta);
        RealType z1 = r * std::sin(theta);
        
        // Cache z1 for next call
        m_spare = z1;
        m_has_spare = true;
        
        return m_mu + m_sigma * z0;
    }
    
    /**
     * @brief Get the mean (μ) of the distribution
     * 
     * @return RealType Mean
     */
    RealType mean() const { return m_mu; }
    
    /**
     * @brief Get the standard deviation (σ) of the distribution
     * 
     * @return RealType Standard deviation
     */
    RealType standard_deviation() const { return m_sigma; }
    
    /**
     * @brief Get the variance (σ²) of the distribution
     * 
     * @return RealType Variance
     */
    RealType variance() const { return m_sigma * m_sigma; }
    
    /**
     * @brief Probability density function
     * 
     * @param x Point at which to evaluate PDF
     * @return RealType PDF value at x
     */
    RealType pdf(RealType x) const {
        RealType diff = x - m_mu;
        RealType exponent = -0.5 * (diff * diff) / (m_sigma * m_sigma);
        return std::exp(exponent) / (m_sigma * std::sqrt(2.0 * M_PI));
    }
    
    /**
     * @brief Cumulative distribution function
     * 
     * @param x Point at which to evaluate CDF
     * @return RealType CDF value at x
     */
    RealType cdf(RealType x) const {
        RealType z = (x - m_mu) / m_sigma;
        return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
    }
    
    /**
     * @brief Reset the cached spare value
     */
    void reset() {
        m_has_spare = false;
        m_spare = 0.0;
    }
    
private:
    RealType m_mu;           /**< Mean */
    RealType m_sigma;        /**< Standard deviation */
    AleamCore& m_rng;        /**< Random number generator */
    bool m_has_spare;        /**< Whether a spare value is cached */
    RealType m_spare;        /**< Cached spare value */
};

/**
 * @brief Generate a standard normal random number (μ=0, σ=1)
 * 
 * @param rng Reference to AleamCore instance
 * @return double Random number from N(0, 1)
 */
inline double standard_normal(AleamCore& rng) {
    double u1 = rng.random();
    double u2 = rng.random();
    double r = std::sqrt(-2.0 * std::log(u1));
    double theta = 2.0 * M_PI * u2;
    return r * std::cos(theta);
}

}  // namespace distributions
}  // namespace aleam

#endif /* ALEAM_DISTRIBUTIONS_NORMAL_H */