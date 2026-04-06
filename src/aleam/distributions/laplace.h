/**
 * @file laplace.h
 * @brief Laplace (Double Exponential) distribution using Aleam true randomness
 * @license MIT
 * 
 * Implements the Laplace distribution using inverse transform sampling.
 * The Laplace distribution is symmetric and has heavier tails than the
 * normal distribution.
 */

#ifndef ALEAM_DISTRIBUTIONS_LAPLACE_H
#define ALEAM_DISTRIBUTIONS_LAPLACE_H

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include "../core/aleam_core.h"

namespace aleam {
namespace distributions {

/**
 * @brief Laplace (Double Exponential) distribution class
 * 
 * Represents a Laplace distribution with location parameter μ and scale parameter b.
 * 
 * Probability density function:
 *     f(x) = 1/(2b) * exp(-|x-μ|/b)
 * 
 * Mean = μ
 * Variance = 2b²
 * 
 * @tparam RealType Floating point type (float or double)
 */
template<typename RealType = double>
class LaplaceDistribution {
public:
    /**
     * @brief Construct a Laplace distribution
     * 
     * @param loc Location parameter (μ)
     * @param scale Scale parameter (b) (> 0)
     * @param rng Reference to AleamCore instance
     */
    LaplaceDistribution(RealType loc = 0.0, RealType scale = 1.0, AleamCore& rng)
        : m_loc(loc)
        , m_scale(scale)
        , m_rng(rng) {
        if (scale <= 0.0) {
            throw std::invalid_argument("scale must be > 0");
        }
    }
    
    /**
     * @brief Generate a random number from the distribution
     * 
     * Uses inverse transform sampling:
     *     F⁻¹(u) = μ - b·sign(u-0.5)·ln(1 - 2|u-0.5|)
     * 
     * @return RealType Random value from Laplace(μ, b)
     */
    RealType operator()() {
        RealType u = static_cast<RealType>(m_rng.random());
        RealType u_centered = u - 0.5;
        
        // sign(u-0.5) * ln(1 - 2|u-0.5|)
        RealType sign = (u_centered > 0.0) ? 1.0 : -1.0;
        RealType abs_u = std::abs(u_centered);
        
        // Avoid log(0) by clamping
        RealType val = 1.0 - 2.0 * abs_u;
        if (val <= 0.0) val = 1e-15;
        
        return m_loc - m_scale * sign * std::log(val);
    }
    
    /**
     * @brief Get the location parameter (μ)
     * 
     * @return RealType Location parameter
     */
    RealType location() const { return m_loc; }
    
    /**
     * @brief Get the scale parameter (b)
     * 
     * @return RealType Scale parameter
     */
    RealType scale() const { return m_scale; }
    
    /**
     * @brief Get the mean of the distribution (μ)
     * 
     * @return RealType Mean
     */
    RealType mean() const { return m_loc; }
    
    /**
     * @brief Get the variance of the distribution (2b²)
     * 
     * @return RealType Variance
     */
    RealType variance() const { return 2.0 * m_scale * m_scale; }
    
    /**
     * @brief Get the standard deviation (√(2)·b)
     * 
     * @return RealType Standard deviation
     */
    RealType standard_deviation() const {
        return std::sqrt(2.0) * m_scale;
    }
    
    /**
     * @brief Probability density function
     * 
     * @param x Point at which to evaluate PDF
     * @return RealType PDF value at x
     */
    RealType pdf(RealType x) const {
        RealType diff = std::abs(x - m_loc);
        return std::exp(-diff / m_scale) / (2.0 * m_scale);
    }
    
    /**
     * @brief Cumulative distribution function
     * 
     * @param x Point at which to evaluate CDF
     * @return RealType CDF value at x
     */
    RealType cdf(RealType x) const {
        RealType z = (x - m_loc) / m_scale;
        if (z <= 0.0) {
            return 0.5 * std::exp(z);
        } else {
            return 1.0 - 0.5 * std::exp(-z);
        }
    }
    
    /**
     * @brief Quantile function (inverse CDF)
     * 
     * @param p Probability (0 ≤ p ≤ 1)
     * @return RealType Value x such that CDF(x) = p
     */
    RealType quantile(RealType p) const {
        if (p < 0.0 || p > 1.0) {
            throw std::invalid_argument("p must be between 0 and 1");
        }
        
        if (p < 0.5) {
            return m_loc + m_scale * std::log(2.0 * p);
        } else {
            return m_loc - m_scale * std::log(2.0 * (1.0 - p));
        }
    }
    
private:
    RealType m_loc;          /**< Location parameter (μ) */
    RealType m_scale;        /**< Scale parameter (b) */
    AleamCore& m_rng;        /**< Random number generator */
};

/**
 * @brief Generate a Laplace distributed random number
 * 
 * @param rng Reference to AleamCore instance
 * @param loc Location parameter (μ)
 * @param scale Scale parameter (b > 0)
 * @return double Random value from Laplace(μ, b)
 */
inline double laplace(AleamCore& rng, double loc, double scale) {
    LaplaceDistribution<double> dist(loc, scale, rng);
    return dist();
}

}  // namespace distributions
}  // namespace aleam

#endif /* ALEAM_DISTRIBUTIONS_LAPLACE_H */