/**
 * @file weibull.h
 * @brief Weibull distribution using Aleam true randomness
 * @license MIT
 * 
 * Implements the Weibull distribution using inverse transform sampling.
 * The Weibull distribution is widely used in reliability engineering
 * and survival analysis.
 */

#ifndef ALEAM_DISTRIBUTIONS_WEIBULL_H
#define ALEAM_DISTRIBUTIONS_WEIBULL_H

#include <cmath>
#include <stdexcept>
#include "../core/aleam_core.h"

namespace aleam {
namespace distributions {

/**
 * @brief Weibull distribution class
 * 
 * Represents a Weibull distribution with shape parameter k and scale parameter λ.
 * 
 * Probability density function:
 *     f(x) = (k/λ) * (x/λ)^(k-1) * e^{-(x/λ)^k} for x ≥ 0
 * 
 * Mean = λ * Γ(1 + 1/k)
 * Variance = λ² * [Γ(1 + 2/k) - (Γ(1 + 1/k))²]
 * 
 * @tparam RealType Floating point type (float or double)
 */
template<typename RealType = double>
class WeibullDistribution {
public:
    /**
     * @brief Construct a Weibull distribution
     * 
     * @param shape Shape parameter (k) (> 0)
     * @param scale Scale parameter (λ) (> 0)
     * @param rng Reference to AleamCore instance
     */
    WeibullDistribution(RealType shape, RealType scale = 1.0, AleamCore& rng)
        : m_shape(shape)
        , m_scale(scale)
        , m_rng(rng) {
        if (shape <= 0.0) {
            throw std::invalid_argument("shape must be > 0");
        }
        if (scale <= 0.0) {
            throw std::invalid_argument("scale must be > 0");
        }
    }
    
    /**
     * @brief Generate a random number from the distribution
     * 
     * Uses inverse transform sampling:
     *     F⁻¹(u) = λ * (-ln(1-u))^(1/k)
     * 
     * @return RealType Random value from Weibull(k, λ)
     */
    RealType operator()() {
        RealType u = static_cast<RealType>(m_rng.random());
        
        // Use 1-u to avoid log(0)
        RealType val = 1.0 - u;
        if (val <= 0.0) val = 1e-15;
        
        return m_scale * std::pow(-std::log(val), 1.0 / m_shape);
    }
    
    /**
     * @brief Get the shape parameter (k)
     * 
     * @return RealType Shape parameter
     */
    RealType shape() const { return m_shape; }
    
    /**
     * @brief Get the scale parameter (λ)
     * 
     * @return RealType Scale parameter
     */
    RealType scale() const { return m_scale; }
    
    /**
     * @brief Get the mean of the distribution
     * 
     * @return RealType Mean = λ * Γ(1 + 1/k)
     */
    RealType mean() const {
        return m_scale * std::tgamma(1.0 + 1.0 / m_shape);
    }
    
    /**
     * @brief Get the variance of the distribution
     * 
     * @return RealType Variance = λ² * [Γ(1 + 2/k) - (Γ(1 + 1/k))²]
     */
    RealType variance() const {
        RealType gamma1 = std::tgamma(1.0 + 1.0 / m_shape);
        RealType gamma2 = std::tgamma(1.0 + 2.0 / m_shape);
        return m_scale * m_scale * (gamma2 - gamma1 * gamma1);
    }
    
    /**
     * @brief Get the median of the distribution
     * 
     * @return RealType Median = λ * (ln 2)^(1/k)
     */
    RealType median() const {
        return m_scale * std::pow(std::log(2.0), 1.0 / m_shape);
    }
    
    /**
     * @brief Probability density function
     * 
     * @param x Point at which to evaluate PDF (x ≥ 0)
     * @return RealType PDF value at x
     */
    RealType pdf(RealType x) const {
        if (x < 0.0) return 0.0;
        
        RealType z = x / m_scale;
        RealType z_pow = std::pow(z, m_shape - 1.0);
        RealType exp_term = std::exp(-std::pow(z, m_shape));
        
        return (m_shape / m_scale) * z_pow * exp_term;
    }
    
    /**
     * @brief Cumulative distribution function
     * 
     * @param x Point at which to evaluate CDF (x ≥ 0)
     * @return RealType CDF value at x
     */
    RealType cdf(RealType x) const {
        if (x < 0.0) return 0.0;
        return 1.0 - std::exp(-std::pow(x / m_scale, m_shape));
    }
    
    /**
     * @brief Hazard (failure rate) function
     * 
     * @param x Point at which to evaluate hazard (x ≥ 0)
     * @return RealType Hazard rate at x
     */
    RealType hazard(RealType x) const {
        if (x < 0.0) return 0.0;
        return (m_shape / m_scale) * std::pow(x / m_scale, m_shape - 1.0);
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
        
        if (p == 0.0) return 0.0;
        if (p == 1.0) return std::numeric_limits<RealType>::infinity();
        
        return m_scale * std::pow(-std::log(1.0 - p), 1.0 / m_shape);
    }
    
private:
    RealType m_shape;        /**< Shape parameter (k) */
    RealType m_scale;        /**< Scale parameter (λ) */
    AleamCore& m_rng;        /**< Random number generator */
};

/**
 * @brief Generate a Weibull distributed random number
 * 
 * @param rng Reference to AleamCore instance
 * @param shape Shape parameter (k > 0)
 * @param scale Scale parameter (λ > 0)
 * @return double Random value from Weibull(k, λ)
 */
inline double weibull(AleamCore& rng, double shape, double scale) {
    WeibullDistribution<double> dist(shape, scale, rng);
    return dist();
}

}  // namespace distributions
}  // namespace aleam

#endif /* ALEAM_DISTRIBUTIONS_WEIBULL_H */