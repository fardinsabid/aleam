/**
 * @file exponential.h
 * @brief Exponential distribution using Aleam true randomness
 * @license MIT
 * 
 * Implements the exponential distribution using inverse transform sampling.
 * The exponential distribution models waiting times between events in
 * a Poisson process.
 */

#ifndef ALEAM_DISTRIBUTIONS_EXPONENTIAL_H
#define ALEAM_DISTRIBUTIONS_EXPONENTIAL_H

#include <cmath>
#include <stdexcept>
#include "../core/aleam_core.h"

namespace aleam {
namespace distributions {

/**
 * @brief Exponential distribution class
 * 
 * Represents an exponential distribution with rate parameter λ.
 * 
 * Probability density function:
 *     f(x) = λ·e^{-λ·x} for x ≥ 0
 * 
 * Mean = 1/λ
 * Variance = 1/λ²
 * 
 * @tparam RealType Floating point type (float or double)
 */
template<typename RealType = double>
class ExponentialDistribution {
public:
    /**
     * @brief Construct an exponential distribution
     * 
     * @param rate Rate parameter (λ) (> 0)
     * @param rng Reference to AleamCore instance
     */
    ExponentialDistribution(RealType rate = 1.0, AleamCore& rng)
        : m_rate(rate)
        , m_rng(rng) {
        if (rate <= 0.0) {
            throw std::invalid_argument("rate must be > 0");
        }
    }
    
    /**
     * @brief Generate a random number from the distribution
     * 
     * Uses inverse transform sampling:
     *     F⁻¹(u) = -ln(1 - u) / λ
     * 
     * Using 1-u instead of u ensures numerical stability near 0.
     * 
     * @return RealType Random value from Exp(λ)
     */
    RealType operator()() {
        RealType u = static_cast<RealType>(m_rng.random());
        // Use 1-u to avoid log(0) when u = 0
        return -std::log(1.0 - u) / m_rate;
    }
    
    /**
     * @brief Get the rate parameter (λ)
     * 
     * @return RealType Rate parameter
     */
    RealType rate() const { return m_rate; }
    
    /**
     * @brief Get the mean of the distribution (1/λ)
     * 
     * @return RealType Mean
     */
    RealType mean() const { return 1.0 / m_rate; }
    
    /**
     * @brief Get the variance of the distribution (1/λ²)
     * 
     * @return RealType Variance
     */
    RealType variance() const { return 1.0 / (m_rate * m_rate); }
    
    /**
     * @brief Probability density function
     * 
     * @param x Point at which to evaluate PDF (x ≥ 0)
     * @return RealType PDF value at x
     */
    RealType pdf(RealType x) const {
        if (x < 0.0) return 0.0;
        return m_rate * std::exp(-m_rate * x);
    }
    
    /**
     * @brief Cumulative distribution function
     * 
     * @param x Point at which to evaluate CDF (x ≥ 0)
     * @return RealType CDF value at x
     */
    RealType cdf(RealType x) const {
        if (x < 0.0) return 0.0;
        return 1.0 - std::exp(-m_rate * x);
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
        return -std::log(1.0 - p) / m_rate;
    }
    
private:
    RealType m_rate;         /**< Rate parameter (λ) */
    AleamCore& m_rng;        /**< Random number generator */
};

/**
 * @brief Generate an exponentially distributed random number
 * 
 * @param rng Reference to AleamCore instance
 * @param rate Rate parameter (λ > 0)
 * @return double Random value from Exp(λ)
 */
inline double exponential(AleamCore& rng, double rate) {
    ExponentialDistribution<double> dist(rate, rng);
    return dist();
}

}  // namespace distributions
}  // namespace aleam

#endif /* ALEAM_DISTRIBUTIONS_EXPONENTIAL_H */