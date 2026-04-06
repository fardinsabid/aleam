/**
 * @file lognormal.h
 * @brief Log-Normal distribution using Aleam true randomness
 * @license MIT
 * 
 * Implements the Log-Normal distribution using the relationship
 * between normal and log-normal distributions.
 * 
 * If X ~ N(μ, σ²), then Y = e^X ~ LogNormal(μ, σ²)
 */

#ifndef ALEAM_DISTRIBUTIONS_LOGNORMAL_H
#define ALEAM_DISTRIBUTIONS_LOGNORMAL_H

#include <cmath>
#include <stdexcept>
#include "../core/aleam_core.h"
#include "normal.h"

namespace aleam {
namespace distributions {

/**
 * @brief Log-Normal distribution class
 * 
 * Represents a Log-Normal distribution with parameters μ and σ.
 * 
 * If X ~ N(μ, σ²), then Y = e^X ~ LogNormal(μ, σ²)
 * 
 * Probability density function:
 *     f(x) = 1/(xσ√(2π)) * exp(-(ln x - μ)²/(2σ²)) for x > 0
 * 
 * Mean = e^{μ + σ²/2}
 * Variance = (e^{σ²} - 1) * e^{2μ + σ²}
 * 
 * @tparam RealType Floating point type (float or double)
 */
template<typename RealType = double>
class LogNormalDistribution {
public:
    /**
     * @brief Construct a Log-Normal distribution
     * 
     * @param mu Mean of the underlying normal distribution
     * @param sigma Standard deviation of the underlying normal distribution (> 0)
     * @param rng Reference to AleamCore instance
     */
    LogNormalDistribution(RealType mu = 0.0, RealType sigma = 1.0, AleamCore& rng)
        : m_mu(mu)
        , m_sigma(sigma)
        , m_rng(rng) {
        if (sigma <= 0.0) {
            throw std::invalid_argument("sigma must be > 0");
        }
    }
    
    /**
     * @brief Generate a random number from the distribution
     * 
     * Algorithm:
     *     X ~ N(μ, σ²)
     *     Y = e^X
     * 
     * @return RealType Random value from LogNormal(μ, σ²)
     */
    RealType operator()() {
        NormalDistribution<RealType> normal(m_mu, m_sigma, m_rng);
        RealType x = normal();
        return std::exp(x);
    }
    
    /**
     * @brief Get the mu parameter (mean of underlying normal)
     * 
     * @return RealType Mu parameter
     */
    RealType mu() const { return m_mu; }
    
    /**
     * @brief Get the sigma parameter (std dev of underlying normal)
     * 
     * @return RealType Sigma parameter
     */
    RealType sigma() const { return m_sigma; }
    
    /**
     * @brief Get the mean of the distribution
     * 
     * @return RealType Mean = e^{μ + σ²/2}
     */
    RealType mean() const {
        return std::exp(m_mu + m_sigma * m_sigma / 2.0);
    }
    
    /**
     * @brief Get the variance of the distribution
     * 
     * @return RealType Variance = (e^{σ²} - 1) * e^{2μ + σ²}
     */
    RealType variance() const {
        RealType exp_sigma_sq = std::exp(m_sigma * m_sigma);
        return (exp_sigma_sq - 1.0) * std::exp(2.0 * m_mu + m_sigma * m_sigma);
    }
    
    /**
     * @brief Get the median of the distribution
     * 
     * @return RealType Median = e^{μ}
     */
    RealType median() const {
        return std::exp(m_mu);
    }
    
    /**
     * @brief Get the mode of the distribution
     * 
     * @return RealType Mode = e^{μ - σ²}
     */
    RealType mode() const {
        return std::exp(m_mu - m_sigma * m_sigma);
    }
    
    /**
     * @brief Probability density function
     * 
     * @param x Point at which to evaluate PDF (x > 0)
     * @return RealType PDF value at x
     */
    RealType pdf(RealType x) const {
        if (x <= 0.0) return 0.0;
        
        RealType ln_x = std::log(x);
        RealType diff = ln_x - m_mu;
        RealType exponent = -0.5 * (diff * diff) / (m_sigma * m_sigma);
        
        return std::exp(exponent) / (x * m_sigma * std::sqrt(2.0 * M_PI));
    }
    
    /**
     * @brief Cumulative distribution function
     * 
     * @param x Point at which to evaluate CDF (x > 0)
     * @return RealType CDF value at x
     */
    RealType cdf(RealType x) const {
        if (x <= 0.0) return 0.0;
        
        RealType ln_x = std::log(x);
        RealType z = (ln_x - m_mu) / m_sigma;
        return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
    }
    
private:
    RealType m_mu;           /**< Mean of underlying normal */
    RealType m_sigma;        /**< Std dev of underlying normal */
    AleamCore& m_rng;        /**< Random number generator */
};

/**
 * @brief Generate a Log-Normal distributed random number
 * 
 * @param rng Reference to AleamCore instance
 * @param mu Mean of underlying normal distribution
 * @param sigma Standard deviation of underlying normal distribution (> 0)
 * @return double Random value from LogNormal(μ, σ²)
 */
inline double lognormal(AleamCore& rng, double mu, double sigma) {
    LogNormalDistribution<double> dist(mu, sigma, rng);
    return dist();
}

}  // namespace distributions
}  // namespace aleam

#endif /* ALEAM_DISTRIBUTIONS_LOGNORMAL_H */