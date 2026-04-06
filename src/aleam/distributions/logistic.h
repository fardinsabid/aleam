/**
 * @file logistic.h
 * @brief Logistic distribution using Aleam true randomness
 * @license MIT
 * 
 * Implements the Logistic distribution using inverse transform sampling.
 * The logistic distribution is similar to the normal distribution but
 * has heavier tails. It is commonly used in logistic regression and
 * growth models.
 */

#ifndef ALEAM_DISTRIBUTIONS_LOGISTIC_H
#define ALEAM_DISTRIBUTIONS_LOGISTIC_H

#include <cmath>
#include <stdexcept>
#include "../core/aleam_core.h"

namespace aleam {
namespace distributions {

/**
 * @brief Logistic distribution class
 * 
 * Represents a Logistic distribution with location parameter μ and scale parameter s.
 * 
 * Probability density function:
 *     f(x) = e^{-(x-μ)/s} / [s * (1 + e^{-(x-μ)/s})²]
 * 
 * Mean = μ
 * Variance = s²π²/3
 * 
 * @tparam RealType Floating point type (float or double)
 */
template<typename RealType = double>
class LogisticDistribution {
public:
    /**
     * @brief Construct a Logistic distribution
     * 
     * @param loc Location parameter (μ)
     * @param scale Scale parameter (s) (> 0)
     * @param rng Reference to AleamCore instance
     */
    LogisticDistribution(RealType loc = 0.0, RealType scale = 1.0, AleamCore& rng)
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
     *     F⁻¹(u) = μ + s·ln(u / (1-u))
     * 
     * @return RealType Random value from Logistic(μ, s)
     */
    RealType operator()() {
        RealType u = static_cast<RealType>(m_rng.random());
        
        // Avoid division by zero and log(0)
        u = std::clamp(u, 1e-12, 1.0 - 1e-12);
        
        return m_loc + m_scale * std::log(u / (1.0 - u));
    }
    
    /**
     * @brief Get the location parameter (μ)
     * 
     * @return RealType Location parameter
     */
    RealType location() const { return m_loc; }
    
    /**
     * @brief Get the scale parameter (s)
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
     * @brief Get the variance of the distribution (s²π²/3)
     * 
     * @return RealType Variance
     */
    RealType variance() const {
        static const RealType PI_SQ = M_PI * M_PI;
        return m_scale * m_scale * PI_SQ / 3.0;
    }
    
    /**
     * @brief Get the standard deviation (s·π/√3)
     * 
     * @return RealType Standard deviation
     */
    RealType standard_deviation() const {
        return m_scale * M_PI / std::sqrt(3.0);
    }
    
    /**
     * @brief Probability density function
     * 
     * @param x Point at which to evaluate PDF
     * @return RealType PDF value at x
     */
    RealType pdf(RealType x) const {
        RealType z = (x - m_loc) / m_scale;
        RealType exp_neg_z = std::exp(-z);
        RealType denominator = m_scale * (1.0 + exp_neg_z) * (1.0 + exp_neg_z);
        return exp_neg_z / denominator;
    }
    
    /**
     * @brief Cumulative distribution function
     * 
     * @param x Point at which to evaluate CDF
     * @return RealType CDF value at x
     */
    RealType cdf(RealType x) const {
        RealType z = (x - m_loc) / m_scale;
        return 1.0 / (1.0 + std::exp(-z));
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
        
        // Avoid division by zero
        p = std::clamp(p, 1e-12, 1.0 - 1e-12);
        
        return m_loc + m_scale * std::log(p / (1.0 - p));
    }
    
private:
    RealType m_loc;          /**< Location parameter (μ) */
    RealType m_scale;        /**< Scale parameter (s) */
    AleamCore& m_rng;        /**< Random number generator */
};

/**
 * @brief Generate a Logistic distributed random number
 * 
 * @param rng Reference to AleamCore instance
 * @param loc Location parameter (μ)
 * @param scale Scale parameter (s > 0)
 * @return double Random value from Logistic(μ, s)
 */
inline double logistic(AleamCore& rng, double loc, double scale) {
    LogisticDistribution<double> dist(loc, scale, rng);
    return dist();
}

}  // namespace distributions
}  // namespace aleam

#endif /* ALEAM_DISTRIBUTIONS_LOGISTIC_H */