/**
 * @file pareto.h
 * @brief Pareto distribution using Aleam true randomness
 * @license MIT
 * 
 * Implements the Pareto distribution using inverse transform sampling.
 * The Pareto distribution is a power-law distribution used in economics,
 * finance, and other fields to model phenomena like wealth distribution.
 */

#ifndef ALEAM_DISTRIBUTIONS_PARETO_H
#define ALEAM_DISTRIBUTIONS_PARETO_H

#include <cmath>
#include <stdexcept>
#include <limits>
#include "../core/aleam_core.h"

namespace aleam {
namespace distributions {

/**
 * @brief Pareto distribution class
 * 
 * Represents a Pareto distribution with shape parameter α and scale parameter x_m.
 * 
 * Probability density function:
 *     f(x) = α * x_m^α / x^(α+1) for x ≥ x_m
 * 
 * Mean = α * x_m / (α - 1) for α > 1
 * Variance = (α * x_m^2) / ((α - 1)^2 * (α - 2)) for α > 2
 * 
 * @tparam RealType Floating point type (float or double)
 */
template<typename RealType = double>
class ParetoDistribution {
public:
    /**
     * @brief Construct a Pareto distribution
     * 
     * @param alpha Shape parameter (α) (> 0)
     * @param scale Scale parameter (x_m) (> 0) - minimum possible value
     * @param rng Reference to AleamCore instance
     */
    ParetoDistribution(RealType alpha, RealType scale = 1.0, AleamCore& rng)
        : m_alpha(alpha)
        , m_scale(scale)
        , m_rng(rng) {
        if (alpha <= 0.0) {
            throw std::invalid_argument("alpha must be > 0");
        }
        if (scale <= 0.0) {
            throw std::invalid_argument("scale must be > 0");
        }
    }
    
    /**
     * @brief Generate a random number from the distribution
     * 
     * Uses inverse transform sampling:
     *     F⁻¹(u) = x_m / u^(1/α)
     * 
     * @return RealType Random value from Pareto(α, x_m)
     */
    RealType operator()() {
        RealType u = static_cast<RealType>(m_rng.random());
        
        // Avoid division by zero
        if (u <= 0.0) u = 1e-15;
        
        return m_scale / std::pow(u, 1.0 / m_alpha);
    }
    
    /**
     * @brief Get the shape parameter (α)
     * 
     * @return RealType Alpha parameter
     */
    RealType alpha() const { return m_alpha; }
    
    /**
     * @brief Get the scale parameter (x_m)
     * 
     * @return RealType Scale parameter (minimum value)
     */
    RealType scale() const { return m_scale; }
    
    /**
     * @brief Get the minimum value of the distribution (x_m)
     * 
     * @return RealType Minimum value
     */
    RealType min() const { return m_scale; }
    
    /**
     * @brief Get the mean of the distribution (for α > 1)
     * 
     * @return RealType Mean = α·x_m/(α-1)
     * @throws std::runtime_error if α ≤ 1 (mean infinite/undefined)
     */
    RealType mean() const {
        if (m_alpha <= 1.0) {
            throw std::runtime_error("Mean is infinite for alpha <= 1");
        }
        return m_alpha * m_scale / (m_alpha - 1.0);
    }
    
    /**
     * @brief Get the variance of the distribution (for α > 2)
     * 
     * @return RealType Variance = (α·x_m²)/((α-1)²(α-2))
     * @throws std::runtime_error if α ≤ 2 (variance infinite/undefined)
     */
    RealType variance() const {
        if (m_alpha <= 2.0) {
            throw std::runtime_error("Variance is infinite for alpha <= 2");
        }
        RealType alpha_minus_1 = m_alpha - 1.0;
        return (m_alpha * m_scale * m_scale) / 
               (alpha_minus_1 * alpha_minus_1 * (m_alpha - 2.0));
    }
    
    /**
     * @brief Get the median of the distribution
     * 
     * @return RealType Median = x_m * 2^(1/α)
     */
    RealType median() const {
        return m_scale * std::pow(2.0, 1.0 / m_alpha);
    }
    
    /**
     * @brief Get the mode of the distribution
     * 
     * @return RealType Mode = x_m
     */
    RealType mode() const {
        return m_scale;
    }
    
    /**
     * @brief Probability density function
     * 
     * @param x Point at which to evaluate PDF (x ≥ x_m)
     * @return RealType PDF value at x
     */
    RealType pdf(RealType x) const {
        if (x < m_scale) return 0.0;
        return m_alpha * std::pow(m_scale, m_alpha) / std::pow(x, m_alpha + 1.0);
    }
    
    /**
     * @brief Cumulative distribution function
     * 
     * @param x Point at which to evaluate CDF (x ≥ x_m)
     * @return RealType CDF value at x
     */
    RealType cdf(RealType x) const {
        if (x < m_scale) return 0.0;
        return 1.0 - std::pow(m_scale / x, m_alpha);
    }
    
    /**
     * @brief Survival function (1 - CDF)
     * 
     * @param x Point at which to evaluate survival (x ≥ x_m)
     * @return RealType Survival function value at x
     */
    RealType survival(RealType x) const {
        if (x < m_scale) return 1.0;
        return std::pow(m_scale / x, m_alpha);
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
        
        if (p == 0.0) return m_scale;
        if (p == 1.0) return std::numeric_limits<RealType>::infinity();
        
        return m_scale / std::pow(1.0 - p, 1.0 / m_alpha);
    }
    
private:
    RealType m_alpha;        /**< Shape parameter (α) */
    RealType m_scale;        /**< Scale parameter (x_m) */
    AleamCore& m_rng;        /**< Random number generator */
};

/**
 * @brief Generate a Pareto distributed random number
 * 
 * @param rng Reference to AleamCore instance
 * @param alpha Shape parameter (α > 0)
 * @param scale Scale parameter (x_m > 0)
 * @return double Random value from Pareto(α, x_m)
 */
inline double pareto(AleamCore& rng, double alpha, double scale) {
    ParetoDistribution<double> dist(alpha, scale, rng);
    return dist();
}

}  // namespace distributions
}  // namespace aleam

#endif /* ALEAM_DISTRIBUTIONS_PARETO_H */