/**
 * @file beta.h
 * @brief Beta distribution using Aleam true randomness
 * @license MIT
 * 
 * Implements the Beta distribution using the gamma method.
 * If X ~ Gamma(α, 1) and Y ~ Gamma(β, 1), then X/(X+Y) ~ Beta(α, β).
 * 
 * The Beta distribution is defined on the interval [0, 1] and is
 * commonly used as a conjugate prior for binomial proportions.
 */

#ifndef ALEAM_DISTRIBUTIONS_BETA_H
#define ALEAM_DISTRIBUTIONS_BETA_H

#include <cmath>
#include <stdexcept>
#include "../core/aleam_core.h"
#include "gamma.h"

namespace aleam {
namespace distributions {

/**
 * @brief Beta distribution class
 * 
 * Represents a Beta distribution with shape parameters α and β.
 * 
 * Probability density function:
 *     f(x) = [x^(α-1) * (1-x)^(β-1)] / B(α, β)
 * 
 * where B(α, β) is the beta function.
 * 
 * Support: x ∈ [0, 1]
 * Mean = α / (α + β)
 * Variance = αβ / [(α+β)²(α+β+1)]
 * 
 * @tparam RealType Floating point type (float or double)
 */
template<typename RealType = double>
class BetaDistribution {
public:
    /**
     * @brief Construct a Beta distribution
     * 
     * @param alpha Shape parameter α (> 0)
     * @param beta Shape parameter β (> 0)
     * @param rng Reference to AleamCore instance
     */
    BetaDistribution(RealType alpha, RealType beta, AleamCore& rng)
        : m_alpha(alpha)
        , m_beta(beta)
        , m_rng(rng) {
        if (alpha <= 0.0) {
            throw std::invalid_argument("alpha must be > 0");
        }
        if (beta <= 0.0) {
            throw std::invalid_argument("beta must be > 0");
        }
    }
    
    /**
     * @brief Generate a random number from the distribution
     * 
     * Algorithm:
     *     X ~ Gamma(α, 1)
     *     Y ~ Gamma(β, 1)
     *     return X / (X + Y)
     * 
     * @return RealType Random value from Beta(α, β)
     */
    RealType operator()() {
        GammaDistribution<RealType> gamma_alpha(m_alpha, 1.0, m_rng);
        GammaDistribution<RealType> gamma_beta(m_beta, 1.0, m_rng);
        
        RealType x = gamma_alpha();
        RealType y = gamma_beta();
        
        return x / (x + y);
    }
    
    /**
     * @brief Get the alpha shape parameter (α)
     * 
     * @return RealType Alpha parameter
     */
    RealType alpha() const { return m_alpha; }
    
    /**
     * @brief Get the beta shape parameter (β)
     * 
     * @return RealType Beta parameter
     */
    RealType beta() const { return m_beta; }
    
    /**
     * @brief Get the mean of the distribution
     * 
     * @return RealType Mean = α / (α + β)
     */
    RealType mean() const {
        return m_alpha / (m_alpha + m_beta);
    }
    
    /**
     * @brief Get the variance of the distribution
     * 
     * @return RealType Variance = αβ / [(α+β)²(α+β+1)]
     */
    RealType variance() const {
        RealType sum = m_alpha + m_beta;
        return (m_alpha * m_beta) / (sum * sum * (sum + 1.0));
    }
    
    /**
     * @brief Get the mode of the distribution
     * 
     * @return RealType Mode = (α-1) / (α+β-2) for α>1, β>1
     */
    RealType mode() const {
        if (m_alpha <= 1.0 || m_beta <= 1.0) {
            return 0.0;  // Mode at boundary or undefined
        }
        return (m_alpha - 1.0) / (m_alpha + m_beta - 2.0);
    }
    
    /**
     * @brief Probability density function
     * 
     * @param x Point at which to evaluate PDF (0 ≤ x ≤ 1)
     * @return RealType PDF value at x
     */
    RealType pdf(RealType x) const {
        if (x < 0.0 || x > 1.0) return 0.0;
        
        // Using log to avoid overflow
        RealType log_pdf = (m_alpha - 1.0) * std::log(x) +
                           (m_beta - 1.0) * std::log(1.0 - x) -
                           std::lgamma(m_alpha) -
                           std::lgamma(m_beta) +
                           std::lgamma(m_alpha + m_beta);
        
        return std::exp(log_pdf);
    }
    
private:
    RealType m_alpha;        /**< Shape parameter α */
    RealType m_beta;         /**< Shape parameter β */
    AleamCore& m_rng;        /**< Random number generator */
};

/**
 * @brief Generate a Beta distributed random number
 * 
 * @param rng Reference to AleamCore instance
 * @param alpha Shape parameter α (> 0)
 * @param beta Shape parameter β (> 0)
 * @return double Random value from Beta(α, β)
 */
inline double beta(AleamCore& rng, double alpha, double beta) {
    BetaDistribution<double> dist(alpha, beta, rng);
    return dist();
}

}  // namespace distributions
}  // namespace aleam

#endif /* ALEAM_DISTRIBUTIONS_BETA_H */