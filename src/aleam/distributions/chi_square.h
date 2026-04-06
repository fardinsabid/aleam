/**
 * @file chi_square.h
 * @brief Chi-square distribution using Aleam true randomness
 * @license MIT
 * 
 * Implements the Chi-square distribution using the Gamma distribution
 * relationship: χ²(k) = Gamma(k/2, 2)
 * 
 * The Chi-square distribution is widely used in hypothesis testing,
 * confidence intervals, and goodness-of-fit tests.
 */

#ifndef ALEAM_DISTRIBUTIONS_CHI_SQUARE_H
#define ALEAM_DISTRIBUTIONS_CHI_SQUARE_H

#include <cmath>
#include <stdexcept>
#include "../core/aleam_core.h"
#include "gamma.h"

namespace aleam {
namespace distributions {

/**
 * @brief Chi-square distribution class
 * 
 * Represents a Chi-square distribution with degrees of freedom k.
 * 
 * Probability density function:
 *     f(x) = x^(k/2 - 1) * e^(-x/2) / [2^(k/2) * Γ(k/2)] for x ≥ 0
 * 
 * Mean = k
 * Variance = 2k
 * 
 * @tparam RealType Floating point type (float or double)
 */
template<typename RealType = double>
class ChiSquareDistribution {
public:
    /**
     * @brief Construct a Chi-square distribution
     * 
     * @param df Degrees of freedom (k) (> 0)
     * @param rng Reference to AleamCore instance
     */
    ChiSquareDistribution(RealType df, AleamCore& rng)
        : m_df(df)
        , m_rng(rng) {
        if (df <= 0.0) {
            throw std::invalid_argument("degrees of freedom must be > 0");
        }
    }
    
    /**
     * @brief Generate a random number from the distribution
     * 
     * Uses Gamma distribution: χ²(k) = Gamma(k/2, 2)
     * 
     * @return RealType Random value from χ²(k)
     */
    RealType operator()() {
        GammaDistribution<RealType> gamma(m_df / 2.0, 2.0, m_rng);
        return gamma();
    }
    
    /**
     * @brief Get the degrees of freedom (k)
     * 
     * @return RealType Degrees of freedom
     */
    RealType degrees_of_freedom() const { return m_df; }
    
    /**
     * @brief Get the mean of the distribution (k)
     * 
     * @return RealType Mean
     */
    RealType mean() const { return m_df; }
    
    /**
     * @brief Get the variance of the distribution (2k)
     * 
     * @return RealType Variance
     */
    RealType variance() const { return 2.0 * m_df; }
    
    /**
     * @brief Get the standard deviation (√(2k))
     * 
     * @return RealType Standard deviation
     */
    RealType standard_deviation() const {
        return std::sqrt(2.0 * m_df);
    }
    
    /**
     * @brief Get the mode of the distribution
     * 
     * @return RealType Mode = max(0, k-2)
     */
    RealType mode() const {
        return (m_df > 2.0) ? (m_df - 2.0) : 0.0;
    }
    
    /**
     * @brief Probability density function
     * 
     * @param x Point at which to evaluate PDF (x ≥ 0)
     * @return RealType PDF value at x
     */
    RealType pdf(RealType x) const {
        if (x < 0.0) return 0.0;
        
        // Using log to avoid overflow
        RealType log_pdf = (m_df / 2.0 - 1.0) * std::log(x) -
                           x / 2.0 -
                           (m_df / 2.0) * std::log(2.0) -
                           std::lgamma(m_df / 2.0);
        
        return std::exp(log_pdf);
    }
    
    /**
     * @brief Cumulative distribution function
     * 
     * @param x Point at which to evaluate CDF (x ≥ 0)
     * @return RealType CDF value at x
     */
    RealType cdf(RealType x) const {
        if (x < 0.0) return 0.0;
        
        // Use regularized gamma function (lower incomplete gamma)
        // For simplicity, using std::gamma for now
        // In production, use a more accurate implementation
        return std::tgamma(m_df / 2.0, x / 2.0) / std::tgamma(m_df / 2.0);
    }
    
    /**
     * @brief Generate a random number from the distribution (static method)
     * 
     * @param rng Reference to AleamCore instance
     * @param df Degrees of freedom
     * @return RealType Random value from χ²(k)
     */
    static RealType sample(AleamCore& rng, RealType df) {
        ChiSquareDistribution dist(df, rng);
        return dist();
    }
    
private:
    RealType m_df;           /**< Degrees of freedom (k) */
    AleamCore& m_rng;        /**< Random number generator */
};

/**
 * @brief Generate a Chi-square distributed random number
 * 
 * @param rng Reference to AleamCore instance
 * @param df Degrees of freedom (k > 0)
 * @return double Random value from χ²(k)
 */
inline double chi_square(AleamCore& rng, double df) {
    ChiSquareDistribution<double> dist(df, rng);
    return dist();
}

}  // namespace distributions
}  // namespace aleam

#endif /* ALEAM_DISTRIBUTIONS_CHI_SQUARE_H */