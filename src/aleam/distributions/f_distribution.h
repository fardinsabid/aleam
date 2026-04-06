/**
 * @file f_distribution.h
 * @brief F-distribution using Aleam true randomness
 * @license MIT
 * 
 * Implements the F-distribution using the relationship:
 *     F(d₁, d₂) = (χ²₁/d₁) / (χ²₂/d₂)
 * where χ²₁ ~ χ²(d₁) and χ²₂ ~ χ²(d₂)
 * 
 * The F-distribution is used in ANOVA, regression analysis,
 * and comparing variances.
 */

#ifndef ALEAM_DISTRIBUTIONS_F_DISTRIBUTION_H
#define ALEAM_DISTRIBUTIONS_F_DISTRIBUTION_H

#include <cmath>
#include <stdexcept>
#include <limits>
#include "../core/aleam_core.h"
#include "chi_square.h"

namespace aleam {
namespace distributions {

/**
 * @brief F-distribution class
 * 
 * Represents an F-distribution with numerator degrees of freedom d₁
 * and denominator degrees of freedom d₂.
 * 
 * Probability density function:
 *     f(x) = √((d₁ x)^{d₁} d₂^{d₂} / (d₁ x + d₂)^{d₁+d₂}) / (x B(d₁/2, d₂/2))
 * 
 * Mean = d₂/(d₂-2) (for d₂ > 2)
 * Variance = 2d₂²(d₁+d₂-2) / (d₁(d₂-2)²(d₂-4)) (for d₂ > 4)
 * 
 * @tparam RealType Floating point type (float or double)
 */
template<typename RealType = double>
class FDistribution {
public:
    /**
     * @brief Construct an F-distribution
     * 
     * @param df1 Numerator degrees of freedom (d₁) (> 0)
     * @param df2 Denominator degrees of freedom (d₂) (> 0)
     * @param rng Reference to AleamCore instance
     */
    FDistribution(RealType df1, RealType df2, AleamCore& rng)
        : m_df1(df1)
        , m_df2(df2)
        , m_rng(rng) {
        if (df1 <= 0.0) {
            throw std::invalid_argument("df1 must be > 0");
        }
        if (df2 <= 0.0) {
            throw std::invalid_argument("df2 must be > 0");
        }
    }
    
    /**
     * @brief Generate a random number from the distribution
     * 
     * Algorithm:
     *     χ²₁ ~ χ²(d₁)
     *     χ²₂ ~ χ²(d₂)
     *     F = (χ²₁/d₁) / (χ²₂/d₂)
     * 
     * @return RealType Random value from F(d₁, d₂)
     */
    RealType operator()() {
        ChiSquareDistribution<RealType> chi1(m_df1, m_rng);
        ChiSquareDistribution<RealType> chi2(m_df2, m_rng);
        
        RealType chi2_1 = chi1();
        RealType chi2_2 = chi2();
        
        return (chi2_1 / m_df1) / (chi2_2 / m_df2);
    }
    
    /**
     * @brief Get numerator degrees of freedom (d₁)
     * 
     * @return RealType Numerator degrees of freedom
     */
    RealType df1() const { return m_df1; }
    
    /**
     * @brief Get denominator degrees of freedom (d₂)
     * 
     * @return RealType Denominator degrees of freedom
     */
    RealType df2() const { return m_df2; }
    
    /**
     * @brief Get the mean of the distribution (for d₂ > 2)
     * 
     * @return RealType Mean = d₂/(d₂-2)
     * @throws std::runtime_error if d₂ ≤ 2 (mean infinite/undefined)
     */
    RealType mean() const {
        if (m_df2 <= 2.0) {
            throw std::runtime_error("Mean is undefined for df2 <= 2");
        }
        return m_df2 / (m_df2 - 2.0);
    }
    
    /**
     * @brief Get the variance of the distribution (for d₂ > 4)
     * 
     * @return RealType Variance = 2d₂²(d₁+d₂-2) / (d₁(d₂-2)²(d₂-4))
     * @throws std::runtime_error if d₂ ≤ 4 (variance infinite/undefined)
     */
    RealType variance() const {
        if (m_df2 <= 4.0) {
            throw std::runtime_error("Variance is undefined for df2 <= 4");
        }
        
        RealType numerator = 2.0 * m_df2 * m_df2 * (m_df1 + m_df2 - 2.0);
        RealType denominator = m_df1 * (m_df2 - 2.0) * (m_df2 - 2.0) * (m_df2 - 4.0);
        
        return numerator / denominator;
    }
    
    /**
     * @brief Get the mode of the distribution
     * 
     * @return RealType Mode = (d₁-2)/d₁ * d₂/(d₂+2) (for d₁ > 2)
     */
    RealType mode() const {
        if (m_df1 <= 2.0) {
            return 0.0;  // Mode at 0 for d₁ ≤ 2
        }
        return (m_df1 - 2.0) / m_df1 * m_df2 / (m_df2 + 2.0);
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
        RealType log_pdf = (m_df1 / 2.0) * std::log(m_df1) +
                           (m_df2 / 2.0) * std::log(m_df2) +
                           (m_df1 / 2.0 - 1.0) * std::log(x) -
                           ((m_df1 + m_df2) / 2.0) * std::log(m_df1 * x + m_df2) -
                           std::lgamma(m_df1 / 2.0) -
                           std::lgamma(m_df2 / 2.0) +
                           std::lgamma((m_df1 + m_df2) / 2.0);
        
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
        
        // Regularized incomplete beta function
        RealType beta = (m_df1 * x) / (m_df1 * x + m_df2);
        
        // Simplified approximation using beta distribution CDF
        // In production, use a more accurate implementation
        if (beta <= 0.0) return 0.0;
        if (beta >= 1.0) return 1.0;
        
        // Approximate using incomplete beta
        return std::tgamma((m_df1 + m_df2) / 2.0) /
               (std::tgamma(m_df1 / 2.0) * std::tgamma(m_df2 / 2.0)) *
               std::pow(beta, m_df1 / 2.0) *
               std::pow(1.0 - beta, m_df2 / 2.0) / (m_df1 / 2.0);
    }
    
    /**
     * @brief Generate a random number (static method)
     * 
     * @param rng Reference to AleamCore instance
     * @param df1 Numerator degrees of freedom
     * @param df2 Denominator degrees of freedom
     * @return RealType Random value from F(d₁, d₂)
     */
    static RealType sample(AleamCore& rng, RealType df1, RealType df2) {
        FDistribution dist(df1, df2, rng);
        return dist();
    }
    
private:
    RealType m_df1;          /**< Numerator degrees of freedom (d₁) */
    RealType m_df2;          /**< Denominator degrees of freedom (d₂) */
    AleamCore& m_rng;        /**< Random number generator */
};

/**
 * @brief Generate an F-distributed random number
 * 
 * @param rng Reference to AleamCore instance
 * @param df1 Numerator degrees of freedom (d₁ > 0)
 * @param df2 Denominator degrees of freedom (d₂ > 0)
 * @return double Random value from F(d₁, d₂)
 */
inline double f_distribution(AleamCore& rng, double df1, double df2) {
    FDistribution<double> dist(df1, df2, rng);
    return dist();
}

}  // namespace distributions
}  // namespace aleam

#endif /* ALEAM_DISTRIBUTIONS_F_DISTRIBUTION_H */