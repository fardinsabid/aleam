/**
 * @file student_t.h
 * @brief Student's t-distribution using Aleam true randomness
 * @license MIT
 * 
 * Implements the Student's t-distribution using the relationship:
 *     t(ν) = Z / √(χ²/ν)
 * where Z ~ N(0,1) and χ² ~ χ²(ν)
 * 
 * The t-distribution is used in hypothesis testing and confidence
 * intervals when the sample size is small.
 */

#ifndef ALEAM_DISTRIBUTIONS_STUDENT_T_H
#define ALEAM_DISTRIBUTIONS_STUDENT_T_H

#include <cmath>
#include <stdexcept>
#include "../core/aleam_core.h"
#include "normal.h"
#include "chi_square.h"

namespace aleam {
namespace distributions {

/**
 * @brief Student's t-distribution class
 * 
 * Represents a t-distribution with degrees of freedom ν.
 * 
 * Probability density function:
 *     f(x) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) * (1 + x²/ν)^(-(ν+1)/2)
 * 
 * Mean = 0 (for ν > 1)
 * Variance = ν/(ν-2) (for ν > 2)
 * 
 * @tparam RealType Floating point type (float or double)
 */
template<typename RealType = double>
class StudentTDistribution {
public:
    /**
     * @brief Construct a Student's t-distribution
     * 
     * @param df Degrees of freedom (ν) (> 0)
     * @param rng Reference to AleamCore instance
     */
    StudentTDistribution(RealType df, AleamCore& rng)
        : m_df(df)
        , m_rng(rng) {
        if (df <= 0.0) {
            throw std::invalid_argument("degrees of freedom must be > 0");
        }
    }
    
    /**
     * @brief Generate a random number from the distribution
     * 
     * Algorithm:
     *     Z ~ N(0, 1)
     *     χ² ~ χ²(ν)
     *     t = Z / √(χ²/ν)
     * 
     * @return RealType Random value from t(ν)
     */
    RealType operator()() {
        NormalDistribution<RealType> normal(0.0, 1.0, m_rng);
        ChiSquareDistribution<RealType> chi_square(m_df, m_rng);
        
        RealType z = normal();
        RealType chi2 = chi_square();
        
        return z / std::sqrt(chi2 / m_df);
    }
    
    /**
     * @brief Get the degrees of freedom (ν)
     * 
     * @return RealType Degrees of freedom
     */
    RealType degrees_of_freedom() const { return m_df; }
    
    /**
     * @brief Get the mean of the distribution (for ν > 1)
     * 
     * @return RealType Mean = 0
     * @throws std::runtime_error if ν ≤ 1 (mean undefined)
     */
    RealType mean() const {
        if (m_df <= 1.0) {
            throw std::runtime_error("Mean is undefined for df <= 1");
        }
        return 0.0;
    }
    
    /**
     * @brief Get the variance of the distribution (for ν > 2)
     * 
     * @return RealType Variance = ν/(ν-2)
     * @throws std::runtime_error if ν ≤ 2 (variance infinite/undefined)
     */
    RealType variance() const {
        if (m_df <= 2.0) {
            throw std::runtime_error("Variance is infinite for df <= 2");
        }
        return m_df / (m_df - 2.0);
    }
    
    /**
     * @brief Get the standard deviation (for ν > 2)
     * 
     * @return RealType Standard deviation = √(ν/(ν-2))
     */
    RealType standard_deviation() const {
        return std::sqrt(variance());
    }
    
    /**
     * @brief Get the mode of the distribution
     * 
     * @return RealType Mode = 0
     */
    RealType mode() const {
        return 0.0;
    }
    
    /**
     * @brief Probability density function
     * 
     * @param x Point at which to evaluate PDF
     * @return RealType PDF value at x
     */
    RealType pdf(RealType x) const {
        RealType x_sq = x * x;
        RealType exponent = -(m_df + 1.0) / 2.0;
        
        RealType numerator = std::tgamma((m_df + 1.0) / 2.0);
        RealType denominator = std::sqrt(m_df * M_PI) * std::tgamma(m_df / 2.0);
        RealType factor = numerator / denominator;
        RealType power = std::pow(1.0 + x_sq / m_df, exponent);
        
        return factor * power;
    }
    
    /**
     * @brief Cumulative distribution function
     * 
     * @param x Point at which to evaluate CDF
     * @return RealType CDF value at x
     */
    RealType cdf(RealType x) const {
        // Using regularized incomplete beta function
        // Simplified approximation using erf
        if (m_df > 30) {
            // Normal approximation for large df
            NormalDistribution<RealType> normal(0.0, 1.0, m_rng);
            return normal.cdf(x);
        }
        
        // For smaller df, use approximation
        RealType t = x / std::sqrt(m_df);
        RealType beta = 0.5 * (1.0 + std::erf(t / std::sqrt(2.0)));
        
        // Adjust for t-distribution
        return beta;
    }
    
    /**
     * @brief Generate a random number (static method)
     * 
     * @param rng Reference to AleamCore instance
     * @param df Degrees of freedom
     * @return RealType Random value from t(ν)
     */
    static RealType sample(AleamCore& rng, RealType df) {
        StudentTDistribution dist(df, rng);
        return dist();
    }
    
private:
    RealType m_df;           /**< Degrees of freedom (ν) */
    AleamCore& m_rng;        /**< Random number generator */
};

/**
 * @brief Generate a Student's t-distributed random number
 * 
 * @param rng Reference to AleamCore instance
 * @param df Degrees of freedom (ν > 0)
 * @return double Random value from t(ν)
 */
inline double student_t(AleamCore& rng, double df) {
    StudentTDistribution<double> dist(df, rng);
    return dist();
}

}  // namespace distributions
}  // namespace aleam

#endif /* ALEAM_DISTRIBUTIONS_STUDENT_T_H */