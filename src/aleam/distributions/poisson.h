/**
 * @file poisson.h
 * @brief Poisson distribution using Aleam true randomness
 * @license MIT
 * 
 * Implements the Poisson distribution using Knuth's algorithm for
 * small lambda and normal approximation for large lambda.
 * 
 * The Poisson distribution models the number of events occurring
 * in a fixed interval of time or space.
 */

#ifndef ALEAM_DISTRIBUTIONS_POISSON_H
#define ALEAM_DISTRIBUTIONS_POISSON_H

#include <cmath>
#include <stdexcept>
#include <random>
#include "../core/aleam_core.h"
#include "normal.h"

namespace aleam {
namespace distributions {

/**
 * @brief Poisson distribution class
 * 
 * Represents a Poisson distribution with mean parameter λ.
 * 
 * Probability mass function:
 *     P(X = k) = e^{-λ} * λ^k / k! for k = 0, 1, 2, ...
 * 
 * Mean = λ
 * Variance = λ
 * 
 * @tparam IntType Integer type (int, long, etc.)
 */
template<typename IntType = int>
class PoissonDistribution {
public:
    /**
     * @brief Construct a Poisson distribution
     * 
     * @param lambda Mean parameter (λ) (> 0)
     * @param rng Reference to AleamCore instance
     */
    PoissonDistribution(double lambda, AleamCore& rng)
        : m_lambda(lambda)
        , m_rng(rng) {
        if (lambda <= 0.0) {
            throw std::invalid_argument("lambda must be > 0");
        }
        
        // Precompute for Knuth's algorithm
        m_L = std::exp(-m_lambda);
    }
    
    /**
     * @brief Generate a random number from the distribution
     * 
     * Uses Knuth's algorithm for λ < 10.
     * Uses normal approximation with continuity correction for λ >= 10.
     * 
     * @return IntType Random value from Poisson(λ)
     */
    IntType operator()() {
        if (m_lambda < 10.0) {
            return knuth_method();
        }
        return normal_approximation();
    }
    
    /**
     * @brief Get the mean parameter (λ)
     * 
     * @return double Lambda parameter
     */
    double lambda() const { return m_lambda; }
    
    /**
     * @brief Get the mean of the distribution
     * 
     * @return double Mean (λ)
     */
    double mean() const { return m_lambda; }
    
    /**
     * @brief Get the variance of the distribution
     * 
     * @return double Variance (λ)
     */
    double variance() const { return m_lambda; }
    
    /**
     * @brief Probability mass function
     * 
     * @param k Non-negative integer
     * @return double PMF value at k
     */
    double pmf(IntType k) const {
        if (k < 0) return 0.0;
        
        // Using log to avoid overflow
        double log_pmf = -m_lambda + k * std::log(m_lambda) - std::lgamma(k + 1.0);
        return std::exp(log_pmf);
    }
    
    /**
     * @brief Cumulative distribution function
     * 
     * @param k Non-negative integer
     * @return double CDF value at k
     */
    double cdf(IntType k) const {
        if (k < 0) return 0.0;
        
        // Sum of PMF from 0 to k
        double sum = 0.0;
        for (IntType i = 0; i <= k; ++i) {
            sum += pmf(i);
        }
        return sum;
    }
    
private:
    /**
     * @brief Knuth's algorithm for Poisson generation
     * 
     * Algorithm:
     *     L = e^(-λ)
     *     k = 0
     *     p = 1
     *     while p > L:
     *         p *= U ~ Uniform(0,1)
     *         k++
     *     return k - 1
     * 
     * @return IntType Poisson random value
     */
    IntType knuth_method() {
        IntType k = 0;
        double p = 1.0;
        
        while (p > m_L) {
            p *= m_rng.random();
            k++;
        }
        
        return k - 1;
    }
    
    /**
     * @brief Normal approximation for large lambda
     * 
     * Uses continuity correction: X ≈ round(N(λ, √λ))
     * 
     * @return IntType Poisson random value
     */
    IntType normal_approximation() {
        NormalDistribution<double> normal(m_lambda, std::sqrt(m_lambda), m_rng);
        double x = normal();
        IntType result = static_cast<IntType>(std::round(x));
        
        // Ensure non-negative result
        if (result < 0) result = 0;
        
        return result;
    }
    
    double m_lambda;         /**< Mean parameter (λ) */
    AleamCore& m_rng;        /**< Random number generator */
    double m_L;              /**< e^{-λ} precomputed for Knuth */
};

/**
 * @brief Generate a Poisson distributed random integer
 * 
 * @param rng Reference to AleamCore instance
 * @param lambda Mean parameter (λ > 0)
 * @return int Random value from Poisson(λ)
 */
inline int poisson(AleamCore& rng, double lambda) {
    PoissonDistribution<int> dist(lambda, rng);
    return dist();
}

}  // namespace distributions
}  // namespace aleam

#endif /* ALEAM_DISTRIBUTIONS_POISSON_H */