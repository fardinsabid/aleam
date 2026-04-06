/**
 * @file dirichlet.h
 * @brief Dirichlet distribution using Aleam true randomness
 * @license MIT
 * 
 * Implements the Dirichlet distribution using the relationship
 * between Gamma and Dirichlet distributions.
 * 
 * If X_i ~ Gamma(α_i, 1), then (X_i / ΣX_j) ~ Dirichlet(α)
 * 
 * The Dirichlet distribution is the conjugate prior for the
 * categorical and multinomial distributions.
 */

#ifndef ALEAM_DISTRIBUTIONS_DIRICHLET_H
#define ALEAM_DISTRIBUTIONS_DIRICHLET_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include "../core/aleam_core.h"
#include "gamma.h"

namespace aleam {
namespace distributions {

/**
 * @brief Dirichlet distribution class
 * 
 * Represents a Dirichlet distribution with concentration parameters α.
 * 
 * Probability density function:
 *     f(x₁, ..., x_K) = [Γ(Σα_i) / ΠΓ(α_i)] * Π x_i^(α_i - 1)
 * 
 * Support: x_i > 0, Σx_i = 1
 * Mean: E[X_i] = α_i / Σα_j
 * Variance: Var[X_i] = α_i(Σα_j - α_i) / (Σα_j²(Σα_j + 1))
 * 
 * @tparam RealType Floating point type (float or double)
 */
template<typename RealType = double>
class DirichletDistribution {
public:
    /**
     * @brief Construct a Dirichlet distribution
     * 
     * @param alpha Vector of concentration parameters (all > 0)
     * @param rng Reference to AleamCore instance
     */
    DirichletDistribution(const std::vector<RealType>& alpha, AleamCore& rng)
        : m_alpha(alpha)
        , m_rng(rng) {
        if (alpha.empty()) {
            throw std::invalid_argument("alpha must not be empty");
        }
        
        for (size_t i = 0; i < alpha.size(); ++i) {
            if (alpha[i] <= 0.0) {
                throw std::invalid_argument("all alpha values must be > 0");
            }
        }
        
        // Precompute total alpha
        m_alpha_sum = std::accumulate(alpha.begin(), alpha.end(), 0.0);
    }
    
    /**
     * @brief Generate a random sample from the distribution
     * 
     * Algorithm:
     *     For each i, generate X_i ~ Gamma(α_i, 1)
     *     Sum all X_i to get total S
     *     Return vector of X_i / S
     * 
     * @return std::vector<RealType> Probability vector summing to 1
     */
    std::vector<RealType> operator()() {
        std::vector<RealType> samples(m_alpha.size());
        RealType total = 0.0;
        
        // Generate Gamma variates
        for (size_t i = 0; i < m_alpha.size(); ++i) {
            GammaDistribution<RealType> gamma(m_alpha[i], 1.0, m_rng);
            samples[i] = gamma();
            total += samples[i];
        }
        
        // Normalize to sum to 1
        if (total > 0.0) {
            for (size_t i = 0; i < samples.size(); ++i) {
                samples[i] /= total;
            }
        }
        
        return samples;
    }
    
    /**
     * @brief Get the concentration parameters (α)
     * 
     * @return const std::vector<RealType>& Alpha parameters
     */
    const std::vector<RealType>& alpha() const { return m_alpha; }
    
    /**
     * @brief Get the sum of all concentration parameters
     * 
     * @return RealType Sum of α_i
     */
    RealType alpha_sum() const { return m_alpha_sum; }
    
    /**
     * @brief Get the number of dimensions (K)
     * 
     * @return size_t Number of dimensions
     */
    size_t dimension() const { return m_alpha.size(); }
    
    /**
     * @brief Get the mean of the i-th component
     * 
     * @param i Component index
     * @return RealType Mean = α_i / Σα_j
     */
    RealType mean(size_t i) const {
        if (i >= m_alpha.size()) {
            throw std::out_of_range("index out of range");
        }
        return m_alpha[i] / m_alpha_sum;
    }
    
    /**
     * @brief Get the variance of the i-th component
     * 
     * @param i Component index
     * @return RealType Variance = α_i(α_sum - α_i) / (α_sum²(α_sum + 1))
     */
    RealType variance(size_t i) const {
        if (i >= m_alpha.size()) {
            throw std::out_of_range("index out of range");
        }
        RealType alpha_i = m_alpha[i];
        RealType numerator = alpha_i * (m_alpha_sum - alpha_i);
        RealType denominator = m_alpha_sum * m_alpha_sum * (m_alpha_sum + 1.0);
        return numerator / denominator;
    }
    
    /**
     * @brief Get the covariance between components i and j
     * 
     * @param i First component index
     * @param j Second component index
     * @return RealType Covariance = -α_i α_j / (α_sum²(α_sum + 1))
     */
    RealType covariance(size_t i, size_t j) const {
        if (i >= m_alpha.size() || j >= m_alpha.size()) {
            throw std::out_of_range("index out of range");
        }
        if (i == j) {
            return variance(i);
        }
        return -m_alpha[i] * m_alpha[j] / 
               (m_alpha_sum * m_alpha_sum * (m_alpha_sum + 1.0));
    }
    
    /**
     * @brief Get the mode of the distribution
     * 
     * @return std::vector<RealType> Mode vector = (α_i - 1) / (Σα_j - K)
     */
    std::vector<RealType> mode() const {
        std::vector<RealType> mode_vec(m_alpha.size());
        RealType denominator = m_alpha_sum - static_cast<RealType>(m_alpha.size());
        
        if (denominator <= 0.0) {
            // Mode is on the boundary
            for (size_t i = 0; i < m_alpha.size(); ++i) {
                mode_vec[i] = (m_alpha[i] > 1.0) ? 1.0 : 0.0;
            }
        } else {
            for (size_t i = 0; i < m_alpha.size(); ++i) {
                mode_vec[i] = (m_alpha[i] - 1.0) / denominator;
            }
        }
        
        return mode_vec;
    }
    
    /**
     * @brief Probability density function
     * 
     * @param x Vector of probabilities (must sum to 1, all > 0)
     * @return RealType PDF value at x
     */
    RealType pdf(const std::vector<RealType>& x) const {
        if (x.size() != m_alpha.size()) {
            throw std::invalid_argument("x dimension must match alpha dimension");
        }
        
        RealType sum_x = std::accumulate(x.begin(), x.end(), 0.0);
        if (std::abs(sum_x - 1.0) > 1e-10) {
            return 0.0;  // Not a valid probability vector
        }
        
        // Using log to avoid overflow
        RealType log_pdf = std::lgamma(m_alpha_sum);
        
        for (size_t i = 0; i < m_alpha.size(); ++i) {
            if (x[i] <= 0.0) return 0.0;
            log_pdf -= std::lgamma(m_alpha[i]);
            log_pdf += (m_alpha[i] - 1.0) * std::log(x[i]);
        }
        
        return std::exp(log_pdf);
    }
    
    /**
     * @brief Generate a random sample (static method)
     * 
     * @param rng Reference to AleamCore instance
     * @param alpha Concentration parameters
     * @return std::vector<RealType> Probability vector
     */
    static std::vector<RealType> sample(AleamCore& rng, const std::vector<RealType>& alpha) {
        DirichletDistribution dist(alpha, rng);
        return dist();
    }
    
private:
    std::vector<RealType> m_alpha;      /**< Concentration parameters */
    RealType m_alpha_sum;                /**< Sum of all alpha */
    AleamCore& m_rng;                    /**< Random number generator */
};

/**
 * @brief Generate a Dirichlet distributed probability vector
 * 
 * @param rng Reference to AleamCore instance
 * @param alpha Vector of concentration parameters (all > 0)
 * @return std::vector<double> Probability vector summing to 1
 */
inline std::vector<double> dirichlet(AleamCore& rng, const std::vector<double>& alpha) {
    DirichletDistribution<double> dist(alpha, rng);
    return dist();
}

}  // namespace distributions
}  // namespace aleam

#endif /* ALEAM_DISTRIBUTIONS_DIRICHLET_H */