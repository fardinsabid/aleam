/**
 * @file gamma.h
 * @brief Gamma distribution using Aleam true randomness
 * @license MIT
 * 
 * Implements the Gamma distribution using Marsaglia & Tsang's
 * rejection sampling algorithm for shape >= 1, and Johnk's method
 * for shape < 1.
 * 
 * The Gamma distribution is a two-parameter family of continuous
 * probability distributions. It generalizes the exponential
 * distribution (shape=1) and chi-square distribution (shape=k/2, scale=2).
 */

#ifndef ALEAM_DISTRIBUTIONS_GAMMA_H
#define ALEAM_DISTRIBUTIONS_GAMMA_H

#include <cmath>
#include <stdexcept>
#include "../core/aleam_core.h"
#include "normal.h"

namespace aleam {
namespace distributions {

/**
 * @brief Gamma distribution class
 * 
 * Represents a Gamma distribution with shape parameter k and scale parameter θ.
 * 
 * Probability density function:
 *     f(x) = x^(k-1) * e^(-x/θ) / [Γ(k) * θ^k] for x ≥ 0
 * 
 * Mean = kθ
 * Variance = kθ²
 * 
 * @tparam RealType Floating point type (float or double)
 */
template<typename RealType = double>
class GammaDistribution {
public:
    /**
     * @brief Construct a Gamma distribution
     * 
     * @param shape Shape parameter (k) (> 0)
     * @param scale Scale parameter (θ) (> 0)
     * @param rng Reference to AleamCore instance
     */
    GammaDistribution(RealType shape, RealType scale, AleamCore& rng)
        : m_shape(shape)
        , m_scale(scale)
        , m_rng(rng) {
        if (shape <= 0.0) {
            throw std::invalid_argument("shape must be > 0");
        }
        if (scale <= 0.0) {
            throw std::invalid_argument("scale must be > 0");
        }
        
        // Precompute constants for Marsaglia & Tsang method
        if (m_shape >= 1.0) {
            m_d = m_shape - 1.0 / 3.0;
            m_c = 1.0 / std::sqrt(9.0 * m_d);
        }
    }
    
    /**
     * @brief Generate a random number from the distribution
     * 
     * Uses Marsaglia & Tsang's rejection sampling for shape >= 1.
     * Uses Johnk's acceptance-rejection for shape < 1.
     * 
     * @return RealType Random value from Gamma(k, θ)
     */
    RealType operator()() {
        if (m_shape < 1.0) {
            return sample_shape_less_than_one();
        }
        return sample_shape_ge_one();
    }
    
    /**
     * @brief Get the shape parameter (k)
     * 
     * @return RealType Shape parameter
     */
    RealType shape() const { return m_shape; }
    
    /**
     * @brief Get the scale parameter (θ)
     * 
     * @return RealType Scale parameter
     */
    RealType scale() const { return m_scale; }
    
    /**
     * @brief Get the mean of the distribution (kθ)
     * 
     * @return RealType Mean
     */
    RealType mean() const { return m_shape * m_scale; }
    
    /**
     * @brief Get the variance of the distribution (kθ²)
     * 
     * @return RealType Variance
     */
    RealType variance() const { return m_shape * m_scale * m_scale; }
    
    /**
     * @brief Probability density function
     * 
     * @param x Point at which to evaluate PDF (x ≥ 0)
     * @return RealType PDF value at x
     */
    RealType pdf(RealType x) const {
        if (x < 0.0) return 0.0;
        
        // Using log to avoid overflow
        RealType log_pdf = (m_shape - 1.0) * std::log(x) -
                           x / m_scale -
                           std::lgamma(m_shape) -
                           m_shape * std::log(m_scale);
        
        return std::exp(log_pdf);
    }
    
private:
    /**
     * @brief Sample from Gamma for shape >= 1 (Marsaglia & Tsang method)
     * 
     * @return RealType Random value
     */
    RealType sample_shape_ge_one() {
        NormalDistribution<RealType> normal(0.0, 1.0, m_rng);
        
        while (true) {
            RealType v = normal();
            RealType x = (1.0 + m_c * v);
            
            // x must be positive
            if (x <= 0.0) {
                continue;
            }
            
            x = x * x * x;  // x³
            RealType u = static_cast<RealType>(m_rng.random());
            
            // Quick accept using quadratic bound
            if (u < 1.0 - 0.0331 * (v * v * v * v)) {
                return m_d * x * m_scale;
            }
            
            // Slower accept using logarithms
            if (std::log(u) < 0.5 * (v * v) + m_d * (1.0 - x + std::log(x))) {
                return m_d * x * m_scale;
            }
        }
    }
    
    /**
     * @brief Sample from Gamma for shape < 1 (Johnk's method)
     * 
     * Uses Gamma(shape+1) and rejection sampling.
     * 
     * @return RealType Random value
     */
    RealType sample_shape_less_than_one() {
        GammaDistribution<RealType> gamma_plus_one(m_shape + 1.0, 1.0, m_rng);
        
        while (true) {
            RealType u = static_cast<RealType>(m_rng.random());
            RealType v = static_cast<RealType>(m_rng.random());
            RealType x = std::pow(u, 1.0 / m_shape);
            RealType y = std::pow(v, 1.0 / (1.0 - m_shape));
            
            if (x + y <= 1.0) {
                RealType z = x / (x + y);
                RealType w = gamma_plus_one();
                return w * z * m_scale;
            }
        }
    }
    
    RealType m_shape;        /**< Shape parameter (k) */
    RealType m_scale;        /**< Scale parameter (θ) */
    AleamCore& m_rng;        /**< Random number generator */
    
    // Precomputed for shape >= 1
    RealType m_d;            /**< d = shape - 1/3 */
    RealType m_c;            /**< c = 1 / sqrt(9 * d) */
};

/**
 * @brief Generate a Gamma distributed random number
 * 
 * @param rng Reference to AleamCore instance
 * @param shape Shape parameter (k > 0)
 * @param scale Scale parameter (θ > 0)
 * @return double Random value from Gamma(k, θ)
 */
inline double gamma(AleamCore& rng, double shape, double scale) {
    GammaDistribution<double> dist(shape, scale, rng);
    return dist();
}

}  // namespace distributions
}  // namespace aleam

#endif /* ALEAM_DISTRIBUTIONS_GAMMA_H */