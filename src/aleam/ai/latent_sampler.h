/**
 * @file latent_sampler.h
 * @brief Latent space sampling for generative models
 * @license MIT
 * 
 * Implements latent space sampling for variational autoencoders (VAEs),
 * generative adversarial networks (GANs), and other generative models.
 * Supports normal and uniform distributions with interpolation.
 */

#ifndef ALEAM_AI_LATENT_SAMPLER_H
#define ALEAM_AI_LATENT_SAMPLER_H

#include <vector>
#include <string>
#include <cmath>
#include "../core/aleam_core.h"
#include "../distributions/normal.h"
#include "../distributions/uniform.h"

namespace aleam {
namespace ai {

/**
 * @brief Latent space sampler for generative models
 * 
 * Provides true random sampling from latent space with support for:
 * - Normal distribution (standard Gaussian)
 * - Uniform distribution ([-1, 1] range)
 * - Batch sampling
 * - Linear interpolation between latent vectors
 * 
 * Example:
 * @code
 *   LatentSampler sampler(128, "normal");
 *   auto z1 = sampler.sample_one();
 *   auto z2 = sampler.sample_one();
 *   auto interpolated = sampler.interpolate(z1, z2, 10);
 * @endcode
 */
template<typename RealType = double>
class LatentSampler {
public:
    /**
     * @brief Construct a LatentSampler object
     * 
     * @param latent_dim Dimension of latent space (> 0)
     * @param distribution Distribution type ("normal" or "uniform")
     * @param rng Reference to AleamCore instance
     */
    LatentSampler(size_t latent_dim, 
                  const std::string& distribution = "normal",
                  AleamCore& rng)
        : m_latent_dim(latent_dim)
        , m_distribution(distribution)
        , m_rng(rng) {
        if (latent_dim == 0) {
            throw std::invalid_argument("latent_dim must be > 0");
        }
        if (distribution != "normal" && distribution != "uniform") {
            throw std::invalid_argument("distribution must be 'normal' or 'uniform'");
        }
    }
    
    /**
     * @brief Sample a single latent vector
     * 
     * @return std::vector<RealType> Latent vector of dimension latent_dim
     */
    std::vector<RealType> sample_one() {
        std::vector<RealType> vec(m_latent_dim);
        
        if (m_distribution == "normal") {
            distributions::NormalDistribution<RealType> normal(0.0, 1.0, m_rng);
            for (size_t i = 0; i < m_latent_dim; ++i) {
                vec[i] = normal();
            }
        } else {  // uniform
            distributions::UniformDistribution<RealType> uniform(-1.0, 1.0, m_rng);
            for (size_t i = 0; i < m_latent_dim; ++i) {
                vec[i] = uniform();
            }
        }
        
        return vec;
    }
    
    /**
     * @brief Sample multiple latent vectors
     * 
     * @param n Number of vectors to sample
     * @return std::vector<std::vector<RealType>> Batch of latent vectors
     */
    std::vector<std::vector<RealType>> sample(size_t n) {
        std::vector<std::vector<RealType>> samples;
        samples.reserve(n);
        
        for (size_t i = 0; i < n; ++i) {
            samples.push_back(sample_one());
        }
        
        return samples;
    }
    
    /**
     * @brief Sample a batch as a flattened 1D array
     * 
     * @param n Number of vectors to sample
     * @return std::vector<RealType> Flattened batch (size = n * latent_dim)
     */
    std::vector<RealType> sample_flat(size_t n) {
        std::vector<RealType> flat(n * m_latent_dim);
        
        if (m_distribution == "normal") {
            distributions::NormalDistribution<RealType> normal(0.0, 1.0, m_rng);
            for (size_t i = 0; i < n * m_latent_dim; ++i) {
                flat[i] = normal();
            }
        } else {  // uniform
            distributions::UniformDistribution<RealType> uniform(-1.0, 1.0, m_rng);
            for (size_t i = 0; i < n * m_latent_dim; ++i) {
                flat[i] = uniform();
            }
        }
        
        return flat;
    }
    
    /**
     * @brief Linear interpolation between two latent vectors
     * 
     * @param z1 First latent vector
     * @param z2 Second latent vector
     * @param steps Number of interpolation steps (including endpoints)
     * @return std::vector<std::vector<RealType>> Interpolated vectors
     */
    std::vector<std::vector<RealType>> interpolate(
        const std::vector<RealType>& z1,
        const std::vector<RealType>& z2,
        size_t steps = 10) {
        
        if (z1.size() != m_latent_dim || z2.size() != m_latent_dim) {
            throw std::invalid_argument("vector dimensions must match latent_dim");
        }
        if (steps < 2) {
            throw std::invalid_argument("steps must be at least 2");
        }
        
        std::vector<std::vector<RealType>> result;
        result.reserve(steps);
        
        for (size_t i = 0; i < steps; ++i) {
            RealType alpha = static_cast<RealType>(i) / static_cast<RealType>(steps - 1);
            std::vector<RealType> vec(m_latent_dim);
            
            for (size_t j = 0; j < m_latent_dim; ++j) {
                vec[j] = (1.0 - alpha) * z1[j] + alpha * z2[j];
            }
            
            result.push_back(vec);
        }
        
        return result;
    }
    
    /**
     * @brief Spherical linear interpolation (slerp) between two latent vectors
     * 
     * Maintains constant velocity along the geodesic on the unit sphere.
     * Recommended for normal distributions.
     * 
     * @param z1 First latent vector (will be normalized)
     * @param z2 Second latent vector (will be normalized)
     * @param steps Number of interpolation steps
     * @return std::vector<std::vector<RealType>> Interpolated vectors
     */
    std::vector<std::vector<RealType>> slerp(
        const std::vector<RealType>& z1,
        const std::vector<RealType>& z2,
        size_t steps = 10) {
        
        if (z1.size() != m_latent_dim || z2.size() != m_latent_dim) {
            throw std::invalid_argument("vector dimensions must match latent_dim");
        }
        if (steps < 2) {
            throw std::invalid_argument("steps must be at least 2");
        }
        
        // Normalize vectors
        RealType norm1 = 0.0, norm2 = 0.0;
        for (size_t i = 0; i < m_latent_dim; ++i) {
            norm1 += z1[i] * z1[i];
            norm2 += z2[i] * z2[i];
        }
        norm1 = std::sqrt(norm1);
        norm2 = std::sqrt(norm2);
        
        std::vector<RealType> u1(m_latent_dim), u2(m_latent_dim);
        for (size_t i = 0; i < m_latent_dim; ++i) {
            u1[i] = z1[i] / norm1;
            u2[i] = z2[i] / norm2;
        }
        
        // Compute dot product and angle
        RealType dot = 0.0;
        for (size_t i = 0; i < m_latent_dim; ++i) {
            dot += u1[i] * u2[i];
        }
        dot = std::clamp(dot, -1.0, 1.0);
        RealType theta = std::acos(dot);
        
        std::vector<std::vector<RealType>> result;
        result.reserve(steps);
        
        for (size_t i = 0; i < steps; ++i) {
            RealType t = static_cast<RealType>(i) / static_cast<RealType>(steps - 1);
            RealType sin_theta = std::sin(theta);
            
            RealType a = std::sin((1.0 - t) * theta) / sin_theta;
            RealType b = std::sin(t * theta) / sin_theta;
            
            std::vector<RealType> vec(m_latent_dim);
            for (size_t j = 0; j < m_latent_dim; ++j) {
                vec[j] = a * u1[j] + b * u2[j];
            }
            
            // Scale to original norms
            RealType norm = (1.0 - t) * norm1 + t * norm2;
            for (size_t j = 0; j < m_latent_dim; ++j) {
                vec[j] *= norm;
            }
            
            result.push_back(vec);
        }
        
        return result;
    }
    
    /**
     * @brief Get latent dimension
     * 
     * @return size_t Latent dimension
     */
    size_t latent_dim() const {
        return m_latent_dim;
    }
    
    /**
     * @brief Get distribution type
     * 
     * @return std::string Distribution name ("normal" or "uniform")
     */
    std::string distribution() const {
        return m_distribution;
    }
    
    /**
     * @brief Set distribution type
     * 
     * @param distribution "normal" or "uniform"
     */
    void set_distribution(const std::string& distribution) {
        if (distribution != "normal" && distribution != "uniform") {
            throw std::invalid_argument("distribution must be 'normal' or 'uniform'");
        }
        m_distribution = distribution;
    }
    
private:
    size_t m_latent_dim;                 /**< Dimension of latent space */
    std::string m_distribution;          /**< Distribution type */
    AleamCore& m_rng;                    /**< Random number generator */
};

}  // namespace ai
}  // namespace aleam

#endif /* ALEAM_AI_LATENT_SAMPLER_H */