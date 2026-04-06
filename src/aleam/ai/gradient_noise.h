/**
 * @file gradient_noise.h
 * @brief Gradient noise injection for training neural networks
 * @license MIT
 * 
 * Implements gradient noise injection with exponential decay.
 * Adding true random noise to gradients during training helps
 * escape local minima and improves generalization.
 */

#ifndef ALEAM_AI_GRADIENT_NOISE_H
#define ALEAM_AI_GRADIENT_NOISE_H

#include <vector>
#include <cmath>
#include "../core/aleam_core.h"
#include "../distributions/normal.h"

namespace aleam {
namespace ai {

/**
 * @brief Gradient noise injection with exponential decay
 * 
 * This class adds true random Gaussian noise to gradients during
 * training. The noise scale decays exponentially over time:
 * 
 *     scale(t) = initial_scale * decay^t
 * 
 * This allows for more exploration early in training and
 * fine-tuning later.
 * 
 * Example:
 * @code
 *   GradientNoise noise(0.01, 0.99);
 *   for (int step = 0; step < steps; ++step) {
 *       auto gradients = compute_gradients();
 *       auto noisy_gradients = noise.add_noise(gradients);
 *       optimizer.update(noisy_gradients);
 *   }
 * @endcode
 */
template<typename RealType = double>
class GradientNoise {
public:
    /**
     * @brief Construct a GradientNoise object
     * 
     * @param initial_scale Initial noise standard deviation (> 0)
     * @param decay Decay factor per step (0 < decay ≤ 1)
     * @param rng Reference to AleamCore instance
     */
    GradientNoise(RealType initial_scale = 0.01, 
                  RealType decay = 0.99,
                  AleamCore& rng)
        : m_initial_scale(initial_scale)
        , m_decay(decay)
        , m_rng(rng)
        , m_step(0) {
        if (initial_scale <= 0.0) {
            throw std::invalid_argument("initial_scale must be > 0");
        }
        if (decay <= 0.0 || decay > 1.0) {
            throw std::invalid_argument("decay must be between 0 and 1");
        }
    }
    
    /**
     * @brief Get current noise scale
     * 
     * @return RealType Current noise standard deviation
     */
    RealType current_scale() const {
        return m_initial_scale * std::pow(m_decay, static_cast<RealType>(m_step));
    }
    
    /**
     * @brief Add noise to a 1D gradient vector
     * 
     * @param gradients Input gradient vector
     * @return std::vector<RealType> Gradients with added noise
     */
    std::vector<RealType> add_noise(const std::vector<RealType>& gradients) {
        RealType scale = current_scale();
        distributions::NormalDistribution<RealType> normal(0.0, scale, m_rng);
        
        std::vector<RealType> result(gradients.size());
        for (size_t i = 0; i < gradients.size(); ++i) {
            result[i] = gradients[i] + normal();
        }
        
        m_step++;
        return result;
    }
    
    /**
     * @brief Add noise to a 2D gradient matrix
     * 
     * @param gradients Input 2D gradient matrix
     * @return std::vector<std::vector<RealType>> Gradients with added noise
     */
    std::vector<std::vector<RealType>> add_noise(
        const std::vector<std::vector<RealType>>& gradients) {
        
        if (gradients.empty()) {
            return std::vector<std::vector<RealType>>();
        }
        
        RealType scale = current_scale();
        distributions::NormalDistribution<RealType> normal(0.0, scale, m_rng);
        
        size_t rows = gradients.size();
        size_t cols = gradients[0].size();
        
        std::vector<std::vector<RealType>> result(rows, std::vector<RealType>(cols));
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] = gradients[i][j] + normal();
            }
        }
        
        m_step++;
        return result;
    }
    
    /**
     * @brief Add noise in-place to a 1D gradient vector
     * 
     * @param gradients Gradient vector (modified in-place)
     */
    void add_noise_inplace(std::vector<RealType>& gradients) {
        RealType scale = current_scale();
        distributions::NormalDistribution<RealType> normal(0.0, scale, m_rng);
        
        for (size_t i = 0; i < gradients.size(); ++i) {
            gradients[i] += normal();
        }
        
        m_step++;
    }
    
    /**
     * @brief Add noise in-place to a 2D gradient matrix
     * 
     * @param gradients Gradient matrix (modified in-place)
     */
    void add_noise_inplace(std::vector<std::vector<RealType>>& gradients) {
        if (gradients.empty()) return;
        
        RealType scale = current_scale();
        distributions::NormalDistribution<RealType> normal(0.0, scale, m_rng);
        
        for (auto& row : gradients) {
            for (auto& val : row) {
                val += normal();
            }
        }
        
        m_step++;
    }
    
    /**
     * @brief Reset step counter
     * 
     * Resets the step count to 0, causing noise scale to return
     * to initial_scale on the next call.
     */
    void reset() {
        m_step = 0;
    }
    
    /**
     * @brief Get current step number
     * 
     * @return size_t Current step count
     */
    size_t step() const {
        return m_step;
    }
    
    /**
     * @brief Set step number manually
     * 
     * @param step New step number
     */
    void set_step(size_t step) {
        m_step = step;
    }
    
    /**
     * @brief Get initial noise scale
     * 
     * @return RealType Initial scale
     */
    RealType initial_scale() const {
        return m_initial_scale;
    }
    
    /**
     * @brief Get decay factor
     * 
     * @return RealType Decay factor
     */
    RealType decay() const {
        return m_decay;
    }
    
    /**
     * @brief Set new initial scale (resets step to 0)
     * 
     * @param scale New initial scale
     */
    void set_initial_scale(RealType scale) {
        if (scale <= 0.0) {
            throw std::invalid_argument("scale must be > 0");
        }
        m_initial_scale = scale;
        m_step = 0;
    }
    
    /**
     * @brief Set new decay factor
     * 
     * @param decay New decay factor
     */
    void set_decay(RealType decay) {
        if (decay <= 0.0 || decay > 1.0) {
            throw std::invalid_argument("decay must be between 0 and 1");
        }
        m_decay = decay;
    }
    
private:
    RealType m_initial_scale;   /**< Initial noise scale */
    RealType m_decay;            /**< Decay factor per step */
    AleamCore& m_rng;            /**< Random number generator */
    size_t m_step;               /**< Current step counter */
};

}  // namespace ai
}  // namespace aleam

#endif /* ALEAM_AI_GRADIENT_NOISE_H */