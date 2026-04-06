/**
 * @file ai.h
 * @brief AI/ML specific randomness features using Aleam true randomness
 * @license MIT
 * 
 * This file provides AI/ML specific random utilities that leverage
 * true randomness for improved exploration, generalization, and creativity
 * in machine learning models.
 * 
 * Features:
 * - Gradient noise injection for training
 * - Latent space sampling for generative models
 * - Dropout mask generation
 * - Data augmentation parameters
 * - Mini-batch sampling
 * - Reinforcement learning exploration noise
 */

#ifndef ALEAM_AI_H
#define ALEAM_AI_H

#include <vector>
#include <cstdint>
#include <string>
#include <functional>
#include <stdexcept>
#include "../core/aleam_core.h"

namespace aleam {
namespace ai {

/* ============================================================================
 * AIRandom - AI-Specific Random Utilities
 * ============================================================================ */

/**
 * @brief AI-specific random utilities using true randomness
 * 
 * This class provides methods for common AI/ML random operations
 * such as gradient noise, latent vector sampling, dropout masks,
 * and data augmentation parameters.
 */
class AIRandom {
public:
    /**
     * @brief Construct a new AIRandom object
     * 
     * @param rng Pointer to AleamCore instance (uses thread-local if nullptr)
     */
    explicit AIRandom(AleamCore* rng = nullptr);
    
    /**
     * @brief Destroy AIRandom object
     */
    ~AIRandom() = default;
    
    /* ========================================================================
     * Gradient Noise
     * ======================================================================== */
    
    /**
     * @brief Generate noise for gradient perturbation
     * 
     * Adds true random Gaussian noise to gradients to help escape
     * local minima and improve generalization.
     * 
     * @param shape Number of elements in gradient tensor
     * @param scale Standard deviation of noise
     * @return std::vector<double> Noise array (0-centered Gaussian)
     */
    std::vector<double> gradient_noise(size_t shape, double scale = 0.1);
    
    /**
     * @brief Generate 2D gradient noise
     * 
     * @param rows Number of rows
     * @param cols Number of columns
     * @param scale Standard deviation of noise
     * @return std::vector<std::vector<double>> 2D noise array
     */
    std::vector<std::vector<double>> gradient_noise_2d(size_t rows, size_t cols, double scale = 0.1);
    
    /* ========================================================================
     * Latent Space Vectors
     * ======================================================================== */
    
    /**
     * @brief Generate latent space vector
     * 
     * Samples a vector from either normal or uniform distribution
     * for use in latent space of generative models (VAEs, GANs).
     * 
     * @param dim Dimension of latent space
     * @param distribution "normal" or "uniform"
     * @return std::vector<double> Latent vector
     */
    std::vector<double> latent_vector(size_t dim, const std::string& distribution = "normal");
    
    /**
     * @brief Generate batch of latent vectors
     * 
     * @param batch_size Number of vectors to generate
     * @param dim Dimension of latent space
     * @param distribution "normal" or "uniform"
     * @return std::vector<std::vector<double>> Batch of latent vectors
     */
    std::vector<std::vector<double>> latent_batch(size_t batch_size, size_t dim,
                                                    const std::string& distribution = "normal");
    
    /* ========================================================================
     * Dropout Masks
     * ======================================================================== */
    
    /**
     * @brief Generate dropout mask
     * 
     * Creates a binary mask where each element is 1 (keep) with probability p
     * and 0 (drop) with probability 1-p. Used for dropout regularization.
     * 
     * @param size Number of neurons
     * @param keep_prob Probability of keeping a neuron (0.5 = keep 50%)
     * @return std::vector<uint8_t> Binary mask (1 = keep, 0 = drop)
     */
    std::vector<uint8_t> dropout_mask(size_t size, double keep_prob = 0.5);
    
    /**
     * @brief Generate 2D dropout mask
     * 
     * @param rows Number of rows
     * @param cols Number of columns
     * @param keep_prob Probability of keeping a neuron
     * @return std::vector<std::vector<uint8_t>> 2D binary mask
     */
    std::vector<std::vector<uint8_t>> dropout_mask_2d(size_t rows, size_t cols, double keep_prob = 0.5);
    
    /* ========================================================================
     * Data Augmentation Parameters
     * ======================================================================== */
    
    /**
     * @brief Structure holding data augmentation parameters
     */
    struct AugmentationParams {
        double rotation;        /**< Rotation angle in degrees (-30 to 30) */
        double scale;           /**< Scale factor (0.8 to 1.2) */
        double brightness;      /**< Brightness adjustment (0.7 to 1.3) */
        double contrast;        /**< Contrast adjustment (0.8 to 1.2) */
        bool flip_horizontal;   /**< Whether to flip horizontally */
        bool flip_vertical;     /**< Whether to flip vertically */
    };
    
    /**
     * @brief Generate random data augmentation parameters
     * 
     * @return AugmentationParams Random parameters for image augmentation
     */
    AugmentationParams augmentation_params();
    
    /* ========================================================================
     * Mini-Batch Sampling
     * ======================================================================== */
    
    /**
     * @brief Sample mini-batch indices without replacement
     * 
     * @param dataset_size Total size of dataset
     * @param batch_size Desired batch size
     * @return std::vector<size_t> Batch indices
     */
    std::vector<size_t> mini_batch(size_t dataset_size, size_t batch_size);
    
    /**
     * @brief Sample multiple mini-batches
     * 
     * @param dataset_size Total size of dataset
     * @param batch_size Desired batch size
     * @param num_batches Number of batches to generate
     * @return std::vector<std::vector<size_t>> List of batch indices
     */
    std::vector<std::vector<size_t>> mini_batches(size_t dataset_size, size_t batch_size, size_t num_batches);
    
    /* ========================================================================
     * Reinforcement Learning Exploration
     * ======================================================================== */
    
    /**
     * @brief Generate exploration noise for reinforcement learning
     * 
     * Adds true random noise to actions for better exploration
     * in RL environments.
     * 
     * @param action_dim Dimension of action space
     * @param scale Standard deviation of noise
     * @return std::vector<double> Noise vector
     */
    std::vector<double> exploration_noise(size_t action_dim, double scale = 0.2);
    
private:
    AleamCore* m_rng;               /**< Pointer to AleamCore instance */
    bool m_owns_rng;                /**< Whether we own the RNG instance */
    
    /**
     * @brief Get RNG instance (creates if needed)
     * 
     * @return AleamCore& Reference to RNG
     */
    AleamCore& get_rng();
};

/* ============================================================================
 * GradientNoise - Gradient Noise with Decay
 * ============================================================================ */

/**
 * @brief Gradient noise injection for training with decay
 * 
 * Adds true random noise to gradients during training with
 * decreasing scale over time to help escape local minima
 * early and fine-tune later.
 */
class GradientNoise {
public:
    /**
     * @brief Construct a new GradientNoise object
     * 
     * @param initial_scale Initial noise standard deviation
     * @param decay Decay factor per step (e.g., 0.99)
     * @param rng Pointer to AleamCore instance
     */
    GradientNoise(double initial_scale = 0.01, double decay = 0.99, AleamCore* rng = nullptr);
    
    /**
     * @brief Destroy GradientNoise object
     */
    ~GradientNoise() = default;
    
    /**
     * @brief Add true random noise to gradients
     * 
     * @param gradients Input gradient array
     * @return std::vector<double> Gradients with added noise
     */
    std::vector<double> add_noise(const std::vector<double>& gradients);
    
    /**
     * @brief Add true random noise to 2D gradients
     * 
     * @param gradients Input 2D gradient array
     * @return std::vector<std::vector<double>> Gradients with added noise
     */
    std::vector<std::vector<double>> add_noise_2d(const std::vector<std::vector<double>>& gradients);
    
    /**
     * @brief Reset step counter
     */
    void reset();
    
    /**
     * @brief Get current step
     * 
     * @return size_t Current step count
     */
    size_t get_step() const { return m_step; }
    
    /**
     * @brief Get current noise scale
     * 
     * @return double Current noise standard deviation
     */
    double get_current_scale() const;
    
private:
    double m_initial_scale;     /**< Initial noise scale */
    double m_decay;              /**< Decay factor per step */
    AleamCore* m_rng;           /**< Pointer to AleamCore instance */
    bool m_owns_rng;            /**< Whether we own the RNG */
    size_t m_step;              /**< Current step counter */
    
    AleamCore& get_rng();
};

/* ============================================================================
 * LatentSampler - Latent Space Sampling
 * ============================================================================ */

/**
 * @brief Latent space sampler for generative models
 * 
 * Provides true random sampling from latent space with
 * interpolation capabilities for smooth transitions.
 */
class LatentSampler {
public:
    /**
     * @brief Construct a new LatentSampler object
     * 
     * @param latent_dim Dimension of latent space
     * @param distribution "normal" or "uniform"
     * @param rng Pointer to AleamCore instance
     */
    LatentSampler(size_t latent_dim, const std::string& distribution = "normal", AleamCore* rng = nullptr);
    
    /**
     * @brief Destroy LatentSampler object
     */
    ~LatentSampler() = default;
    
    /**
     * @brief Sample n latent vectors
     * 
     * @param n Number of vectors to sample
     * @return std::vector<std::vector<double>> Sampled vectors
     */
    std::vector<std::vector<double>> sample(size_t n = 1);
    
    /**
     * @brief Sample a single latent vector
     * 
     * @return std::vector<double> Single latent vector
     */
    std::vector<double> sample_one();
    
    /**
     * @brief Interpolate between two latent vectors
     * 
     * @param z1 First latent vector
     * @param z2 Second latent vector
     * @param steps Number of interpolation steps
     * @return std::vector<std::vector<double>> Interpolated vectors
     */
    std::vector<std::vector<double>> interpolate(const std::vector<double>& z1,
                                                   const std::vector<double>& z2,
                                                   size_t steps = 10);
    
    /**
     * @brief Get latent dimension
     * 
     * @return size_t Latent dimension
     */
    size_t get_latent_dim() const { return m_latent_dim; }
    
    /**
     * @brief Get distribution type
     * 
     * @return std::string Distribution name
     */
    std::string get_distribution() const { return m_distribution; }
    
private:
    size_t m_latent_dim;            /**< Dimension of latent space */
    std::string m_distribution;     /**< Distribution type ("normal" or "uniform") */
    AleamCore* m_rng;               /**< Pointer to AleamCore instance */
    bool m_owns_rng;                /**< Whether we own the RNG */
    
    AleamCore& get_rng();
    
    /**
     * @brief Generate a single latent value
     * 
     * @return double Random value from selected distribution
     */
    double generate_value();
};

}  // namespace ai
}  // namespace aleam

#endif /* ALEAM_AI_H */