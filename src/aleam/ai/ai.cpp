/**
 * @file ai.cpp
 * @brief Implementation of AI/ML specific randomness features
 * @license MIT
 * 
 * This file implements the AI/ML random utilities using true randomness
 * from AleamCore.
 */

#include "ai.h"
#include "../distributions/distributions.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace aleam {
namespace ai {

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * @brief Generate a random double in [0, 1)
 */
inline double rand_double(AleamCore& rng) {
    return rng.random();
}

/**
 * @brief Generate a random integer in [0, n-1]
 */
inline size_t rand_index(AleamCore& rng, size_t n) {
    return static_cast<size_t>(rng.random() * n);
}

/* ============================================================================
 * AIRandom Implementation
 * ============================================================================ */

AIRandom::AIRandom(AleamCore* rng)
    : m_rng(rng)
    , m_owns_rng(false) {
    if (m_rng == nullptr) {
        m_rng = &get_thread_local_instance();
        m_owns_rng = false;  // Thread-local instance not owned
    }
}

AleamCore& AIRandom::get_rng() {
    return *m_rng;
}

std::vector<double> AIRandom::gradient_noise(size_t shape, double scale) {
    if (scale <= 0.0) {
        throw std::invalid_argument("scale must be > 0");
    }
    
    std::vector<double> noise(shape);
    AleamCore& rng = get_rng();
    
    for (size_t i = 0; i < shape; i++) {
        noise[i] = distributions::normal(rng, 0.0, scale);
    }
    
    return noise;
}

std::vector<std::vector<double>> AIRandom::gradient_noise_2d(size_t rows, size_t cols, double scale) {
    if (rows == 0 || cols == 0) {
        return std::vector<std::vector<double>>();
    }
    
    std::vector<std::vector<double>> noise(rows, std::vector<double>(cols));
    AleamCore& rng = get_rng();
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            noise[i][j] = distributions::normal(rng, 0.0, scale);
        }
    }
    
    return noise;
}

std::vector<double> AIRandom::latent_vector(size_t dim, const std::string& distribution) {
    if (dim == 0) {
        throw std::invalid_argument("dimension must be > 0");
    }
    
    std::vector<double> vec(dim);
    AleamCore& rng = get_rng();
    
    if (distribution == "normal") {
        for (size_t i = 0; i < dim; i++) {
            vec[i] = distributions::normal(rng, 0.0, 1.0);
        }
    } else if (distribution == "uniform") {
        for (size_t i = 0; i < dim; i++) {
            vec[i] = distributions::uniform(rng, -1.0, 1.0);
        }
    } else {
        throw std::invalid_argument("distribution must be 'normal' or 'uniform'");
    }
    
    return vec;
}

std::vector<std::vector<double>> AIRandom::latent_batch(size_t batch_size, size_t dim,
                                                          const std::string& distribution) {
    std::vector<std::vector<double>> batch;
    batch.reserve(batch_size);
    
    for (size_t i = 0; i < batch_size; i++) {
        batch.push_back(latent_vector(dim, distribution));
    }
    
    return batch;
}

std::vector<uint8_t> AIRandom::dropout_mask(size_t size, double keep_prob) {
    if (keep_prob < 0.0 || keep_prob > 1.0) {
        throw std::invalid_argument("keep_prob must be between 0 and 1");
    }
    
    std::vector<uint8_t> mask(size);
    AleamCore& rng = get_rng();
    
    for (size_t i = 0; i < size; i++) {
        mask[i] = (rng.random() < keep_prob) ? 1 : 0;
    }
    
    return mask;
}

std::vector<std::vector<uint8_t>> AIRandom::dropout_mask_2d(size_t rows, size_t cols, double keep_prob) {
    if (rows == 0 || cols == 0) {
        return std::vector<std::vector<uint8_t>>();
    }
    
    std::vector<std::vector<uint8_t>> mask(rows, std::vector<uint8_t>(cols));
    AleamCore& rng = get_rng();
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            mask[i][j] = (rng.random() < keep_prob) ? 1 : 0;
        }
    }
    
    return mask;
}

AIRandom::AugmentationParams AIRandom::augmentation_params() {
    AleamCore& rng = get_rng();
    AugmentationParams params;
    
    params.rotation = distributions::uniform(rng, -30.0, 30.0);
    params.scale = distributions::uniform(rng, 0.8, 1.2);
    params.brightness = distributions::uniform(rng, 0.7, 1.3);
    params.contrast = distributions::uniform(rng, 0.8, 1.2);
    params.flip_horizontal = (rng.random() > 0.5);
    params.flip_vertical = (rng.random() > 0.5);
    
    return params;
}

std::vector<size_t> AIRandom::mini_batch(size_t dataset_size, size_t batch_size) {
    if (batch_size > dataset_size) {
        throw std::invalid_argument("batch_size cannot exceed dataset_size");
    }
    
    /* Create shuffled indices */
    std::vector<size_t> indices(dataset_size);
    for (size_t i = 0; i < dataset_size; i++) {
        indices[i] = i;
    }
    
    /* Fisher-Yates shuffle on first batch_size elements */
    AleamCore& rng = get_rng();
    for (size_t i = 0; i < batch_size; i++) {
        size_t j = i + rand_index(rng, dataset_size - i);
        std::swap(indices[i], indices[j]);
    }
    
    /* Return first batch_size elements */
    indices.resize(batch_size);
    return indices;
}

std::vector<std::vector<size_t>> AIRandom::mini_batches(size_t dataset_size, size_t batch_size, size_t num_batches) {
    std::vector<std::vector<size_t>> batches;
    batches.reserve(num_batches);
    
    for (size_t i = 0; i < num_batches; i++) {
        batches.push_back(mini_batch(dataset_size, batch_size));
    }
    
    return batches;
}

std::vector<double> AIRandom::exploration_noise(size_t action_dim, double scale) {
    if (scale <= 0.0) {
        throw std::invalid_argument("scale must be > 0");
    }
    
    std::vector<double> noise(action_dim);
    AleamCore& rng = get_rng();
    
    for (size_t i = 0; i < action_dim; i++) {
        noise[i] = distributions::normal(rng, 0.0, scale);
    }
    
    return noise;
}

/* ============================================================================
 * GradientNoise Implementation
 * ============================================================================ */

GradientNoise::GradientNoise(double initial_scale, double decay, AleamCore* rng)
    : m_initial_scale(initial_scale)
    , m_decay(decay)
    , m_rng(rng)
    , m_owns_rng(false)
    , m_step(0) {
    if (initial_scale <= 0.0) {
        throw std::invalid_argument("initial_scale must be > 0");
    }
    if (decay <= 0.0 || decay > 1.0) {
        throw std::invalid_argument("decay must be between 0 and 1");
    }
    
    if (m_rng == nullptr) {
        m_rng = &get_thread_local_instance();
        m_owns_rng = false;
    }
}

AleamCore& GradientNoise::get_rng() {
    return *m_rng;
}

double GradientNoise::get_current_scale() const {
    return m_initial_scale * std::pow(m_decay, static_cast<double>(m_step));
}

std::vector<double> GradientNoise::add_noise(const std::vector<double>& gradients) {
    double current_scale = get_current_scale();
    std::vector<double> result(gradients.size());
    AleamCore& rng = get_rng();
    
    for (size_t i = 0; i < gradients.size(); i++) {
        double noise = distributions::normal(rng, 0.0, current_scale);
        result[i] = gradients[i] + noise;
    }
    
    m_step++;
    return result;
}

std::vector<std::vector<double>> GradientNoise::add_noise_2d(const std::vector<std::vector<double>>& gradients) {
    if (gradients.empty()) {
        return std::vector<std::vector<double>>();
    }
    
    double current_scale = get_current_scale();
    size_t rows = gradients.size();
    size_t cols = gradients[0].size();
    
    std::vector<std::vector<double>> result(rows, std::vector<double>(cols));
    AleamCore& rng = get_rng();
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double noise = distributions::normal(rng, 0.0, current_scale);
            result[i][j] = gradients[i][j] + noise;
        }
    }
    
    m_step++;
    return result;
}

void GradientNoise::reset() {
    m_step = 0;
}

/* ============================================================================
 * LatentSampler Implementation
 * ============================================================================ */

LatentSampler::LatentSampler(size_t latent_dim, const std::string& distribution, AleamCore* rng)
    : m_latent_dim(latent_dim)
    , m_distribution(distribution)
    , m_rng(rng)
    , m_owns_rng(false) {
    if (latent_dim == 0) {
        throw std::invalid_argument("latent_dim must be > 0");
    }
    if (distribution != "normal" && distribution != "uniform") {
        throw std::invalid_argument("distribution must be 'normal' or 'uniform'");
    }
    
    if (m_rng == nullptr) {
        m_rng = &get_thread_local_instance();
        m_owns_rng = false;
    }
}

AleamCore& LatentSampler::get_rng() {
    return *m_rng;
}

double LatentSampler::generate_value() {
    AleamCore& rng = get_rng();
    
    if (m_distribution == "normal") {
        return distributions::normal(rng, 0.0, 1.0);
    } else {  // uniform
        return distributions::uniform(rng, -1.0, 1.0);
    }
}

std::vector<double> LatentSampler::sample_one() {
    std::vector<double> vec(m_latent_dim);
    
    for (size_t i = 0; i < m_latent_dim; i++) {
        vec[i] = generate_value();
    }
    
    return vec;
}

std::vector<std::vector<double>> LatentSampler::sample(size_t n) {
    std::vector<std::vector<double>> samples;
    samples.reserve(n);
    
    for (size_t i = 0; i < n; i++) {
        samples.push_back(sample_one());
    }
    
    return samples;
}

std::vector<std::vector<double>> LatentSampler::interpolate(const std::vector<double>& z1,
                                                              const std::vector<double>& z2,
                                                              size_t steps) {
    if (z1.size() != m_latent_dim || z2.size() != m_latent_dim) {
        throw std::invalid_argument("vector dimensions must match latent_dim");
    }
    if (steps < 2) {
        throw std::invalid_argument("steps must be at least 2");
    }
    
    std::vector<std::vector<double>> interpolations;
    interpolations.reserve(steps);
    
    for (size_t i = 0; i < steps; i++) {
        double alpha = static_cast<double>(i) / static_cast<double>(steps - 1);
        std::vector<double> vec(m_latent_dim);
        
        for (size_t j = 0; j < m_latent_dim; j++) {
            vec[j] = (1.0 - alpha) * z1[j] + alpha * z2[j];
        }
        
        interpolations.push_back(vec);
    }
    
    return interpolations;
}

}  // namespace ai
}  // namespace aleam