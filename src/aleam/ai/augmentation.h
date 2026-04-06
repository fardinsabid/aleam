/**
 * @file augmentation.h
 * @brief Data augmentation parameter generation for computer vision
 * @license MIT
 * 
 * Implements random parameter generation for data augmentation
 * in computer vision pipelines. All parameters are generated
 * using true randomness from Aleam.
 */

#ifndef ALEAM_AI_AUGMENTATION_H
#define ALEAM_AI_AUGMENTATION_H

#include <vector>
#include <random>
#include "../core/aleam_core.h"
#include "../distributions/uniform.h"

namespace aleam {
namespace ai {

/**
 * @brief Data augmentation parameters for a single image
 * 
 * Contains all parameters needed for common image augmentation
 * operations.
 */
template<typename RealType = double>
struct AugmentationParams {
    RealType rotation;           /**< Rotation angle in degrees (-30 to 30) */
    RealType scale;              /**< Scale factor (0.8 to 1.2) */
    RealType shear_x;            /**< X-axis shear (-0.2 to 0.2) */
    RealType shear_y;            /**< Y-axis shear (-0.2 to 0.2) */
    RealType translation_x;      /**< X translation as fraction (-0.1 to 0.1) */
    RealType translation_y;      /**< Y translation as fraction (-0.1 to 0.1) */
    RealType brightness;         /**< Brightness adjustment (0.7 to 1.3) */
    RealType contrast;           /**< Contrast adjustment (0.8 to 1.2) */
    RealType saturation;         /**< Saturation adjustment (0.7 to 1.3) */
    RealType hue;                /**< Hue shift in degrees (-15 to 15) */
    bool flip_horizontal;        /**< Horizontal flip flag */
    bool flip_vertical;          /**< Vertical flip flag */
    bool grayscale;              /**< Convert to grayscale flag */
    
    /**
     * @brief Check if any geometric transformation is applied
     * 
     * @return bool True if rotation, scale, shear, or translation is non-zero
     */
    bool has_geometric() const {
        return rotation != 0.0 || scale != 1.0 || 
               shear_x != 0.0 || shear_y != 0.0 ||
               translation_x != 0.0 || translation_y != 0.0;
    }
    
    /**
     * @brief Check if any color transformation is applied
     * 
     * @return bool True if brightness, contrast, saturation, or hue is non-zero
     */
    bool has_color() const {
        return brightness != 1.0 || contrast != 1.0 || 
               saturation != 1.0 || hue != 0.0;
    }
    
    /**
     * @brief Check if any flip is applied
     * 
     * @return bool True if horizontal or vertical flip is true
     */
    bool has_flip() const {
        return flip_horizontal || flip_vertical;
    }
};

/**
 * @brief Data augmentation parameter generator
 * 
 * Generates random augmentation parameters for training
 * computer vision models. All parameters are within reasonable
 * ranges for common augmentation strategies.
 */
template<typename RealType = double>
class AugmentationGenerator {
public:
    /**
     * @brief Construct an AugmentationGenerator
     * 
     * @param rng Reference to AleamCore instance
     */
    explicit AugmentationGenerator(AleamCore& rng)
        : m_rng(rng)
        , m_rotation_range(30.0)
        , m_scale_range(0.8, 1.2)
        , m_shear_range(0.2)
        , m_translation_range(0.1)
        , m_brightness_range(0.7, 1.3)
        , m_contrast_range(0.8, 1.2)
        , m_saturation_range(0.7, 1.3)
        , m_hue_range(15.0)
        , m_flip_probability(0.5)
        , m_grayscale_probability(0.1) {}
    
    /**
     * @brief Generate a random augmentation parameter set
     * 
     * @return AugmentationParams<RealType> Random parameters
     */
    AugmentationParams<RealType> generate() {
        distributions::UniformDistribution<RealType> uniform(0.0, 1.0, m_rng);
        
        AugmentationParams<RealType> params;
        
        // Geometric parameters
        params.rotation = generate_rotation();
        params.scale = generate_scale();
        params.shear_x = generate_shear();
        params.shear_y = generate_shear();
        params.translation_x = generate_translation();
        params.translation_y = generate_translation();
        
        // Color parameters
        params.brightness = generate_brightness();
        params.contrast = generate_contrast();
        params.saturation = generate_saturation();
        params.hue = generate_hue();
        
        // Flip flags
        params.flip_horizontal = uniform() < m_flip_probability;
        params.flip_vertical = uniform() < m_flip_probability;
        params.grayscale = uniform() < m_grayscale_probability;
        
        return params;
    }
    
    /**
     * @brief Generate a batch of augmentation parameters
     * 
     * @param batch_size Number of parameter sets to generate
     * @return std::vector<AugmentationParams<RealType>> Batch of parameters
     */
    std::vector<AugmentationParams<RealType>> generate_batch(size_t batch_size) {
        std::vector<AugmentationParams<RealType>> batch;
        batch.reserve(batch_size);
        
        for (size_t i = 0; i < batch_size; ++i) {
            batch.push_back(generate());
        }
        
        return batch;
    }
    
    /**
     * @brief Set rotation range
     * 
     * @param max_angle Maximum rotation angle in degrees
     */
    void set_rotation_range(RealType max_angle) {
        m_rotation_range = std::abs(max_angle);
    }
    
    /**
     * @brief Set scale range
     * 
     * @param min_scale Minimum scale factor
     * @param max_scale Maximum scale factor
     */
    void set_scale_range(RealType min_scale, RealType max_scale) {
        m_scale_range.first = min_scale;
        m_scale_range.second = max_scale;
    }
    
    /**
     * @brief Set shear range
     * 
     * @param max_shear Maximum shear value (symmetric)
     */
    void set_shear_range(RealType max_shear) {
        m_shear_range = std::abs(max_shear);
    }
    
    /**
     * @brief Set translation range
     * 
     * @param max_translation Maximum translation as fraction of image size
     */
    void set_translation_range(RealType max_translation) {
        m_translation_range = std::abs(max_translation);
    }
    
    /**
     * @brief Set brightness range
     * 
     * @param min_brightness Minimum brightness factor
     * @param max_brightness Maximum brightness factor
     */
    void set_brightness_range(RealType min_brightness, RealType max_brightness) {
        m_brightness_range.first = min_brightness;
        m_brightness_range.second = max_brightness;
    }
    
    /**
     * @brief Set contrast range
     * 
     * @param min_contrast Minimum contrast factor
     * @param max_contrast Maximum contrast factor
     */
    void set_contrast_range(RealType min_contrast, RealType max_contrast) {
        m_contrast_range.first = min_contrast;
        m_contrast_range.second = max_contrast;
    }
    
    /**
     * @brief Set saturation range
     * 
     * @param min_saturation Minimum saturation factor
     * @param max_saturation Maximum saturation factor
     */
    void set_saturation_range(RealType min_saturation, RealType max_saturation) {
        m_saturation_range.first = min_saturation;
        m_saturation_range.second = max_saturation;
    }
    
    /**
     * @brief Set hue range
     * 
     * @param max_hue Maximum hue shift in degrees
     */
    void set_hue_range(RealType max_hue) {
        m_hue_range = std::abs(max_hue);
    }
    
    /**
     * @brief Set flip probability
     * 
     * @param prob Probability of flipping (0 to 1)
     */
    void set_flip_probability(RealType prob) {
        m_flip_probability = std::clamp(prob, 0.0, 1.0);
    }
    
    /**
     * @brief Set grayscale probability
     * 
     * @param prob Probability of grayscale conversion (0 to 1)
     */
    void set_grayscale_probability(RealType prob) {
        m_grayscale_probability = std::clamp(prob, 0.0, 1.0);
    }
    
private:
    /**
     * @brief Generate random rotation angle
     */
    RealType generate_rotation() {
        distributions::UniformDistribution<RealType> uniform(-m_rotation_range, m_rotation_range, m_rng);
        return uniform();
    }
    
    /**
     * @brief Generate random scale factor
     */
    RealType generate_scale() {
        distributions::UniformDistribution<RealType> uniform(m_scale_range.first, m_scale_range.second, m_rng);
        return uniform();
    }
    
    /**
     * @brief Generate random shear value
     */
    RealType generate_shear() {
        distributions::UniformDistribution<RealType> uniform(-m_shear_range, m_shear_range, m_rng);
        return uniform();
    }
    
    /**
     * @brief Generate random translation value
     */
    RealType generate_translation() {
        distributions::UniformDistribution<RealType> uniform(-m_translation_range, m_translation_range, m_rng);
        return uniform();
    }
    
    /**
     * @brief Generate random brightness factor
     */
    RealType generate_brightness() {
        distributions::UniformDistribution<RealType> uniform(m_brightness_range.first, m_brightness_range.second, m_rng);
        return uniform();
    }
    
    /**
     * @brief Generate random contrast factor
     */
    RealType generate_contrast() {
        distributions::UniformDistribution<RealType> uniform(m_contrast_range.first, m_contrast_range.second, m_rng);
        return uniform();
    }
    
    /**
     * @brief Generate random saturation factor
     */
    RealType generate_saturation() {
        distributions::UniformDistribution<RealType> uniform(m_saturation_range.first, m_saturation_range.second, m_rng);
        return uniform();
    }
    
    /**
     * @brief Generate random hue shift
     */
    RealType generate_hue() {
        distributions::UniformDistribution<RealType> uniform(-m_hue_range, m_hue_range, m_rng);
        return uniform();
    }
    
    AleamCore& m_rng;                                    /**< Random number generator */
    
    // Range parameters
    RealType m_rotation_range;                           /**< Max rotation angle */
    std::pair<RealType, RealType> m_scale_range;         /**< Min/max scale */
    RealType m_shear_range;                              /**< Max shear */
    RealType m_translation_range;                        /**< Max translation fraction */
    std::pair<RealType, RealType> m_brightness_range;    /**< Min/max brightness */
    std::pair<RealType, RealType> m_contrast_range;      /**< Min/max contrast */
    std::pair<RealType, RealType> m_saturation_range;    /**< Min/max saturation */
    RealType m_hue_range;                                /**< Max hue shift */
    
    // Probabilities
    RealType m_flip_probability;                         /**< Probability of flipping */
    RealType m_grayscale_probability;                    /**< Probability of grayscale */
};

}  // namespace ai
}  // namespace aleam

#endif /* ALEAM_AI_AUGMENTATION_H */