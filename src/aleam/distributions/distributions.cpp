/**
 * @file distributions.cpp
 * @brief Implementation of statistical distributions using Aleam true randomness
 * @license MIT
 * 
 * This file implements all the statistical distribution functions declared
 * in distributions.h. Each function uses true random numbers from AleamCore
 * to generate samples from the specified distribution.
 */

#include "distributions.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace aleam {
namespace distributions {

/* ============================================================================
 * Normal Distribution (Box-Muller Transform)
 * ============================================================================ */

/**
 * @brief Generate normally distributed random number
 * 
 * Implementation of the Box-Muller transform:
 *     Z = √(-2·ln U₁) · cos(2π·U₂)
 * 
 * This produces one standard normal variable. The second variable (sine term)
 * is discarded. This is acceptable for single-variable generation.
 * 
 * @param rng Reference to AleamCore instance
 * @param mu Mean of the distribution
 * @param sigma Standard deviation (must be > 0)
 * @return double Normally distributed value
 */
double normal(AleamCore& rng, double mu, double sigma) {
    if (sigma <= 0.0) {
        throw std::invalid_argument("sigma must be > 0");
    }
    
    /* Generate two uniform random numbers */
    double u1 = rng.random();
    double u2 = rng.random();
    
    /* Box-Muller transform */
    double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    
    /* Scale and shift */
    return mu + sigma * z;
}

/* ============================================================================
 * Exponential Distribution
 * ============================================================================ */

/**
 * @brief Generate exponentially distributed random number
 * 
 * Inverse transform sampling:
 *     F⁻¹(u) = -ln(1 - u) / λ
 * 
 * Using 1-u instead of u ensures numerical stability near 0.
 * 
 * @param rng Reference to AleamCore instance
 * @param rate Rate parameter λ (> 0)
 * @return double Exponentially distributed value (≥ 0)
 */
double exponential(AleamCore& rng, double rate) {
    if (rate <= 0.0) {
        throw std::invalid_argument("rate must be > 0");
    }
    
    double u = rng.random();
    return -std::log(1.0 - u) / rate;
}

/* ============================================================================
 * Beta Distribution (Gamma Method)
 * ============================================================================ */

/**
 * @brief Generate Beta distributed random number
 * 
 * Algorithm:
 *     X ~ Gamma(α, 1)
 *     Y ~ Gamma(β, 1)
 *     return X / (X + Y)
 * 
 * @param rng Reference to AleamCore instance
 * @param alpha Shape parameter (> 0)
 * @param beta Shape parameter (> 0)
 * @return double Beta distributed value in [0, 1]
 */
double beta(AleamCore& rng, double alpha, double beta) {
    if (alpha <= 0.0 || beta <= 0.0) {
        throw std::invalid_argument("alpha and beta must be > 0");
    }
    
    double x = gamma(rng, alpha, 1.0);
    double y = gamma(rng, beta, 1.0);
    return x / (x + y);
}

/* ============================================================================
 * Gamma Distribution (Marsaglia & Tsang Method)
 * ============================================================================ */

/**
 * @brief Generate Gamma distributed random number
 * 
 * For shape >= 1, uses Marsaglia & Tsang's rejection sampling algorithm.
 * For shape < 1, uses Johnk's acceptance-rejection method.
 * 
 * Marsaglia & Tsang algorithm (shape >= 1):
 *     d = shape - 1/3
 *     c = 1 / sqrt(9d)
 *     Generate v ~ N(0,1), x = (1 + c·v)³
 *     Accept with probability...
 * 
 * @param rng Reference to AleamCore instance
 * @param shape Shape parameter k (> 0)
 * @param scale Scale parameter θ (> 0)
 * @return double Gamma distributed value (≥ 0)
 */
double gamma(AleamCore& rng, double shape, double scale) {
    if (shape <= 0.0 || scale <= 0.0) {
        throw std::invalid_argument("shape and scale must be > 0");
    }
    
    /* For shape < 1, use Johnk's method with Gamma(shape+1) */
    if (shape < 1.0) {
        while (true) {
            double u = rng.random();
            double v = rng.random();
            double x = std::pow(u, 1.0 / shape);
            double y = std::pow(v, 1.0 / (1.0 - shape));
            
            if (x + y <= 1.0) {
                double z = x / (x + y);
                double w = gamma(rng, shape + 1.0, 1.0);
                return w * z * scale;
            }
        }
    }
    
    /* Marsaglia & Tsang method for shape >= 1 */
    double d = shape - 1.0 / 3.0;
    double c = 1.0 / std::sqrt(9.0 * d);
    
    while (true) {
        double v = normal(rng, 0.0, 1.0);
        double x = (1.0 + c * v);
        
        /* x must be positive */
        if (x <= 0.0) {
            continue;
        }
        
        x = x * x * x;  /* x³ */
        
        double u = rng.random();
        
        /* Quick accept using quadratic bound */
        if (u < 1.0 - 0.0331 * (v * v * v * v)) {
            return d * x * scale;
        }
        
        /* Slower accept using logarithms */
        if (std::log(u) < 0.5 * (v * v) + d * (1.0 - x + std::log(x))) {
            return d * x * scale;
        }
    }
}

/* ============================================================================
 * Poisson Distribution
 * ============================================================================ */

/**
 * @brief Generate Poisson distributed integer
 * 
 * For lambda < 10, uses Knuth's algorithm (sequential Poisson generation).
 * For lambda >= 10, uses normal approximation with continuity correction.
 * 
 * Knuth's algorithm:
 *     L = e^(-λ)
 *     k = 0, p = 1
 *     while p > L:
 *         p *= U ~ Uniform(0,1)
 *         k++
 *     return k - 1
 * 
 * @param rng Reference to AleamCore instance
 * @param lambda Mean parameter λ (> 0)
 * @return int Poisson distributed integer (≥ 0)
 */
int poisson(AleamCore& rng, double lambda) {
    if (lambda <= 0.0) {
        throw std::invalid_argument("lambda must be > 0");
    }
    
    /* Knuth's algorithm for small lambda */
    if (lambda < 10.0) {
        double L = std::exp(-lambda);
        int k = 0;
        double p = 1.0;
        
        while (p > L) {
            p *= rng.random();
            k++;
        }
        return k - 1;
    }
    
    /* Normal approximation with continuity correction for large lambda */
    double x = normal(rng, lambda, std::sqrt(lambda));
    int result = static_cast<int>(x + 0.5);
    
    /* Ensure non-negative result */
    return (result < 0) ? 0 : result;
}

/* ============================================================================
 * Laplace Distribution (Double Exponential)
 * ============================================================================ */

/**
 * @brief Generate Laplace distributed random number
 * 
 * Inverse transform sampling for Laplace distribution:
 *     F⁻¹(u) = μ - b·sign(u-0.5)·ln(1 - 2|u-0.5|)
 * 
 * @param rng Reference to AleamCore instance
 * @param loc Location parameter μ
 * @param scale Scale parameter b (> 0)
 * @return double Laplace distributed value
 */
double laplace(AleamCore& rng, double loc, double scale) {
    if (scale <= 0.0) {
        throw std::invalid_argument("scale must be > 0");
    }
    
    double u = rng.random() - 0.5;
    double sign = (u > 0.0) ? 1.0 : -1.0;
    return loc - scale * sign * std::log(1.0 - 2.0 * std::abs(u));
}

/* ============================================================================
 * Logistic Distribution
 * ============================================================================ */

/**
 * @brief Generate Logistic distributed random number
 * 
 * Inverse transform sampling for Logistic distribution:
 *     F⁻¹(u) = μ + s·ln(u / (1-u))
 * 
 * @param rng Reference to AleamCore instance
 * @param loc Location parameter μ
 * @param scale Scale parameter s (> 0)
 * @return double Logistic distributed value
 */
double logistic(AleamCore& rng, double loc, double scale) {
    if (scale <= 0.0) {
        throw std::invalid_argument("scale must be > 0");
    }
    
    double u = rng.random();
    return loc + scale * std::log(u / (1.0 - u));
}

/* ============================================================================
 * Weibull Distribution
 * ============================================================================ */

/**
 * @brief Generate Weibull distributed random number
 * 
 * Inverse transform sampling for Weibull distribution:
 *     F⁻¹(u) = λ·(-ln(1-u))^(1/k)
 * 
 * @param rng Reference to AleamCore instance
 * @param shape Shape parameter k (> 0)
 * @param scale Scale parameter λ (> 0)
 * @return double Weibull distributed value (≥ 0)
 */
double weibull(AleamCore& rng, double shape, double scale) {
    if (shape <= 0.0 || scale <= 0.0) {
        throw std::invalid_argument("shape and scale must be > 0");
    }
    
    double u = rng.random();
    return scale * std::pow(-std::log(1.0 - u), 1.0 / shape);
}

/* ============================================================================
 * Pareto Distribution
 * ============================================================================ */

/**
 * @brief Generate Pareto distributed random number
 * 
 * Inverse transform sampling for Pareto distribution:
 *     F⁻¹(u) = x_m / u^(1/α)
 * 
 * @param rng Reference to AleamCore instance
 * @param alpha Shape parameter α (> 0)
 * @param scale Scale parameter x_m (> 0)
 * @return double Pareto distributed value (≥ scale)
 */
double pareto(AleamCore& rng, double alpha, double scale) {
    if (alpha <= 0.0 || scale <= 0.0) {
        throw std::invalid_argument("alpha and scale must be > 0");
    }
    
    double u = rng.random();
    return scale / std::pow(u, 1.0 / alpha);
}

/* ============================================================================
 * Student's t-Distribution
 * ============================================================================ */

/**
 * @brief Generate Student's t-distributed random number
 * 
 * Algorithm:
 *     Z ~ N(0, 1)
 *     χ² ~ χ²(ν)
 *     t = Z / √(χ²/ν)
 * 
 * @param rng Reference to AleamCore instance
 * @param df Degrees of freedom ν (> 0)
 * @return double Student's t distributed value
 */
double student_t(AleamCore& rng, double df) {
    if (df <= 0.0) {
        throw std::invalid_argument("degrees of freedom must be > 0");
    }
    
    double z = normal(rng, 0.0, 1.0);
    double chi2 = chi_square(rng, df);
    return z / std::sqrt(chi2 / df);
}

/* ============================================================================
 * F-Distribution
 * ============================================================================ */

/**
 * @brief Generate F-distributed random number
 * 
 * Algorithm:
 *     χ²₁ ~ χ²(d₁)
 *     χ²₂ ~ χ²(d₂)
 *     F = (χ²₁/d₁) / (χ²₂/d₂)
 * 
 * @param rng Reference to AleamCore instance
 * @param df1 Numerator degrees of freedom d₁ (> 0)
 * @param df2 Denominator degrees of freedom d₂ (> 0)
 * @return double F-distributed value (≥ 0)
 */
double f_distribution(AleamCore& rng, double df1, double df2) {
    if (df1 <= 0.0 || df2 <= 0.0) {
        throw std::invalid_argument("degrees of freedom must be > 0");
    }
    
    double chi2_1 = chi_square(rng, df1);
    double chi2_2 = chi_square(rng, df2);
    return (chi2_1 / df1) / (chi2_2 / df2);
}

/* ============================================================================
 * Dirichlet Distribution
 * ============================================================================ */

/**
 * @brief Generate Dirichlet distributed probability vector
 * 
 * Algorithm:
 *     For each α_i, generate X_i ~ Gamma(α_i, 1)
 *     Sum all X_i to get total S
 *     Return vector of X_i / S
 * 
 * The result is a probability vector (all values in [0,1], sum to 1).
 * 
 * @param rng Reference to AleamCore instance
 * @param alpha Vector of concentration parameters (all > 0)
 * @return std::vector<double> Probability vector summing to 1
 */
std::vector<double> dirichlet(AleamCore& rng, const std::vector<double>& alpha) {
    if (alpha.empty()) {
        throw std::invalid_argument("alpha must not be empty");
    }
    
    /* Generate Gamma variates */
    std::vector<double> samples;
    samples.reserve(alpha.size());
    double sum = 0.0;
    
    for (double a : alpha) {
        if (a <= 0.0) {
            throw std::invalid_argument("all alpha values must be > 0");
        }
        double x = gamma(rng, a, 1.0);
        samples.push_back(x);
        sum += x;
    }
    
    /* Normalize to sum to 1 */
    for (double& x : samples) {
        x /= sum;
    }
    
    return samples;
}

}  // namespace distributions
}  // namespace aleam