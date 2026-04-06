/**
 * @file distributions.h
 * @brief Statistical distributions using Aleam true randomness
 * @license MIT
 * 
 * This file provides a comprehensive set of statistical distributions
 * all powered by Aleam's true random number generator.
 * 
 * Each distribution function takes an AleamCore reference as the first
 * parameter, allowing them to use true entropy instead of pseudo-random
 * numbers from a PRNG.
 * 
 * Available distributions:
 * - Normal (Gaussian)
 * - Uniform
 * - Exponential
 * - Beta
 * - Gamma
 * - Poisson
 * - Laplace (Double Exponential)
 * - Logistic
 * - Log-Normal
 * - Weibull
 * - Pareto
 * - Chi-square
 * - Student's t
 * - F-distribution
 * - Dirichlet
 */

#ifndef ALEAM_DISTRIBUTIONS_H
#define ALEAM_DISTRIBUTIONS_H

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include "../core/aleam_core.h"

namespace aleam {
namespace distributions {

/* ============================================================================
 * Core Distribution Helper Functions
 * ============================================================================ */

/**
 * @brief Generate a random double in [0, 1) from AleamCore
 * 
 * Convenience wrapper for AleamCore::random()
 * 
 * @param rng Reference to AleamCore instance
 * @return double Random number in [0, 1)
 */
inline double uniform_01(AleamCore& rng) {
    return rng.random();
}

/* ============================================================================
 * Normal (Gaussian) Distribution
 * ============================================================================ */

/**
 * @brief Generate normally distributed random number using Box-Muller transform
 * 
 * The Box-Muller transform converts two independent uniform random variables
 * into two independent standard normal variables.
 * 
 * Algorithm:
 *     U1, U2 ~ Uniform(0, 1)
 *     Z = √(-2·ln U1) · cos(2π·U2)
 *     Z ~ N(0, 1)
 * 
 * @param rng Reference to AleamCore instance
 * @param mu Mean of the distribution
 * @param sigma Standard deviation (must be > 0)
 * @return double Normally distributed value
 */
double normal(AleamCore& rng, double mu = 0.0, double sigma = 1.0);

/* ============================================================================
 * Uniform Distribution
 * ============================================================================ */

/**
 * @brief Generate uniformly distributed random number in [low, high]
 * 
 * Uses inverse transform sampling.
 * 
 * @param rng Reference to AleamCore instance
 * @param low Lower bound (inclusive)
 * @param high Upper bound (inclusive)
 * @return double Uniformly distributed value in [low, high]
 */
inline double uniform(AleamCore& rng, double low, double high) {
    return low + rng.random() * (high - low);
}

/* ============================================================================
 * Exponential Distribution
 * ============================================================================ */

/**
 * @brief Generate exponentially distributed random number
 * 
 * PDF: f(x) = λ·e^{-λ·x} for x ≥ 0
 * 
 * Uses inverse transform sampling:
 *     F⁻¹(u) = -ln(1 - u) / λ
 * 
 * @param rng Reference to AleamCore instance
 * @param rate Rate parameter λ (> 0)
 * @return double Exponentially distributed value (≥ 0)
 */
double exponential(AleamCore& rng, double rate = 1.0);

/* ============================================================================
 * Beta Distribution
 * ============================================================================ */

/**
 * @brief Generate Beta distributed random number in [0, 1]
 * 
 * PDF: f(x) = [x^(α-1)·(1-x)^(β-1)] / B(α, β)
 * 
 * Uses the gamma method:
 *     If X ~ Gamma(α, 1) and Y ~ Gamma(β, 1),
 *     then X/(X+Y) ~ Beta(α, β)
 * 
 * @param rng Reference to AleamCore instance
 * @param alpha Shape parameter (> 0)
 * @param beta Shape parameter (> 0)
 * @return double Beta distributed value in [0, 1]
 */
double beta(AleamCore& rng, double alpha, double beta);

/* ============================================================================
 * Gamma Distribution
 * ============================================================================ */

/**
 * @brief Generate Gamma distributed random number
 * 
 * PDF: f(x) = x^(k-1)·e^(-x/θ) / [Γ(k)·θ^k] for x ≥ 0
 * 
 * Uses Marsaglia & Tsang method for k ≥ 1, and Johnk's method for k < 1.
 * 
 * @param rng Reference to AleamCore instance
 * @param shape Shape parameter k (> 0)
 * @param scale Scale parameter θ (> 0)
 * @return double Gamma distributed value (≥ 0)
 */
double gamma(AleamCore& rng, double shape, double scale = 1.0);

/* ============================================================================
 * Poisson Distribution
 * ============================================================================ */

/**
 * @brief Generate Poisson distributed integer
 * 
 * PMF: P(X = k) = e^{-λ}·λ^k / k!
 * 
 * Uses Knuth's algorithm for λ < 10, normal approximation for larger λ.
 * 
 * @param rng Reference to AleamCore instance
 * @param lambda Mean parameter λ (> 0)
 * @return int Poisson distributed integer (≥ 0)
 */
int poisson(AleamCore& rng, double lambda = 1.0);

/* ============================================================================
 * Laplace (Double Exponential) Distribution
 * ============================================================================ */

/**
 * @brief Generate Laplace distributed random number
 * 
 * PDF: f(x) = 1/(2b)·exp(-|x-μ|/b)
 * 
 * Uses inverse transform sampling.
 * 
 * @param rng Reference to AleamCore instance
 * @param loc Location parameter μ
 * @param scale Scale parameter b (> 0)
 * @return double Laplace distributed value
 */
double laplace(AleamCore& rng, double loc = 0.0, double scale = 1.0);

/* ============================================================================
 * Logistic Distribution
 * ============================================================================ */

/**
 * @brief Generate Logistic distributed random number
 * 
 * PDF: f(x) = e^{-(x-μ)/s} / [s·(1 + e^{-(x-μ)/s})²]
 * 
 * Uses inverse transform sampling.
 * 
 * @param rng Reference to AleamCore instance
 * @param loc Location parameter μ
 * @param scale Scale parameter s (> 0)
 * @return double Logistic distributed value
 */
double logistic(AleamCore& rng, double loc = 0.0, double scale = 1.0);

/* ============================================================================
 * Log-Normal Distribution
 * ============================================================================ */

/**
 * @brief Generate Log-Normal distributed random number
 * 
 * If X ~ N(μ, σ²), then Y = e^X ~ LogNormal(μ, σ²)
 * 
 * @param rng Reference to AleamCore instance
 * @param mu Mean of underlying normal distribution
 * @param sigma Standard deviation of underlying normal distribution (> 0)
 * @return double Log-Normal distributed value (> 0)
 */
inline double lognormal(AleamCore& rng, double mu = 0.0, double sigma = 1.0) {
    return std::exp(normal(rng, mu, sigma));
}

/* ============================================================================
 * Weibull Distribution
 * ============================================================================ */

/**
 * @brief Generate Weibull distributed random number
 * 
 * PDF: f(x) = (k/λ)·(x/λ)^(k-1)·e^{-(x/λ)^k} for x ≥ 0
 * 
 * Uses inverse transform sampling:
 *     F⁻¹(u) = λ·(-ln(1-u))^(1/k)
 * 
 * @param rng Reference to AleamCore instance
 * @param shape Shape parameter k (> 0)
 * @param scale Scale parameter λ (> 0)
 * @return double Weibull distributed value (≥ 0)
 */
double weibull(AleamCore& rng, double shape, double scale = 1.0);

/* ============================================================================
 * Pareto Distribution
 * ============================================================================ */

/**
 * @brief Generate Pareto distributed random number
 * 
 * PDF: f(x) = α·x_m^α / x^(α+1) for x ≥ x_m
 * 
 * Uses inverse transform sampling:
 *     F⁻¹(u) = x_m / u^(1/α)
 * 
 * @param rng Reference to AleamCore instance
 * @param alpha Shape parameter α (> 0)
 * @param scale Scale parameter x_m (> 0)
 * @return double Pareto distributed value (≥ scale)
 */
double pareto(AleamCore& rng, double alpha, double scale = 1.0);

/* ============================================================================
 * Chi-Square Distribution
 * ============================================================================ */

/**
 * @brief Generate Chi-square distributed random number
 * 
 * χ²(k) = Gamma(k/2, 2)
 * 
 * @param rng Reference to AleamCore instance
 * @param df Degrees of freedom k (> 0)
 * @return double Chi-square distributed value (≥ 0)
 */
inline double chi_square(AleamCore& rng, double df) {
    return gamma(rng, df / 2.0, 2.0);
}

/* ============================================================================
 * Student's t-Distribution
 * ============================================================================ */

/**
 * @brief Generate Student's t-distributed random number
 * 
 * t(ν) = Z / √(χ²/ν)
 * where Z ~ N(0,1) and χ² ~ χ²(ν)
 * 
 * @param rng Reference to AleamCore instance
 * @param df Degrees of freedom ν (> 0)
 * @return double Student's t distributed value
 */
double student_t(AleamCore& rng, double df);

/* ============================================================================
 * F-Distribution
 * ============================================================================ */

/**
 * @brief Generate F-distributed random number
 * 
 * F(d₁, d₂) = (χ²₁/d₁) / (χ²₂/d₂)
 * 
 * @param rng Reference to AleamCore instance
 * @param df1 Numerator degrees of freedom d₁ (> 0)
 * @param df2 Denominator degrees of freedom d₂ (> 0)
 * @return double F-distributed value (≥ 0)
 */
double f_distribution(AleamCore& rng, double df1, double df2);

/* ============================================================================
 * Dirichlet Distribution
 * ============================================================================ */

/**
 * @brief Generate Dirichlet distributed probability vector
 * 
 * If X_i ~ Gamma(α_i, 1), then (X_i / ΣX_j) ~ Dirichlet(α)
 * 
 * @param rng Reference to AleamCore instance
 * @param alpha Vector of concentration parameters (all > 0)
 * @return std::vector<double> Probability vector summing to 1
 */
std::vector<double> dirichlet(AleamCore& rng, const std::vector<double>& alpha);

}  // namespace distributions
}  // namespace aleam

#endif /* ALEAM_DISTRIBUTIONS_H */