/**
 * @file pymc_integration.h
 * @brief PyMC integration for Aleam true randomness
 * @license MIT
 * 
 * Provides true random sampling for PyMC Bayesian models.
 * PyMC is a Python library for Bayesian statistical modeling.
 * 
 * Example:
 * @code
 *   #include "aleam/integrations/pymc_integration.h"
 *   
 *   aleam::AleamCore rng;
 *   aleam::integrations::PyMCGenerator gen(rng);
 *   auto samples = gen.normal(1000);
 * @endcode
 */

#ifndef ALEAM_INTEGRATIONS_PYMC_INTEGRATION_H
#define ALEAM_INTEGRATIONS_PYMC_INTEGRATION_H

#include <vector>
#include <string>
#include <cstdint>
#include "../core/aleam_core.h"

namespace aleam {
namespace integrations {

/**
 * @brief PyMC-compatible random generator using true randomness
 * 
 * Provides true random sampling for PyMC Bayesian models.
 * Supports common distributions used in PyMC.
 */
class PyMCGenerator {
public:
    /**
     * @brief Construct a PyMCGenerator
     * 
     * @param rng Reference to AleamCore instance
     */
    explicit PyMCGenerator(AleamCore& rng);
    
    /**
     * @brief Generate normal (Gaussian) samples
     * 
     * @param n Number of samples
     * @param mu Mean
     * @param sigma Standard deviation
     * @return std::vector<double> Normal samples
     */
    std::vector<double> normal(size_t n, double mu = 0.0, double sigma = 1.0);
    
    /**
     * @brief Generate uniform samples
     * 
     * @param n Number of samples
     * @param lower Lower bound
     * @param upper Upper bound
     * @return std::vector<double> Uniform samples
     */
    std::vector<double> uniform(size_t n, double lower = 0.0, double upper = 1.0);
    
    /**
     * @brief Generate exponential samples
     * 
     * @param n Number of samples
     * @param rate Rate parameter
     * @return std::vector<double> Exponential samples
     */
    std::vector<double> exponential(size_t n, double rate = 1.0);
    
    /**
     * @brief Generate gamma samples
     * 
     * @param n Number of samples
     * @param alpha Shape parameter
     * @param beta Rate parameter (1/scale)
     * @return std::vector<double> Gamma samples
     */
    std::vector<double> gamma(size_t n, double alpha, double beta = 1.0);
    
    /**
     * @brief Generate beta samples
     * 
     * @param n Number of samples
     * @param alpha First shape parameter
     * @param beta Second shape parameter
     * @return std::vector<double> Beta samples in [0, 1]
     */
    std::vector<double> beta(size_t n, double alpha, double beta);
    
    /**
     * @brief Generate Poisson samples
     * 
     * @param n Number of samples
     * @param mu Mean parameter
     * @return std::vector<int64_t> Poisson samples
     */
    std::vector<int64_t> poisson(size_t n, double mu);
    
    /**
     * @brief Generate binomial samples
     * 
     * @param n Number of samples
     * @param trials Number of trials
     * @param p Probability of success
     * @return std::vector<int64_t> Binomial samples
     */
    std::vector<int64_t> binomial(size_t n, int64_t trials, double p);
    
    /**
     * @brief Generate Bernoulli samples
     * 
     * @param n Number of samples
     * @param p Probability of success
     * @return std::vector<int64_t> Bernoulli samples (0 or 1)
     */
    std::vector<int64_t> bernoulli(size_t n, double p);
    
    /**
     * @brief Generate categorical samples
     * 
     * @param n Number of samples
     * @param probabilities Category probabilities (must sum to 1)
     * @return std::vector<size_t> Category indices
     */
    std::vector<size_t> categorical(size_t n, const std::vector<double>& probabilities);
    
    /**
     * @brief Generate Dirichlet samples
     * 
     * @param n Number of samples
     * @param alpha Concentration parameters
     * @return std::vector<std::vector<double>> Probability vectors
     */
    std::vector<std::vector<double>> dirichlet(size_t n, const std::vector<double>& alpha);
    
    /**
     * @brief Generate Wald (inverse Gaussian) samples
     * 
     * @param n Number of samples
     * @param mu Mean parameter
     * @param lam Shape parameter
     * @return std::vector<double> Wald samples
     */
    std::vector<double> wald(size_t n, double mu, double lam);
    
private:
    /**
     * @brief Box-Muller transform for standard normal
     * 
     * @return double Standard normal value
     */
    double standard_normal();
    
    /**
     * @brief Gamma sampling using Marsaglia & Tsang method
     * 
     * @param shape Shape parameter
     * @param scale Scale parameter
     * @return double Gamma value
     */
    double sample_gamma(double shape, double scale);
    
    /**
     * @brief Poisson sampling using Knuth's algorithm
     * 
     * @param lambda Mean parameter
     * @return int64_t Poisson value
     */
    int64_t sample_poisson(double lambda);
    
    AleamCore& m_rng;        /**< Random number generator */
};

}  // namespace integrations
}  // namespace aleam

#endif /* ALEAM_INTEGRATIONS_PYMC_INTEGRATION_H */