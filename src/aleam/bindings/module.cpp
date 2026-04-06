/**
 * @file module.cpp
 * @brief Pybind11 module definition for Aleam C++ core
 * @license MIT
 * 
 * This file implements the Python bindings for Aleam's C++ core.
 * It exports all classes and functions to Python as the _c_core module.
 * 
 * The Python package 'aleam' imports this module and re-exports
 * everything with the same API as the original pure Python version.
 */

#include "exports.h"
#include "../core/aleam_core.h"
#include "../distributions/distributions.h"
#include "../arrays/arrays.h"
#include "../ai/ai.h"
#include "../integrations/integrations.h"

namespace py = pybind11;

/* ============================================================================
 * NumPy Array Conversion Helpers Implementation
 * ============================================================================ */

py::array_t<double> vector_to_numpy(const std::vector<double>& vec) {
    py::array_t<double> result(vec.size());
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    std::copy(vec.begin(), vec.end(), ptr);
    return result;
}

py::array_t<int64_t> vector_int64_to_numpy(const std::vector<int64_t>& vec) {
    py::array_t<int64_t> result(vec.size());
    auto buf = result.request();
    int64_t* ptr = static_cast<int64_t*>(buf.ptr);
    std::copy(vec.begin(), vec.end(), ptr);
    return result;
}

py::array_t<double> vector2d_to_numpy(const std::vector<std::vector<double>>& vec) {
    if (vec.empty() || vec[0].empty()) {
        return py::array_t<double>();
    }
    
    size_t rows = vec.size();
    size_t cols = vec[0].size();
    
    py::array_t<double> result({rows, cols});
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            ptr[i * cols + j] = vec[i][j];
        }
    }
    
    return result;
}

py::array_t<uint8_t> mask_to_numpy(const std::vector<std::vector<uint8_t>>& vec) {
    if (vec.empty() || vec[0].empty()) {
        return py::array_t<uint8_t>();
    }
    
    size_t rows = vec.size();
    size_t cols = vec[0].size();
    
    py::array_t<uint8_t> result({rows, cols});
    auto buf = result.request();
    uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            ptr[i * cols + j] = vec[i][j];
        }
    }
    
    return result;
}

/* ============================================================================
 * Exception Translation
 * ============================================================================ */

void register_exception_translators() {
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::invalid_argument& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const std::out_of_range& e) {
            PyErr_SetString(PyExc_IndexError, e.what());
        } catch (const std::runtime_error& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_Exception, e.what());
        }
    });
}

/* ============================================================================
 * AleamCore Binding
 * ============================================================================ */

void bind_aleam_core(py::module& m) {
    py::class_<aleam::AleamCore>(m, "AleamCore")
        .def(py::init<>())
        // Core random methods
        .def("random", &aleam::AleamCore::random)
        .def("random_uint64", &aleam::AleamCore::random_uint64)
        .def("random_batch", [](aleam::AleamCore& self, size_t count) {
            std::vector<double> result(count);
            self.random_batch(result.data(), count);
            return vector_to_numpy(result);
        })
        .def("random_uint64_batch", [](aleam::AleamCore& self, size_t count) {
            std::vector<uint64_t> result(count);
            self.random_uint64_batch(result.data(), count);
            return py::array_t<uint64_t>(result.size(), result.data());
        })
        // Batch cache management
        .def("set_batch_size", &aleam::AleamCore::set_batch_size)
        .def("get_batch_size", &aleam::AleamCore::get_batch_size)
        .def("clear_cache", &aleam::AleamCore::clear_cache)
        // Statistics
        .def("get_stats", &aleam::AleamCore::get_stats)
        .def("reset_stats", &aleam::AleamCore::reset_stats)
        // Integer methods
        .def("randint", &aleam::AleamCore::randint)
        .def("random_bytes", &aleam::AleamCore::random_bytes)
        // Distribution methods
        .def("uniform", &aleam::AleamCore::uniform)
        .def("gauss", &aleam::AleamCore::gauss)
        .def("exponential", &aleam::AleamCore::exponential)
        .def("beta", &aleam::AleamCore::beta)
        .def("gamma", &aleam::AleamCore::gamma)
        .def("poisson", &aleam::AleamCore::poisson)
        .def("laplace", &aleam::AleamCore::laplace)
        .def("logistic", &aleam::AleamCore::logistic)
        .def("lognormal", &aleam::AleamCore::lognormal)
        .def("weibull", &aleam::AleamCore::weibull)
        .def("pareto", &aleam::AleamCore::pareto)
        .def("chi_square", &aleam::AleamCore::chi_square)
        .def("student_t", &aleam::AleamCore::student_t)
        .def("f_distribution", &aleam::AleamCore::f_distribution)
        .def("dirichlet", &aleam::AleamCore::dirichlet)
        // Sequence methods
        .def("choice", [](aleam::AleamCore& self, py::list seq) {
            if (py::len(seq) == 0) throw std::invalid_argument("Cannot choose from empty sequence");
            size_t idx = static_cast<size_t>(self.random() * py::len(seq));
            return seq[idx];
        })
        .def("sample", [](aleam::AleamCore& self, py::list population, size_t k) {
            if (k > py::len(population)) throw std::invalid_argument("Sample larger than population");
            py::list result;
            std::vector<size_t> indices(py::len(population));
            for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
            for (size_t i = 0; i < k; ++i) {
                size_t j = i + static_cast<size_t>(self.random() * (indices.size() - i));
                if (j >= indices.size()) j = indices.size() - 1;
                std::swap(indices[i], indices[j]);
                result.append(population[indices[i]]);
            }
            return result;
        })
        .def("shuffle", [](aleam::AleamCore& self, py::list lst) {
            size_t n = py::len(lst);
            for (size_t i = n - 1; i > 0; --i) {
                size_t j = static_cast<size_t>(self.random() * (i + 1));
                if (j > i) j = i;
                py::object tmp = lst[i];
                lst[i] = lst[j];
                lst[j] = tmp;
            }
        });
    
    /* Bind the stats structure */
    py::class_<aleam::AleamCore::Stats>(m, "Stats")
        .def_readonly("calls", &aleam::AleamCore::Stats::calls)
        .def_readonly("batch_size", &aleam::AleamCore::Stats::batch_size)
        .def_readonly("cache_hits", &aleam::AleamCore::Stats::cache_hits)
        .def_readonly("cache_misses", &aleam::AleamCore::Stats::cache_misses)
        .def_readonly("algorithm", &aleam::AleamCore::Stats::algorithm)
        .def_readonly("entropy_source", &aleam::AleamCore::Stats::entropy_source)
        .def_readonly("entropy_bits_per_call", &aleam::AleamCore::Stats::entropy_bits_per_call);
    
    /* Thread-local instance getter */
    m.def("get_thread_local_instance", &aleam::get_thread_local_instance,
          py::return_value_policy::reference);
}

/* ============================================================================
 * Distributions Binding
 * ============================================================================ */

void bind_distributions(py::module& m) {
    /* Create a submodule for distributions */
    py::module dist = m.def_submodule("distributions", "Statistical distributions");
    
    dist.def("normal", [](aleam::AleamCore& rng, double mu, double sigma) {
        return aleam::distributions::normal(rng, mu, sigma);
    }, py::arg("rng"), py::arg("mu") = 0.0, py::arg("sigma") = 1.0);
    
    dist.def("uniform", [](aleam::AleamCore& rng, double low, double high) {
        return aleam::distributions::uniform(rng, low, high);
    }, py::arg("rng"), py::arg("low"), py::arg("high"));
    
    dist.def("exponential", [](aleam::AleamCore& rng, double rate) {
        return aleam::distributions::exponential(rng, rate);
    }, py::arg("rng"), py::arg("rate") = 1.0);
    
    dist.def("beta", [](aleam::AleamCore& rng, double alpha, double beta) {
        return aleam::distributions::beta(rng, alpha, beta);
    }, py::arg("rng"), py::arg("alpha"), py::arg("beta"));
    
    dist.def("gamma", [](aleam::AleamCore& rng, double shape, double scale) {
        return aleam::distributions::gamma(rng, shape, scale);
    }, py::arg("rng"), py::arg("shape"), py::arg("scale") = 1.0);
    
    dist.def("poisson", [](aleam::AleamCore& rng, double lambda) {
        return aleam::distributions::poisson(rng, lambda);
    }, py::arg("rng"), py::arg("lambda") = 1.0);
    
    dist.def("laplace", [](aleam::AleamCore& rng, double loc, double scale) {
        return aleam::distributions::laplace(rng, loc, scale);
    }, py::arg("rng"), py::arg("loc") = 0.0, py::arg("scale") = 1.0);
    
    dist.def("logistic", [](aleam::AleamCore& rng, double loc, double scale) {
        return aleam::distributions::logistic(rng, loc, scale);
    }, py::arg("rng"), py::arg("loc") = 0.0, py::arg("scale") = 1.0);
    
    dist.def("lognormal", [](aleam::AleamCore& rng, double mu, double sigma) {
        return aleam::distributions::lognormal(rng, mu, sigma);
    }, py::arg("rng"), py::arg("mu") = 0.0, py::arg("sigma") = 1.0);
    
    dist.def("weibull", [](aleam::AleamCore& rng, double shape, double scale) {
        return aleam::distributions::weibull(rng, shape, scale);
    }, py::arg("rng"), py::arg("shape"), py::arg("scale") = 1.0);
    
    dist.def("pareto", [](aleam::AleamCore& rng, double alpha, double scale) {
        return aleam::distributions::pareto(rng, alpha, scale);
    }, py::arg("rng"), py::arg("alpha"), py::arg("scale") = 1.0);
    
    dist.def("chi_square", [](aleam::AleamCore& rng, double df) {
        return aleam::distributions::chi_square(rng, df);
    }, py::arg("rng"), py::arg("df"));
    
    dist.def("student_t", [](aleam::AleamCore& rng, double df) {
        return aleam::distributions::student_t(rng, df);
    }, py::arg("rng"), py::arg("df"));
    
    dist.def("f_distribution", [](aleam::AleamCore& rng, double df1, double df2) {
        return aleam::distributions::f_distribution(rng, df1, df2);
    }, py::arg("rng"), py::arg("df1"), py::arg("df2"));
    
    dist.def("dirichlet", [](aleam::AleamCore& rng, std::vector<double> alpha) {
        return aleam::distributions::dirichlet(rng, alpha);
    }, py::arg("rng"), py::arg("alpha"));
}

/* ============================================================================
 * Arrays Binding
 * ============================================================================ */

void bind_arrays(py::module& m) {
    py::module arr = m.def_submodule("arrays", "Array operations");
    
    arr.def("random_array_1d", [](aleam::AleamCore& rng, size_t size) {
        return vector_to_numpy(aleam::arrays::random_array_1d(rng, size));
    }, py::arg("rng"), py::arg("size"));
    
    arr.def("random_array_2d", [](aleam::AleamCore& rng, size_t rows, size_t cols) {
        return vector2d_to_numpy(aleam::arrays::random_array_2d(rng, rows, cols));
    }, py::arg("rng"), py::arg("rows"), py::arg("cols"));
    
    arr.def("randn_array_1d", [](aleam::AleamCore& rng, size_t size, double mu, double sigma) {
        return vector_to_numpy(aleam::arrays::randn_array_1d(rng, size, mu, sigma));
    }, py::arg("rng"), py::arg("size"), py::arg("mu") = 0.0, py::arg("sigma") = 1.0);
    
    arr.def("randn_array_2d", [](aleam::AleamCore& rng, size_t rows, size_t cols, double mu, double sigma) {
        return vector2d_to_numpy(aleam::arrays::randn_array_2d(rng, rows, cols, mu, sigma));
    }, py::arg("rng"), py::arg("rows"), py::arg("cols"), py::arg("mu") = 0.0, py::arg("sigma") = 1.0);
    
    arr.def("randint_array_1d", [](aleam::AleamCore& rng, size_t size, int64_t low, int64_t high) {
        return vector_int64_to_numpy(aleam::arrays::randint_array_1d(rng, size, low, high));
    }, py::arg("rng"), py::arg("size"), py::arg("low"), py::arg("high"));
    
    arr.def("randint_array_2d", [](aleam::AleamCore& rng, size_t rows, size_t cols, int64_t low, int64_t high) {
        auto result = aleam::arrays::randint_array_2d(rng, rows, cols, low, high);
        /* Convert to Python list of lists for simplicity */
        py::list py_result;
        for (const auto& row : result) {
            py::list py_row;
            for (auto val : row) {
                py_row.append(val);
            }
            py_result.append(py_row);
        }
        return py_result;
    }, py::arg("rng"), py::arg("rows"), py::arg("cols"), py::arg("low"), py::arg("high"));
}

/* ============================================================================
 * AI Bindings
 * ============================================================================ */

void bind_ai(py::module& m) {
    py::module ai = m.def_submodule("ai", "AI/ML randomness features");
    
    /* AIRandom class */
    py::class_<aleam::ai::AIRandom>(ai, "AIRandom")
        .def(py::init<aleam::AleamCore*>(), py::arg("rng") = nullptr)
        .def("gradient_noise", [](aleam::ai::AIRandom& self, size_t shape, double scale) {
            return vector_to_numpy(self.gradient_noise(shape, scale));
        }, py::arg("shape"), py::arg("scale") = 0.1)
        .def("latent_vector", [](aleam::ai::AIRandom& self, size_t dim, const std::string& dist) {
            return vector_to_numpy(self.latent_vector(dim, dist));
        }, py::arg("dim"), py::arg("distribution") = "normal")
        .def("dropout_mask", [](aleam::ai::AIRandom& self, size_t size, double keep_prob) {
            auto mask = self.dropout_mask(size, keep_prob);
            return py::array_t<uint8_t>(mask.size(), mask.data());
        }, py::arg("size"), py::arg("keep_prob") = 0.5)
        .def("augmentation_params", &aleam::ai::AIRandom::augmentation_params)
        .def("mini_batch", &aleam::ai::AIRandom::mini_batch)
        .def("exploration_noise", [](aleam::ai::AIRandom& self, size_t action_dim, double scale) {
            return vector_to_numpy(self.exploration_noise(action_dim, scale));
        }, py::arg("action_dim"), py::arg("scale") = 0.2);
    
    /* AugmentationParams struct */
    py::class_<aleam::ai::AIRandom::AugmentationParams>(ai, "AugmentationParams")
        .def_readwrite("rotation", &aleam::ai::AIRandom::AugmentationParams::rotation)
        .def_readwrite("scale", &aleam::ai::AIRandom::AugmentationParams::scale)
        .def_readwrite("brightness", &aleam::ai::AIRandom::AugmentationParams::brightness)
        .def_readwrite("contrast", &aleam::ai::AIRandom::AugmentationParams::contrast)
        .def_readwrite("flip_horizontal", &aleam::ai::AIRandom::AugmentationParams::flip_horizontal)
        .def_readwrite("flip_vertical", &aleam::ai::AIRandom::AugmentationParams::flip_vertical);
    
    /* GradientNoise class */
    py::class_<aleam::ai::GradientNoise>(ai, "GradientNoise")
        .def(py::init<double, double, aleam::AleamCore*>(),
             py::arg("initial_scale") = 0.01,
             py::arg("decay") = 0.99,
             py::arg("rng") = nullptr)
        .def("add_noise", [](aleam::ai::GradientNoise& self, py::array_t<double> gradients) {
            auto buf = gradients.request();
            double* ptr = static_cast<double*>(buf.ptr);
            std::vector<double> vec(ptr, ptr + buf.size);
            auto result = self.add_noise(vec);
            return vector_to_numpy(result);
        })
        .def("reset", &aleam::ai::GradientNoise::reset)
        .def("get_step", &aleam::ai::GradientNoise::get_step)
        .def("get_current_scale", &aleam::ai::GradientNoise::get_current_scale);
    
    /* LatentSampler class */
    py::class_<aleam::ai::LatentSampler>(ai, "LatentSampler")
        .def(py::init<size_t, const std::string&, aleam::AleamCore*>(),
             py::arg("latent_dim"),
             py::arg("distribution") = "normal",
             py::arg("rng") = nullptr)
        .def("sample", [](aleam::ai::LatentSampler& self, size_t n) {
            auto samples = self.sample(n);
            /* Convert to Python list of arrays */
            py::list result;
            for (const auto& vec : samples) {
                result.append(vector_to_numpy(vec));
            }
            return result;
        }, py::arg("n") = 1)
        .def("sample_one", [](aleam::ai::LatentSampler& self) {
            return vector_to_numpy(self.sample_one());
        })
        .def("interpolate", [](aleam::ai::LatentSampler& self, 
                                py::array_t<double> z1, py::array_t<double> z2, size_t steps) {
            auto buf1 = z1.request();
            auto buf2 = z2.request();
            std::vector<double> vec1(static_cast<double*>(buf1.ptr),
                                      static_cast<double*>(buf1.ptr) + buf1.size);
            std::vector<double> vec2(static_cast<double*>(buf2.ptr),
                                      static_cast<double*>(buf2.ptr) + buf2.size);
            auto interpolations = self.interpolate(vec1, vec2, steps);
            py::list result;
            for (const auto& vec : interpolations) {
                result.append(vector_to_numpy(vec));
            }
            return result;
        }, py::arg("z1"), py::arg("z2"), py::arg("steps") = 10)
        .def("get_latent_dim", &aleam::ai::LatentSampler::get_latent_dim)
        .def("get_distribution", &aleam::ai::LatentSampler::get_distribution);
}

/* ============================================================================
 * Integrations Binding
 * ============================================================================ */

void bind_integrations(py::module& m) {
    py::module integ = m.def_submodule("integrations", "Framework integrations");
    
    /* BaseGenerator (not directly exposed) */
    
    /* TorchGenerator */
    py::class_<aleam::integrations::TorchGenerator>(integ, "TorchGenerator")
        .def(py::init<const std::string&, aleam::AleamCore*>(),
             py::arg("device") = "cpu",
             py::arg("rng") = nullptr)
        .def_property("device", &aleam::integrations::TorchGenerator::device,
                      &aleam::integrations::TorchGenerator::set_device)
        .def("rand", [](aleam::integrations::TorchGenerator& self, size_t size) {
            return vector_to_numpy(self.rand(size));
        })
        .def("randn", [](aleam::integrations::TorchGenerator& self, size_t size) {
            return vector_to_numpy(self.randn(size));
        })
        .def("randint", [](aleam::integrations::TorchGenerator& self, int64_t low, int64_t high, size_t size) {
            auto result = self.randint(low, high, size);
            return py::array_t<int64_t>(result.size(), result.data());
        })
        .def("manual_seed", &aleam::integrations::TorchGenerator::manual_seed);
    
    /* TFGenerator */
    py::class_<aleam::integrations::TFGenerator>(integ, "TFGenerator")
        .def(py::init<aleam::AleamCore*>(), py::arg("rng") = nullptr)
        .def("normal", [](aleam::integrations::TFGenerator& self, size_t shape, double mean, double stddev) {
            return vector_to_numpy(self.normal(shape, mean, stddev));
        }, py::arg("shape"), py::arg("mean") = 0.0, py::arg("stddev") = 1.0)
        .def("uniform", [](aleam::integrations::TFGenerator& self, size_t shape, double minval, double maxval) {
            return vector_to_numpy(self.uniform(shape, minval, maxval));
        }, py::arg("shape"), py::arg("minval") = 0.0, py::arg("maxval") = 1.0)
        .def("randint", [](aleam::integrations::TFGenerator& self, size_t shape, int64_t minval, int64_t maxval) {
            auto result = self.randint(shape, minval, maxval);
            return py::array_t<int64_t>(result.size(), result.data());
        })
        .def("truncated_normal", [](aleam::integrations::TFGenerator& self, size_t shape, double mean, double stddev) {
            return vector_to_numpy(self.truncated_normal(shape, mean, stddev));
        }, py::arg("shape"), py::arg("mean") = 0.0, py::arg("stddev") = 1.0);
    
    /* JAXGenerator */
    py::class_<aleam::integrations::JAXGenerator>(integ, "JAXGenerator")
        .def(py::init<aleam::AleamCore*>(), py::arg("rng") = nullptr)
        .def("key", &aleam::integrations::JAXGenerator::key)
        .def("keys", &aleam::integrations::JAXGenerator::keys)
        .def("normal", [](aleam::integrations::JAXGenerator& self, size_t shape, double mean, double stddev) {
            return vector_to_numpy(self.normal(shape, mean, stddev));
        }, py::arg("shape"), py::arg("mean") = 0.0, py::arg("stddev") = 1.0)
        .def("uniform", [](aleam::integrations::JAXGenerator& self, size_t shape, double minval, double maxval) {
            return vector_to_numpy(self.uniform(shape, minval, maxval));
        }, py::arg("shape"), py::arg("minval") = 0.0, py::arg("maxval") = 1.0);
    
    /* CuPyGenerator */
    py::class_<aleam::integrations::CuPyGenerator>(integ, "CuPyGenerator")
        .def(py::init<aleam::AleamCore*>(), py::arg("rng") = nullptr)
        .def("random", [](aleam::integrations::CuPyGenerator& self, size_t size, const std::string& dtype) {
            return vector_to_numpy(self.random(size, dtype));
        }, py::arg("size"), py::arg("dtype") = "float32")
        .def("randn", [](aleam::integrations::CuPyGenerator& self, size_t size, double mu, double sigma, const std::string& dtype) {
            return vector_to_numpy(self.randn(size, mu, sigma, dtype));
        }, py::arg("size"), py::arg("mu") = 0.0, py::arg("sigma") = 1.0, py::arg("dtype") = "float32")
        .def("randint", [](aleam::integrations::CuPyGenerator& self, size_t size, int64_t low, int64_t high) {
            auto result = self.randint(size, low, high);
            return py::array_t<int64_t>(result.size(), result.data());
        });
    
    /* PandasGenerator */
    py::class_<aleam::integrations::PandasGenerator>(integ, "PandasGenerator")
        .def(py::init<aleam::AleamCore*>(), py::arg("rng") = nullptr)
        .def("series", [](aleam::integrations::PandasGenerator& self, size_t n, const std::string& dist, const std::string& params) {
            return vector_to_numpy(self.series(n, dist, params));
        }, py::arg("n"), py::arg("distribution") = "uniform", py::arg("params") = "")
        .def("shuffle_indices", &aleam::integrations::PandasGenerator::shuffle_indices);
    
    /* DaskGenerator */
    py::class_<aleam::integrations::DaskGenerator>(integ, "DaskGenerator")
        .def(py::init<aleam::AleamCore*>(), py::arg("rng") = nullptr)
        .def("block_random", [](aleam::integrations::DaskGenerator& self, size_t block_shape, const std::string& dist) {
            return vector_to_numpy(self.block_random(block_shape, dist));
        }, py::arg("block_shape"), py::arg("distribution") = "uniform");
}

/* ============================================================================
 * CUDA Bindings (Conditional)
 * ============================================================================ */

void bind_cuda(py::module& m) {
#ifdef ALEAM_WITH_CUDA
    py::module cuda = m.def_submodule("cuda", "CUDA GPU acceleration");
    
    /* Check if CUDA is available */
    cuda.def("is_available", &is_cuda_available);
    cuda.def("get_device_count", &get_cuda_device_count);
    
    cuda.def("get_device_name", [](int device_id) {
        char name[256];
        get_cuda_device_name(device_id, name, sizeof(name));
        return std::string(name);
    });
    
    /* Note: Actual kernel launching functions would be bound here,
       but they require careful memory management. For simplicity,
       the Python CUDA integration uses the existing CuPy/PyTorch
       backends which are already optimized. */
#else
    /* CUDA not available - provide stub */
    py::module cuda = m.def_submodule("cuda", "CUDA GPU acceleration (not available)");
    cuda.def("is_available", []() { return false; });
#endif
}

/* ============================================================================
 * Main Bindings Entry Point
 * ============================================================================ */

void bind_all(py::module_& m) {
    /* Register exception translators */
    register_exception_translators();
    
    /* Bind all components */
    bind_aleam_core(m);
    bind_distributions(m);
    bind_arrays(m);
    bind_ai(m);
    bind_integrations(m);
    bind_cuda(m);
}

/* ============================================================================
 * Main Module Definition
 * ============================================================================ */

PYBIND11_MODULE(_c_core, m) {
    /* Module documentation */
    m.doc() = "Aleam C++ Core - True random number generator";
    
    /* Bind all components */
    bind_all(m);
    
    /* ========================================================================
     * Module-Level Convenience Functions (using default instance)
     * ======================================================================== */
    
    // Create a default RNG instance for module-level functions
    static aleam::AleamCore default_rng;
    
    // Core random functions
    m.def("random", []() { return default_rng.random(); });
    m.def("random_uint64", []() { return default_rng.random_uint64(); });
    m.def("randint", [](int64_t a, int64_t b) { return default_rng.randint(a, b); });
    m.def("choice", [](py::list seq) {
        if (py::len(seq) == 0) throw std::invalid_argument("Cannot choose from empty sequence");
        size_t idx = static_cast<size_t>(default_rng.random() * py::len(seq));
        return seq[idx];
    });
    m.def("uniform", [](double low, double high) { return default_rng.uniform(low, high); });
    m.def("gauss", [](double mu, double sigma) { return default_rng.gauss(mu, sigma); });
    m.def("shuffle", [](py::list lst) {
        size_t n = py::len(lst);
        for (size_t i = n - 1; i > 0; --i) {
            size_t j = static_cast<size_t>(default_rng.random() * (i + 1));
            if (j > i) j = i;
            py::object tmp = lst[i];
            lst[i] = lst[j];
            lst[j] = tmp;
        }
    });
    m.def("sample", [](py::list population, size_t k) {
        if (k > py::len(population)) throw std::invalid_argument("Sample larger than population");
        py::list result;
        std::vector<size_t> indices(py::len(population));
        for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
        for (size_t i = 0; i < k; ++i) {
            size_t j = i + static_cast<size_t>(default_rng.random() * (indices.size() - i));
            if (j >= indices.size()) j = indices.size() - 1;
            std::swap(indices[i], indices[j]);
            result.append(population[indices[i]]);
        }
        return result;
    });
    m.def("random_bytes", [](size_t n) {
        std::vector<uint8_t> bytes(n);
        for (size_t i = 0; i < n; ++i) {
            bytes[i] = static_cast<uint8_t>(default_rng.randint(0, 255));
        }
        return py::bytes(reinterpret_cast<char*>(bytes.data()), n);
    });
    
    // Array functions
    m.def("random_array", [](py::tuple shape) {
        std::vector<ssize_t> shape_vec;
        size_t total = 1;
        for (auto dim : shape) { auto s = dim.cast<ssize_t>(); shape_vec.push_back(s); total *= s; }
        std::vector<double> data(total);
        default_rng.random_batch(data.data(), total);
        return py::array_t<double>(shape_vec, data.data());
    });
    m.def("randn_array", [](py::tuple shape, double mu, double sigma) {
        std::vector<ssize_t> shape_vec;
        size_t total = 1;
        for (auto dim : shape) { auto s = dim.cast<ssize_t>(); shape_vec.push_back(s); total *= s; }
        std::vector<double> data(total);
        for (size_t i = 0; i < total; ++i) {
            data[i] = default_rng.gauss(mu, sigma);
        }
        return py::array_t<double>(shape_vec, data.data());
    }, py::arg("shape"), py::arg("mu") = 0.0, py::arg("sigma") = 1.0);
    m.def("randint_array", [](py::tuple shape, int64_t low, int64_t high) {
        std::vector<ssize_t> shape_vec;
        size_t total = 1;
        for (auto dim : shape) { auto s = dim.cast<ssize_t>(); shape_vec.push_back(s); total *= s; }
        std::vector<int64_t> data(total);
        for (size_t i = 0; i < total; ++i) {
            data[i] = default_rng.randint(low, high);
        }
        return py::array_t<int64_t>(shape_vec, data.data());
    }, py::arg("shape"), py::arg("low"), py::arg("high"));
    
    // Distribution functions
    m.def("exponential", [](double rate) { return default_rng.exponential(rate); }, py::arg("rate") = 1.0);
    m.def("beta", [](double alpha, double beta) { return default_rng.beta(alpha, beta); });
    m.def("gamma", [](double shape, double scale) { return default_rng.gamma(shape, scale); }, py::arg("shape"), py::arg("scale") = 1.0);
    m.def("poisson", [](double lam) { return default_rng.poisson(lam); }, py::arg("lam") = 1.0);
    m.def("laplace", [](double loc, double scale) { return default_rng.laplace(loc, scale); }, py::arg("loc") = 0.0, py::arg("scale") = 1.0);
    m.def("logistic", [](double loc, double scale) { return default_rng.logistic(loc, scale); }, py::arg("loc") = 0.0, py::arg("scale") = 1.0);
    m.def("lognormal", [](double mu, double sigma) { return default_rng.lognormal(mu, sigma); }, py::arg("mu") = 0.0, py::arg("sigma") = 1.0);
    m.def("weibull", [](double shape, double scale) { return default_rng.weibull(shape, scale); }, py::arg("shape"), py::arg("scale") = 1.0);
    m.def("pareto", [](double alpha, double scale) { return default_rng.pareto(alpha, scale); }, py::arg("alpha"), py::arg("scale") = 1.0);
    m.def("chi_square", [](double df) { return default_rng.chi_square(df); }, py::arg("df"));
    m.def("student_t", [](double df) { return default_rng.student_t(df); }, py::arg("df"));
    m.def("f_distribution", [](double df1, double df2) { return default_rng.f_distribution(df1, df2); });
    m.def("dirichlet", [](std::vector<double> alpha) { return default_rng.dirichlet(alpha); });
    
    // AI functions (using default RNG)
    m.def("AIRandom", [](aleam::AleamCore* rng) {
        return aleam::ai::AIRandom(rng ? rng : &default_rng);
    }, py::arg("rng") = nullptr);
    m.def("GradientNoise", [](double scale, double decay, aleam::AleamCore* rng) {
        return aleam::ai::GradientNoise(scale, decay, rng ? rng : &default_rng);
    }, py::arg("initial_scale") = 0.01, py::arg("decay") = 0.99, py::arg("rng") = nullptr);
    m.def("LatentSampler", [](size_t latent_dim, const std::string& distribution, aleam::AleamCore* rng) {
        return aleam::ai::LatentSampler(latent_dim, distribution, rng ? rng : &default_rng);
    }, py::arg("latent_dim"), py::arg("distribution") = "normal", py::arg("rng") = nullptr);
    
    /* Module-level constants */
    m.attr("__version__") = "1.0.3";
    m.attr("__algorithm__") = "Ψ(t) = BLAKE2s( (Φ × Ξ(t)) ⊕ τ(t) )";
}