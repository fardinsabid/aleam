/**
 * @file exports.h
 * @brief Pybind11 binding declarations for Aleam C++ core
 * @license MIT
 * 
 * This file declares the pybind11 module initialization function and
 * helper functions for exposing Aleam's C++ classes to Python.
 * 
 * The actual bindings are implemented in module.cpp.
 */

#ifndef ALEAM_EXPORTS_H
#define ALEAM_EXPORTS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

namespace py = pybind11;

/* ============================================================================
 * Module Initialization Declaration
 * ============================================================================ */

/**
 * @brief Pybind11 module initialization function
 * 
 * This function is called when Python imports the _c_core module.
 * It defines all the classes and functions exposed to Python.
 * 
 * @param m Module object to populate
 */
// PYBIND11_MODULE is defined ONLY in module.cpp - not here

/* ============================================================================
 * Forward Declarations for Binding Functions
 * ============================================================================ */

/**
 * @brief Main binding function - called from module.cpp
 * 
 * @param m Module object
 */
void bind_all(py::module_& m);

/**
 * @brief Bind AleamCore class to Python
 * 
 * @param m Module object
 */
void bind_aleam_core(py::module& m);

/**
 * @brief Bind distribution functions to Python
 * 
 * @param m Module object
 */
void bind_distributions(py::module& m);

/**
 * @brief Bind array functions to Python
 * 
 * @param m Module object
 */
void bind_arrays(py::module& m);

/**
 * @brief Bind AI/ML classes to Python
 * 
 * @param m Module object
 */
void bind_ai(py::module& m);

/**
 * @brief Bind framework integrations to Python
 * 
 * @param m Module object
 */
void bind_integrations(py::module& m);

/**
 * @brief Bind CUDA functions to Python (if CUDA available)
 * 
 * @param m Module object
 */
void bind_cuda(py::module& m);

/* ============================================================================
 * NumPy Array Conversion Helpers
 * ============================================================================ */

/**
 * @brief Convert std::vector<double> to NumPy array
 * 
 * @param vec Vector to convert
 * @return py::array_t<double> NumPy array
 */
py::array_t<double> vector_to_numpy(const std::vector<double>& vec);

/**
 * @brief Convert std::vector<int64_t> to NumPy array
 * 
 * @param vec Vector to convert
 * @return py::array_t<int64_t> NumPy array
 */
py::array_t<int64_t> vector_int64_to_numpy(const std::vector<int64_t>& vec);

/**
 * @brief Convert std::vector<std::vector<double>> to 2D NumPy array
 * 
 * @param vec 2D vector to convert
 * @return py::array_t<double> 2D NumPy array
 */
py::array_t<double> vector2d_to_numpy(const std::vector<std::vector<double>>& vec);

/**
 * @brief Convert 2D vector of uint8_t to NumPy array
 * 
 * @param vec 2D vector to convert
 * @return py::array_t<uint8_t> 2D NumPy array
 */
py::array_t<uint8_t> mask_to_numpy(const std::vector<std::vector<uint8_t>>& vec);

/* ============================================================================
 * Python Exception Translation
 * ============================================================================ */

/**
 * @brief Register C++ exception translators for Python
 * 
 * Converts C++ exceptions to appropriate Python exceptions.
 */
void register_exception_translators();

#endif /* ALEAM_EXPORTS_H */