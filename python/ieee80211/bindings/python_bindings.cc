/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <pybind11/pybind11.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace py = pybind11;

// Headers for binding functions
/**************************************/
// The following comment block is used for
// gr_modtool to insert function prototypes
// Please do not delete
/**************************************/
// BINDING_FUNCTION_PROTOTYPES(
    void bind_trigger(py::module& m);
    void bind_sync(py::module& m);
    void bind_signal(py::module& m);
    void bind_modulation(py::module& m);
    void bind_demod(py::module& m);
    void bind_decode(py::module& m);
    void bind_encode(py::module& m);
    void bind_signal2(py::module& m);
    void bind_demod2(py::module& m);
    void bind_preproc(py::module& m);
    void bind_demodcu(py::module& m);
    void bind_demodcu2(py::module& m);
// ) END BINDING_FUNCTION_PROTOTYPES


// We need this hack because import_array() returns NULL
// for newer Python versions.
// This function is also necessary because it ensures access to the C API
// and removes a warning.
void* init_numpy()
{
    import_array();
    return NULL;
}

PYBIND11_MODULE(ieee80211_python, m)
{
    // Initialize the numpy C API
    // (otherwise we will see segmentation faults)
    init_numpy();

    // Allow access to base block methods
    py::module::import("gnuradio.gr");

    /**************************************/
    // The following comment block is used for
    // gr_modtool to insert binding function calls
    // Please do not delete
    /**************************************/
    // BINDING_FUNCTION_CALLS(
    bind_trigger(m);
    bind_sync(m);
    bind_signal(m);
    bind_modulation(m);
    bind_demod(m);
    bind_decode(m);
    bind_encode(m);
    bind_signal2(m);
    bind_demod2(m);
    bind_preproc(m);
    bind_demodcu(m);
    bind_demodcu2(m);
    // ) END BINDING_FUNCTION_CALLS
}