/*
 * Copyright 2023 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/***********************************************************************************/
/* This file is automatically generated using bindtool and can be manually edited  */
/* The following lines can be configured to regenerate this file during cmake      */
/* If manual edits are made, the following tags should be modified accordingly.    */
/* BINDTOOL_GEN_AUTOMATIC(0)                                                       */
/* BINDTOOL_USE_PYGCCXML(0)                                                        */
/* BINDTOOL_HEADER_FILE(pad.h)                                        */
/* BINDTOOL_HEADER_FILE_HASH(ddf5ba72069a2ab2f7dea3993c515c37)                     */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/ieee80211/pad.h>
// pydoc.h is automatically generated in the build directory
#include <pad_pydoc.h>

void bind_pad(py::module& m)
{

    using pad    = gr::ieee80211::pad;


    py::class_<pad, gr::block, gr::basic_block,
        std::shared_ptr<pad>>(m, "pad", D(pad))

        .def(py::init(&pad::make),
           D(pad,make)
        )
        



        ;




}







