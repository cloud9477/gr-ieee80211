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
/* BINDTOOL_HEADER_FILE(decode.h)                                        */
/* BINDTOOL_HEADER_FILE_HASH(a361cf4d308060a5eddc043fa1d668da)                     */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/ieee80211/decode.h>
// pydoc.h is automatically generated in the build directory
#include <decode_pydoc.h>

void bind_decode(py::module& m)
{

    using decode    = ::gr::ieee80211::decode;


    py::class_<decode, gr::block, gr::basic_block,
        std::shared_ptr<decode>>(m, "decode", D(decode))

        .def(py::init(&decode::make),
           D(decode,make)
        )
        



        ;




}








