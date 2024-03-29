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
/* BINDTOOL_HEADER_FILE(encode2.h)                                        */
/* BINDTOOL_HEADER_FILE_HASH(e7e232f56e917fcf5fb658db37536fde)                     */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/ieee80211/encode2.h>
// pydoc.h is automatically generated in the build directory
#include <encode2_pydoc.h>

void bind_encode2(py::module& m)
{

    using encode2    = ::gr::ieee80211::encode2;


    py::class_<encode2, gr::block, gr::basic_block,
        std::shared_ptr<encode2>>(m, "encode2", D(encode2))

        .def(py::init(&encode2::make),
           D(encode2,make)
        )
        



        ;




}








