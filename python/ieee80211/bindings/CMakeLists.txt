# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT ieee80211_sources)
    MESSAGE(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT ieee80211_sources)

########################################################################
# Check for pygccxml
########################################################################
GR_PYTHON_CHECK_MODULE_RAW(
    "pygccxml"
    "import pygccxml"
    PYGCCXML_FOUND
    )

include(GrPybind)

########################################################################
# Python Bindings
########################################################################

list(APPEND ieee80211_python_files
    trigger_python.cc
    sync_python.cc
    signal_python.cc
    modulation_python.cc
    demod_python.cc
    decode_python.cc
    encode_python.cc
    signal2_python.cc
    demod2_python.cc
    preproc_python.cc python_bindings.cc)

GR_PYBIND_MAKE_OOT(ieee80211
   ../../..
   gr::ieee80211
   "${ieee80211_python_files}")

# copy bindings extension for use in QA test module
add_custom_command(TARGET ieee80211_python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:ieee80211_python>
    ${PROJECT_BINARY_DIR}/test_modules/gnuradio/ieee80211/
)

install(TARGETS ieee80211_python DESTINATION ${GR_PYTHON_DIR}/gnuradio/ieee80211 COMPONENT pythonapi)
