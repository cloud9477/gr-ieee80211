#!/bin/bash

nvcc -o cudaPrintHardware.cubin cudaPrintHardware.cu
nvcc -o cudaComplexTest.cubin cudaComplexTest.cu
nvcc -o cudaFirstC.cubin cudaFirstC.cu
nvcc -o cudaCopyTest.cubin cudaCopyTest.cu
nvcc -o cuda2dArray.cubin cuda2dArray.cu
nvcc -o cudaViterbi.cubin cudaViterbi.cu