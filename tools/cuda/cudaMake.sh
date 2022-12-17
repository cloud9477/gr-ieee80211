#!/bin/bash

nvcc -o cudaPrintHardware.cubin cudaPrintHardware.cu
nvcc -o cudaComplexTest.cubin cudaComplexTest.cu
nvcc -o cudaFirstC.cubin cudaFirstC.cu
nvcc -o cudaCopyTest.cubin cudaCopyTest.cu