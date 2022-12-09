#include <stdio.h> 
#include <iostream>
#include "cuComplex.h"

__global__
void cuCompMultiKernel(int n, cuFloatComplex* x, cuFloatComplex* y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    y[i] = cuCmulf(x[i], y[i]);
  }
}

int main()
{
    std::cout<<"cuComplex Type Size: "<<sizeof(cuFloatComplex)<<std::endl;

    int n = 16;

    cuFloatComplex* signalX;
    cuFloatComplex* signalY;
    cudaMalloc(&signalX, n*sizeof(cuFloatComplex));
    cudaMalloc(&signalY, n*sizeof(cuFloatComplex));

    cuFloatComplex* x = (cuFloatComplex*)malloc(n*sizeof(cuFloatComplex));
    cuFloatComplex* y = (cuFloatComplex*)malloc(n*sizeof(cuFloatComplex));

    std::cout<<"original cuda complex type, real and imag of a cuFloatComplex"<<std::endl;
    for (int i = 0; i < n; i++) {
        x[i] = make_cuFloatComplex((float)i, (float)-i);
        std::cout <<cuCrealf(x[i]) << ", " << cuCimagf(x[i])<<std::endl;
        y[i] = make_cuFloatComplex((float)i, (float)i * 2.0f);
    }

    float* x2 = (float*) malloc(n*2*sizeof(float));
    float* y2 = (float*) malloc(n*2*sizeof(float));

    std::cout<<"convert cuFloatComplex to float array, real and imag"<<std::endl;
    for (int i = 0; i < n; i++) {
        x2[i*2] = (float)i;
        x2[i*2+1] = (float)-i;
        std::cout << x2[i*2] << ", " << x2[i*2+1] <<std::endl;
        y2[i*2] = (float)i;
        y2[i*2+1] = (float)i * 2.0f;
    }

    cudaMemcpy(signalX, x, n*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(signalY, y, n*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    cuCompMultiKernel<<<(n+1024)/1024, 1024>>>(n, signalX, signalY);

    cudaMemcpy(y, signalY, n*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    std::cout<<"cuda results"<<std::endl;
    for (int i = 0; i < n; i++) {
        std::cout <<cuCrealf(y[i]) << ", " << cuCimagf(y[i])<<std::endl;
    }

    cudaMemcpy(signalX, (cuFloatComplex*)x2, n*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(signalY, (cuFloatComplex*)y2, n*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    cuCompMultiKernel<<<(n+1024)/1024, 1024>>>(n, signalX, signalY);

    cudaMemcpy((cuFloatComplex*)y2, signalY, n*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    std::cout<<"cuda results 2"<<std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << y2[i*2] << ", " << y2[i*2+1] <<std::endl;
    }

    free(x);
    free(y);
    free(x2);
    free(y2);

    cudaFree(signalX);
    cudaFree(signalY);
    return 0;
}