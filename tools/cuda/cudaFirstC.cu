#include <stdio.h>
#include <chrono>
#include <iostream>

__global__ void saxpy(int n, float a, float* x, float* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

std::chrono::_V2::system_clock::time_point d_te0;
std::chrono::_V2::system_clock::time_point d_te1;
std::chrono::_V2::system_clock::time_point d_te2;
std::chrono::_V2::system_clock::time_point d_te3;
std::chrono::_V2::system_clock::time_point d_te4;
std::chrono::_V2::system_clock::time_point d_te5;
std::chrono::_V2::system_clock::time_point d_te6;

int main(void)
{
    int N = 1 << 16;
    float *x, *y, *d_x, *d_y;

    d_te0 = std::chrono::high_resolution_clock::now();

    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));

    d_te1 = std::chrono::high_resolution_clock::now();

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    d_te2 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    d_te3 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    d_te4 = std::chrono::high_resolution_clock::now();

    // Perform SAXPY on 1M elements
    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);

    d_te5 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    d_te6 = std::chrono::high_resolution_clock::now();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = max(maxError, abs(y[i] - 4.0f));
    printf("Max error: %f\n", maxError);

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

    std::cout<<"mall:"<<std::chrono::duration_cast<std::chrono::microseconds>(d_te1 - d_te0).count()<<", ";
    std::cout<<"cuda mall:"<<std::chrono::duration_cast<std::chrono::microseconds>(d_te2 - d_te1).count()<<", ";
    std::cout<<"mem h2d:"<<std::chrono::duration_cast<std::chrono::microseconds>(d_te4 - d_te3).count()<<", ";
    std::cout<<"kernel:"<<std::chrono::duration_cast<std::chrono::microseconds>(d_te5 - d_te4).count()<<", ";
    std::cout<<"mem d2h:"<<std::chrono::duration_cast<std::chrono::microseconds>(d_te6 - d_te5).count()<<std::endl;
}