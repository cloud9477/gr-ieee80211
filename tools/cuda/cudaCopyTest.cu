#include <stdio.h>
#include <chrono>
#include <iostream>

/*
this is to test
copy only fft samples
copy 
*/

__global__ void saxpy(int n, float a, float* x, float* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

std::chrono::_V2::system_clock::time_point d_te0;
std::chrono::_V2::system_clock::time_point d_te1;
std::chrono::_V2::system_clock::time_point d_te2;
int main(void)
{
    int N = 1640 * 80;
    float *x, *y, *d_x, *d_y;
    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    d_te0 = std::chrono::high_resolution_clock::now();
    
    int p = 0, q = 0;
    for(int i=0; i<1640; i++)
    {
        cudaMemcpy(d_x + p, x + q, 64 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y + p, y + q, 64 * sizeof(float), cudaMemcpyHostToDevice);
        p += 64;
        q += 80;
    }

    d_te1 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    d_te2 = std::chrono::high_resolution_clock::now();

    // Perform SAXPY on 1M elements
    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = max(maxError, abs(y[i] - 4.0f));
    printf("Max error: %f\n", maxError);

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

    std::cout<<"copy time1:"<<std::chrono::duration_cast<std::chrono::microseconds>(d_te1 - d_te0).count()<<std::endl;
    std::cout<<"copy time2:"<<std::chrono::duration_cast<std::chrono::microseconds>(d_te2 - d_te1).count()<<std::endl;
}