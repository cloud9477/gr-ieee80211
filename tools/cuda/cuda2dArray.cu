#include <stdio.h>
#include <chrono>
#include <iostream>

int main(void)
{
    int N = 64;
    float x[64][64];
    float y[65536];
    float *d_x;

    cudaMalloc(&d_x, N * N * sizeof(float));

    for (int i = 0; i < N; i++) {
        for(int j=0;j<N;j++)
        {
            x[i][j] = 2.0f;
        }
    }

    cudaMemcpy(d_x, x, N * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(y, d_x, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N*N; i++)
        maxError = max(maxError, abs(y[i] - 2.0f));
    printf("Max error: %f\n", maxError);

    cudaFree(d_x);
}