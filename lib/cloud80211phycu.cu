/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 20M bw and upto 2x2
 *     PHY utilization functions and parameters CUDA Version
 *     Copyright (C) Dec 1, 2022  Zelin Yun
 *
 *     This program is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU Affero General Public License as published
 *     by the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU Affero General Public License for more details.
 *
 *     You should have received a copy of the GNU Affero General Public License
 *     along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "cloud80211phycu.cuh"

__global__
void cuMultiplyKernel(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i];
}

void cuMultiply(int n, float a, const float *x, float *y)
{
  float *d_x, *d_y;
  cudaMalloc(&d_x, n*sizeof(float)); 
  cudaMalloc(&d_y, n*sizeof(float));
  cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, n*sizeof(float), cudaMemcpyHostToDevice);
  // Perform SAXPY on 1M elements
  // N+255 means requires at least 1 block, and 256 means 256 threads in each block
  cuMultiplyKernel<<<(n+255)/256, 256>>>(n, 1.5f, d_x, d_y);
  cudaMemcpy(y, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_y);
}

