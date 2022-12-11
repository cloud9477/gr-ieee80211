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

const int FFT_26_DEMAP[64] = {
	48, 24, 25, 26, 27, 28, 29, 49, 30, 31, 32, 33, 34, 35, 36, 37, 
	38, 39, 40, 41, 42, 50, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 
	56, 57, 58, 59, 60, 61, 0, 1, 2, 3, 4, 62, 5, 6, 7, 8, 
	9, 10, 11, 12, 13, 14, 15, 16, 17, 63, 18, 19, 20, 21, 22, 23
};

const float LTF_L_26_F_FLOAT[64] = {
    0.0f, 1.0f, -1.0f, -1.0f, 
    1.0f, 1.0f, -1.0f, 1.0f, 
    -1.0f, 1.0f, -1.0f, -1.0f, 
    -1.0f, -1.0f, -1.0f, 1.0f, 
    1.0f, -1.0f, -1.0f, 1.0f, 
    -1.0f, 1.0f, -1.0f, 1.0f, 
    1.0f, 1.0f, 1.0f, 0.0f, 
    0.0f, 0.0f, 0.0f, 0.0f, 
    0.0f, 0.0f, 0.0f, 0.0f, 
    0.0f, 0.0f, 1.0f, 1.0f, 
    -1.0f, -1.0f, 1.0f, 1.0f, 
    -1.0f, 1.0f, -1.0f, 1.0f, 
    1.0f, 1.0f, 1.0f, 1.0f, 
    1.0f, -1.0f, -1.0f, 1.0f, 
    1.0f, -1.0f, 1.0f, -1.0f, 
    1.0f, 1.0f, 1.0f, 1.0f};

cuFloatComplex* signalX;
cuFloatComplex* signalY;
cuFloatComplex* signalA;
cuFloatComplex* signalB;
cuFloatComplex* signalHSig;
float* signalLLR;
cufftHandle signalPlan;
cuFloatComplex* signalLtf;
int* signalLLRMap;

void signalMall()
{
  cudaMalloc(&signalX, 8192*sizeof(cuFloatComplex));
  cudaMalloc(&signalY, 8192*sizeof(cuFloatComplex));
  cudaMalloc(&signalA, 240*sizeof(cuFloatComplex));
  cudaMalloc(&signalB, 192*sizeof(cuFloatComplex));
  cudaMalloc(&signalHSig, 128*sizeof(cuFloatComplex));
  cudaMalloc(&signalLtf, 64*sizeof(cuFloatComplex));
  cudaMalloc(&signalLLR, 64*sizeof(float));
  cudaMalloc(&signalLLRMap, 64*sizeof(int));
  cuFloatComplex signalLtfTmp[64];
  for(int i=0;i<64;i++)
  {
    signalLtfTmp[i] = make_cuComplex(LTF_L_26_F_FLOAT[i]*2.0f, 0.0f);
  }
  cudaMemcpy(signalLtf, signalLtfTmp, 64*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
  cudaMemcpy(signalLLRMap, FFT_26_DEMAP, 64*sizeof(int), cudaMemcpyHostToDevice);
  cufftPlan1d(&signalPlan, 64, CUFFT_C2C, 3);
}

void signalFree()
{
  cudaFree(signalX);
  cudaFree(signalY);
  cudaFree(signalA);
  cudaFree(signalB);
  cudaFree(signalHSig);
  cudaFree(signalLtf);
  cudaFree(signalLLR);
  cudaFree(signalLLRMap);
  cufftDestroy(signalPlan);
}

__global__
void cuSignalKernel(int n, int s, float radStep, cuFloatComplex *x, cuFloatComplex *y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    y[i] = cuCmulf(x[i], make_cuFloatComplex(cosf( ((float)s+i) * radStep), sinf(((float)s+i) * radStep)));
  }
}

void cuSignalCfoCompen(int n, int s, float radStep, const cuFloatComplex *x, cuFloatComplex *y)
{
  cudaMemcpy(signalX, x, n*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
  // N+255 means requires at least 1 block, and 256 means 256 threads in each block
  cuSignalKernel<<<(n+1024)/1024, 1024>>>(n, s, radStep, signalX, signalY);
  cudaMemcpy(y, signalY, n*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
}

__global__
void cuSignalChannelCfo(int s, float radStep, cuFloatComplex *x, cuFloatComplex *y)
{
  int i = threadIdx.x;
  if(i >= 8 && i < 72)
  {
    y[i-8] = cuCmulf(x[i], make_cuFloatComplex(cosf( ((float)s+i) * radStep), sinf(((float)s+i) * radStep)));
  }
  if(i >= 72 && i < 136)
  {
    y[i-8] = cuCmulf(x[i], make_cuFloatComplex(cosf( ((float)s+i) * radStep), sinf(((float)s+i) * radStep)));
  }
  if(i >= 152 && i < 216)
  {
    y[i-24] = cuCmulf(x[i], make_cuFloatComplex(cosf( ((float)s+i) * radStep), sinf(((float)s+i) * radStep)));
  }
}

__global__
void cuSignalChannel(cuFloatComplex* inSig, cuFloatComplex* hsig, cuFloatComplex* ltf, float*llr, int* demap)
{
  int i = threadIdx.x;
  hsig[i] = cuCdivf( cuCaddf(inSig[i], inSig[i+64]), ltf[i]);
  hsig[i + 64] = cuCdivf(inSig[i+128], hsig[i]);
  llr[demap[i]] = cuCrealf(hsig[i + 64]);
}

void cuSignalChannel(int s, float radStep, const cuFloatComplex *sig, cuFloatComplex *h, float* llr)
{
  cudaMemcpy(signalA, sig, 240*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
  cuSignalChannelCfo<<<1, 216>>>(s, radStep, signalA, signalB);
  cufftExecC2C(signalPlan, signalB, signalB, CUFFT_FORWARD);
  cuSignalChannel<<<1, 64>>>(signalB, signalHSig, signalLtf, signalLLR, signalLLRMap);
  cudaMemcpy(h, signalHSig, 64*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
  cudaMemcpy(llr, signalLLR, 48*sizeof(float), cudaMemcpyDeviceToHost);
}



/*--------------------------------------------------------------------------------------------------------*/

cuFloatComplex* ppSig;
cuFloatComplex* ppSigConj;
cuFloatComplex* ppSigConjAvg;
float* ppSigConjAvgMag;
float* ppSigMag2;
float* ppSigMag2Avg;
float* ppOut;

cuFloatComplex ppTest[PREPROC_MAX];

void preprocMall()
{
  cudaMalloc(&ppSig, PREPROC_MAX*sizeof(cuFloatComplex));
  cudaMalloc(&ppSigConj, PREPROC_MAX*sizeof(cuFloatComplex));
  cudaMalloc(&ppSigConjAvg, PREPROC_MAX*sizeof(cuFloatComplex));
  cudaMalloc(&ppSigConjAvgMag, PREPROC_MAX*sizeof(float));
  cudaMalloc(&ppSigMag2, PREPROC_MAX*sizeof(float));
  cudaMalloc(&ppSigMag2Avg, PREPROC_MAX*sizeof(float));
  cudaMalloc(&ppOut, PREPROC_MAX*sizeof(float));
}

void preprocFree()
{
  cudaFree(ppSig);
  cudaFree(ppSigConj);
  cudaFree(ppSigConjAvg);
  cudaFree(ppSigConjAvgMag);
  cudaFree(ppSigMag2);
  cudaFree(ppSigMag2Avg);
  cudaFree(ppOut);
}

__global__
void cuPreProcConj(int n, cuFloatComplex* inSig, cuFloatComplex* inSigConj)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < (n-16))
  {
    inSigConj[i] = cuCmulf(inSig[i], make_cuFloatComplex (cuCrealf(inSig[i+16]), -cuCimagf(inSig[i+16])));
  }
}

__global__
void cuPreProcMag2(int n, cuFloatComplex* inSig, float* inSigMag2)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    inSigMag2[i] = cuCabsf(inSig[i]);
    inSigMag2[i] = inSigMag2[i] * inSigMag2[i];
  }
}

__global__
void cuPreProcConjAvgMag(int n, cuFloatComplex* inSigConj, cuFloatComplex* inSigConjAvg, float* inSigConjAvgMag)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < (n-48))
  {
    inSigConjAvg[i] = make_cuFloatComplex(0.0f, 0.0f);
    for(int j=0;j<48;j++)
    {
      inSigConjAvg[i] = cuCaddf(inSigConjAvg[i], inSigConj[i+j]);
    }
    inSigConjAvg[i] = cuCdivf(inSigConjAvg[i], make_cuFloatComplex(48.0f, 0.0f));
    inSigConjAvgMag[i] = cuCabsf(inSigConjAvg[i]);
  }
}

__global__
void cuPreProcMag2AvgOut(int n, float* inSigMag2, float* inSigMag2Avg, float* inSigConjAvgMag, float* out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < (n-80))
  {
    inSigMag2Avg[i+16] = 0.0f;
    for(int j=0;j<64;j++)
    {
      inSigMag2Avg[i+16] += inSigMag2[i+16+j];
    }
    inSigMag2Avg[i+16] = inSigMag2Avg[i+16] / 64.0f;
    out[i] = inSigConjAvgMag[i] / inSigMag2Avg[i+16];
  }
}

void cuPreProc(int n, const cuFloatComplex *sig, float* ac, cuFloatComplex* conj)
{
  if(n > 80)
  {
    // for(int i=0;i<n-16;i++)
    // {
    //   std::cout<<cuCrealf(sig[i])<<" ";
    // }
    cudaMemcpy(ppSig, sig, n*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cuPreProcConj<<<n/1024 + 1, 1024>>>(n, ppSig, ppSigConj);
    // cudaMemcpy(ppTest, ppSigConj, n*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    // for(int i=0;i<n-16;i++)
    // {
    //   std::cout<<cuCrealf(ppTest[i])<<" ";
    // }
    cuPreProcMag2<<<n/1024 + 1, 1024>>>(n, ppSig, ppSigMag2);
    cuPreProcConjAvgMag<<<n/1024 + 1, 1024>>>(n, ppSigConj, ppSigConjAvg, ppSigConjAvgMag);
    cuPreProcMag2AvgOut<<<n/1024 + 1, 1024>>>(n, ppSigMag2, ppSigMag2Avg, ppSigConjAvgMag, ppOut);
    cudaMemcpy(ac, ppOut, (n - 80)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(conj, ppSigConjAvg, (n - 80)*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
  }
}