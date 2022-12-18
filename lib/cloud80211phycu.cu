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


/*--------------------------------------------------------------------------------------------------------*/

cuFloatComplex* ppSig;
cuFloatComplex* ppSigConj;
cuFloatComplex* ppSigConjAvg;
float* ppSigConjAvgMag;
float* ppSigMag2;
float* ppSigMag2Avg;
float* ppOut;

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
    inSigConjAvgMag[i] = cuCabsf(inSigConjAvg[i]);
  }
}

__global__
void cuPreProcMag2AvgOut(int n, float* inSigMag2, float* inSigMag2Avg, float* inSigConjAvgMag, float* out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < (n-64))
  {
    inSigMag2Avg[i] = 0.0f;
    for(int j=0;j<64;j++)
    {
      inSigMag2Avg[i] += inSigMag2[i+j];
    }
    out[i] = inSigConjAvgMag[i] / inSigMag2Avg[i];
  }
}

void cuPreProc(int n, const cuFloatComplex *sig, float* ac, cuFloatComplex* conj)
{
  if(n > 64 && n < PREPROC_MAX)
  {
    cudaMemcpy(ppSig, sig, n*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cuPreProcConj<<<n/1024 + 1, 1024>>>(n, ppSig, ppSigConj);
    cuPreProcMag2<<<n/1024 + 1, 1024>>>(n, ppSig, ppSigMag2);
    cuPreProcConjAvgMag<<<n/1024 + 1, 1024>>>(n, ppSigConj, ppSigConjAvg, ppSigConjAvgMag);
    cuPreProcMag2AvgOut<<<n/1024 + 1, 1024>>>(n, ppSigMag2, ppSigMag2Avg, ppSigConjAvgMag, ppOut);
    cudaMemcpy(ac, ppOut, (n - 64)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(conj, ppSigConjAvg, (n - 64)*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
  }
}

/*--------------------------------------------------------------------------------------------------------*/
int mapDeshiftFftLegacy[64] = {
  0, 24, 25, 26, 27, 28, 29, 0, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 0, 43, 44, 45, 46, 47, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 18, 19, 20, 21, 22, 23};
int mapDeshiftFftNonlegacy[64] = {
  0, 26, 27, 28, 29, 30, 31, 0, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 0, 45, 46, 47, 48, 49, 50, 51, 0, 0, 0, 
  0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 20, 21, 22, 23, 24, 25};
cuFloatComplex* demodChanSiso;
cuFloatComplex* demodSig;
cuFloatComplex* demodSigFft;
cufftHandle demodPlan;
float* demodSigLlr;
cuFloatComplex pListTmp[127];
cuFloatComplex* pList;

int* demodDemapFftL;
int* demodDemapBpskL;
int* demodDemapQpskL;
int* demodDemap16QamL;
int* demodDemap64QamL;

int* demodDemapFftNL;
int* demodDemapBpskNL;
int* demodDemapQpskNL;
int* demodDemap16QamNL;
int* demodDemap64QamNL;
int* demodDemap256QamNL;

__global__
void cuDemodChopSamp(int n, cuFloatComplex* sig, cuFloatComplex* sigfft)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = i / 80;       // symbol index
  int offset = i % 80;  
  if(i < n && offset >= 8 && offset < 72)
  {
    sigfft[j*64 + offset - 8] = sig[i];
  }
}

__global__
void cuDemodChanComp(int n, cuFloatComplex* sigfft, cuFloatComplex* chan)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = i % 64;
  if(i < n)
  {
    sigfft[i] = cuCdivf(sigfft[i], chan[offset]);
  }
}

// __global__
// void cuDemodChanComp(int n, cuFloatComplex* sigfft, cuFloatComplex* chan)
// {
//   int i = threadIdx.x;
//   int I = blockIdx.x * blockDim.x + threadIdx.x;
//   int offset = I % 64;
//   __shared__ cuFloatComplex chanIn[64];
//   if(I >= n)
//   {
//     return;
//   }
//   if(i < 64)
//   {
//     chanIn[i] = chan[i];
//   }
//   __syncthreads();
//   sigfft[I] = cuCdivf(sigfft[I], chanIn[offset]);
// }

__global__
void cuDemodQamToLlrLegacy(int n, int nBPSCS, cuFloatComplex* sigfft, float* llr, cuFloatComplex* p, int* deshift, int* deint)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = i / 64;       // sym index
  int offset = i % 64;  // sample index
  int pIndex = (j+1) % 127;
  cuFloatComplex qam;
  float qamReal = 0.0f, qamImag = 0.0f;
  int scIndex = 0;

  __shared__ cuFloatComplex pilot;
  __shared__ float pilotAbs;

  if(i < n)
  {
    return;
  }

  if(offset == 0)
  {
    pilot = make_cuFloatComplex(0.0f, 0.0f);
    pilot = cuCaddf(pilot, cuCmulf(sigfft[j*64 + 43], cuCmulf(make_cuFloatComplex(1.0f, 0.0f), p[pIndex])));
    pilot = cuCaddf(pilot, cuCmulf(sigfft[j*64 + 57], cuCmulf(make_cuFloatComplex(1.0f, 0.0f), p[pIndex])));
    pilot = cuCaddf(pilot, cuCmulf(sigfft[j*64 + 7], cuCmulf(make_cuFloatComplex(1.0f, 0.0f), p[pIndex])));
    pilot = cuCaddf(pilot, cuCmulf(sigfft[j*64 + 21], cuCmulf(make_cuFloatComplex(-1.0f, 0.0f), p[pIndex])));
    pilotAbs = cuCabsf(pilot);
    pilot = cuConjf(pilot);
  }
  __syncthreads();
  if(offset==0 || (offset>=27 && offset<=37) || offset==7 || offset==21 || offset==43 || offset==57)
  {}
  else
  {
    qam = cuCdivf(cuCmulf(sigfft[i], pilot), make_cuFloatComplex(pilotAbs, 0.0f));
    scIndex = deshift[offset];      // sc after fft to data sc index
    if(nBPSCS == 1)
    {
      llr[j*48 + deint[scIndex]] = qamReal;
    }
    else if(nBPSCS == 2)
    {
      qam = cuCmulf(qam, make_cuFloatComplex(1.4142135623730951f, 0.0f));
      qamReal = cuCrealf(qam);
      qamImag = cuCimagf(qam);
      llr[j*96 + deint[scIndex*2]] = qamReal;
      llr[j*96 + deint[scIndex*2+1]] = qamImag;
    }
    else if(nBPSCS == 4)
    {
      qam = cuCmulf(qam, make_cuFloatComplex(3.1622776601683795f, 0.0f));
      qamReal = cuCrealf(qam);
      qamImag = cuCimagf(qam);
      llr[j*192 + deint[scIndex*4]] = qamReal;
      llr[j*192 + deint[scIndex*4+1]] = 2.0f - fabsf(qamReal);
      llr[j*192 + deint[scIndex*4+1]] = qamImag;
      llr[j*192 + deint[scIndex*4+1]] = 2.0f - fabsf(qamImag);
    }
    else
    {
      qam = cuCmulf(qam, make_cuFloatComplex(6.48074069840786f, 0.0f));
      qamReal = cuCrealf(qam);
      qamImag = cuCimagf(qam);
      llr[j*288 + deint[scIndex*6]] = qamReal;
      llr[j*288 + deint[scIndex*6+1]] = 4.0f - fabsf(qamReal);
      llr[j*288 + deint[scIndex*6+2]] = 2.0f - fabsf(4.0f - fabsf(qamReal));
      llr[j*288 + deint[scIndex*6+3]] = qamImag;
      llr[j*288 + deint[scIndex*6+4]] = 4.0f - fabsf(qamImag);
      llr[j*288 + deint[scIndex*6+5]] = 2.0f - fabsf(4.0f - fabsf(qamImag));
    }
  }
}

__global__
void cuDemodQamToLlrNonlegacy(int n, int nBPSCS, cuFloatComplex* sigfft, float* llr, cuFloatComplex* p, int* deshift, int* deint, int format)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = i / 64;       // sym index
  int offset = i % 64;  // sample index
  int pIndex;
  if(format == C8P_F_HT)
  {
    pIndex = (j+3) % 127;
  }
  else
  {
    pIndex = (j+4) % 127;
  }
  cuFloatComplex qam;
  float qamReal = 0.0f, qamImag = 0.0f;
  int scIndex;

  __shared__ cuFloatComplex pilot;
  __shared__ float pilotAbs;

  if(i < n)
  {
    return;
  }

  if(offset == 0)
  {
    float tmpPilot[4];
    if((j%4) == 0)
    {
      tmpPilot[0] = 1.0f; tmpPilot[1] = 1.0f; tmpPilot[2] = 1.0f; tmpPilot[3] = -1.0f;
    }
    else if((j%4) == 1)
    {
      tmpPilot[0] = 1.0f; tmpPilot[1] = 1.0f; tmpPilot[2] = -1.0f; tmpPilot[3] = 1.0f;
    }
    else if((j%4) == 2)
    {
      tmpPilot[0] = 1.0f; tmpPilot[1] = -1.0f; tmpPilot[2] = 1.0f; tmpPilot[3] = 1.0f;
    }
    else
    {
      tmpPilot[0] = -1.0f; tmpPilot[1] = 1.0f; tmpPilot[2] = 1.0f; tmpPilot[3] = 1.0f;
    }
    pilot = make_cuFloatComplex(0.0f, 0.0f);
    pilot = cuCaddf(pilot, cuCmulf(sigfft[j*64 + 43], cuCmulf(make_cuFloatComplex(tmpPilot[0], 0.0f), p[pIndex])));
    pilot = cuCaddf(pilot, cuCmulf(sigfft[j*64 + 57], cuCmulf(make_cuFloatComplex(tmpPilot[1], 0.0f), p[pIndex])));
    pilot = cuCaddf(pilot, cuCmulf(sigfft[j*64 + 7], cuCmulf(make_cuFloatComplex(tmpPilot[2], 0.0f), p[pIndex])));
    pilot = cuCaddf(pilot, cuCmulf(sigfft[j*64 + 21], cuCmulf(make_cuFloatComplex(tmpPilot[3], 0.0f), p[pIndex])));
    pilotAbs = cuCabsf(pilot);
    pilot = cuConjf(pilot);
  }
  __syncthreads();
  if(offset==0 || (offset>=29 && offset<=35) || offset==7 || offset==21 || offset==43 || offset==57)
  {}
  else
  {
    qam = cuCdivf(cuCmulf(sigfft[i], pilot), make_cuFloatComplex(pilotAbs, 0.0f));
    scIndex = deshift[offset];      // sc after fft to data sc index
    if(nBPSCS == 1)
    {
      llr[j*52 + deint[scIndex]] = qamReal;
    }
    else if(nBPSCS == 2)
    {
      qam = cuCmulf(qam, make_cuFloatComplex(1.4142135623730951f, 0.0f));
      qamReal = cuCrealf(qam);
      qamImag = cuCimagf(qam);
      llr[j*104 + deint[scIndex*2]] = qamReal;
      llr[j*104 + deint[scIndex*2+1]] = qamImag;
    }
    else if(nBPSCS == 4)
    {
      qam = cuCmulf(qam, make_cuFloatComplex(3.1622776601683795f, 0.0f));
      qamReal = cuCrealf(qam);
      qamImag = cuCimagf(qam);
      llr[j*208 + deint[scIndex*4]] = qamReal;
      llr[j*208 + deint[scIndex*4+1]] = 2.0f - fabsf(qamReal);
      llr[j*208 + deint[scIndex*4+1]] = qamImag;
      llr[j*208 + deint[scIndex*4+1]] = 2.0f - fabsf(qamImag);
    }
    else if(nBPSCS == 6)
    {
      qam = cuCmulf(qam, make_cuFloatComplex(6.48074069840786f, 0.0f));
      qamReal = cuCrealf(qam);
      qamImag = cuCimagf(qam);
      llr[j*312 + deint[scIndex*6]] = qamReal;
      llr[j*312 + deint[scIndex*6+1]] = 4.0f - fabsf(qamReal);
      llr[j*312 + deint[scIndex*6+2]] = 2.0f - fabsf(4.0f - fabsf(qamReal));
      llr[j*312 + deint[scIndex*6+3]] = qamImag;
      llr[j*312 + deint[scIndex*6+4]] = 4.0f - fabsf(qamImag);
      llr[j*312 + deint[scIndex*6+5]] = 2.0f - fabsf(4.0f - fabsf(qamImag));
    }
    else
    {
      qam = cuCmulf(qam, make_cuFloatComplex(13.038404810405298f, 0.0f));
      qamReal = cuCrealf(qam);
      qamImag = cuCimagf(qam);
      llr[j*416 + deint[scIndex*8]] = qamReal;
      llr[j*416 + deint[scIndex*8+1]] = 8.0f - fabsf(qamReal);
      llr[j*416 + deint[scIndex*8+2]] = 4.0f - fabsf(8.0f - fabsf(qamReal));
      llr[j*416 + deint[scIndex*8+3]] = 2.0f - fabsf(4.0f - fabsf(8.0f - fabsf(qamReal)));
      llr[j*416 + deint[scIndex*8+4]] = qamImag;
      llr[j*416 + deint[scIndex*8+5]] = 8.0f - fabsf(qamImag);
      llr[j*416 + deint[scIndex*8+6]] = 4.0f - fabsf(8.0f - fabsf(qamImag));
      llr[j*416 + deint[scIndex*8+7]] = 2.0f - fabsf(4.0f - fabsf(8.0f - fabsf(qamImag)));
    }
  }
}

void cuDemodMall()
{
  cudaMalloc(&demodChanSiso, sizeof(cuFloatComplex) * 64);
  cudaMalloc(&demodSig, sizeof(cuFloatComplex) * CUDEMOD_S_MAX * 80);
  cudaMalloc(&demodSigFft, sizeof(cuFloatComplex) * CUDEMOD_S_MAX * 64);
  if(cufftPlan1d(&demodPlan, 64, CUFFT_C2C, CUDEMOD_FFT_BATCH) != CUFFT_SUCCESS){
    std::cout<<"cloud80211 cufft, plan creation failed"<<std::endl;
  }
  cudaMalloc(&demodSigLlr, sizeof(float) * CUDEMOD_S_MAX * 52 * 8);
  for(int i=0;i<127;i++)
  {
    pListTmp[i] = make_cuFloatComplex((float)PILOT_P[i], 0.0f);
  }
  cudaMalloc(&pList, sizeof(cuFloatComplex) * 127);
  cudaMemcpy(pList, pListTmp, 127*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
  
  cudaMalloc(&demodDemapFftL, sizeof(int) * 64);
  cudaMemcpy(demodDemapFftL, mapDeshiftFftLegacy, 64*sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&demodDemapBpskL, sizeof(int) * 48);
  cudaMalloc(&demodDemapQpskL, sizeof(int) * 96);
  cudaMalloc(&demodDemap16QamL, sizeof(int) * 192);
  cudaMalloc(&demodDemap64QamL, sizeof(int) * 288);

  cudaMalloc(&demodDemapFftNL, sizeof(int) * 64);
  cudaMemcpy(demodDemapFftNL, mapDeshiftFftNonlegacy, 64*sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&demodDemapBpskNL, sizeof(int) * 52);
  cudaMalloc(&demodDemapQpskNL, sizeof(int) * 104);
  cudaMalloc(&demodDemap16QamNL, sizeof(int) * 208);
  cudaMalloc(&demodDemap64QamNL, sizeof(int) * 312);
  cudaMalloc(&demodDemap256QamNL, sizeof(int) * 416);

  cudaMemcpy(demodDemapBpskL, mapDeintLegacyBpsk, 48*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(demodDemapQpskL, mapDeintLegacyQpsk, 96*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(demodDemap16QamL, mapDeintLegacy16Qam, 192*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(demodDemap64QamL, mapDeintLegacy64Qam, 288*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(demodDemapBpskNL, mapDeintNonlegacyBpsk, 52*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(demodDemapQpskNL, mapDeintNonlegacyQpsk, 104*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(demodDemap16QamNL, mapDeintNonlegacy16Qam, 208*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(demodDemap64QamNL, mapDeintNonlegacy64Qam, 312*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(demodDemap256QamNL, mapDeintNonlegacy256Qam, 416*sizeof(int), cudaMemcpyHostToDevice);
}

void cuDemodFree()
{
  cudaFree(demodChanSiso);
  cudaFree(demodSig);
  cudaFree(demodSigFft);
  cufftDestroy(demodPlan);
  cudaFree(demodSigLlr);
  cudaFree(pList);

  cudaFree(demodDemapFftL);
  cudaFree(demodDemapBpskL);
  cudaFree(demodDemapQpskL);
  cudaFree(demodDemap16QamL);
  cudaFree(demodDemap64QamL);
  cudaFree(demodDemapBpskNL);
  cudaFree(demodDemapQpskNL);
  cudaFree(demodDemap16QamNL);
  cudaFree(demodDemap64QamNL);
  cudaFree(demodDemap256QamNL);
}

void cuDemodChanSiso(cuFloatComplex *chan)
{
  cudaMemcpy(demodChanSiso, chan, 64*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
}

void cuDemodSigCopy(int i, int n, const cuFloatComplex *sig)
{
  if(i >= 0 && n >= 0 && (i+n) < (CUDEMOD_S_MAX * 80))
  {
    cudaMemcpy(&demodSig[i], sig, n*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
  }
}

void cuDemodSiso(c8p_mod* m)
{
  cuDemodChopSamp<<<(m->nSym * m->nSymSamp)/1024 + 1, 1024>>>(m->nSym * m->nSymSamp, demodSig, demodSigFft);
  for(int symIter=0; symIter < ((m->nSym + CUDEMOD_FFT_BATCH - 1) / CUDEMOD_FFT_BATCH); symIter++ )   // each round inlcudes 256 batches
  {
    cufftExecC2C(demodPlan, &demodSigFft[symIter*CUDEMOD_FFT_BATCH*64], &demodSigFft[symIter*CUDEMOD_FFT_BATCH*64], CUFFT_FORWARD);
  }
  cuDemodChanComp<<<(m->nSym * 64)/1024 + 1, 1024>>>(m->nSym * 64, demodSigFft, demodChanSiso);
  if(m->format == C8P_F_L)
  {
    if(m->nBPSCS == 1)
    {
      cuDemodQamToLlrLegacy<<<(m->nSym * 64)/1024 + 1, 1024>>>(m->nSym * 64, m->nBPSCS, demodSigFft, demodSigLlr, pList, demodDemapFftL, demodDemapBpskL);
    }
    else if(m->nBPSCS == 2)
    {
      cuDemodQamToLlrLegacy<<<(m->nSym * 64)/1024 + 1, 1024>>>(m->nSym * 64, m->nBPSCS, demodSigFft, demodSigLlr, pList, demodDemapFftL, demodDemapQpskL);
    }
    else if(m->nBPSCS == 4)
    {
      cuDemodQamToLlrLegacy<<<(m->nSym * 64)/1024 + 1, 1024>>>(m->nSym * 64, m->nBPSCS, demodSigFft, demodSigLlr, pList, demodDemapFftL, demodDemap16QamL);
    }
    else
    {
      cuDemodQamToLlrLegacy<<<(m->nSym * 64)/1024 + 1, 1024>>>(m->nSym * 64, m->nBPSCS, demodSigFft, demodSigLlr, pList, demodDemapFftL, demodDemap64QamL);
    }
  }
  else
  {
    if(m->nBPSCS == 1)
    {
      cuDemodQamToLlrNonlegacy<<<(m->nSym * 64)/1024 + 1, 1024>>>(m->nSym * 64, m->nBPSCS, demodSigFft, demodSigLlr, pList, demodDemapFftL, demodDemapBpskNL, m->format);
    }
    else if(m->nBPSCS == 2)
    {
      cuDemodQamToLlrNonlegacy<<<(m->nSym * 64)/1024 + 1, 1024>>>(m->nSym * 64, m->nBPSCS, demodSigFft, demodSigLlr, pList, demodDemapFftL, demodDemapQpskNL, m->format);
    }
    else if(m->nBPSCS == 4)
    {
      cuDemodQamToLlrNonlegacy<<<(m->nSym * 64)/1024 + 1, 1024>>>(m->nSym * 64, m->nBPSCS, demodSigFft, demodSigLlr, pList, demodDemapFftL, demodDemap16QamNL, m->format);
    }
    else if(m->nBPSCS == 6)
    {
      cuDemodQamToLlrNonlegacy<<<(m->nSym * 64)/1024 + 1, 1024>>>(m->nSym * 64, m->nBPSCS, demodSigFft, demodSigLlr, pList, demodDemapFftL, demodDemap64QamNL, m->format);
    }
    else
    {
      cuDemodQamToLlrNonlegacy<<<(m->nSym * 64)/1024 + 1, 1024>>>(m->nSym * 64, m->nBPSCS, demodSigFft, demodSigLlr, pList, demodDemapFftL, demodDemap64QamNL, m->format);
    }
  }
}

void cuDemodDebug(int n, cuFloatComplex* outcomp, int m, float* outfloat)
{
  cudaMemcpy(outcomp, demodSigFft, n*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
  // cudaMemcpy(outfloat, demodSigFft, m*sizeof(float), cudaMemcpyDeviceToHost);
}