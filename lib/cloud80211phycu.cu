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
  -1, 24, 25, 26, 27, 28, 29, -1, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, -1, 43, 44, 45, 46, 47, -1, -1, -1, -1, -1, 
  -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, -1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, -1, 18, 19, 20, 21, 22, 23};
int mapDeshiftFftNonlegacy[64] = {
  -1, 26, 27, 28, 29, 30, 31, -1, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, -1, 45, 46, 47, 48, 49, 50, 51, -1, -1, -1, 
  -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, -1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, -1, 20, 21, 22, 23, 24, 25};
cuFloatComplex* demodChanSiso;
cuFloatComplex* demodSig;
cuFloatComplex* demodSigFft;
cufftHandle demodPlan;
float* demodSigLlr;
cuFloatComplex* pilotsLegacy;
cuFloatComplex* pilotsHt;
cuFloatComplex* pilotsVht;

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

int* cuv_seq;
int* cuv_bits;
int* cuv_state_his;
int* cuv_state_bit;
int* cuv_state_next;
int* cuv_state_output;
int* cuv_cr_punc;

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
void cuDemodQamToLlr(int n, int nCBPSS, cuFloatComplex* sigfft, float* llr, cuFloatComplex* p, int* deshift, int* deint)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = i / 64;  // sym index
  int k = i % 64;  // sample index
  int llrOffset = j * nCBPSS;
  cuFloatComplex pilotConj = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex pilotAbs;
  cuFloatComplex qam;
  float qamReal, qamImag;
  int scIndex = deshift[k];      // sc after fft to data sc index

  if(i >= n || scIndex < 0)
  {
    return;
  }

  pilotConj = cuCaddf(pilotConj, cuCmulf(sigfft[j*64 + 43], p[j*4]));
  pilotConj = cuCaddf(pilotConj, cuCmulf(sigfft[j*64 + 57], p[j*4+1]));
  pilotConj = cuCaddf(pilotConj, cuCmulf(sigfft[j*64 +  7], p[j*4+2]));
  pilotConj = cuCaddf(pilotConj, cuCmulf(sigfft[j*64 + 21], p[j*4+3]));
  pilotAbs = make_cuFloatComplex(cuCabsf(pilotConj), 0.0f);
  pilotConj = cuConjf(pilotConj);

  qam = cuCdivf(cuCmulf(sigfft[i], pilotConj), pilotAbs);

  if(nCBPSS == 48 || nCBPSS == 52)
  {
    qamReal = cuCrealf(qam);
    qamImag = cuCimagf(qam);
    llr[llrOffset + deint[scIndex]] = qamReal;
  }
  else if(nCBPSS == 96 || nCBPSS == 104)
  {
    qam = cuCmulf(qam, make_cuFloatComplex(1.4142135623730951f, 0.0f));
    qamReal = cuCrealf(qam);
    qamImag = cuCimagf(qam);
    llr[llrOffset + deint[scIndex*2]] = qamReal;
    llr[llrOffset + deint[scIndex*2+1]] = qamImag;
  }
  else if(nCBPSS == 192 || nCBPSS == 208)
  {
    qam = cuCmulf(qam, make_cuFloatComplex(3.1622776601683795f, 0.0f));
    qamReal = cuCrealf(qam);
    qamImag = cuCimagf(qam);
    llr[llrOffset + deint[scIndex*4]] = qamReal;
    llr[llrOffset + deint[scIndex*4+1]] = 2.0f - fabsf(qamReal);
    llr[llrOffset + deint[scIndex*4+2]] = qamImag;
    llr[llrOffset + deint[scIndex*4+3]] = 2.0f - fabsf(qamImag);
  }
  else if(nCBPSS == 288 || nCBPSS == 312)
  {
    qam = cuCmulf(qam, make_cuFloatComplex(6.48074069840786f, 0.0f));
    qamReal = cuCrealf(qam);
    qamImag = cuCimagf(qam);
    llr[llrOffset + deint[scIndex*6]] = qamReal;
    llr[llrOffset + deint[scIndex*6+1]] = 4.0f - fabsf(qamReal);
    llr[llrOffset + deint[scIndex*6+2]] = 2.0f - fabsf(4.0f - fabsf(qamReal));
    llr[llrOffset + deint[scIndex*6+3]] = qamImag;
    llr[llrOffset + deint[scIndex*6+4]] = 4.0f - fabsf(qamImag);
    llr[llrOffset + deint[scIndex*6+5]] = 2.0f - fabsf(4.0f - fabsf(qamImag));
  }
  else
  {
    qam = cuCmulf(qam, make_cuFloatComplex(13.038404810405298f, 0.0f));
    qamReal = cuCrealf(qam);
    qamImag = cuCimagf(qam);
    llr[llrOffset + deint[scIndex*8]] = qamReal;
    llr[llrOffset + deint[scIndex*8+1]] = 8.0f - fabsf(qamReal);
    llr[llrOffset + deint[scIndex*8+2]] = 4.0f - fabsf(8.0f - fabsf(qamReal));
    llr[llrOffset + deint[scIndex*8+3]] = 2.0f - fabsf(4.0f - fabsf(8.0f - fabsf(qamReal)));
    llr[llrOffset + deint[scIndex*8+4]] = qamImag;
    llr[llrOffset + deint[scIndex*8+5]] = 8.0f - fabsf(qamImag);
    llr[llrOffset + deint[scIndex*8+6]] = 4.0f - fabsf(8.0f - fabsf(qamImag));
    llr[llrOffset + deint[scIndex*8+7]] = 2.0f - fabsf(4.0f - fabsf(8.0f - fabsf(qamImag)));
  }
  // sigfft[i] = make_cuFloatComplex(llrOffset + scIndex*2, llrOffset + scIndex*2 + 1);
}

__global__ void cuDecodeViterbi(float* llr,
                                int len,
                                int trellis,
                                int crlen,
                                int* punc,
                                int* s_his,
                                int* s_output,
                                int* s_next)
{
    int i = threadIdx.x;
    int v_cr_p = 0;
    int tmpUsed = 0;
    float *v_ae_pPre, *v_ae_pCur, *v_ae_pTmp;
    float v_acc_tmp0, v_acc_tmp1;
    int v_next0, v_next1;
    int v_t = 0;
    __shared__ float v_accum_err0[64];
    __shared__ float v_accum_err1[64];
    __shared__ float v_tab_t[4];
    __shared__ int v_punc[10];
    __shared__ int v_output[128];
    __shared__ int v_next[128];

    if (i < crlen) {
        v_punc[i] = punc[i];
    }

    v_output[i * 2] = s_output[i * 2];
    v_output[i * 2 + 1] = s_output[i * 2 + 1];
    v_next[i * 2] = s_next[i * 2];
    v_next[i * 2 + 1] = s_next[i * 2 + 1];

    if (i == 0) {
        v_accum_err0[i] = 0.0f;
    } else {
        v_accum_err0[i] = -1000000000000000.0f;
    }
    v_accum_err1[i] = -1000000000000000.0f;
    v_ae_pCur = v_accum_err1;
    v_ae_pPre = v_accum_err0;

    while ((tmpUsed + v_punc[v_cr_p] + v_punc[v_cr_p + 1]) <= len && v_t < trellis) {
        if (i == 0) {
            v_tab_t[0] = 0.0f;
            if (v_punc[v_cr_p]) {
                v_tab_t[2] = llr[tmpUsed];
                v_tab_t[3] = llr[tmpUsed];
                tmpUsed++;
            } else {
                v_tab_t[2] = 0.0f;
                v_tab_t[3] = 0.0f;
            }
            if (v_punc[v_cr_p + 1]) {
                v_tab_t[1] = llr[tmpUsed];
                v_tab_t[3] += llr[tmpUsed];
                tmpUsed++;
            } else {
                v_tab_t[1] = 0.0f;
            }
        }
        __syncthreads();

        v_acc_tmp0 = v_ae_pPre[i] + v_tab_t[v_output[i * 2]];
        v_acc_tmp1 = v_ae_pPre[i] + v_tab_t[v_output[i * 2 + 1]];

        if ((i % 2) == 0) {
            v_next0 = v_next[i * 2];
            v_next1 = v_next[i * 2 + 1];
            if (v_acc_tmp0 > v_ae_pCur[v_next0]) {
                v_ae_pCur[v_next0] = v_acc_tmp0;
                s_his[(v_t + 1) * 64 + v_next0] = i;
            }
            if (v_acc_tmp1 > v_ae_pCur[v_next1]) {
                v_ae_pCur[v_next1] = v_acc_tmp1;
                s_his[(v_t + 1) * 64 + v_next1] = i;
            }
        }
        __syncthreads();

        if ((i % 2) == 1) {
            v_next0 = v_next[i * 2];
            v_next1 = v_next[i * 2 + 1];
            if (v_acc_tmp0 > v_ae_pCur[v_next0]) {
                v_ae_pCur[v_next0] = v_acc_tmp0;
                s_his[(v_t + 1) * 64 + v_next0] = i;
            }
            if (v_acc_tmp1 > v_ae_pCur[v_next1]) {
                v_ae_pCur[v_next1] = v_acc_tmp1;
                s_his[(v_t + 1) * 64 + v_next1] = i;
            }
        }

        v_ae_pTmp = v_ae_pPre;
        v_ae_pPre = v_ae_pCur;
        v_ae_pCur = v_ae_pTmp;

        v_ae_pCur[i] = -1000000000000000.0f;

        v_cr_p += 2;
        if (v_cr_p >= crlen) {
            v_cr_p = 0;
        }

        v_t++;
    }
}

__global__ void cuDecodeTb1(int trellis, int* s_his, int* s_seq)
{
    s_seq[trellis] = 0;
    for (int j = trellis; j > 0; j--) {
        s_seq[j - 1] = s_his[j * 64 + s_seq[j]];
    }
}

__global__ void cuDecodeTb2(int trellis, int* s_seq, int* s_next, int* bits)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= trellis) {
        return;
    }
    if (s_seq[i + 1] == s_next[s_seq[i] * 2 + 1]) {
        bits[i] = 1;
    } else {
        bits[i] = 0;
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
  cudaMemset(demodSigLlr, 0, sizeof(float) * CUDEMOD_S_MAX * 52 * 8);

  cuFloatComplex pListTmp[CUDEMOD_S_MAX * 4];
  cudaMalloc(&pilotsLegacy, sizeof(cuFloatComplex) * CUDEMOD_S_MAX * 4);
  cudaMalloc(&pilotsHt, sizeof(cuFloatComplex) * CUDEMOD_S_MAX * 4);
  cudaMalloc(&pilotsVht, sizeof(cuFloatComplex) * CUDEMOD_S_MAX * 4);
  for(int i=0;i<CUDEMOD_S_MAX;i++)
  {
    pListTmp[i*4] = make_cuFloatComplex(1.0f * PILOT_P[(i+1)%127], 0.0f);
    pListTmp[i*4+1] = make_cuFloatComplex(1.0f * PILOT_P[(i+1)%127], 0.0f);
    pListTmp[i*4+2] = make_cuFloatComplex(1.0f * PILOT_P[(i+1)%127], 0.0f);
    pListTmp[i*4+3] = make_cuFloatComplex(-1.0f * PILOT_P[(i+1)%127], 0.0f);
  }
  cudaMemcpy(pilotsLegacy, pListTmp, sizeof(cuFloatComplex) * CUDEMOD_S_MAX * 4, cudaMemcpyHostToDevice);
  float pTmp[4] = {1.0f, 1.0f, 1.0f, -1.0f};
  for(int i=0;i<CUDEMOD_S_MAX;i++)
  {
    pListTmp[i*4] = make_cuFloatComplex(pTmp[0] * PILOT_P[(i+3)%127], 0.0f);
    pListTmp[i*4+1] = make_cuFloatComplex(pTmp[1] * PILOT_P[(i+3)%127], 0.0f);
    pListTmp[i*4+2] = make_cuFloatComplex(pTmp[2] * PILOT_P[(i+3)%127], 0.0f);
    pListTmp[i*4+3] = make_cuFloatComplex(pTmp[3] * PILOT_P[(i+3)%127], 0.0f);

    float tmpPilot = pTmp[0];
    pTmp[0] = pTmp[1];
    pTmp[1] = pTmp[2];
    pTmp[2] = pTmp[3];
    pTmp[3] = tmpPilot;
  }
  cudaMemcpy(pilotsHt, pListTmp, sizeof(cuFloatComplex) * CUDEMOD_S_MAX * 4, cudaMemcpyHostToDevice);

  float pTmp2[4] = {1.0f, 1.0f, 1.0f, -1.0f};
  for(int i=0;i<CUDEMOD_S_MAX;i++)
  {
    pListTmp[i*4] = make_cuFloatComplex(pTmp2[0] * PILOT_P[(i+4)%127], 0.0f);
    pListTmp[i*4+1] = make_cuFloatComplex(pTmp2[1] * PILOT_P[(i+4)%127], 0.0f);
    pListTmp[i*4+2] = make_cuFloatComplex(pTmp2[2] * PILOT_P[(i+4)%127], 0.0f);
    pListTmp[i*4+3] = make_cuFloatComplex(pTmp2[3] * PILOT_P[(i+4)%127], 0.0f);

    float tmpPilot = pTmp2[0];
    pTmp2[0] = pTmp2[1];
    pTmp2[1] = pTmp2[2];
    pTmp2[2] = pTmp2[3];
    pTmp2[3] = tmpPilot;
  }
  cudaMemcpy(pilotsVht, pListTmp, sizeof(cuFloatComplex) * CUDEMOD_S_MAX * 4, cudaMemcpyHostToDevice);
  
  
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

  cudaMalloc(&cuv_seq, sizeof(int) * CUDEMOD_V_MAX);
  cudaMalloc(&cuv_bits, sizeof(int) * CUDEMOD_V_MAX);
  cudaMalloc(&cuv_state_his, sizeof(int) * 64 * (CUDEMOD_V_MAX + 1));
  cudaMalloc(&cuv_state_bit, sizeof(int) * (CUDEMOD_V_MAX + 1));
  cudaMalloc(&cuv_state_next, sizeof(int) * 128);
  cudaMemcpy(cuv_state_next, SV_STATE_NEXT, 128 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&cuv_state_output, sizeof(int) * 128);
  cudaMemcpy(
      cuv_state_output, SV_STATE_OUTPUT, 128 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&cuv_cr_punc, sizeof(int) * 22);
  int tmpPunc[22] = {
      1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1
  };
  cudaMemcpy(cuv_cr_punc, tmpPunc, 22 * sizeof(int), cudaMemcpyHostToDevice);
}

void cuDemodFree()
{
  cudaFree(demodChanSiso);
  cudaFree(demodSig);
  cudaFree(demodSigFft);
  cufftDestroy(demodPlan);
  cudaFree(demodSigLlr);
  cudaFree(pilotsLegacy);
  cudaFree(pilotsHt);
  cudaFree(pilotsVht);

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

  cudaFree(cuv_seq);
  cudaFree(cuv_bits);
  cudaFree(cuv_state_his);
  cudaFree(cuv_state_bit);
  cudaFree(cuv_state_next);
  cudaFree(cuv_state_output);
  cudaFree(cuv_cr_punc);
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
  int cuv_llr_len = m->nSym * m->nCBPS;
  int* cuv_cr_punc_p;
  int cuv_cr_len;
  int cuv_trellis;

  cuDemodChopSamp<<<(m->nSym * m->nSymSamp)/1024 + 1, 1024>>>(m->nSym * m->nSymSamp, demodSig, demodSigFft);
  for(int symIter=0; symIter < ((m->nSym + CUDEMOD_FFT_BATCH - 1) / CUDEMOD_FFT_BATCH); symIter++ )   // each round inlcudes 256 batches
  {
    cufftExecC2C(demodPlan, &demodSigFft[symIter*CUDEMOD_FFT_BATCH*64], &demodSigFft[symIter*CUDEMOD_FFT_BATCH*64], CUFFT_FORWARD);
  }
  cuDemodChanComp<<<(m->nSym * 64)/1024 + 1, 1024>>>(m->nSym * 64, demodSigFft, demodChanSiso);
  if(m->format == C8P_F_L)
  {
    if(m->mod == C8P_QAM_BPSK)
    {
      cuDemodQamToLlr<<<(m->nSym * 64)/256 + 1, 256>>>(m->nSym * 64, m->nCBPSS, demodSigFft, demodSigLlr, pilotsLegacy, demodDemapFftL, demodDemapBpskL);
    }
    else if(m->mod == C8P_QAM_QPSK)
    {
      cuDemodQamToLlr<<<(m->nSym * 64)/256 + 1, 256>>>(m->nSym * 64, m->nCBPSS, demodSigFft, demodSigLlr, pilotsLegacy, demodDemapFftL, demodDemapQpskL);
    }
    else if(m->mod == C8P_QAM_16QAM)
    {
      cuDemodQamToLlr<<<(m->nSym * 64)/256 + 1, 256>>>(m->nSym * 64, m->nCBPSS, demodSigFft, demodSigLlr, pilotsLegacy, demodDemapFftL, demodDemap16QamL);
    }
    else
    {
      cuDemodQamToLlr<<<(m->nSym * 64)/256 + 1, 256>>>(m->nSym * 64, m->nCBPSS, demodSigFft, demodSigLlr, pilotsLegacy, demodDemapFftL, demodDemap64QamL);
    }
  }
  else
  {
    if(m->mod == C8P_QAM_BPSK)
    {
      if(m->format == C8P_F_HT)
      {cuDemodQamToLlr<<<(m->nSym * 64)/1024 + 1, 1024>>>(m->nSym * 64, m->nCBPSS, demodSigFft, demodSigLlr, pilotsHt, demodDemapFftNL, demodDemapBpskNL);}
      else
      {cuDemodQamToLlr<<<(m->nSym * 64)/1024 + 1, 1024>>>(m->nSym * 64, m->nCBPSS, demodSigFft, demodSigLlr, pilotsVht, demodDemapFftNL, demodDemapBpskNL);}
    }
    else if(m->mod == C8P_QAM_QPSK)
    {
      if(m->format == C8P_F_HT)
      {cuDemodQamToLlr<<<(m->nSym * 64)/1024 + 1, 1024>>>(m->nSym * 64, m->nCBPSS, demodSigFft, demodSigLlr, pilotsHt, demodDemapFftNL, demodDemapQpskNL);}
      else
      {cuDemodQamToLlr<<<(m->nSym * 64)/1024 + 1, 1024>>>(m->nSym * 64, m->nCBPSS, demodSigFft, demodSigLlr, pilotsVht, demodDemapFftNL, demodDemapQpskNL);}
    }
    else if(m->mod == C8P_QAM_16QAM)
    {
      if(m->format == C8P_F_HT)
      {cuDemodQamToLlr<<<(m->nSym * 64)/1024 + 1, 1024>>>(m->nSym * 64, m->nCBPSS, demodSigFft, demodSigLlr, pilotsHt, demodDemapFftNL, demodDemap16QamNL);}
      else
      {cuDemodQamToLlr<<<(m->nSym * 64)/1024 + 1, 1024>>>(m->nSym * 64, m->nCBPSS, demodSigFft, demodSigLlr, pilotsVht, demodDemapFftNL, demodDemap16QamNL);}
    }
    else if(m->mod == C8P_QAM_64QAM)
    {
      if(m->format == C8P_F_HT)
      {cuDemodQamToLlr<<<(m->nSym * 64)/1024 + 1, 1024>>>(m->nSym * 64, m->nCBPSS, demodSigFft, demodSigLlr, pilotsHt, demodDemapFftNL, demodDemap64QamNL);}
      else
      {cuDemodQamToLlr<<<(m->nSym * 64)/1024 + 1, 1024>>>(m->nSym * 64, m->nCBPSS, demodSigFft, demodSigLlr, pilotsVht, demodDemapFftNL, demodDemap64QamNL);}
    }
    else
    {
      cuDemodQamToLlr<<<(m->nSym * 64)/1024 + 1, 1024>>>(m->nSym * 64, m->nCBPSS, demodSigFft, demodSigLlr, pilotsVht, demodDemapFftNL, demodDemap256QamNL);
    }
  }

  if(m->format == C8P_F_L || m->format == C8P_F_HT)
  {
    cuv_trellis = 22 + m->len*8;
  }
  else
  {
    cuv_trellis = m->nSym * m->nDBPS;
  }
  
  if (m->cr == C8P_CR_12) {
      cuv_cr_len = 2;
      cuv_cr_punc_p = &cuv_cr_punc[0];
  } else if (m->cr == C8P_CR_23) {
      cuv_cr_len = 4;
      cuv_cr_punc_p = &cuv_cr_punc[2];
  } else if (m->cr == C8P_CR_34) {
      cuv_cr_len = 6;
      cuv_cr_punc_p = &cuv_cr_punc[6];
  } else {
      cuv_cr_len = 10;
      cuv_cr_punc_p = &cuv_cr_punc[12];
  }
  cudaMemset(cuv_state_his, 0, sizeof(int) * 64 * (cuv_trellis + 1));
  cuDecodeViterbi<<<1, 64>>>(demodSigLlr,
                              cuv_llr_len,
                              cuv_trellis,
                              cuv_cr_len,
                              cuv_cr_punc_p,
                              cuv_state_his,
                              cuv_state_output,
                              cuv_state_next);
  cuDecodeTb1<<<1, 1>>>(cuv_trellis, cuv_state_his, cuv_seq);
  cuDecodeTb2<<<(cuv_trellis + 1023) / 1024, 1024>>>(cuv_trellis, cuv_seq, cuv_state_next, cuv_bits);

}

void cuDemodDebug(int n, cuFloatComplex* outcomp, int m, float* outfloat, int o, int* outint)
{
  cudaMemcpy(outcomp, demodSigFft, n*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
  cudaMemcpy(outfloat, demodSigLlr, m*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(outint, cuv_bits, o*sizeof(int), cudaMemcpyDeviceToHost);
}