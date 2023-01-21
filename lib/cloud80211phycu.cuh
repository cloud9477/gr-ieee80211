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

#ifndef INCLUDED_CLOUD80211PHYCU_H
#define INCLUDED_CLOUD80211PHYCU_H

#include <iostream>
#include <math.h>
#include <cuComplex.h>
#include <cufft.h>
#include "cloud80211phy.h"

#define PREPROC_MIN 1024
#define PREPROC_MAX 8192

#define CUDEMOD_B_MAX 4095     // max PSDU byte len
#define CUDEMOD_T_MAX 32782    // max trellis len, psdu * 8 + 22
#define CUDEMOD_L_MAX 65728    // max llr len, VHT CR1/2
#define CUDEMOD_S_MAX 1408     // max symbol number, legacy nDBPS 24 is 1366, round to multiple of fft batch 1408
#define CUDEMOD_FFT_BATCH 64    // each execution 64 symbols

#define CUDEMOD_VTB_LEN 80

void preprocMall();
void preprocFree();
void cuPreProc(int n, const cuFloatComplex *sig, float* ac, cuFloatComplex* conj);

void cuDemodMall();
void cuDemodMall2();
void cuDemodFree();
void cuDemodFree2();
void cuDemodChanSiso(cuFloatComplex *chan);
void cuDemodChanMimo(cuFloatComplex *chan, cuFloatComplex *chaninv, cuFloatComplex *pilotsltf);
void cuDemodSigCopy(int i, int n, const cuFloatComplex *sig);
void cuDemodSigCopy2(int i, int j, int n, const cuFloatComplex *sig, const cuFloatComplex *sig2);
void cuDemodSiso(c8p_mod* m, unsigned char* psduBytes);
void cuDemodMimo(c8p_mod* m, unsigned char* psduBytes);

#endif /* INCLUDED_IEEE80211_SIGNAL_IMPL_H */