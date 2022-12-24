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

/*
Legacy  max MPDU 4095
HT      max MPDU 7935     AMPDU 65535
VHT     max MPDU 11454    AMPDU 1048575
The max ampdu len can receive is indicated by different devices in HT Cap or VHT Cap
For VHT, max is 2^20-1=1048575, the minimal su packet apeplen/4 is 2^17=131072, 131072*4=524288
This phy only supports 20MHz, so we use 524288 here
However, for real usage, this may not necessary

#define DECODE_B_MAX 524288
#define DECODE_V_MAX 4196000    // max llr len
#define DECODE_D_MAX 11454      // max mpdu len
*/
#define CUDEMOD_S_MAX 512      // max symbol number, each fft batch is 256, this is 256 * 6
#define CUDEMOD_B_MAX 5000     // max apmdu len
#define CUDEMOD_V_MAX 82000    // max llr len
#define CUDEMOD_D_MAX 1600      // max mpdu len
#define CUDEMOD_FFT_BATCH 64

void preprocMall();
void preprocFree();
void cuPreProc(int n, const cuFloatComplex *sig, float* ac, cuFloatComplex* conj);

void cuDemodMall();
void cuDemodFree();
void cuDemodChanSiso(cuFloatComplex *chan);
void cuDemodSigCopy(int i, int n, const cuFloatComplex *sig);
void cuDemodSiso(c8p_mod* m);
void cuDemodDebug(int n, cuFloatComplex* outcomp, int m, float* outfloat, int o, int* outint);

#endif /* INCLUDED_IEEE80211_SIGNAL_IMPL_H */