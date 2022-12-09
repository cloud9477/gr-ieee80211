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

void signalMall();
void signalFree();
void cuSignalCfoCompen(int n, int s, float radStep, const cuFloatComplex *x, cuFloatComplex *y);
void cuSignalChannel(int s, float radStep, const cuFloatComplex *sig, cuFloatComplex *h, float* llr);

#endif /* INCLUDED_IEEE80211_SIGNAL_IMPL_H */