/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 20M bw and upto 2x2
 *     PHY utilization functions and parameters
 *     Copyright (C) June 1, 2022  Zelin Yun
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

#include "cloud80211phy.h"

/***************************************************/
/* training field */
/***************************************************/

const int C8P_LEGACY_DP_SC[64] = {
	0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

const int C8P_LEGACY_D_SC[64] = {
	0, 24, 25, 26, 27, 28, 29, 0, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 0, 43, 44, 45, 46, 47, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 18, 19, 20, 21, 22, 23
};

const int FFT_26_DEMAP[64] = {
	48, 24, 25, 26, 27, 28, 29, 49, 30, 31, 32, 33, 34, 35, 36, 37, 
	38, 39, 40, 41, 42, 50, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 
	56, 57, 58, 59, 60, 61, 0, 1, 2, 3, 4, 62, 5, 6, 7, 8, 
	9, 10, 11, 12, 13, 14, 15, 16, 17, 63, 18, 19, 20, 21, 22, 23
};

const int FFT_26_SHIFT_DEMAP[128] = {
	-1, 24, 25, 26, 27, 28, 29, -1, 30, 31, 32, 33, 34, 35, 36, 37, 
	38, 39, 40, 41, 42, -1, 43, 44, 45, 46, 47, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1,  0,  1,  2,  3,  4, -1,  5,  6,  7,  8,  
	 9, 10, 11, 12, 13, 14, 15, 16, 17, -1, 18, 19, 20, 21, 22, 23, 
	-1, 72, 73, 74, 75, 76, 77, -1, 78, 79, 80, 81, 82, 83, 84, 85, 
	86, 87, 88, 89, 90, -1, 91, 92, 93, 94, 95, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, 48, 49, 50, 51, 52, -1, 53, 54, 55, 56, 
	57, 58, 59, 60, 61, 62, 63, 64, 65, -1, 66, 67, 68, 69, 70, 71
};

const gr_complex LTF_L_26_F_COMP[64] = {
    gr_complex(0.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
    gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
    gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
    gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
    gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
    gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
    gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(0.0f, 0.0f), 
    gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), 
    gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), 
    gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
    gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
    gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
    gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
    gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
    gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
    gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f)};

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

const float LTF_NL_28_F_FLOAT[64] = {
    0.0f, 1.0f, -1.0f, -1.0f, 
    1.0f, 1.0f, -1.0f, 1.0f, 
    -1.0f, 1.0f, -1.0f, -1.0f, 
    -1.0f, -1.0f, -1.0f, 1.0f, 
    1.0f, -1.0f, -1.0f, 1.0f, 
    -1.0f, 1.0f, -1.0f, 1.0f, 
    1.0f, 1.0f, 1.0f, -1.0f, 
    -1.0f, 0.0f, 0.0f, 0.0f, 
    0.0f, 0.0f, 0.0f, 0.0f, 
    1.0f, 1.0f, 1.0f, 1.0f, 
    -1.0f, -1.0f, 1.0f, 1.0f, 
    -1.0f, 1.0f, -1.0f, 1.0f, 
    1.0f, 1.0f, 1.0f, 1.0f, 
    1.0f, -1.0f, -1.0f, 1.0f, 
    1.0f, -1.0f, 1.0f, -1.0f, 
    1.0f, 1.0f, 1.0f, 1.0f};

const float LTF_NL_28_F_FLOAT2[64] = {
    0.0f, 0.5f, -0.5f, -0.5f, 
    0.5f, 0.5f, -0.5f, 0.5f, 
    -0.5f, 0.5f, -0.5f, -0.5f, 
    -0.5f, -0.5f, -0.5f, 0.5f, 
    0.5f, -0.5f, -0.5f, 0.5f, 
    -0.5f, 0.5f, -0.5f, 0.5f, 
    0.5f, 0.5f, 0.5f, -0.5f, 
    -0.5f, 0.0f, 0.0f, 0.0f, 
    0.0f, 0.0f, 0.0f, 0.0f, 
    0.5f, 0.5f, 0.5f, 0.5f, 
    -0.5f, -0.5f, 0.5f, 0.5f, 
    -0.5f, 0.5f, -0.5f, 0.5f, 
    0.5f, 0.5f, 0.5f, 0.5f, 
    0.5f, -0.5f, -0.5f, 0.5f, 
    0.5f, -0.5f, 0.5f, -0.5f, 
    0.5f, 0.5f, 0.5f, 0.5f};

const float PILOT_P[127] = {
	 1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
	-1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
	 1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f,
	-1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
	-1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f,
	-1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
	-1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f,
	-1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};

const float PILOT_L[4] = {1.0f, 1.0f, 1.0f, -1.0f};
const float PILOT_HT_2_1[4] = {1.0f, 1.0f, -1.0f, -1.0f};
const float PILOT_HT_2_2[4] = {1.0f, -1.0f, -1.0f, 1.0f};
const float PILOT_VHT[4] = {1.0f, 1.0f, 1.0f, -1.0f};
const uint8_t EOF_PAD_SUBFRAME[32] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0};

const uint8_t LEGACY_RATE_BITS[8][4] = {
	{1, 1, 0, 1},
	{1, 1, 1, 1},
	{0, 1, 0, 1},
	{0, 1, 1, 1},
	{1, 0, 0, 1},
	{1, 0, 1, 1},
	{0, 0, 0, 1},
	{0, 0, 1, 1}
};

const uint8_t VHT_NDP_SIGB_20_BITS[26] = {
	0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
};

const gr_complex C8P_STF_F[64] = {
	gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(1.0f, 1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(-1.0f, -1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(1.0f, 1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(-1.0f, -1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(-1.0f, -1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(1.0f, 1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f),

	gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(-1.0f, -1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(-1.0f, -1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(1.0f, 1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(1.0f, 1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(1.0f, 1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(1.0f, 1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f)
};

const gr_complex C8P_LTF_L_F[64] = {
	gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), 
	gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f),

	gr_complex(0.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(0.0f, 0.0f), 
	gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f)
};

const gr_complex C8P_LTF_NL_F[64] = {
	gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f),

	gr_complex(0.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f),
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f),
	gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f),
	gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f)
};

const gr_complex C8P_LTF_NL_F_N[64] = {
	gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 

	gr_complex(0.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f)
};

const gr_complex C8P_LTF_NL_F_VHT22[64] = {
	gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f),

	gr_complex(0.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f)
};

const gr_complex C8P_QAM_TAB_BPSK[2] = {gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f)};
const gr_complex C8P_QAM_TAB_QBPSK[2] = {gr_complex(0.0f, -1.0f), gr_complex(0.0f, 1.0f)};
const gr_complex C8P_QAM_TAB_QPSK[4] = {
	gr_complex(-1.0f, -1.0f)/sqrtf(2.0f), gr_complex(1.0f, -1.0f)/sqrtf(2.0f), 
	gr_complex(-1.0f, 1.0f)/sqrtf(2.0f), gr_complex(1.0f, 1.0f)/sqrtf(2.0f)
};
const gr_complex C8P_QAM_TAB_16QAM[16] = {
	gr_complex(-3.0f, -3.0f)/sqrtf(10.0f), gr_complex(3.0f, -3.0f)/sqrtf(10.0f), 
	gr_complex(-1.0f, -3.0f)/sqrtf(10.0f), gr_complex(1.0f, -3.0f)/sqrtf(10.0f), 
	gr_complex(-3.0f, 3.0f)/sqrtf(10.0f), gr_complex(3.0f, 3.0f)/sqrtf(10.0f), 
	gr_complex(-1.0f, 3.0f)/sqrtf(10.0f), gr_complex(1.0f, 3.0f)/sqrtf(10.0f), 
	gr_complex(-3.0f, -1.0f)/sqrtf(10.0f), gr_complex(3.0f, -1.0f)/sqrtf(10.0f), 
	gr_complex(-1.0f, -1.0f)/sqrtf(10.0f), gr_complex(1.0f, -1.0f)/sqrtf(10.0f), 
	gr_complex(-3.0f, 1.0f)/sqrtf(10.0f), gr_complex(3.0f, 1.0f)/sqrtf(10.0f), 
	gr_complex(-1.0f, 1.0f)/sqrtf(10.0f), gr_complex(1.0f, 1.0f)/sqrtf(10.0f)
};
const gr_complex C8P_QAM_TAB_64QAM[64] = {
	gr_complex(-7.0f, -7.0f)/sqrtf(42.0f), gr_complex(7.0f, -7.0f)/sqrtf(42.0f), 
	gr_complex(-1.0f, -7.0f)/sqrtf(42.0f), gr_complex(1.0f, -7.0f)/sqrtf(42.0f), 
	gr_complex(-5.0f, -7.0f)/sqrtf(42.0f), gr_complex(5.0f, -7.0f)/sqrtf(42.0f), 
	gr_complex(-3.0f, -7.0f)/sqrtf(42.0f), gr_complex(3.0f, -7.0f)/sqrtf(42.0f), 
	gr_complex(-7.0f, 7.0f)/sqrtf(42.0f), gr_complex(7.0f, 7.0f)/sqrtf(42.0f), 
	gr_complex(-1.0f, 7.0f)/sqrtf(42.0f), gr_complex(1.0f, 7.0f)/sqrtf(42.0f), 
	gr_complex(-5.0f, 7.0f)/sqrtf(42.0f), gr_complex(5.0f, 7.0f)/sqrtf(42.0f), 
	gr_complex(-3.0f, 7.0f)/sqrtf(42.0f), gr_complex(3.0f, 7.0f)/sqrtf(42.0f), 
	gr_complex(-7.0f, -1.0f)/sqrtf(42.0f), gr_complex(7.0f, -1.0f)/sqrtf(42.0f), 
	gr_complex(-1.0f, -1.0f)/sqrtf(42.0f), gr_complex(1.0f, -1.0f)/sqrtf(42.0f), 
	gr_complex(-5.0f, -1.0f)/sqrtf(42.0f), gr_complex(5.0f, -1.0f)/sqrtf(42.0f), 
	gr_complex(-3.0f, -1.0f)/sqrtf(42.0f), gr_complex(3.0f, -1.0f)/sqrtf(42.0f), 
	gr_complex(-7.0f, 1.0f)/sqrtf(42.0f), gr_complex(7.0f, 1.0f)/sqrtf(42.0f), 
	gr_complex(-1.0f, 1.0f)/sqrtf(42.0f), gr_complex(1.0f, 1.0f)/sqrtf(42.0f), 
	gr_complex(-5.0f, 1.0f)/sqrtf(42.0f), gr_complex(5.0f, 1.0f)/sqrtf(42.0f), 
	gr_complex(-3.0f, 1.0f)/sqrtf(42.0f), gr_complex(3.0f, 1.0f)/sqrtf(42.0f), 
	gr_complex(-7.0f, -5.0f)/sqrtf(42.0f), gr_complex(7.0f, -5.0f)/sqrtf(42.0f), 
	gr_complex(-1.0f, -5.0f)/sqrtf(42.0f), gr_complex(1.0f, -5.0f)/sqrtf(42.0f), 
	gr_complex(-5.0f, -5.0f)/sqrtf(42.0f), gr_complex(5.0f, -5.0f)/sqrtf(42.0f), 
	gr_complex(-3.0f, -5.0f)/sqrtf(42.0f), gr_complex(3.0f, -5.0f)/sqrtf(42.0f), 
	gr_complex(-7.0f, 5.0f)/sqrtf(42.0f), gr_complex(7.0f, 5.0f)/sqrtf(42.0f), 
	gr_complex(-1.0f, 5.0f)/sqrtf(42.0f), gr_complex(1.0f, 5.0f)/sqrtf(42.0f), 
	gr_complex(-5.0f, 5.0f)/sqrtf(42.0f), gr_complex(5.0f, 5.0f)/sqrtf(42.0f), 
	gr_complex(-3.0f, 5.0f)/sqrtf(42.0f), gr_complex(3.0f, 5.0f)/sqrtf(42.0f), 
	gr_complex(-7.0f, -3.0f)/sqrtf(42.0f), gr_complex(7.0f, -3.0f)/sqrtf(42.0f), 
	gr_complex(-1.0f, -3.0f)/sqrtf(42.0f), gr_complex(1.0f, -3.0f)/sqrtf(42.0f), 
	gr_complex(-5.0f, -3.0f)/sqrtf(42.0f), gr_complex(5.0f, -3.0f)/sqrtf(42.0f), 
	gr_complex(-3.0f, -3.0f)/sqrtf(42.0f), gr_complex(3.0f, -3.0f)/sqrtf(42.0f), 
	gr_complex(-7.0f, 3.0f)/sqrtf(42.0f), gr_complex(7.0f, 3.0f)/sqrtf(42.0f), 
	gr_complex(-1.0f, 3.0f)/sqrtf(42.0f), gr_complex(1.0f, 3.0f)/sqrtf(42.0f), 
	gr_complex(-5.0f, 3.0f)/sqrtf(42.0f), gr_complex(5.0f, 3.0f)/sqrtf(42.0f), 
	gr_complex(-3.0f, 3.0f)/sqrtf(42.0f), gr_complex(3.0f, 3.0f)/sqrtf(42.0f)
};

const gr_complex C8P_QAM_TAB_256QAM[256] = {
	gr_complex(-15.0f, -15.0f)/sqrtf(170.0f), gr_complex(15.0f, -15.0f)/sqrtf(170.0f), 
	gr_complex(-1.0f, -15.0f)/sqrtf(170.0f), gr_complex(1.0f, -15.0f)/sqrtf(170.0f), 
	gr_complex(-9.0f, -15.0f)/sqrtf(170.0f), gr_complex(9.0f, -15.0f)/sqrtf(170.0f), 
	gr_complex(-7.0f, -15.0f)/sqrtf(170.0f), gr_complex(7.0f, -15.0f)/sqrtf(170.0f), 
	gr_complex(-13.0f, -15.0f)/sqrtf(170.0f), gr_complex(13.0f, -15.0f)/sqrtf(170.0f), 
	gr_complex(-3.0f, -15.0f)/sqrtf(170.0f), gr_complex(3.0f, -15.0f)/sqrtf(170.0f), 
	gr_complex(-11.0f, -15.0f)/sqrtf(170.0f), gr_complex(11.0f, -15.0f)/sqrtf(170.0f), 
	gr_complex(-5.0f, -15.0f)/sqrtf(170.0f), gr_complex(5.0f, -15.0f)/sqrtf(170.0f), 
	gr_complex(-15.0f, 15.0f)/sqrtf(170.0f), gr_complex(15.0f, 15.0f)/sqrtf(170.0f), 
	gr_complex(-1.0f, 15.0f)/sqrtf(170.0f), gr_complex(1.0f, 15.0f)/sqrtf(170.0f), 
	gr_complex(-9.0f, 15.0f)/sqrtf(170.0f), gr_complex(9.0f, 15.0f)/sqrtf(170.0f), 
	gr_complex(-7.0f, 15.0f)/sqrtf(170.0f), gr_complex(7.0f, 15.0f)/sqrtf(170.0f), 
	gr_complex(-13.0f, 15.0f)/sqrtf(170.0f), gr_complex(13.0f, 15.0f)/sqrtf(170.0f), 
	gr_complex(-3.0f, 15.0f)/sqrtf(170.0f), gr_complex(3.0f, 15.0f)/sqrtf(170.0f), 
	gr_complex(-11.0f, 15.0f)/sqrtf(170.0f), gr_complex(11.0f, 15.0f)/sqrtf(170.0f), 
	gr_complex(-5.0f, 15.0f)/sqrtf(170.0f), gr_complex(5.0f, 15.0f)/sqrtf(170.0f), 
	gr_complex(-15.0f, -1.0f)/sqrtf(170.0f), gr_complex(15.0f, -1.0f)/sqrtf(170.0f), 
	gr_complex(-1.0f, -1.0f)/sqrtf(170.0f), gr_complex(1.0f, -1.0f)/sqrtf(170.0f), 
	gr_complex(-9.0f, -1.0f)/sqrtf(170.0f), gr_complex(9.0f, -1.0f)/sqrtf(170.0f), 
	gr_complex(-7.0f, -1.0f)/sqrtf(170.0f), gr_complex(7.0f, -1.0f)/sqrtf(170.0f), 
	gr_complex(-13.0f, -1.0f)/sqrtf(170.0f), gr_complex(13.0f, -1.0f)/sqrtf(170.0f), 
	gr_complex(-3.0f, -1.0f)/sqrtf(170.0f), gr_complex(3.0f, -1.0f)/sqrtf(170.0f), 
	gr_complex(-11.0f, -1.0f)/sqrtf(170.0f), gr_complex(11.0f, -1.0f)/sqrtf(170.0f), 
	gr_complex(-5.0f, -1.0f)/sqrtf(170.0f), gr_complex(5.0f, -1.0f)/sqrtf(170.0f), 
	gr_complex(-15.0f, 1.0f)/sqrtf(170.0f), gr_complex(15.0f, 1.0f)/sqrtf(170.0f), 
	gr_complex(-1.0f, 1.0f)/sqrtf(170.0f), gr_complex(1.0f, 1.0f)/sqrtf(170.0f), 
	gr_complex(-9.0f, 1.0f)/sqrtf(170.0f), gr_complex(9.0f, 1.0f)/sqrtf(170.0f), 
	gr_complex(-7.0f, 1.0f)/sqrtf(170.0f), gr_complex(7.0f, 1.0f)/sqrtf(170.0f), 
	gr_complex(-13.0f, 1.0f)/sqrtf(170.0f), gr_complex(13.0f, 1.0f)/sqrtf(170.0f), 
	gr_complex(-3.0f, 1.0f)/sqrtf(170.0f), gr_complex(3.0f, 1.0f)/sqrtf(170.0f), 
	gr_complex(-11.0f, 1.0f)/sqrtf(170.0f), gr_complex(11.0f, 1.0f)/sqrtf(170.0f), 
	gr_complex(-5.0f, 1.0f)/sqrtf(170.0f), gr_complex(5.0f, 1.0f)/sqrtf(170.0f), 
	gr_complex(-15.0f, -9.0f)/sqrtf(170.0f), gr_complex(15.0f, -9.0f)/sqrtf(170.0f), 
	gr_complex(-1.0f, -9.0f)/sqrtf(170.0f), gr_complex(1.0f, -9.0f)/sqrtf(170.0f), 
	gr_complex(-9.0f, -9.0f)/sqrtf(170.0f), gr_complex(9.0f, -9.0f)/sqrtf(170.0f), 
	gr_complex(-7.0f, -9.0f)/sqrtf(170.0f), gr_complex(7.0f, -9.0f)/sqrtf(170.0f), 
	gr_complex(-13.0f, -9.0f)/sqrtf(170.0f), gr_complex(13.0f, -9.0f)/sqrtf(170.0f), 
	gr_complex(-3.0f, -9.0f)/sqrtf(170.0f), gr_complex(3.0f, -9.0f)/sqrtf(170.0f), 
	gr_complex(-11.0f, -9.0f)/sqrtf(170.0f), gr_complex(11.0f, -9.0f)/sqrtf(170.0f), 
	gr_complex(-5.0f, -9.0f)/sqrtf(170.0f), gr_complex(5.0f, -9.0f)/sqrtf(170.0f), 
	gr_complex(-15.0f, 9.0f)/sqrtf(170.0f), gr_complex(15.0f, 9.0f)/sqrtf(170.0f), 
	gr_complex(-1.0f, 9.0f)/sqrtf(170.0f), gr_complex(1.0f, 9.0f)/sqrtf(170.0f), 
	gr_complex(-9.0f, 9.0f)/sqrtf(170.0f), gr_complex(9.0f, 9.0f)/sqrtf(170.0f), 
	gr_complex(-7.0f, 9.0f)/sqrtf(170.0f), gr_complex(7.0f, 9.0f)/sqrtf(170.0f), 
	gr_complex(-13.0f, 9.0f)/sqrtf(170.0f), gr_complex(13.0f, 9.0f)/sqrtf(170.0f), 
	gr_complex(-3.0f, 9.0f)/sqrtf(170.0f), gr_complex(3.0f, 9.0f)/sqrtf(170.0f), 
	gr_complex(-11.0f, 9.0f)/sqrtf(170.0f), gr_complex(11.0f, 9.0f)/sqrtf(170.0f), 
	gr_complex(-5.0f, 9.0f)/sqrtf(170.0f), gr_complex(5.0f, 9.0f)/sqrtf(170.0f), 
	gr_complex(-15.0f, -7.0f)/sqrtf(170.0f), gr_complex(15.0f, -7.0f)/sqrtf(170.0f), 
	gr_complex(-1.0f, -7.0f)/sqrtf(170.0f), gr_complex(1.0f, -7.0f)/sqrtf(170.0f), 
	gr_complex(-9.0f, -7.0f)/sqrtf(170.0f), gr_complex(9.0f, -7.0f)/sqrtf(170.0f), 
	gr_complex(-7.0f, -7.0f)/sqrtf(170.0f), gr_complex(7.0f, -7.0f)/sqrtf(170.0f), 
	gr_complex(-13.0f, -7.0f)/sqrtf(170.0f), gr_complex(13.0f, -7.0f)/sqrtf(170.0f), 
	gr_complex(-3.0f, -7.0f)/sqrtf(170.0f), gr_complex(3.0f, -7.0f)/sqrtf(170.0f), 
	gr_complex(-11.0f, -7.0f)/sqrtf(170.0f), gr_complex(11.0f, -7.0f)/sqrtf(170.0f), 
	gr_complex(-5.0f, -7.0f)/sqrtf(170.0f), gr_complex(5.0f, -7.0f)/sqrtf(170.0f), 
	gr_complex(-15.0f, 7.0f)/sqrtf(170.0f), gr_complex(15.0f, 7.0f)/sqrtf(170.0f), 
	gr_complex(-1.0f, 7.0f)/sqrtf(170.0f), gr_complex(1.0f, 7.0f)/sqrtf(170.0f), 
	gr_complex(-9.0f, 7.0f)/sqrtf(170.0f), gr_complex(9.0f, 7.0f)/sqrtf(170.0f), 
	gr_complex(-7.0f, 7.0f)/sqrtf(170.0f), gr_complex(7.0f, 7.0f)/sqrtf(170.0f), 
	gr_complex(-13.0f, 7.0f)/sqrtf(170.0f), gr_complex(13.0f, 7.0f)/sqrtf(170.0f), 
	gr_complex(-3.0f, 7.0f)/sqrtf(170.0f), gr_complex(3.0f, 7.0f)/sqrtf(170.0f), 
	gr_complex(-11.0f, 7.0f)/sqrtf(170.0f), gr_complex(11.0f, 7.0f)/sqrtf(170.0f), 
	gr_complex(-5.0f, 7.0f)/sqrtf(170.0f), gr_complex(5.0f, 7.0f)/sqrtf(170.0f), 
	gr_complex(-15.0f, -13.0f)/sqrtf(170.0f), gr_complex(15.0f, -13.0f)/sqrtf(170.0f), 
	gr_complex(-1.0f, -13.0f)/sqrtf(170.0f), gr_complex(1.0f, -13.0f)/sqrtf(170.0f), 
	gr_complex(-9.0f, -13.0f)/sqrtf(170.0f), gr_complex(9.0f, -13.0f)/sqrtf(170.0f), 
	gr_complex(-7.0f, -13.0f)/sqrtf(170.0f), gr_complex(7.0f, -13.0f)/sqrtf(170.0f), 
	gr_complex(-13.0f, -13.0f)/sqrtf(170.0f), gr_complex(13.0f, -13.0f)/sqrtf(170.0f), 
	gr_complex(-3.0f, -13.0f)/sqrtf(170.0f), gr_complex(3.0f, -13.0f)/sqrtf(170.0f), 
	gr_complex(-11.0f, -13.0f)/sqrtf(170.0f), gr_complex(11.0f, -13.0f)/sqrtf(170.0f), 
	gr_complex(-5.0f, -13.0f)/sqrtf(170.0f), gr_complex(5.0f, -13.0f)/sqrtf(170.0f), 
	gr_complex(-15.0f, 13.0f)/sqrtf(170.0f), gr_complex(15.0f, 13.0f)/sqrtf(170.0f), 
	gr_complex(-1.0f, 13.0f)/sqrtf(170.0f), gr_complex(1.0f, 13.0f)/sqrtf(170.0f), 
	gr_complex(-9.0f, 13.0f)/sqrtf(170.0f), gr_complex(9.0f, 13.0f)/sqrtf(170.0f), 
	gr_complex(-7.0f, 13.0f)/sqrtf(170.0f), gr_complex(7.0f, 13.0f)/sqrtf(170.0f), 
	gr_complex(-13.0f, 13.0f)/sqrtf(170.0f), gr_complex(13.0f, 13.0f)/sqrtf(170.0f), 
	gr_complex(-3.0f, 13.0f)/sqrtf(170.0f), gr_complex(3.0f, 13.0f)/sqrtf(170.0f), 
	gr_complex(-11.0f, 13.0f)/sqrtf(170.0f), gr_complex(11.0f, 13.0f)/sqrtf(170.0f), 
	gr_complex(-5.0f, 13.0f)/sqrtf(170.0f), gr_complex(5.0f, 13.0f)/sqrtf(170.0f), 
	gr_complex(-15.0f, -3.0f)/sqrtf(170.0f), gr_complex(15.0f, -3.0f)/sqrtf(170.0f), 
	gr_complex(-1.0f, -3.0f)/sqrtf(170.0f), gr_complex(1.0f, -3.0f)/sqrtf(170.0f), 
	gr_complex(-9.0f, -3.0f)/sqrtf(170.0f), gr_complex(9.0f, -3.0f)/sqrtf(170.0f), 
	gr_complex(-7.0f, -3.0f)/sqrtf(170.0f), gr_complex(7.0f, -3.0f)/sqrtf(170.0f), 
	gr_complex(-13.0f, -3.0f)/sqrtf(170.0f), gr_complex(13.0f, -3.0f)/sqrtf(170.0f), 
	gr_complex(-3.0f, -3.0f)/sqrtf(170.0f), gr_complex(3.0f, -3.0f)/sqrtf(170.0f), 
	gr_complex(-11.0f, -3.0f)/sqrtf(170.0f), gr_complex(11.0f, -3.0f)/sqrtf(170.0f), 
	gr_complex(-5.0f, -3.0f)/sqrtf(170.0f), gr_complex(5.0f, -3.0f)/sqrtf(170.0f), 
	gr_complex(-15.0f, 3.0f)/sqrtf(170.0f), gr_complex(15.0f, 3.0f)/sqrtf(170.0f), 
	gr_complex(-1.0f, 3.0f)/sqrtf(170.0f), gr_complex(1.0f, 3.0f)/sqrtf(170.0f), 
	gr_complex(-9.0f, 3.0f)/sqrtf(170.0f), gr_complex(9.0f, 3.0f)/sqrtf(170.0f), 
	gr_complex(-7.0f, 3.0f)/sqrtf(170.0f), gr_complex(7.0f, 3.0f)/sqrtf(170.0f), 
	gr_complex(-13.0f, 3.0f)/sqrtf(170.0f), gr_complex(13.0f, 3.0f)/sqrtf(170.0f), 
	gr_complex(-3.0f, 3.0f)/sqrtf(170.0f), gr_complex(3.0f, 3.0f)/sqrtf(170.0f), 
	gr_complex(-11.0f, 3.0f)/sqrtf(170.0f), gr_complex(11.0f, 3.0f)/sqrtf(170.0f), 
	gr_complex(-5.0f, 3.0f)/sqrtf(170.0f), gr_complex(5.0f, 3.0f)/sqrtf(170.0f), 
	gr_complex(-15.0f, -11.0f)/sqrtf(170.0f), gr_complex(15.0f, -11.0f)/sqrtf(170.0f), 
	gr_complex(-1.0f, -11.0f)/sqrtf(170.0f), gr_complex(1.0f, -11.0f)/sqrtf(170.0f), 
	gr_complex(-9.0f, -11.0f)/sqrtf(170.0f), gr_complex(9.0f, -11.0f)/sqrtf(170.0f), 
	gr_complex(-7.0f, -11.0f)/sqrtf(170.0f), gr_complex(7.0f, -11.0f)/sqrtf(170.0f), 
	gr_complex(-13.0f, -11.0f)/sqrtf(170.0f), gr_complex(13.0f, -11.0f)/sqrtf(170.0f), 
	gr_complex(-3.0f, -11.0f)/sqrtf(170.0f), gr_complex(3.0f, -11.0f)/sqrtf(170.0f), 
	gr_complex(-11.0f, -11.0f)/sqrtf(170.0f), gr_complex(11.0f, -11.0f)/sqrtf(170.0f), 
	gr_complex(-5.0f, -11.0f)/sqrtf(170.0f), gr_complex(5.0f, -11.0f)/sqrtf(170.0f), 
	gr_complex(-15.0f, 11.0f)/sqrtf(170.0f), gr_complex(15.0f, 11.0f)/sqrtf(170.0f), 
	gr_complex(-1.0f, 11.0f)/sqrtf(170.0f), gr_complex(1.0f, 11.0f)/sqrtf(170.0f), 
	gr_complex(-9.0f, 11.0f)/sqrtf(170.0f), gr_complex(9.0f, 11.0f)/sqrtf(170.0f), 
	gr_complex(-7.0f, 11.0f)/sqrtf(170.0f), gr_complex(7.0f, 11.0f)/sqrtf(170.0f), 
	gr_complex(-13.0f, 11.0f)/sqrtf(170.0f), gr_complex(13.0f, 11.0f)/sqrtf(170.0f), 
	gr_complex(-3.0f, 11.0f)/sqrtf(170.0f), gr_complex(3.0f, 11.0f)/sqrtf(170.0f), 
	gr_complex(-11.0f, 11.0f)/sqrtf(170.0f), gr_complex(11.0f, 11.0f)/sqrtf(170.0f), 
	gr_complex(-5.0f, 11.0f)/sqrtf(170.0f), gr_complex(5.0f, 11.0f)/sqrtf(170.0f), 
	gr_complex(-15.0f, -5.0f)/sqrtf(170.0f), gr_complex(15.0f, -5.0f)/sqrtf(170.0f), 
	gr_complex(-1.0f, -5.0f)/sqrtf(170.0f), gr_complex(1.0f, -5.0f)/sqrtf(170.0f), 
	gr_complex(-9.0f, -5.0f)/sqrtf(170.0f), gr_complex(9.0f, -5.0f)/sqrtf(170.0f), 
	gr_complex(-7.0f, -5.0f)/sqrtf(170.0f), gr_complex(7.0f, -5.0f)/sqrtf(170.0f), 
	gr_complex(-13.0f, -5.0f)/sqrtf(170.0f), gr_complex(13.0f, -5.0f)/sqrtf(170.0f), 
	gr_complex(-3.0f, -5.0f)/sqrtf(170.0f), gr_complex(3.0f, -5.0f)/sqrtf(170.0f), 
	gr_complex(-11.0f, -5.0f)/sqrtf(170.0f), gr_complex(11.0f, -5.0f)/sqrtf(170.0f), 
	gr_complex(-5.0f, -5.0f)/sqrtf(170.0f), gr_complex(5.0f, -5.0f)/sqrtf(170.0f), 
	gr_complex(-15.0f, 5.0f)/sqrtf(170.0f), gr_complex(15.0f, 5.0f)/sqrtf(170.0f), 
	gr_complex(-1.0f, 5.0f)/sqrtf(170.0f), gr_complex(1.0f, 5.0f)/sqrtf(170.0f), 
	gr_complex(-9.0f, 5.0f)/sqrtf(170.0f), gr_complex(9.0f, 5.0f)/sqrtf(170.0f), 
	gr_complex(-7.0f, 5.0f)/sqrtf(170.0f), gr_complex(7.0f, 5.0f)/sqrtf(170.0f), 
	gr_complex(-13.0f, 5.0f)/sqrtf(170.0f), gr_complex(13.0f, 5.0f)/sqrtf(170.0f), 
	gr_complex(-3.0f, 5.0f)/sqrtf(170.0f), gr_complex(3.0f, 5.0f)/sqrtf(170.0f), 
	gr_complex(-11.0f, 5.0f)/sqrtf(170.0f), gr_complex(11.0f, 5.0f)/sqrtf(170.0f), 
	gr_complex(-5.0f, 5.0f)/sqrtf(170.0f), gr_complex(5.0f, 5.0f)/sqrtf(170.0f)
};

/***************************************************/
/* signal field */
/***************************************************/

void signalNlDemodDecode(gr_complex *sym1, gr_complex *sym2, gr_complex *h, float *llrht, float *llrvht)
{
	gr_complex tmpM1, tmpM2;
	gr_complex tmpPilotSum1 = std::conj(sym1[7] / h[7] - sym1[21] / h[21] + sym1[43] / h[43] + sym1[57] / h[57]);
	gr_complex tmpPilotSum2 = std::conj(sym2[7] / h[7] - sym2[21] / h[21] + sym2[43] / h[43] + sym2[57] / h[57]);
	float tmpPilotSumAbs1 = std::abs(tmpPilotSum1);
	float tmpPilotSumAbs2 = std::abs(tmpPilotSum2);
	for(int i=0;i<64;i++)
	{
		if(FFT_26_SHIFT_DEMAP[i] > -1)
		{
			tmpM1 = sym1[i] / h[i] * tmpPilotSum1 / tmpPilotSumAbs1;
			tmpM2 = sym2[i] / h[i] * tmpPilotSum2 / tmpPilotSumAbs2;
			llrht[FFT_26_SHIFT_DEMAP[i]] = tmpM1.imag();
			llrht[FFT_26_SHIFT_DEMAP[i + 64]] = tmpM2.imag();
			llrvht[FFT_26_SHIFT_DEMAP[i]] = tmpM1.real();
			llrvht[FFT_26_SHIFT_DEMAP[i + 64]] = tmpM2.imag();
		}
	}
}

bool signalCheckLegacy(uint8_t* inBits, int* mcs, int* len, int* nDBPS)
{
	uint8_t tmpSumP = 0;
	int tmpRate = 0;

	if(!inBits[3])
	{
		return false;
	}

	if(inBits[4])
	{
		return false;
	}

	for(int i=0;i<17;i++)
	{
		tmpSumP += inBits[i];
	}
	if((tmpSumP & 0x01) ^ inBits[17])
	{
		return false;
	}

	for(int i=0;i<4;i++)
	{
		tmpRate |= (((int)inBits[i])<<i);
	}
	switch(tmpRate)
	{
		case 11:	// 0b1101
			*mcs = 0;
			*nDBPS = 24;
			break;
		case 15:	// 0b1111
			*mcs = 1;
			*nDBPS = 36;
			break;
		case 10:	// 0b0101
			*mcs = 2;
			*nDBPS = 48;
			break;
		case 14:	// 0b0111
			*mcs = 3;
			*nDBPS = 72;
			break;
		case 9:		// 0b1001
			*mcs = 4;
			*nDBPS = 96;
			break;
		case 13:	// 0b1011
			*mcs = 5;
			*nDBPS = 144;
			break;
		case 8:		// 0b0001
			*mcs = 6;
			*nDBPS = 192;
			break;
		case 12:	// 0b0011
			*mcs = 7;
			*nDBPS = 216;
			break;
		default:
			*mcs = 0;
			*nDBPS = 24;
			break;
	}

	*len = 0;
	for(int i=0;i<12;i++)
	{
		*len |= (((int)inBits[i+5])<<i);
	}
	if(*len > 4095 || *len < 14)
	{
		return false;
	}
	return true;
}

bool signalCheckHt(uint8_t* inBits)
{
	// correctness check
	if(inBits[26] != 1)
	{
		//std::cout<<"ht check error 1"<<std::endl;
		return false;
	}
	if(!checkBitCrc8(inBits, 34, &inBits[34]))
	{
		//std::cout<<"ht check error 2"<<std::endl;
		return false;
	}
	// supporting check
	if(inBits[5] + inBits[6] + inBits[7] + inBits[28] + inBits[29] + inBits[30] + inBits[32] + inBits[33])
	{
		//std::cout<<"ht check error 3"<<std::endl;
		// mcs > 31 (bit 5 & 6), 40bw (bit 7), stbc, ldpc and ESS are not supported
		return false;
	}
	return true;
}

bool signalCheckVhtA(uint8_t* inBits)
{
	// correctness check
	if((inBits[2] != 1) || (inBits[23] != 1) || (inBits[33] != 1))
	{
		return false;
	}
	if(!checkBitCrc8(inBits, 34, &inBits[34]))
	{
		return false;
	}
	// support check
	if(inBits[0] + inBits[1])
	{
		// 40, 80, 160 bw (bit 0&1) are not supported
		return false;
	}
	return true;
}

void signalParserL(int mcs, int len, c8p_mod* outMod)
{
	outMod->mcs = mcs;
	switch(mcs)
	{
		case 0:	// 0b1101
			outMod->mod = C8P_QAM_BPSK;
			outMod->cr = C8P_CR_12;
			outMod->nDBPS = 24;
			outMod->nCBPS = 48;
			outMod->nBPSCS = 1;
			break;
		case 1:	// 0b1111
			outMod->mod = C8P_QAM_BPSK;
			outMod->cr = C8P_CR_34;
			outMod->nDBPS = 36;
			outMod->nCBPS = 48;
			outMod->nBPSCS = 1;
			break;
		case 2:	// 0b0101
			outMod->mod = C8P_QAM_QPSK;
			outMod->cr = C8P_CR_12;
			outMod->nDBPS = 48;
			outMod->nCBPS = 96;
			outMod->nBPSCS = 2;
			break;
		case 3:	// 0b0111
			outMod->mod = C8P_QAM_QPSK;
			outMod->cr = C8P_CR_34;
			outMod->nDBPS = 72;
			outMod->nCBPS = 96;
			outMod->nBPSCS = 2;
			break;
		case 4:	// 0b1001
			outMod->mod = C8P_QAM_16QAM;
			outMod->cr = C8P_CR_12;
			outMod->nDBPS = 96;
			outMod->nCBPS = 192;
			outMod->nBPSCS = 4;
			break;
		case 5:	// 0b1011
			outMod->mod = C8P_QAM_16QAM;
			outMod->cr = C8P_CR_34;
			outMod->nDBPS = 144;
			outMod->nCBPS = 192;
			outMod->nBPSCS = 4;
			break;
		case 6:	// 0b0001
			outMod->mod = C8P_QAM_64QAM;
			outMod->cr = C8P_CR_23;
			outMod->nDBPS = 192;
			outMod->nCBPS = 288;
			outMod->nBPSCS = 6;
			break;
		case 7:	// 0b0011
			outMod->mod = C8P_QAM_64QAM;
			outMod->cr = C8P_CR_34;
			outMod->nDBPS = 216;
			outMod->nCBPS = 288;
			outMod->nBPSCS = 6;
			break;
		default:
			// error
			break;
	}
	outMod->len = len;
	outMod->nCBPSS = outMod->nCBPS;
	outMod->nSD = 48;
	outMod->nSP = 4;
	outMod->nSS = 1;		// only 1 ss
	outMod->sumu = 0;		// su
	outMod->nLTF = 0;

	outMod->format = C8P_F_L;
	outMod->nSymSamp = 80;
	outMod->nSym = (outMod->len*8 + 22)/outMod->nDBPS + (((outMod->len*8 + 22)%outMod->nDBPS) != 0);
	outMod->ampdu = 0;
}

void signalParserHt(uint8_t* inBits, c8p_mod* outMod, c8p_sigHt* outSigHt)
{
	// ht signal field
	// 0-6 mcs
	outSigHt->mcs = 0;
	for(int i=0;i<7;i++)
	{
		outSigHt->mcs |= (((int)inBits[i])<<i);
	}
	// 7 bw
	outSigHt->bw = inBits[7];
	// 8-23 len
	outSigHt->len = 0;
	for(int i=0;i<16;i++)
	{
		outSigHt->len |= (((int)inBits[i+8])<<i);
	}
	// 24 smoothing
	outSigHt->smooth = inBits[24];
	// 25 not sounding
	outSigHt->noSound = inBits[25];
	// 26 reserved
	// 27 aggregation
	outSigHt->aggre = inBits[27];
	// 28-29 stbc
	outSigHt->stbc = 0;
	for(int i=0;i<2;i++)
	{
		outSigHt->stbc |= (((int)inBits[i+28])<<i);
	}
	// 30 fec coding
	outSigHt->coding = inBits[30];
	// 31 short GI
	outSigHt->shortGi = inBits[31];
	// 32-33 ESS
	outSigHt->nExtSs = 0;
	for(int i=0;i<2;i++)
	{
		outSigHt->nExtSs |= (((int)inBits[i+32])<<i);
	}

	// ht modulation related
	// format
	outMod->format = C8P_F_HT;
	outMod->sumu = 0;
	// short GI
	outMod->nSymSamp = 80;
	if(outSigHt->shortGi)
	{
		outMod->nSymSamp = 72;
	}
	// AMPDU
	outMod->ampdu = 0;
	if(outSigHt->aggre)
	{
		outMod->ampdu = 1;
	}
	outMod->mcs = outSigHt->mcs;
	switch(outSigHt->mcs % 8)
	{
		case 0:
			outMod->mod = C8P_QAM_BPSK;
			outMod->nBPSCS = 1;
			outMod->cr = C8P_CR_12;
			break;
		case 1:
			outMod->mod = C8P_QAM_QPSK;
			outMod->nBPSCS = 2;
			outMod->cr = C8P_CR_12;
			break;
		case 2:
			outMod->mod = C8P_QAM_QPSK;
			outMod->nBPSCS = 2;
			outMod->cr = C8P_CR_34;
			break;
		case 3:
			outMod->mod = C8P_QAM_16QAM;
			outMod->nBPSCS = 4;
			outMod->cr = C8P_CR_12;
			break;
		case 4:
			outMod->mod = C8P_QAM_16QAM;
			outMod->nBPSCS = 4;
			outMod->cr = C8P_CR_34;
			break;
		case 5:
			outMod->mod = C8P_QAM_64QAM;
			outMod->nBPSCS = 6;
			outMod->cr = C8P_CR_23;
			break;
		case 6:
			outMod->mod = C8P_QAM_64QAM;
			outMod->nBPSCS = 6;
			outMod->cr = C8P_CR_34;
			break;
		case 7:
			outMod->mod = C8P_QAM_64QAM;
			outMod->nBPSCS = 6;
			outMod->cr = C8P_CR_56;
			break;
		default:
			break;
	}
	outMod->len = outSigHt->len;
	outMod->nSS = outSigHt->mcs / 8 + 1;
	outMod->nSD = 52;
	outMod->nSP = 4;
	outMod->nCBPSS = outMod->nBPSCS * outMod->nSD;
	outMod->nCBPS = outMod->nCBPSS * outMod->nSS;
	switch(outMod->cr)
	{
		case C8P_CR_12:
			outMod->nDBPS = outMod->nCBPS / 2;
			break;
		case C8P_CR_23:
			outMod->nDBPS = (outMod->nCBPS * 2) / 3;
			break;
		case C8P_CR_34:
			outMod->nDBPS = (outMod->nCBPS * 3) / 4;
			break;
		case C8P_CR_56:
			outMod->nDBPS = (outMod->nCBPS * 5) / 6;
			break;
		default:
			break;
	}
	outMod->nIntCol = 13;
	outMod->nIntRow = outMod->nBPSCS * 4;
	outMod->nIntRot = 11;
	switch(outMod->nSS)
	{
		case 1:
			outMod->nLTF = 1;
			break;
		case 2:
			outMod->nLTF = 2;
			break;
		case 3:
		case 4:
			outMod->nLTF = 4;
			break;
		default:
			break;
		
	}
	outMod->nSym = ((outMod->len*8 + 22)/outMod->nDBPS + (((outMod->len*8 + 22)%outMod->nDBPS) != 0));
}

void modParserHt(int mcs, c8p_mod* outMod)
{
	outMod->mcs = mcs;
	switch(mcs % 8)
	{
		case 0:
			outMod->mod = C8P_QAM_BPSK;
			outMod->nBPSCS = 1;
			outMod->cr = C8P_CR_12;
			break;
		case 1:
			outMod->mod = C8P_QAM_QPSK;
			outMod->nBPSCS = 2;
			outMod->cr = C8P_CR_12;
			break;
		case 2:
			outMod->mod = C8P_QAM_QPSK;
			outMod->nBPSCS = 2;
			outMod->cr = C8P_CR_34;
			break;
		case 3:
			outMod->mod = C8P_QAM_16QAM;
			outMod->nBPSCS = 4;
			outMod->cr = C8P_CR_12;
			break;
		case 4:
			outMod->mod = C8P_QAM_16QAM;
			outMod->nBPSCS = 4;
			outMod->cr = C8P_CR_34;
			break;
		case 5:
			outMod->mod = C8P_QAM_64QAM;
			outMod->nBPSCS = 6;
			outMod->cr = C8P_CR_23;
			break;
		case 6:
			outMod->mod = C8P_QAM_64QAM;
			outMod->nBPSCS = 6;
			outMod->cr = C8P_CR_34;
			break;
		case 7:
			outMod->mod = C8P_QAM_64QAM;
			outMod->nBPSCS = 6;
			outMod->cr = C8P_CR_56;
			break;
		default:
			break;
	}
	outMod->nSD = 52;
	outMod->nSP = 4;
	outMod->nCBPSS = outMod->nBPSCS * outMod->nSD;
	outMod->nCBPS = outMod->nCBPSS * outMod->nSS;
	switch(outMod->cr)
	{
		case C8P_CR_12:
			outMod->nDBPS = outMod->nCBPS / 2;
			break;
		case C8P_CR_23:
			outMod->nDBPS = (outMod->nCBPS * 2) / 3;
			break;
		case C8P_CR_34:
			outMod->nDBPS = (outMod->nCBPS * 3) / 4;
			break;
		case C8P_CR_56:
			outMod->nDBPS = (outMod->nCBPS * 5) / 6;
			break;
		default:
			break;
	}
	outMod->nIntCol = 13;
	outMod->nIntRow = outMod->nBPSCS * 4;
	outMod->nIntRot = 11;
	switch(outMod->nSS)
	{
		case 1:
			outMod->nLTF = 1;
			break;
		case 2:
			outMod->nLTF = 2;
			break;
		case 3:
		case 4:
			outMod->nLTF = 4;
			break;
		default:
			break;
		
	}
}

void signalParserVhtA(uint8_t* inBits, c8p_mod* outMod, c8p_sigVhtA* outSigVhtA)
{
	// vht signal field
	// 0-1 bw
	outSigVhtA->bw = 0;
	for(int i=0;i<2;i++){outSigVhtA->bw |= (((int)inBits[i])<<i);}
	// 2 reserved
	// 3 stbc
	outSigVhtA->stbc = inBits[3];
	// 4-9 group ID, group ID is used to judge su or mu and filter the packet, only 0 and 63 used for su
	outSigVhtA->groupId = 0;
	for(int i=0;i<6;i++){outSigVhtA->groupId |= (((int)inBits[i+4])<<i);}
	if(outSigVhtA->groupId == 0 || outSigVhtA->groupId == 63)	// su
	{
		// 10-12 nSTS
		outSigVhtA->su_nSTS = 0;
		for(int i=0;i<3;i++){outSigVhtA->su_nSTS |= (((int)inBits[i+10])<<i);}
		// 13-21 partial AID
		outSigVhtA->su_partialAID = 0;
		for(int i=0;i<9;i++){outSigVhtA->su_partialAID |= (((int)inBits[i+13])<<i);}
		// 26 coding
		outSigVhtA->su_coding = inBits[26];
		// 28-31 mcs
		outSigVhtA->su_mcs = 0;
		for(int i=0;i<4;i++){outSigVhtA->su_mcs |= (((int)inBits[i+28])<<i);}
		// 32 beamforming
		outSigVhtA->su_beamformed = inBits[32];
	}
	else
	{
		// 10-12 nSTS 0
		outSigVhtA->mu_nSTS[0] = 0;
		for(int i=0;i<3;i++){outSigVhtA->mu_nSTS[0] |= (((int)inBits[i+10])<<i);}
		// 13-15 nSTS 1
		outSigVhtA->mu_nSTS[1] = 0;
		for(int i=0;i<3;i++){outSigVhtA->mu_nSTS[1] |= (((int)inBits[i+13])<<i);}
		// 16-18 nSTS 2
		outSigVhtA->mu_nSTS[2] = 0;
		for(int i=0;i<3;i++){outSigVhtA->mu_nSTS[2] |= (((int)inBits[i+16])<<i);}
		// 19-21 nSTS 3
		outSigVhtA->mu_nSTS[3] = 0;
		for(int i=0;i<3;i++){outSigVhtA->mu_nSTS[3] |= (((int)inBits[i+19])<<i);}
		// 26 coding 0
		outSigVhtA->mu_coding[0] = inBits[26];
		// 28 coding 1
		outSigVhtA->mu_coding[1] = inBits[28];
		// 29 coding 2
		outSigVhtA->mu_coding[2] = inBits[29];
		// 30 coding 3
		outSigVhtA->mu_coding[3] = inBits[30];
	}
	
	// 22 txop ps not allowed
	outSigVhtA->txoppsNot = inBits[22];
	// 24 short gi
	outSigVhtA->shortGi = inBits[24];
	// 25 short gi nSYM disambiguantion
	outSigVhtA->shortGiNsymDis = inBits[25];
	// 27 ldpc extra
	outSigVhtA->ldpcExtra = inBits[27];

	// modualtion ralated
	// format
	outMod->format = C8P_F_VHT;
	// short GI
	outMod->nSymSamp = 80;
	if(outSigVhtA->shortGi)
	{
		outMod->nSymSamp = 72;
	}
	// AMPDU
	outMod->ampdu = 1;

	if((outSigVhtA->groupId == 0) || (outSigVhtA->groupId == 63))
	{
		outMod->sumu = 0;	// su
		outMod->nSS = outSigVhtA->su_nSTS + 1;
		modParserVht(outSigVhtA->su_mcs, outMod);
		// still need the packet len in sig b
	}
	else
	{
		outMod->sumu = 1;	// mu flag, mod is parsed after sig b
		outMod->nLTF = 2;
		outMod->nSS = 1;
		outMod->nSD = 52;
		outMod->nSP = 4;
	}
}

void signalParserVhtB(uint8_t* inBits, c8p_mod* outMod)
{
	int tmpLen = 0;
	int tmpMcs = 0;
	if(outMod->sumu)
	{
		for(int i=0;i<16;i++){tmpLen |= (((int)inBits[i])<<i);}
		for(int i=0;i<4;i++){tmpMcs |= (((int)inBits[i+16])<<i);}
		modParserVht(tmpMcs, outMod);
		outMod->len = tmpLen * 4;
		outMod->nSym = (outMod->len*8 + 16 + 6) / outMod->nDBPS + (((outMod->len*8 + 16 + 6) % outMod->nDBPS) != 0);
		outMod->nLTF = 2;
	}
	else
	{
		if((inBits[17] + inBits[18] + inBits[19]) == 3)
		{
			for(int i=0;i<17;i++){tmpLen |= (((int)inBits[i])<<i);}
			outMod->len = tmpLen * 4;
			outMod->nSym = (outMod->len*8 + 16 + 6) / outMod->nDBPS + (((outMod->len*8 + 16 + 6) % outMod->nDBPS) != 0);
		}
		else
		{
			uint32_t tmpRxPattern = 0;
			uint32_t tmpEachBit;
			for(int i=0;i<20;i++)
			{
				tmpEachBit = inBits[i];
				tmpRxPattern |= (tmpEachBit << i);
			}
			if(tmpRxPattern == 0b01000010001011100000)
			{
				outMod->len = 0;
				outMod->nSym = 0;
			}
			else
			{
				outMod->len = -1;
				outMod->nSym = -1;
			}
			// std::cout<<"sig b parser, NDP check "<<outMod->len<<" "<<outMod->nSym<<", pattern:"<<tmpRxPattern<<std::endl;
		}
	}
}

void modParserVht(int mcs, c8p_mod* outMod)
{
	outMod->mcs = mcs;
	switch(mcs)
	{
		case 0:
			outMod->mod = C8P_QAM_BPSK;
			outMod->nBPSCS = 1;
			outMod->cr = C8P_CR_12;
			break;
		case 1:
			outMod->mod = C8P_QAM_QPSK;
			outMod->nBPSCS = 2;
			outMod->cr = C8P_CR_12;
			break;
		case 2:
			outMod->mod = C8P_QAM_QPSK;
			outMod->nBPSCS = 2;
			outMod->cr = C8P_CR_34;
			break;
		case 3:
			outMod->mod = C8P_QAM_16QAM;
			outMod->nBPSCS = 4;
			outMod->cr = C8P_CR_12;
			break;
		case 4:
			outMod->mod = C8P_QAM_16QAM;
			outMod->nBPSCS = 4;
			outMod->cr = C8P_CR_34;
			break;
		case 5:
			outMod->mod = C8P_QAM_64QAM;
			outMod->nBPSCS = 6;
			outMod->cr = C8P_CR_23;
			break;
		case 6:
			outMod->mod = C8P_QAM_64QAM;
			outMod->nBPSCS = 6;
			outMod->cr = C8P_CR_34;
			break;
		case 7:
			outMod->mod = C8P_QAM_64QAM;
			outMod->nBPSCS = 6;
			outMod->cr = C8P_CR_56;
			break;
		case 8:
			outMod->mod = C8P_QAM_256QAM;
			outMod->nBPSCS = 8;
			outMod->cr = C8P_CR_34;
			break;
		case 9:
			outMod->mod = C8P_QAM_256QAM;
			outMod->nBPSCS = 8;
			outMod->cr = C8P_CR_56;
			break;
		default:
			break;
	}
	outMod->nSD = 52;
	outMod->nSP = 4;
	outMod->nCBPSS = outMod->nBPSCS * outMod->nSD;
	outMod->nCBPS = outMod->nCBPSS * outMod->nSS;
	switch(outMod->cr)
	{
		case C8P_CR_12:
			outMod->nDBPS = outMod->nCBPS / 2;
			break;
		case C8P_CR_23:
			outMod->nDBPS = (outMod->nCBPS * 2) / 3;
			break;
		case C8P_CR_34:
			outMod->nDBPS = (outMod->nCBPS * 3) / 4;
			break;
		case C8P_CR_56:
			outMod->nDBPS = (outMod->nCBPS * 5) / 6;
			break;
		default:
			break;
	}
	outMod->nIntCol = 13;
	outMod->nIntRow = outMod->nBPSCS * 4;
	outMod->nIntRot = 11;
	switch(outMod->nSS)
	{
		case 1:
			outMod->nLTF = 1;
			break;
		case 2:
			outMod->nLTF = 2;
			break;
		case 3:
		case 4:
			outMod->nLTF = 4;
			break;
		default:
			break;
		
	}
}

/***************************************************/
/* coding */
/***************************************************/

void genCrc8Bits(uint8_t* inBits, uint8_t* outBits, int len)
{
	uint16_t c = 0x00ff;
	for (int i = 0; i < len; i++)
	{
		c = c << 1;
		if (c & 0x0100)
		{
			c = c + 1;
			c = c ^ 0x0006;
		}
		else
		{
			c = c ^ 0x0000;
		}
		if (inBits[i])
		{
			c = c ^ 0x0007;
		}
		else
		{
			c = c ^ 0x0000;
		}
	}
	c = (0x00ff - (c & 0x00ff));
	for (int i = 0; i < 8; i++)
	{
		if (c & (1 << (7-i)))
		{
			outBits[i] = 1;
		}
		else
		{
			outBits[i] = 0;
		}
	}
}

bool checkBitCrc8(uint8_t* inBits, int len, uint8_t* crcBits)
{
	uint16_t c = 0x00ff;
	for (int i = 0; i < len; i++)
	{
		c = c << 1;
		if (c & 0x0100)
		{
			c = c + 1;
			c = c ^ 0x0006;
		}
		else
		{
			c = c ^ 0x0000;
		}
		if (inBits[i])
		{
			c = c ^ 0x0007;
		}
		else
		{
			c = c ^ 0x0000;
		}
	}
	for (int i = 0; i < 8; i++)
	{
		if (crcBits[i])
		{
			c ^= (1 << (7 - i));
		}
	}
	if ((c & 0x00ff) == 0x00ff)
	{
		return true;
	}
	return false;
}


// legacy interleave and deinterleave map
const int mapIntelLegacyBpsk[48] = {
	0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 
	1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 
	2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47
};

const int mapDeintLegacyBpsk[48] = {
    0, 16, 32, 1, 17, 33, 2, 18, 34, 3, 19, 35, 
	4, 20, 36, 5, 21, 37, 6, 22, 38, 7, 23, 39, 
	8, 24, 40, 9, 25, 41, 10, 26, 42, 11, 27, 43, 
	12, 28, 44, 13, 29, 45, 14, 30, 46, 15, 31, 47
};

const int mapIntelLegacyQpsk[96] = {
	0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 
	1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79, 85, 91, 
	2, 8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 80, 86, 92, 
	3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 87, 93, 
	4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70, 76, 82, 88, 94, 
	5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95
};

const int mapDeintLegacyQpsk[96] = {
	0, 16, 32, 48, 64, 80, 1, 17, 33, 49, 65, 81, 2, 18, 34, 50, 66, 82, 3, 19, 35, 51, 67, 83, 
	4, 20, 36, 52, 68, 84, 5, 21, 37, 53, 69, 85, 6, 22, 38, 54, 70, 86, 7, 23, 39, 55, 71, 87, 
	8, 24, 40, 56, 72, 88, 9, 25, 41, 57, 73, 89, 10, 26, 42, 58, 74, 90, 11, 27, 43, 59, 75, 91, 
	12, 28, 44, 60, 76, 92, 13, 29, 45, 61, 77, 93, 14, 30, 46, 62, 78, 94, 15, 31, 47, 63, 79, 95
};

const int mapIntelLegacy16Qam[192] = {
	0, 13, 24, 37, 48, 61, 72, 85, 96, 109, 120, 133, 144, 157, 168, 181, 
	1, 12, 25, 36, 49, 60, 73, 84, 97, 108, 121, 132, 145, 156, 169, 180, 
	2, 15, 26, 39, 50, 63, 74, 87, 98, 111, 122, 135, 146, 159, 170, 183, 
	3, 14, 27, 38, 51, 62, 75, 86, 99, 110, 123, 134, 147, 158, 171, 182, 
	4, 17, 28, 41, 52, 65, 76, 89, 100, 113, 124, 137, 148, 161, 172, 185, 
	5, 16, 29, 40, 53, 64, 77, 88, 101, 112, 125, 136, 149, 160, 173, 184, 
	6, 19, 30, 43, 54, 67, 78, 91, 102, 115, 126, 139, 150, 163, 174, 187, 
	7, 18, 31, 42, 55, 66, 79, 90, 103, 114, 127, 138, 151, 162, 175, 186, 
	8, 21, 32, 45, 56, 69, 80, 93, 104, 117, 128, 141, 152, 165, 176, 189, 
	9, 20, 33, 44, 57, 68, 81, 92, 105, 116, 129, 140, 153, 164, 177, 188, 
	10, 23, 34, 47, 58, 71, 82, 95, 106, 119, 130, 143, 154, 167, 178, 191, 
	11, 22, 35, 46, 59, 70, 83, 94, 107, 118, 131, 142, 155, 166, 179, 190
};

const int mapDeintLegacy16Qam[192] = {
	0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 17, 1, 49, 33, 81, 65, 113, 97, 145, 129, 177, 161, 
	2, 18, 34, 50, 66, 82, 98, 114, 130, 146, 162, 178, 19, 3, 51, 35, 83, 67, 115, 99, 147, 131, 179, 163, 
	4, 20, 36, 52, 68, 84, 100, 116, 132, 148, 164, 180, 21, 5, 53, 37, 85, 69, 117, 101, 149, 133, 181, 165, 
	6, 22, 38, 54, 70, 86, 102, 118, 134, 150, 166, 182, 23, 7, 55, 39, 87, 71, 119, 103, 151, 135, 183, 167, 
	8, 24, 40, 56, 72, 88, 104, 120, 136, 152, 168, 184, 25, 9, 57, 41, 89, 73, 121, 105, 153, 137, 185, 169, 
	10, 26, 42, 58, 74, 90, 106, 122, 138, 154, 170, 186, 27, 11, 59, 43, 91, 75, 123, 107, 155, 139, 187, 171, 
	12, 28, 44, 60, 76, 92, 108, 124, 140, 156, 172, 188, 29, 13, 61, 45, 93, 77, 125, 109, 157, 141, 189, 173, 
	14, 30, 46, 62, 78, 94, 110, 126, 142, 158, 174, 190, 31, 15, 63, 47, 95, 79, 127, 111, 159, 143, 191, 175
};

const int mapIntelLegacy64Qam[288] = {
	0, 20, 37, 54, 74, 91, 108, 128, 145, 162, 182, 199, 216, 236, 253, 270, 
	1, 18, 38, 55, 72, 92, 109, 126, 146, 163, 180, 200, 217, 234, 254, 271, 
	2, 19, 36, 56, 73, 90, 110, 127, 144, 164, 181, 198, 218, 235, 252, 272, 
	3, 23, 40, 57, 77, 94, 111, 131, 148, 165, 185, 202, 219, 239, 256, 273, 
	4, 21, 41, 58, 75, 95, 112, 129, 149, 166, 183, 203, 220, 237, 257, 274, 
	5, 22, 39, 59, 76, 93, 113, 130, 147, 167, 184, 201, 221, 238, 255, 275, 
	6, 26, 43, 60, 80, 97, 114, 134, 151, 168, 188, 205, 222, 242, 259, 276, 
	7, 24, 44, 61, 78, 98, 115, 132, 152, 169, 186, 206, 223, 240, 260, 277, 
	8, 25, 42, 62, 79, 96, 116, 133, 150, 170, 187, 204, 224, 241, 258, 278, 
	9, 29, 46, 63, 83, 100, 117, 137, 154, 171, 191, 208, 225, 245, 262, 279, 
	10, 27, 47, 64, 81, 101, 118, 135, 155, 172, 189, 209, 226, 243, 263, 280, 
	11, 28, 45, 65, 82, 99, 119, 136, 153, 173, 190, 207, 227, 244, 261, 281, 
	12, 32, 49, 66, 86, 103, 120, 140, 157, 174, 194, 211, 228, 248, 265, 282, 
	13, 30, 50, 67, 84, 104, 121, 138, 158, 175, 192, 212, 229, 246, 266, 283, 
	14, 31, 48, 68, 85, 102, 122, 139, 156, 176, 193, 210, 230, 247, 264, 284, 
	15, 35, 52, 69, 89, 106, 123, 143, 160, 177, 197, 214, 231, 251, 268, 285, 
	16, 33, 53, 70, 87, 107, 124, 141, 161, 178, 195, 215, 232, 249, 269, 286, 
	17, 34, 51, 71, 88, 105, 125, 142, 159, 179, 196, 213, 233, 250, 267, 287
};

const int mapDeintLegacy64Qam[288] = {
	0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 17, 33, 
	1, 65, 81, 49, 113, 129, 97, 161, 177, 145, 209, 225, 193, 257, 273, 241, 34, 
	2, 18, 82, 50, 66, 130, 98, 114, 178, 146, 162, 226, 194, 210, 274, 242, 258, 
	3, 19, 35, 51, 67, 83, 99, 115, 131, 147, 163, 179, 195, 211, 227, 243, 259, 275, 20, 36, 
	4, 68, 84, 52, 116, 132, 100, 164, 180, 148, 212, 228, 196, 260, 276, 244, 37, 
	5, 21, 85, 53, 69, 133, 101, 117, 181, 149, 165, 229, 197, 213, 277, 245, 261, 
	6, 22, 38, 54, 70, 86, 102, 118, 134, 150, 166, 182, 198, 214, 230, 246, 262, 278, 23, 39, 
	7, 71, 87, 55, 119, 135, 103, 167, 183, 151, 215, 231, 199, 263, 279, 247, 40, 
	8, 24, 88, 56, 72, 136, 104, 120, 184, 152, 168, 232, 200, 216, 280, 248, 264, 
	9, 25, 41, 57, 73, 89, 105, 121, 137, 153, 169, 185, 201, 217, 233, 249, 265, 281, 26, 42, 
	10, 74, 90, 58, 122, 138, 106, 170, 186, 154, 218, 234, 202, 266, 282, 250, 43, 
	11, 27, 91, 59, 75, 139, 107, 123, 187, 155, 171, 235, 203, 219, 283, 251, 267, 
	12, 28, 44, 60, 76, 92, 108, 124, 140, 156, 172, 188, 204, 220, 236, 252, 268, 284, 29, 45, 
	13, 77, 93, 61, 125, 141, 109, 173, 189, 157, 221, 237, 205, 269, 285, 253, 46, 
	14, 30, 94, 62, 78, 142, 110, 126, 190, 158, 174, 238, 206, 222, 286, 254, 270, 
	15, 31, 47, 63, 79, 95, 111, 127, 143, 159, 175, 191, 207, 223, 239, 255, 271, 287
};

// ht and vht interleave and deinterleave map for siso and 2x2 mimo
const int mapDeintVhtSigB20[52] = {
	0, 13, 26, 39, 1, 14, 27, 40, 2, 15, 28, 41, 3, 16, 29, 42, 4, 17,
	30, 43, 5, 18, 31, 44, 6,19, 32, 45, 7, 20, 33, 46, 8, 21, 34, 47,
	9, 22, 35, 48, 10, 23, 36, 49, 11, 24, 37, 50, 12, 25, 38, 51};

const int mapIntelVhtSigB20[52] = {
	0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 1, 5, 9, 13, 17, 
	21, 25, 29, 33, 37, 41, 45, 49, 2, 6, 10, 14, 18, 22, 26, 30, 34, 
	38, 42, 46, 50, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51};

const int mapIntelNonlegacyBpsk[52] = {
	0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 
	1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 
	2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 
	3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51
};

const int mapIntelNonlegacyBpsk2[52] = {
	30, 34, 38, 42, 46, 50, 2, 6, 10, 14, 18, 22, 26, 
	31, 35, 39, 43, 47, 51, 3, 7, 11, 15, 19, 23, 27, 
	32, 36, 40, 44, 48, 0, 4, 8, 12, 16, 20, 24, 28, 
	33, 37, 41, 45, 49, 1, 5, 9, 13, 17, 21, 25, 29
};

const int mapIntelNonlegacyQpsk[104] = {
	0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 
	1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 
	2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 
	3, 11, 19, 27, 35, 43, 51, 59, 67, 75, 83, 91, 99, 
	4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 
	5, 13, 21, 29, 37, 45, 53, 61, 69, 77, 85, 93, 101, 
	6, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 
	7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103
};

const int mapIntelNonlegacyQpsk2[104] = {
	60, 68, 76, 84, 92, 100, 4, 12, 20, 28, 36, 44, 52, 
	61, 69, 77, 85, 93, 101, 5, 13, 21, 29, 37, 45, 53, 
	62, 70, 78, 86, 94, 102, 6, 14, 22, 30, 38, 46, 54, 
	63, 71, 79, 87, 95, 103, 7, 15, 23, 31, 39, 47, 55, 
	64, 72, 80, 88, 96, 0, 8, 16, 24, 32, 40, 48, 56, 
	65, 73, 81, 89, 97, 1, 9, 17, 25, 33, 41, 49, 57, 
	66, 74, 82, 90, 98, 2, 10, 18, 26, 34, 42, 50, 58, 
	67, 75, 83, 91, 99, 3, 11, 19, 27, 35, 43, 51, 59
};

const int mapIntelNonlegacy16Qam[208] = {
	0, 17, 32, 49, 64, 81, 96, 113, 128, 145, 160, 177, 192, 
	1, 16, 33, 48, 65, 80, 97, 112, 129, 144, 161, 176, 193, 
	2, 19, 34, 51, 66, 83, 98, 115, 130, 147, 162, 179, 194, 
	3, 18, 35, 50, 67, 82, 99, 114, 131, 146, 163, 178, 195, 
	4, 21, 36, 53, 68, 85, 100, 117, 132, 149, 164, 181, 196, 
	5, 20, 37, 52, 69, 84, 101, 116, 133, 148, 165, 180, 197, 
	6, 23, 38, 55, 70, 87, 102, 119, 134, 151, 166, 183, 198, 
	7, 22, 39, 54, 71, 86, 103, 118, 135, 150, 167, 182, 199, 
	8, 25, 40, 57, 72, 89, 104, 121, 136, 153, 168, 185, 200, 
	9, 24, 41, 56, 73, 88, 105, 120, 137, 152, 169, 184, 201, 
	10, 27, 42, 59, 74, 91, 106, 123, 138, 155, 170, 187, 202, 
	11, 26, 43, 58, 75, 90, 107, 122, 139, 154, 171, 186, 203, 
	12, 29, 44, 61, 76, 93, 108, 125, 140, 157, 172, 189, 204, 
	13, 28, 45, 60, 77, 92, 109, 124, 141, 156, 173, 188, 205, 
	14, 31, 46, 63, 78, 95, 110, 127, 142, 159, 174, 191, 206, 
	15, 30, 47, 62, 79, 94, 111, 126, 143, 158, 175, 190, 207
};

const int mapIntelNonlegacy16Qam2[208] = {
	120, 137, 152, 169, 184, 201, 8, 25, 40, 57, 72, 89, 104, 
	121, 136, 153, 168, 185, 200, 9, 24, 41, 56, 73, 88, 105, 
	122, 139, 154, 171, 186, 203, 10, 27, 42, 59, 74, 91, 106, 
	123, 138, 155, 170, 187, 202, 11, 26, 43, 58, 75, 90, 107, 
	124, 141, 156, 173, 188, 205, 12, 29, 44, 61, 76, 93, 108, 
	125, 140, 157, 172, 189, 204, 13, 28, 45, 60, 77, 92, 109, 
	126, 143, 158, 175, 190, 207, 14, 31, 46, 63, 78, 95, 110, 
	127, 142, 159, 174, 191, 206, 15, 30, 47, 62, 79, 94, 111, 
	128, 145, 160, 177, 192, 1, 16, 33, 48, 65, 80, 97, 112, 
	129, 144, 161, 176, 193, 0, 17, 32, 49, 64, 81, 96, 113, 
	130, 147, 162, 179, 194, 3, 18, 35, 50, 67, 82, 99, 114, 
	131, 146, 163, 178, 195, 2, 19, 34, 51, 66, 83, 98, 115, 
	132, 149, 164, 181, 196, 5, 20, 37, 52, 69, 84, 101, 116, 
	133, 148, 165, 180, 197, 4, 21, 36, 53, 68, 85, 100, 117, 
	134, 151, 166, 183, 198, 7, 22, 39, 54, 71, 86, 103, 118, 
	135, 150, 167, 182, 199, 6, 23, 38, 55, 70, 87, 102, 119
};

const int mapIntelNonlegacy64Qam[312] = {
	0, 26, 49, 72, 98, 121, 144, 170, 193, 216, 242, 265, 288, 
	1, 24, 50, 73, 96, 122, 145, 168, 194, 217, 240, 266, 289, 
	2, 25, 48, 74, 97, 120, 146, 169, 192, 218, 241, 264, 290, 
	3, 29, 52, 75, 101, 124, 147, 173, 196, 219, 245, 268, 291, 
	4, 27, 53, 76, 99, 125, 148, 171, 197, 220, 243, 269, 292, 
	5, 28, 51, 77, 100, 123, 149, 172, 195, 221, 244, 267, 293, 
	6, 32, 55, 78, 104, 127, 150, 176, 199, 222, 248, 271, 294, 
	7, 30, 56, 79, 102, 128, 151, 174, 200, 223, 246, 272, 295, 
	8, 31, 54, 80, 103, 126, 152, 175, 198, 224, 247, 270, 296, 
	9, 35, 58, 81, 107, 130, 153, 179, 202, 225, 251, 274, 297, 
	10, 33, 59, 82, 105, 131, 154, 177, 203, 226, 249, 275, 298, 
	11, 34, 57, 83, 106, 129, 155, 178, 201, 227, 250, 273, 299, 
	12, 38, 61, 84, 110, 133, 156, 182, 205, 228, 254, 277, 300, 
	13, 36, 62, 85, 108, 134, 157, 180, 206, 229, 252, 278, 301, 
	14, 37, 60, 86, 109, 132, 158, 181, 204, 230, 253, 276, 302, 
	15, 41, 64, 87, 113, 136, 159, 185, 208, 231, 257, 280, 303, 
	16, 39, 65, 88, 111, 137, 160, 183, 209, 232, 255, 281, 304, 
	17, 40, 63, 89, 112, 135, 161, 184, 207, 233, 256, 279, 305, 
	18, 44, 67, 90, 116, 139, 162, 188, 211, 234, 260, 283, 306, 
	19, 42, 68, 91, 114, 140, 163, 186, 212, 235, 258, 284, 307, 
	20, 43, 66, 92, 115, 138, 164, 187, 210, 236, 259, 282, 308, 
	21, 47, 70, 93, 119, 142, 165, 191, 214, 237, 263, 286, 309, 
	22, 45, 71, 94, 117, 143, 166, 189, 215, 238, 261, 287, 310, 
	23, 46, 69, 95, 118, 141, 167, 190, 213, 239, 262, 285, 311
};

const int mapIntelNonlegacy64Qam2[312] = {
	180, 206, 229, 252, 278, 301, 12, 38, 61, 84, 110, 133, 156, 
	181, 204, 230, 253, 276, 302, 13, 36, 62, 85, 108, 134, 157, 
	182, 205, 228, 254, 277, 300, 14, 37, 60, 86, 109, 132, 158, 
	183, 209, 232, 255, 281, 304, 15, 41, 64, 87, 113, 136, 159, 
	184, 207, 233, 256, 279, 305, 16, 39, 65, 88, 111, 137, 160, 
	185, 208, 231, 257, 280, 303, 17, 40, 63, 89, 112, 135, 161, 
	186, 212, 235, 258, 284, 307, 18, 44, 67, 90, 116, 139, 162, 
	187, 210, 236, 259, 282, 308, 19, 42, 68, 91, 114, 140, 163, 
	188, 211, 234, 260, 283, 306, 20, 43, 66, 92, 115, 138, 164, 
	189, 215, 238, 261, 287, 310, 21, 47, 70, 93, 119, 142, 165, 
	190, 213, 239, 262, 285, 311, 22, 45, 71, 94, 117, 143, 166, 
	191, 214, 237, 263, 286, 309, 23, 46, 69, 95, 118, 141, 167, 
	192, 218, 241, 264, 290, 1, 24, 50, 73, 96, 122, 145, 168, 
	193, 216, 242, 265, 288, 2, 25, 48, 74, 97, 120, 146, 169, 
	194, 217, 240, 266, 289, 0, 26, 49, 72, 98, 121, 144, 170, 
	195, 221, 244, 267, 293, 4, 27, 53, 76, 99, 125, 148, 171, 
	196, 219, 245, 268, 291, 5, 28, 51, 77, 100, 123, 149, 172, 
	197, 220, 243, 269, 292, 3, 29, 52, 75, 101, 124, 147, 173, 
	198, 224, 247, 270, 296, 7, 30, 56, 79, 102, 128, 151, 174, 
	199, 222, 248, 271, 294, 8, 31, 54, 80, 103, 126, 152, 175, 
	200, 223, 246, 272, 295, 6, 32, 55, 78, 104, 127, 150, 176, 
	201, 227, 250, 273, 299, 10, 33, 59, 82, 105, 131, 154, 177, 
	202, 225, 251, 274, 297, 11, 34, 57, 83, 106, 129, 155, 178, 
	203, 226, 249, 275, 298, 9, 35, 58, 81, 107, 130, 153, 179
};

const int mapIntelNonlegacy256Qam[416] = {
	0, 35, 66, 97, 128, 163, 194, 225, 256, 291, 322, 353, 384, 
	1, 32, 67, 98, 129, 160, 195, 226, 257, 288, 323, 354, 385, 
	2, 33, 64, 99, 130, 161, 192, 227, 258, 289, 320, 355, 386, 
	3, 34, 65, 96, 131, 162, 193, 224, 259, 290, 321, 352, 387, 
	4, 39, 70, 101, 132, 167, 198, 229, 260, 295, 326, 357, 388, 
	5, 36, 71, 102, 133, 164, 199, 230, 261, 292, 327, 358, 389, 
	6, 37, 68, 103, 134, 165, 196, 231, 262, 293, 324, 359, 390, 
	7, 38, 69, 100, 135, 166, 197, 228, 263, 294, 325, 356, 391, 
	8, 43, 74, 105, 136, 171, 202, 233, 264, 299, 330, 361, 392, 
	9, 40, 75, 106, 137, 168, 203, 234, 265, 296, 331, 362, 393, 
	10, 41, 72, 107, 138, 169, 200, 235, 266, 297, 328, 363, 394, 
	11, 42, 73, 104, 139, 170, 201, 232, 267, 298, 329, 360, 395, 
	12, 47, 78, 109, 140, 175, 206, 237, 268, 303, 334, 365, 396, 
	13, 44, 79, 110, 141, 172, 207, 238, 269, 300, 335, 366, 397, 
	14, 45, 76, 111, 142, 173, 204, 239, 270, 301, 332, 367, 398, 
	15, 46, 77, 108, 143, 174, 205, 236, 271, 302, 333, 364, 399, 
	16, 51, 82, 113, 144, 179, 210, 241, 272, 307, 338, 369, 400, 
	17, 48, 83, 114, 145, 176, 211, 242, 273, 304, 339, 370, 401, 
	18, 49, 80, 115, 146, 177, 208, 243, 274, 305, 336, 371, 402, 
	19, 50, 81, 112, 147, 178, 209, 240, 275, 306, 337, 368, 403, 
	20, 55, 86, 117, 148, 183, 214, 245, 276, 311, 342, 373, 404, 
	21, 52, 87, 118, 149, 180, 215, 246, 277, 308, 343, 374, 405, 
	22, 53, 84, 119, 150, 181, 212, 247, 278, 309, 340, 375, 406, 
	23, 54, 85, 116, 151, 182, 213, 244, 279, 310, 341, 372, 407, 
	24, 59, 90, 121, 152, 187, 218, 249, 280, 315, 346, 377, 408, 
	25, 56, 91, 122, 153, 184, 219, 250, 281, 312, 347, 378, 409, 
	26, 57, 88, 123, 154, 185, 216, 251, 282, 313, 344, 379, 410, 
	27, 58, 89, 120, 155, 186, 217, 248, 283, 314, 345, 376, 411, 
	28, 63, 94, 125, 156, 191, 222, 253, 284, 319, 350, 381, 412, 
	29, 60, 95, 126, 157, 188, 223, 254, 285, 316, 351, 382, 413, 
	30, 61, 92, 127, 158, 189, 220, 255, 286, 317, 348, 383, 414, 
	31, 62, 93, 124, 159, 190, 221, 252, 287, 318, 349, 380, 415
};

const int mapIntelNonlegacy256Qam2[416] = {
	240, 275, 306, 337, 368, 403, 18, 49, 80, 115, 146, 177, 208, 
	241, 272, 307, 338, 369, 400, 19, 50, 81, 112, 147, 178, 209, 
	242, 273, 304, 339, 370, 401, 16, 51, 82, 113, 144, 179, 210, 
	243, 274, 305, 336, 371, 402, 17, 48, 83, 114, 145, 176, 211, 
	244, 279, 310, 341, 372, 407, 22, 53, 84, 119, 150, 181, 212, 
	245, 276, 311, 342, 373, 404, 23, 54, 85, 116, 151, 182, 213, 
	246, 277, 308, 343, 374, 405, 20, 55, 86, 117, 148, 183, 214, 
	247, 278, 309, 340, 375, 406, 21, 52, 87, 118, 149, 180, 215, 
	248, 283, 314, 345, 376, 411, 26, 57, 88, 123, 154, 185, 216, 
	249, 280, 315, 346, 377, 408, 27, 58, 89, 120, 155, 186, 217, 
	250, 281, 312, 347, 378, 409, 24, 59, 90, 121, 152, 187, 218, 
	251, 282, 313, 344, 379, 410, 25, 56, 91, 122, 153, 184, 219, 
	252, 287, 318, 349, 380, 415, 30, 61, 92, 127, 158, 189, 220, 
	253, 284, 319, 350, 381, 412, 31, 62, 93, 124, 159, 190, 221, 
	254, 285, 316, 351, 382, 413, 28, 63, 94, 125, 156, 191, 222, 
	255, 286, 317, 348, 383, 414, 29, 60, 95, 126, 157, 188, 223, 
	256, 291, 322, 353, 384, 3, 34, 65, 96, 131, 162, 193, 224, 
	257, 288, 323, 354, 385, 0, 35, 66, 97, 128, 163, 194, 225, 
	258, 289, 320, 355, 386, 1, 32, 67, 98, 129, 160, 195, 226, 
	259, 290, 321, 352, 387, 2, 33, 64, 99, 130, 161, 192, 227, 
	260, 295, 326, 357, 388, 7, 38, 69, 100, 135, 166, 197, 228, 
	261, 292, 327, 358, 389, 4, 39, 70, 101, 132, 167, 198, 229, 
	262, 293, 324, 359, 390, 5, 36, 71, 102, 133, 164, 199, 230, 
	263, 294, 325, 356, 391, 6, 37, 68, 103, 134, 165, 196, 231, 
	264, 299, 330, 361, 392, 11, 42, 73, 104, 139, 170, 201, 232, 
	265, 296, 331, 362, 393, 8, 43, 74, 105, 136, 171, 202, 233, 
	266, 297, 328, 363, 394, 9, 40, 75, 106, 137, 168, 203, 234, 
	267, 298, 329, 360, 395, 10, 41, 72, 107, 138, 169, 200, 235, 
	268, 303, 334, 365, 396, 15, 46, 77, 108, 143, 174, 205, 236, 
	269, 300, 335, 366, 397, 12, 47, 78, 109, 140, 175, 206, 237, 
	270, 301, 332, 367, 398, 13, 44, 79, 110, 141, 172, 207, 238, 
	271, 302, 333, 364, 399, 14, 45, 76, 111, 142, 173, 204, 239
};

const int mapDeintNonlegacyBpsk[52] = {
	0, 13, 26, 39, 1, 14, 27, 40, 2, 15, 28, 41, 3, 16, 29, 42, 4, 17, 30, 43, 5, 18, 31, 44, 
	6, 19, 32, 45, 7, 20, 33, 46, 8, 21, 34, 47, 9, 22, 35, 48, 10, 23, 36, 49, 11, 24, 37, 50, 
	12, 25, 38, 51
};

const int mapDeintNonlegacyBpsk2[52] = {
	31, 44, 6, 19, 32, 45, 7, 20, 33, 46, 8, 21, 34, 47, 9, 22, 35, 48, 10, 23, 36, 49, 11, 24, 
	37, 50, 12, 25, 38, 51, 0, 13, 26, 39, 1, 14, 27, 40, 2, 15, 28, 41, 3, 16, 29, 42, 4, 17, 
	30, 43, 5, 18
};

const int mapDeintNonlegacyQpsk[104] = {
	0, 13, 26, 39, 52, 65, 78, 91, 1, 14, 27, 40, 53, 66, 79, 92, 2, 15, 28, 41, 54, 67, 80, 93, 
	3, 16, 29, 42, 55, 68, 81, 94, 4, 17, 30, 43, 56, 69, 82, 95, 5, 18, 31, 44, 57, 70, 83, 96, 
	6, 19, 32, 45, 58, 71, 84, 97, 7, 20, 33, 46, 59, 72, 85, 98, 8, 21, 34, 47, 60, 73, 86, 99, 
	9, 22, 35, 48, 61, 74, 87, 100, 10, 23, 36, 49, 62, 75, 88, 101, 11, 24, 37, 50, 63, 76, 89, 
	102, 12, 25, 38, 51, 64, 77, 90, 103
};

 const int mapDeintNonlegacyQpsk2[104] = {
	57, 70, 83, 96, 6, 19, 32, 45, 58, 71, 84, 97, 7, 20, 33, 46, 59, 72, 85, 98, 8, 21, 34, 47, 
	60, 73, 86, 99, 9, 22, 35, 48, 61, 74, 87, 100, 10, 23, 36, 49, 62, 75, 88, 101, 11, 24, 37, 
	50, 63, 76, 89, 102, 12, 25, 38, 51, 64, 77, 90, 103, 0, 13, 26, 39, 52, 65, 78, 91, 1, 14, 
	27, 40, 53, 66, 79, 92, 2, 15, 28, 41, 54, 67, 80, 93, 3, 16, 29, 42, 55, 68, 81, 94, 4, 17, 
	30, 43, 56, 69, 82, 95, 5, 18, 31, 44
};

const int mapDeintNonlegacy16Qam[208] = {
	0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143, 156, 169, 182, 195, 14, 
	1, 40, 27, 66, 53, 92, 79, 118, 105, 144, 131, 170, 157, 196, 183, 
	2, 15, 28, 41, 54, 67, 80, 93, 106, 119, 132, 145, 158, 171, 184, 197, 16, 
	3, 42, 29, 68, 55, 94, 81, 120, 107, 146, 133, 172, 159, 198, 185, 
	4, 17, 30, 43, 56, 69, 82, 95, 108, 121, 134, 147, 160, 173, 186, 199, 18, 
	5, 44, 31, 70, 57, 96, 83, 122, 109, 148, 135, 174, 161, 200, 187, 
	6, 19, 32, 45, 58, 71, 84, 97, 110, 123, 136, 149, 162, 175, 188, 201, 20, 
	7, 46, 33, 72, 59, 98, 85, 124, 111, 150, 137, 176, 163, 202, 189, 
	8, 21, 34, 47, 60, 73, 86, 99, 112, 125, 138, 151, 164, 177, 190, 203, 22, 
	9, 48, 35, 74, 61, 100, 87, 126, 113, 152, 139, 178, 165, 204, 191, 
	10, 23, 36, 49, 62, 75, 88, 101, 114, 127, 140, 153, 166, 179, 192, 205, 24, 
	11, 50, 37, 76, 63, 102, 89, 128, 115, 154, 141, 180, 167, 206, 193, 
	12, 25, 38, 51, 64, 77, 90, 103, 116, 129, 142, 155, 168, 181, 194, 207
};

const int mapDeintNonlegacy16Qam2[208] = {
	122, 109, 148, 135, 174, 161, 200, 187, 6, 19, 32, 45, 58, 71, 84, 97, 110, 
	123, 136, 149, 162, 175, 188, 201, 20, 7, 46, 33, 72, 59, 98, 85, 124, 111, 
	150, 137, 176, 163, 202, 189, 8, 21, 34, 47, 60, 73, 86, 99, 112, 125, 138, 
	151, 164, 177, 190, 203, 22, 9, 48, 35, 74, 61, 100, 87, 126, 113, 152, 139, 
	178, 165, 204, 191, 10, 23, 36, 49, 62, 75, 88, 101, 114, 127, 140, 153, 166, 
	179, 192, 205, 24, 11, 50, 37, 76, 63, 102, 89, 128, 115, 154, 141, 180, 167, 
	206, 193, 12, 25, 38, 51, 64, 77, 90, 103, 116, 129, 142, 155, 168, 181, 194, 
	207, 0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143, 156, 169, 182, 195, 
	14, 1, 40, 27, 66, 53, 92, 79, 118, 105, 144, 131, 170, 157, 196, 183, 2, 15, 
	28, 41, 54, 67, 80, 93, 106, 119, 132, 145, 158, 171, 184, 197, 16, 3, 42, 29, 
	68, 55, 94, 81, 120, 107, 146, 133, 172, 159, 198, 185, 4, 17, 30, 43, 56, 69, 
	82, 95, 108, 121, 134, 147, 160, 173, 186, 199, 18, 5, 44, 31, 70, 57, 96, 83
};

const int mapDeintNonlegacy64Qam[312] = {
	0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143, 156, 169, 182, 195, 208, 221, 234, 247, 260, 273, 286, 299, 14, 27, 
	1, 53, 66, 40, 92, 105, 79, 131, 144, 118, 170, 183, 157, 209, 222, 196, 248, 261, 235, 287, 300, 274, 28, 
	2, 15, 67, 41, 54, 106, 80, 93, 145, 119, 132, 184, 158, 171, 223, 197, 210, 262, 236, 249, 301, 275, 288, 
	3, 16, 29, 42, 55, 68, 81, 94, 107, 120, 133, 146, 159, 172, 185, 198, 211, 224, 237, 250, 263, 276, 289, 302, 17, 30, 
	4, 56, 69, 43, 95, 108, 82, 134, 147, 121, 173, 186, 160, 212, 225, 199, 251, 264, 238, 290, 303, 277, 31, 
	5, 18, 70, 44, 57, 109, 83, 96, 148, 122, 135, 187, 161, 174, 226, 200, 213, 265, 239, 252, 304, 278, 291, 
	6, 19, 32, 45, 58, 71, 84, 97, 110, 123, 136, 149, 162, 175, 188, 201, 214, 227, 240, 253, 266, 279, 292, 305, 20, 33, 
	7, 59, 72, 46, 98, 111, 85, 137, 150, 124, 176, 189, 163, 215, 228, 202, 254, 267, 241, 293, 306, 280, 34, 
	8, 21, 73, 47, 60, 112, 86, 99, 151, 125, 138, 190, 164, 177, 229, 203, 216, 268, 242, 255, 307, 281, 294, 
	9, 22, 35, 48, 61, 74, 87, 100, 113, 126, 139, 152, 165, 178, 191, 204, 217, 230, 243, 256, 269, 282, 295, 308, 23, 36, 
	10, 62, 75, 49, 101, 114, 88, 140, 153, 127, 179, 192, 166, 218, 231, 205, 257, 270, 244, 296, 309, 283, 37, 
	11, 24, 76, 50, 63, 115, 89, 102, 154, 128, 141, 193, 167, 180, 232, 206, 219, 271, 245, 258, 310, 284, 297, 
	12, 25, 38, 51, 64, 77, 90, 103, 116, 129, 142, 155, 168, 181, 194, 207, 220, 233, 246, 259, 272, 285, 298, 311
};

const int mapDeintNonlegacy64Qam2[312] = {
	187, 161, 174, 226, 200, 213, 265, 239, 252, 304, 278, 291, 6, 19, 32, 45, 58, 71, 84, 97, 110, 123, 136, 149, 162, 175, 
	188, 201, 214, 227, 240, 253, 266, 279, 292, 305, 20, 33, 7, 59, 72, 46, 98, 111, 85, 137, 150, 124, 176, 189, 163, 215, 
	228, 202, 254, 267, 241, 293, 306, 280, 34, 8, 21, 73, 47, 60, 112, 86, 99, 151, 125, 138, 190, 164, 177, 229, 203, 216, 
	268, 242, 255, 307, 281, 294, 9, 22, 35, 48, 61, 74, 87, 100, 113, 126, 139, 152, 165, 178, 191, 204, 217, 230, 243, 256, 
	269, 282, 295, 308, 23, 36, 10, 62, 75, 49, 101, 114, 88, 140, 153, 127, 179, 192, 166, 218, 231, 205, 257, 270, 244, 296, 
	309, 283, 37, 11, 24, 76, 50, 63, 115, 89, 102, 154, 128, 141, 193, 167, 180, 232, 206, 219, 271, 245, 258, 310, 284, 297, 
	12, 25, 38, 51, 64, 77, 90, 103, 116, 129, 142, 155, 168, 181, 194, 207, 220, 233, 246, 259, 272, 285, 298, 311, 0, 13, 
	26, 39, 52, 65, 78, 91, 104, 117, 130, 143, 156, 169, 182, 195, 208, 221, 234, 247, 260, 273, 286, 299, 14, 27, 1, 53, 66, 
	40, 92, 105, 79, 131, 144, 118, 170, 183, 157, 209, 222, 196, 248, 261, 235, 287, 300, 274, 28, 2, 15, 67, 41, 54, 106, 80, 
	93, 145, 119, 132, 184, 158, 171, 223, 197, 210, 262, 236, 249, 301, 275, 288, 3, 16, 29, 42, 55, 68, 81, 94, 107, 120, 133, 
	146, 159, 172, 185, 198, 211, 224, 237, 250, 263, 276, 289, 302, 17, 30, 4, 56, 69, 43, 95, 108, 82, 134, 147, 121, 173, 186, 
	160, 212, 225, 199, 251, 264, 238, 290, 303, 277, 31, 5, 18, 70, 44, 57, 109, 83, 96, 148, 122, 135
};

const int mapDeintNonlegacy256Qam[416] = {
	0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143, 156, 169, 182, 195, 208, 221, 234, 247, 260, 273, 286, 299, 312, 325, 338, 351, 364, 377, 390, 403, 14, 27, 40, 
	1, 66, 79, 92, 53, 118, 131, 144, 105, 170, 183, 196, 157, 222, 235, 248, 209, 274, 287, 300, 261, 326, 339, 352, 313, 378, 391, 404, 365, 28, 41, 
	2, 15, 80, 93, 54, 67, 132, 145, 106, 119, 184, 197, 158, 171, 236, 249, 210, 223, 288, 301, 262, 275, 340, 353, 314, 327, 392, 405, 366, 379, 42, 
	3, 16, 29, 94, 55, 68, 81, 146, 107, 120, 133, 198, 159, 172, 185, 250, 211, 224, 237, 302, 263, 276, 289, 354, 315, 328, 341, 406, 367, 380, 393, 
	4, 17, 30, 43, 56, 69, 82, 95, 108, 121, 134, 147, 160, 173, 186, 199, 212, 225, 238, 251, 264, 277, 290, 303, 316, 329, 342, 355, 368, 381, 394, 407, 18, 31, 44, 
	5, 70, 83, 96, 57, 122, 135, 148, 109, 174, 187, 200, 161, 226, 239, 252, 213, 278, 291, 304, 265, 330, 343, 356, 317, 382, 395, 408, 369, 32, 45, 
	6, 19, 84, 97, 58, 71, 136, 149, 110, 123, 188, 201, 162, 175, 240, 253, 214, 227, 292, 305, 266, 279, 344, 357, 318, 331, 396, 409, 370, 383, 46, 
	7, 20, 33, 98, 59, 72, 85, 150, 111, 124, 137, 202, 163, 176, 189, 254, 215, 228, 241, 306, 267, 280, 293, 358, 319, 332, 345, 410, 371, 384, 397, 
	8, 21, 34, 47, 60, 73, 86, 99, 112, 125, 138, 151, 164, 177, 190, 203, 216, 229, 242, 255, 268, 281, 294, 307, 320, 333, 346, 359, 372, 385, 398, 411, 22, 35, 48, 
	9, 74, 87, 100, 61, 126, 139, 152, 113, 178, 191, 204, 165, 230, 243, 256, 217, 282, 295, 308, 269, 334, 347, 360, 321, 386, 399, 412, 373, 36, 49, 
	10, 23, 88, 101, 62, 75, 140, 153, 114, 127, 192, 205, 166, 179, 244, 257, 218, 231, 296, 309, 270, 283, 348, 361, 322, 335, 400, 413, 374, 387, 50, 
	11, 24, 37, 102, 63, 76, 89, 154, 115, 128, 141, 206, 167, 180, 193, 258, 219, 232, 245, 310, 271, 284, 297, 362, 323, 336, 349, 414, 375, 388, 401, 
	12, 25, 38, 51, 64, 77, 90, 103, 116, 129, 142, 155, 168, 181, 194, 207, 220, 233, 246, 259, 272, 285, 298, 311, 324, 337, 350, 363, 376, 389, 402, 415
};

const int mapDeintNonlegacy256Qam2[416] = {
	226, 239, 252, 213, 278, 291, 304, 265, 330, 343, 356, 317, 382, 395, 408, 369, 32, 45, 6, 19, 84, 97, 58, 71, 136, 149, 110, 123, 188, 201, 162, 175, 
	240, 253, 214, 227, 292, 305, 266, 279, 344, 357, 318, 331, 396, 409, 370, 383, 46, 7, 20, 33, 98, 59, 72, 85, 150, 111, 124, 137, 202, 163, 176, 189, 
	254, 215, 228, 241, 306, 267, 280, 293, 358, 319, 332, 345, 410, 371, 384, 397, 8, 21, 34, 47, 60, 73, 86, 99, 112, 125, 138, 151, 164, 177, 190, 203, 
	216, 229, 242, 255, 268, 281, 294, 307, 320, 333, 346, 359, 372, 385, 398, 411, 22, 35, 48, 9, 74, 87, 100, 61, 126, 139, 152, 113, 178, 191, 204, 165, 
	230, 243, 256, 217, 282, 295, 308, 269, 334, 347, 360, 321, 386, 399, 412, 373, 36, 49, 10, 23, 88, 101, 62, 75, 140, 153, 114, 127, 192, 205, 166, 179, 
	244, 257, 218, 231, 296, 309, 270, 283, 348, 361, 322, 335, 400, 413, 374, 387, 50, 11, 24, 37, 102, 63, 76, 89, 154, 115, 128, 141, 206, 167, 180, 193, 
	258, 219, 232, 245, 310, 271, 284, 297, 362, 323, 336, 349, 414, 375, 388, 401, 12, 25, 38, 51, 64, 77, 90, 103, 116, 129, 142, 155, 168, 181, 194, 207, 
	220, 233, 246, 259, 272, 285, 298, 311, 324, 337, 350, 363, 376, 389, 402, 415, 0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143, 156, 169, 182, 195, 
	208, 221, 234, 247, 260, 273, 286, 299, 312, 325, 338, 351, 364, 377, 390, 403, 14, 27, 40, 1, 66, 79, 92, 53, 118, 131, 144, 105, 170, 183, 196, 157, 
	222, 235, 248, 209, 274, 287, 300, 261, 326, 339, 352, 313, 378, 391, 404, 365, 28, 41, 2, 15, 80, 93, 54, 67, 132, 145, 106, 119, 184, 197, 158, 171, 
	236, 249, 210, 223, 288, 301, 262, 275, 340, 353, 314, 327, 392, 405, 366, 379, 42, 3, 16, 29, 94, 55, 68, 81, 146, 107, 120, 133, 198, 159, 172, 185, 
	250, 211, 224, 237, 302, 263, 276, 289, 354, 315, 328, 341, 406, 367, 380, 393, 4, 17, 30, 43, 56, 69, 82, 95, 108, 121, 134, 147, 160, 173, 186, 199, 
	212, 225, 238, 251, 264, 277, 290, 303, 316, 329, 342, 355, 368, 381, 394, 407, 18, 31, 44, 5, 70, 83, 96, 57, 122, 135, 148, 109, 174, 187, 200, 161
};

void procDeintLegacyBpsk(float* inBits, float* outBits)
{
    for(int i=0;i<48;i++)
    {
        outBits[mapDeintLegacyBpsk[i]] = inBits[i];
    }
}

void procIntelLegacyBpsk(uint8_t* inBits, uint8_t* outBits)
{
	for(int i=0;i<48;i++)
    {
        outBits[mapIntelLegacyBpsk[i]] = inBits[i];
    }
}

void procIntelVhtB20(uint8_t* inBits, uint8_t* outBits)
{
	for(int i=0;i<52;i++)
    {
        outBits[mapIntelVhtSigB20[i]] = inBits[i];
    }
}

const int SV_PUNC_12[2] = {1, 1};
const int SV_PUNC_23[4] = {1, 1, 1, 0};
const int SV_PUNC_34[6] = {1, 1, 1, 0, 0, 1};
const int SV_PUNC_56[10] = {1, 1, 1, 0, 0, 1, 1, 0, 0, 1};


// viterbi, next state of each state with S1 = 0 and 1
const int SV_STATE_NEXT[64][2] =
{
 { 0, 32}, { 0, 32}, { 1, 33}, { 1, 33}, { 2, 34}, { 2, 34}, { 3, 35}, { 3, 35},
 { 4, 36}, { 4, 36}, { 5, 37}, { 5, 37}, { 6, 38}, { 6, 38}, { 7, 39}, { 7, 39},
 { 8, 40}, { 8, 40}, { 9, 41}, { 9, 41}, {10, 42}, {10, 42}, {11, 43}, {11, 43},
 {12, 44}, {12, 44}, {13, 45}, {13, 45}, {14, 46}, {14, 46}, {15, 47}, {15, 47},
 {16, 48}, {16, 48}, {17, 49}, {17, 49}, {18, 50}, {18, 50}, {19, 51}, {19, 51},
 {20, 52}, {20, 52}, {21, 53}, {21, 53}, {22, 54}, {22, 54}, {23, 55}, {23, 55},
 {24, 56}, {24, 56}, {25, 57}, {25, 57}, {26, 58}, {26, 58}, {27, 59}, {27, 59},
 {28, 60}, {28, 60}, {29, 61}, {29, 61}, {30, 62}, {30, 62}, {31, 63}, {31, 63}
};

// viterbi, output coded 2 bits of each state with S1 = 0 and 1
const int SV_STATE_OUTPUT[64][2] =
{
 {0, 3}, {3, 0}, {2, 1}, {1, 2}, {0, 3}, {3, 0}, {2, 1}, {1, 2},
 {3, 0}, {0, 3}, {1, 2}, {2, 1}, {3, 0}, {0, 3}, {1, 2}, {2, 1},
 {3, 0}, {0, 3}, {1, 2}, {2, 1}, {3, 0}, {0, 3}, {1, 2}, {2, 1},
 {0, 3}, {3, 0}, {2, 1}, {1, 2}, {0, 3}, {3, 0}, {2, 1}, {1, 2},
 {1, 2}, {2, 1}, {3, 0}, {0, 3}, {1, 2}, {2, 1}, {3, 0}, {0, 3},
 {2, 1}, {1, 2}, {0, 3}, {3, 0}, {2, 1}, {1, 2}, {0, 3}, {3, 0},
 {2, 1}, {1, 2}, {0, 3}, {3, 0}, {2, 1}, {1, 2}, {0, 3}, {3, 0},
 {1, 2}, {2, 1}, {3, 0}, {0, 3}, {1, 2}, {2, 1}, {3, 0}, {0, 3},
};

// viterbi, soft decoding
void SV_Decode_Sig(float* llrv, uint8_t* decoded_bits, int trellisLen)
{
	int i, j, t;

	/* accumulated error metirc */
	float accum_err_metric0[64];
	float accum_err_metric1[64];
	float *tmp, *pre_accum_err_metric, *cur_accum_err_metric;
	int *state_history[64];			/* state history table */
	int *state_sequence; 					/* state sequence list */
	int op0, op1, next0, next1;
	float acc_tmp0, acc_tmp1, t0, t1;
	float tbl_t[4];

	/* allocate memory for state tables */
	for (i = 0; i < 64; i++)
		state_history[i] = (int*)malloc((trellisLen+1) * sizeof(int));

	state_sequence = (int*)malloc((trellisLen+1) * sizeof(int));

	/* initialize data structures */
	for (i = 0; i < 64; i++)
	{
		for (j = 0; j <= trellisLen; j++)
			state_history[i][j] = 0;

		/* initial the accumulated error metrics */
		accum_err_metric0[i] = -1000000000000000.0f;
		accum_err_metric1[i] = -1000000000000000.0f;
	}
    accum_err_metric0[0] = 0;
    cur_accum_err_metric = &accum_err_metric1[0];
    pre_accum_err_metric = &accum_err_metric0[0];

	/* start viterbi decoding */
	for (t=0; t<trellisLen; t++)
	{
		t0 = *llrv++;
		t1 = *llrv++;

		tbl_t[0] = 0;
		tbl_t[1] = t1;
		tbl_t[2] = t0;
		tbl_t[3] = t1+t0;

		/* repeat for each possible state */
		for (i = 0; i < 64; i++)
		{
			op0 = SV_STATE_OUTPUT[i][0];
			op1 = SV_STATE_OUTPUT[i][1];

			acc_tmp0 = pre_accum_err_metric[i] + tbl_t[op0];
			acc_tmp1 = pre_accum_err_metric[i] + tbl_t[op1];

			next0 = SV_STATE_NEXT[i][0];
			next1 = SV_STATE_NEXT[i][1];

			if (acc_tmp0 > cur_accum_err_metric[next0])
			{
				cur_accum_err_metric[next0] = acc_tmp0;
				state_history[next0][t+1] = i;			//path
			}

			if (acc_tmp1 > cur_accum_err_metric[next1])
			{
				cur_accum_err_metric[next1] = acc_tmp1;
				state_history[next1][t+1] = i;
			}
		}

		/* update accum_err_metric array */
		tmp = pre_accum_err_metric;
		pre_accum_err_metric = cur_accum_err_metric;
		cur_accum_err_metric = tmp;

		for (i = 0; i < 64; i++)
		{
			cur_accum_err_metric[i] = -1000000000000000.0f;
		}
	} // end of t loop

    // The final state should be 0
    state_sequence[trellisLen] = 0;

    for (j = trellisLen; j > 0; j--)
	{
        state_sequence[j-1] = state_history[state_sequence[j]][j];
	}
    
	//memset(decoded_bits, 0, trellisLen * sizeof(int));

	for (j = 0; j < trellisLen; j++)
	{
		if (state_sequence[j+1] == SV_STATE_NEXT[state_sequence[j]][1])
		{
			decoded_bits[j] = 1;
		}
        else
        {
            decoded_bits[j] = 0;
        }
	}

	for (i = 0; i < 64; i++)
	{
		free(state_history[i]);
	}

	free(state_sequence);
}

void procSymQamToLlr(gr_complex* inQam, float* outLlr, c8p_mod* mod)
{
	if(mod->mod == C8P_QAM_BPSK)
	{
		for(int i=0;i<mod->nSD;i++)
		{
			outLlr[i] = inQam[i].real();
		}
	}
	else if(mod->mod == C8P_QAM_QPSK)
	{
		for(int i=0;i<mod->nSD;i++)
		{
			inQam[i] *= 1.4142135623730951f;
			outLlr[i*2] = inQam[i].real();
			outLlr[i*2+1] = inQam[i].imag();
		}
	}
	else if(mod->mod == C8P_QAM_16QAM)
	{
		for(int i=0;i<mod->nSD;i++)
		{
			inQam[i] *= 3.1622776601683795f;
			outLlr[i*4] = inQam[i].real();
			outLlr[i*4+1] = 2.0f - std::abs(inQam[i].real());
			outLlr[i*4+2] = inQam[i].imag();
			outLlr[i*4+3] = 2.0f - std::abs(inQam[i].imag());
		}
	}
	else if(mod->mod == C8P_QAM_64QAM)
	{
		for(int i=0;i<mod->nSD;i++)
		{
			inQam[i] *= 6.48074069840786f;
			outLlr[i*6] = inQam[i].real();
			outLlr[i*6+1] = 4.0f - std::abs(inQam[i].real());
			outLlr[i*6+2] = 2 - std::abs(4.0f - std::abs(inQam[i].real()));
			outLlr[i*6+3] = inQam[i].imag();
			outLlr[i*6+4] = 4.0f - std::abs(inQam[i].imag());
			outLlr[i*6+5] = 2 - std::abs(4.0f - std::abs(inQam[i].imag()));
		}
	}
	else if(mod->mod == C8P_QAM_256QAM)
	{
		for(int i=0;i<mod->nSD;i++)
		{
			inQam[i] *= 13.038404810405298f;
			outLlr[i*8] = inQam[i].real();
			outLlr[i*8+1] = 8.0f - std::abs(inQam[i].real());
			outLlr[i*8+2] = 4 - std::abs(8.0f - std::abs(inQam[i].real()));
			outLlr[i*8+3] = 2 - std::abs(4 - std::abs(8.0f - std::abs(inQam[i].real())));
			outLlr[i*8+4] = inQam[i].imag();
			outLlr[i*8+5] = 8.0f - std::abs(inQam[i].imag());
			outLlr[i*8+6] = 4 - std::abs(8.0f - std::abs(inQam[i].imag()));
			outLlr[i*8+7] = 2 - std::abs(4 - std::abs(8.0f - std::abs(inQam[i].imag())));

		}
	}
}

void procSymDeintL2(float* in, float* out, c8p_mod* mod)
{
	// this version follows standard
	switch(mod->nCBPS)
	{
		case 48:
		{
			for(int i=0; i<48; i++)
			{
				out[mapDeintLegacyBpsk[i]] = in[i];
			}
			return;
		}
		case 96:
		{
			for(int i=0; i<96; i++)
			{
				out[mapDeintLegacyQpsk[i]] = in[i];
			}
			return;
		}
		case 192:
		{
			for(int i=0; i<192; i++)
			{
				out[mapDeintLegacy16Qam[i]] = in[i];
			}
			return;
		}
		case 288:
		{
			for(int i=0; i<288; i++)
			{
				out[mapDeintLegacy64Qam[i]] = in[i];
			}
			return;
		}
		default:
		{
			return;
		}
	}
}

void procSymIntelL2(uint8_t* in, uint8_t* out, c8p_mod* mod)
{
	// this version follows standard
	switch(mod->nCBPS)
	{
		case 48:
		{
			for(int i=0; i<48; i++)
			{
				out[mapIntelLegacyBpsk[i]] = in[i];
			}
			return;
		}
		case 96:
		{
			for(int i=0; i<96; i++)
			{
				out[mapIntelLegacyQpsk[i]] = in[i];
			}
			return;
		}
		case 192:
		{
			for(int i=0; i<192; i++)
			{
				out[mapIntelLegacy16Qam[i]] = in[i];
			}
			return;
		}
		case 288:
		{
			for(int i=0; i<288; i++)
			{
				out[mapIntelLegacy64Qam[i]] = in[i];
			}
			return;
		}
		default:
		{
			return;
		}
	}
}

void procSymDeintNL2SS1(float* in, float* out, c8p_mod* mod)
{
	switch(mod->nCBPSS)
	{
		case 52:
		{
			for(int i=0; i<52; i++)
			{
				out[mapDeintNonlegacyBpsk[i]] = in[i];
			}
			return;
		}
		case 104:
		{
			for(int i=0; i<104; i++)
			{
				out[mapDeintNonlegacyQpsk[i]] = in[i];
			}
			return;
		}
		case 208:
		{
			for(int i=0; i<208; i++)
			{
				out[mapDeintNonlegacy16Qam[i]] = in[i];
			}
			return;
		}
		case 312:
		{
			for(int i=0; i<312; i++)
			{
				out[mapDeintNonlegacy64Qam[i]] = in[i];
			}
			return;
		}
		case 416:
		{
			for(int i=0; i<416; i++)
			{
				out[mapDeintNonlegacy256Qam[i]] = in[i];
			}
			return;
		}
		default:
		{
			return;
		}
	}
}

void procSymDeintNL2SS2(float* in, float* out, c8p_mod* mod)
{
	switch(mod->nCBPSS)
	{
		case 52:
		{
			for(int i=0; i<52; i++)
			{
				out[mapDeintNonlegacyBpsk2[i]] = in[i];
			}
			return;
		}
		case 104:
		{
			for(int i=0; i<104; i++)
			{
				out[mapDeintNonlegacyQpsk2[i]] = in[i];
			}
			return;
		}
		case 208:
		{
			for(int i=0; i<208; i++)
			{
				out[mapDeintNonlegacy16Qam2[i]] = in[i];
			}
			return;
		}
		case 312:
		{
			for(int i=0; i<312; i++)
			{
				out[mapDeintNonlegacy64Qam2[i]] = in[i];
			}
			return;
		}
		case 416:
		{
			for(int i=0; i<416; i++)
			{
				out[mapDeintNonlegacy256Qam2[i]] = in[i];
			}
			return;
		}
		default:
		{
			return;
		}
	}
}

void procSymIntelNL2SS1(uint8_t* in, uint8_t* out, c8p_mod* mod)
{
	switch(mod->nCBPSS)
	{
		case 52:
		{
			for(int i=0; i<52; i++)
			{
				out[mapIntelNonlegacyBpsk[i]] = in[i];
			}
			return;
		}
		case 104:
		{
			for(int i=0; i<104; i++)
			{
				out[mapIntelNonlegacyQpsk[i]] = in[i];
			}
			return;
		}
		case 208:
		{
			for(int i=0; i<208; i++)
			{
				out[mapIntelNonlegacy16Qam[i]] = in[i];
			}
			return;
		}
		case 312:
		{
			for(int i=0; i<312; i++)
			{
				out[mapIntelNonlegacy64Qam[i]] = in[i];
			}
			return;
		}
		case 416:
		{
			for(int i=0; i<416; i++)
			{
				out[mapIntelNonlegacy256Qam[i]] = in[i];
			}
			return;
		}
		default:
		{
			return;
		}
	}
}

void procSymIntelNL2SS2(uint8_t* in, uint8_t* out, c8p_mod* mod)
{
	switch(mod->nCBPSS)
	{
		case 52:
		{
			for(int i=0; i<52; i++)
			{
				out[mapIntelNonlegacyBpsk2[i]] = in[i];
			}
			return;
		}
		case 104:
		{
			for(int i=0; i<104; i++)
			{
				out[mapIntelNonlegacyQpsk2[i]] = in[i];
			}
			return;
		}
		case 208:
		{
			for(int i=0; i<208; i++)
			{
				out[mapIntelNonlegacy16Qam2[i]] = in[i];
			}
			return;
		}
		case 312:
		{
			for(int i=0; i<312; i++)
			{
				out[mapIntelNonlegacy64Qam2[i]] = in[i];
			}
			return;
		}
		case 416:
		{
			for(int i=0; i<416; i++)
			{
				out[mapIntelNonlegacy256Qam2[i]] = in[i];
			}
			return;
		}
		default:
		{
			return;
		}
	}
}

void procSymDepasNL(float in[C8P_MAX_N_SS][C8P_MAX_N_CBPSS], float* out, c8p_mod* mod)
{
	// this ver only for 2 ss
	int s = std::max(mod->nBPSCS/2, 1);
	for(int i=0; i<int(mod->nCBPSS/s); i++)
	{
		memcpy(&out[i*2*s], &in[0][i*s], sizeof(float)*s);
		memcpy(&out[(i*2+1)*s], &in[1][i*s], sizeof(float)*s);
	}
}

int nCodedToUncoded(int nCoded, c8p_mod* mod)
{
	switch(mod->cr)
	{
		case C8P_CR_12:
			return (nCoded/2);
		case C8P_CR_23:
			return (nCoded * 2 / 3);
		case C8P_CR_34:
			return (nCoded * 3 / 4);
		case C8P_CR_56:
			return (nCoded * 5 / 6);
		default:
			return 0;
	}
}

int nUncodedToCoded(int nUncoded, c8p_mod* mod)
{
	switch(mod->cr)
	{
		case C8P_CR_12:
			return (nUncoded * 2);
		case C8P_CR_23:
			return (nUncoded * 3 / 2);
		case C8P_CR_34:
			return (nUncoded * 4 / 3);
		case C8P_CR_56:
			return (nUncoded * 6 / 5);
		default:
			return 0;
	}
}

void formatToModSu(c8p_mod* mod, int format, int mcs, int nss, int len)
{
	// not supporting other bandwidth and short GI in this version
	if(format == C8P_F_L)
	{
		signalParserL(mcs, len, mod);
	}
	else if(format == C8P_F_VHT)
	{
		mod->format = C8P_F_VHT;
		mod->nSS = nss;
		mod->len = len;
		modParserVht(mcs, mod);
		mod->nSymSamp = 80;
		mod->ampdu = 1;
		mod->sumu = 0;
		if(len > 0)
		{
			mod->nSym = (mod->len*8 + 22) / mod->nDBPS + (((mod->len*8 + 22) % mod->nDBPS) != 0);
		}
		else
		{
			mod->nSym = 0;		// NDP
		}
		
	}
	else
	{
		mod->format = C8P_F_HT;
		mod->nSS = nss;
		mod->len = len;
		modParserHt(mcs, mod);
		mod->nSymSamp = 80;
		mod->ampdu = 0;
		mod->sumu = 0;
		mod->nSym = ((mod->len*8 + 22)/mod->nDBPS + (((mod->len*8 + 22)%mod->nDBPS) != 0));
	}
}

void vhtModMuToSu(c8p_mod* mod, int pos)
{
	mod->mcs = mod->mcsMu[pos];
	mod->len = mod->lenMu[pos];
	mod->mod = mod->modMu[pos];
	mod->cr = mod->crMu[pos];
	mod->nBPSCS = mod->nBPSCSMu[pos];
	mod->nDBPS = mod->nDBPSMu[pos];
	mod->nCBPS = mod->nCBPSMu[pos];
	mod->nCBPSS = mod->nCBPSSMu[pos];

	mod->nIntCol = mod->nIntColMu[pos];
	mod->nIntRow = mod->nIntRowMu[pos];
	mod->nIntRot = mod->nIntRotMu[pos];
}

void vhtModSuToMu(c8p_mod* mod, int pos)
{
	mod->mcsMu[pos] = mod->mcs;
	mod->lenMu[pos] = mod->len;
	mod->modMu[pos] = mod->mod;
	mod->crMu[pos] = mod->cr;
	mod->nBPSCSMu[pos] = mod->nBPSCS;
	mod->nDBPSMu[pos] = mod->nDBPS;
	mod->nCBPSMu[pos] = mod->nCBPS;
	mod->nCBPSSMu[pos] = mod->nCBPSS;

	mod->nIntColMu[pos] = mod->nIntCol;
	mod->nIntRowMu[pos] = mod->nIntRow;
	mod->nIntRotMu[pos] = mod->nIntRot;
}

void formatToModMu(c8p_mod* mod, int mcs0, int nSS0, int len0, int mcs1, int nSS1, int len1)
{
	mod->format = C8P_F_VHT;
	mod->sumu = 1;
	mod->ampdu = 1;
	mod->nSymSamp = 80;
	
	mod->nSS = nSS0;
	mod->len = len0;
	modParserVht(mcs0, mod);
	int tmpNSym0 = (mod->len*8 + 22) / mod->nDBPS + (((mod->len*8 + 22) % mod->nDBPS) != 0);
	vhtModSuToMu(mod, 0);

	mod->nSS = nSS1;
	mod->len = len1;
	modParserVht(mcs1, mod);
	int tmpNSym1 = (mod->len*8 + 22) / mod->nDBPS + (((mod->len*8 + 22) % mod->nDBPS) != 0);
	vhtModSuToMu(mod, 1);

	mod->nSym = std::max(tmpNSym0, tmpNSym1);

	// current only 
	mod->nSS = 2;
	mod->nSD = 52;
	mod->nSP = 4;
	mod->nLTF = 2;
}


bool formatCheck(int format, int mcs, int nss)
{
	// to be added

	return true;
}

void scramEncoder(uint8_t* inBits, uint8_t* outBits, int len, int init)
{
	int tmpState = init;
    int tmpFb;

	for(int i=0;i<len;i++)
	{
		tmpFb = (!!(tmpState & 64)) ^ (!!(tmpState & 8));
        outBits[i] = tmpFb ^ inBits[i];
        tmpState = ((tmpState << 1) & 0x7e) | tmpFb;
	}
}

void bccEncoder(uint8_t* inBits, uint8_t* outBits, int len)
{

    int state = 0;
	int count = 0;
    for (int i = 0; i < len; i++) {
        state = ((state << 1) & 0x7e) | inBits[i];
		count = 0;
		for(int j=0;j<7;j++)
		{
			if((state & 0155) & (1 << j))
				count++;
		}
        outBits[i * 2] = count % 2;
		count = 0;
		for(int j=0;j<7;j++)
		{
			if((state & 0117) & (1 << j))
				count++;
		}
        outBits[i * 2 + 1] = count % 2;
    }
}

void punctEncoder(uint8_t* inBits, uint8_t* outBits, int len, c8p_mod* mod)
{
	int tmpPunctIndex = 0;
	if(mod->cr == C8P_CR_12)
	{
		memcpy(outBits, inBits, len);
	}
	else if(mod->cr == C8P_CR_23)
	{
		for(int i=0;i<len;i++)
		{
			if(SV_PUNC_23[i%4])
			{
				outBits[tmpPunctIndex] = inBits[i];
				tmpPunctIndex++;
			}
		}
	}
	else if(mod->cr == C8P_CR_34)
	{
		for(int i=0;i<len;i++)
		{
			if(SV_PUNC_34[i%6])
			{
				outBits[tmpPunctIndex] = inBits[i];
				tmpPunctIndex++;
			}
		}
	}
	else
	{
		for(int i=0;i<len;i++)
		{
			if(SV_PUNC_56[i%10])
			{
				outBits[tmpPunctIndex] = inBits[i];
				tmpPunctIndex++;
			}
		}
	}
}

void streamParser2(uint8_t* inBits, uint8_t* outBits1, uint8_t* outBits2, int len, c8p_mod* mod)
{
	int s = std::max(mod->nBPSCS/2, 1);
	uint8_t* tmpInP = inBits;
	uint8_t* tmpOutP1 = outBits1;
	uint8_t* tmpOutP2 = outBits2;
	for(int i=0;i<(len/2/s);i++)
	{
		memcpy(tmpOutP1, tmpInP, s);
		tmpOutP1 += s;
		tmpInP += s;
		memcpy(tmpOutP2, tmpInP, s);
		tmpOutP2 += s;
		tmpInP += s;
	}
}

void bitsToChips(uint8_t* inBits, uint8_t* outChips, c8p_mod* mod)
{
	int tmpBitIndex = 0;
	int tmpChipIndex = 0;
	int i, tmpChipLen, tmpChipNum;

	switch(mod->mod)
	{
		case C8P_QAM_BPSK:
			tmpChipLen = 1;
			break;
		case C8P_QAM_QPSK:
			tmpChipLen = 2;
			break;
		case C8P_QAM_16QAM:
			tmpChipLen = 4;
			break;
		case C8P_QAM_64QAM:
			tmpChipLen = 6;
			break;
		case C8P_QAM_256QAM:
			tmpChipLen = 8;
			break;
		default:
			tmpChipLen = 1;
			break;
	}

	tmpChipNum = (mod->nSym * mod->nCBPSS) / tmpChipLen;

	while(tmpChipIndex < tmpChipNum)
	{
		outChips[tmpChipIndex] = 0;
		for(i=0;i<tmpChipLen;i++)
		{
			outChips[tmpChipIndex] |= (inBits[tmpBitIndex] << i);
			tmpBitIndex++;
		}
		tmpChipIndex++;
	}
	
}

void procChipsToQam(const uint8_t* inChips,  gr_complex* outQam, int qamType, int len)
{
	if(qamType == C8P_QAM_BPSK)
	{
		for(int i=0;i<len;i++)
		{
			outQam[i] = C8P_QAM_TAB_BPSK[inChips[i]];
		}
	}
	else if(qamType == C8P_QAM_QBPSK)
	{
		for(int i=0;i<len;i++)
		{
			outQam[i] = C8P_QAM_TAB_QBPSK[inChips[i]];
		}
	}
	else if(qamType == C8P_QAM_QPSK)
	{
		for(int i=0;i<len;i++)
		{
			outQam[i] = C8P_QAM_TAB_QPSK[inChips[i]];
		}
	}
	else if(qamType == C8P_QAM_16QAM)
	{
		for(int i=0;i<len;i++)
		{
			outQam[i] = C8P_QAM_TAB_16QAM[inChips[i]];
		}
	}
	else if(qamType == C8P_QAM_64QAM)
	{
		for(int i=0;i<len;i++)
		{
			outQam[i] = C8P_QAM_TAB_64QAM[inChips[i]];
		}
	}
	else if(qamType == C8P_QAM_256QAM)
	{
		for(int i=0;i<len;i++)
		{
			outQam[i] = C8P_QAM_TAB_256QAM[inChips[i]];
		}
	}
	else
	{
		std::cout<<"ieee80211 procQam qam type error."<<std::endl;
	}
}

void procInsertPilotsDc(gr_complex* sigIn, gr_complex* sigOut, gr_complex* pilots, int format)
{
	if(format == C8P_F_L)
	{
		memcpy(&sigOut[0], &sigIn[0], 5*sizeof(gr_complex));
		sigOut[5] = pilots[0];
		memcpy(&sigOut[6], &sigIn[5], 13*sizeof(gr_complex));
		sigOut[19] = pilots[1];
		memcpy(&sigOut[20], &sigIn[18], 6*sizeof(gr_complex));
		sigOut[26] = gr_complex(0.0f, 0.0f);
		memcpy(&sigOut[27], &sigIn[24], 6*sizeof(gr_complex));
		sigOut[33] = pilots[2];
		memcpy(&sigOut[34], &sigIn[30], 13*sizeof(gr_complex));
		sigOut[47] = pilots[3];
		memcpy(&sigOut[48], &sigIn[43], 5*sizeof(gr_complex));
	}
	else
	{
		memcpy(&sigOut[0], &sigIn[0], 7*sizeof(gr_complex));
		sigOut[7] = pilots[0];
		memcpy(&sigOut[8], &sigIn[7], 13*sizeof(gr_complex));
		sigOut[21] = pilots[1];
		memcpy(&sigOut[22], &sigIn[20], 6*sizeof(gr_complex));
		sigOut[28] = gr_complex(0.0f, 0.0f);
		memcpy(&sigOut[29], &sigIn[26], 6*sizeof(gr_complex));
		sigOut[35] = pilots[2];
		memcpy(&sigOut[36], &sigIn[32], 13*sizeof(gr_complex));
		sigOut[49] = pilots[3];
		memcpy(&sigOut[50], &sigIn[45], 7*sizeof(gr_complex));
	}
}

void procNonDataSc(gr_complex* sigIn, gr_complex* sigOut, int format)
{
	if(format == C8P_F_L)
	{
		// memcpy(&sigOut[0], &sigIn[26], 27*sizeof(gr_complex));
		// memset(&sigOut[27], 0, 11*sizeof(gr_complex));
		// memcpy(&sigOut[38], &sigIn[0], 26*sizeof(gr_complex));

		memset((uint8_t*)&sigOut[0],  0, 6*sizeof(gr_complex));
		memcpy((uint8_t*)&sigOut[6], &sigIn[0], 53*sizeof(gr_complex));
		memset((uint8_t*)&sigOut[59], 0, 5*sizeof(gr_complex));
	}
	else
	{
		// memcpy(&sigOut[0], &sigIn[28], 29*sizeof(gr_complex));
		// memset(&sigOut[29], 0, 7*sizeof(gr_complex));
		// memcpy(&sigOut[36], &sigIn[0], 28*sizeof(gr_complex));
		memset((uint8_t*)&sigOut[0],  0, 4*sizeof(gr_complex));
		memcpy((uint8_t*)&sigOut[4], &sigIn[0], 57*sizeof(gr_complex));
		memset((uint8_t*)&sigOut[61], 0, 3*sizeof(gr_complex));
	}
}

void procCSD(gr_complex* sig, int cycShift)
{
	gr_complex tmpStep = gr_complex(0.0f, -2.0f) * (float)M_PI * (float)cycShift * 20.0f * 0.001f;
	for(int i=0;i<64;i++)
	{
		sig[i] = sig[i] * std::exp( tmpStep * (float)(i - 32) / 64.0f);
	}
}

void procToneScaling(gr_complex* sig, int ntf, int nss, int len)
{
	for(int i=0;i<len;i++)
	{
		sig[i] = sig[i] / sqrtf((float)ntf * (float)nss) / 3.55555f;
	}
}

void procNss2SymBfQ(gr_complex* sig0, gr_complex* sig1, gr_complex* bfQ)
{
	gr_complex tmpOut0, tmpOut1;
	for(int i=0;i<64;i++)
	{
		tmpOut0 = sig0[i] * bfQ[i*4 + 0] + sig1[i] * bfQ[i*4 + 1];
		tmpOut1 = sig0[i] * bfQ[i*4 + 2] + sig1[i] * bfQ[i*4 + 3];
		sig0[i] = tmpOut0;
		sig1[i] = tmpOut1;
	}
}

void legacySigBitsGen(uint8_t* sigbits, uint8_t* sigbitscoded, int mcs, int len)
{
	int p = 0;
	// b 0-3 rate
	memcpy(sigbits, LEGACY_RATE_BITS[mcs], 4);
	// b 4 reserved
	sigbits[4] = 0;
	// b 5-16 len
	for(int i=0;i<12;i++)
	{
		sigbits[5+i] = (len >> i) & 0x01;
	}
	// b 17 p
	for(int i=0;i<17;i++)
	{
		if(sigbits[i])
			p++;
	}
	sigbits[17] = (p % 2);
	// b 18-23 tail
	memset(&sigbits[18], 0, 6);

	// ----------------------coding---------------------

	bccEncoder(sigbits, sigbitscoded, 24);
}

void vhtSigABitsGen(uint8_t* sigabits, uint8_t* sigabitscoded, c8p_mod* mod)
{
	// b 0-1, bw
	memset(&sigabits[0], 0, 2);
	// b 2, reserved
	sigabits[2] = 1;
	// b 3, stbc
	sigabits[3] = 0;
	if(mod->sumu)
	{
		// b 4-9, group ID
		for(int i=0;i<6;i++)
		{
			sigabits[4+i] = (mod->groupId >> i) & 0x01;
		}
		// b 10-12 MU 0 nSTS, use 1 in this ver
		for(int i=0;i<3;i++)
		{
			sigabits[10+i] = (1 >> i) & 0x01;
		}
		// b 13-15 MU 1 nSTS, use 1 in this ver
		for(int i=0;i<3;i++)
		{
			sigabits[13+i] = (1 >> i) & 0x01;
		}
		// b 16-21 MU 2,3 nSTS, set 0
		memset(&sigabits[16], 0, 6);
	}
	else
	{
		// b 4-9, group ID
		memset(&sigabits[4], 0, 6);
		// b 10-12 SU nSTS
		for(int i=0;i<3;i++)
		{
			sigabits[10+i] = ((mod->nSS-1) >> i) & 0x01;
		}
		// b 13-21 SU partial AID
		memset(&sigabits[13], 0, 9);
	}
	// b 22 txop ps not allowed, set 0, allowed
	sigabits[22] = 0;
	// b 23 reserved
	sigabits[23] = 1;
	// b 24 short GI
	sigabits[24] = 0;
	// b 25 short GI disam
	sigabits[25] = 0;
	// b 26 SU/MU0 coding, BCC
	sigabits[26] = 0;
	// b 27 LDPC extra
	sigabits[27] = 0;
	if(mod->sumu)
	{
		// b 28 MU1 bcc, b 29,30 not used set 1, b 31 reserved set 1
		sigabits[28] = 0;
		sigabits[29] = 1;
		sigabits[30] = 1;
		sigabits[31] = 1;
		// 32 beamforming, mu-mimo set 1
		sigabits[32] = 1;
	}
	else
	{
		// b 28-31 SU mcs
		for(int i=0;i<4;i++)
		{
			sigabits[28+i] = (mod->mcs >> i) & 0x01;
		}
		// 32 beamforming
		sigabits[32] = 0;
	}
	// 33 reserved
	sigabits[33] = 1;
	// 34-41 crc 8
	genCrc8Bits(sigabits, &sigabits[34], 34);
	// 42-47 tail, all 0
	memset(&sigabits[42], 0, 6);

	// ----------------------coding---------------------

	bccEncoder(sigabits, sigabitscoded, 48);
}

void vhtSigB20BitsGenSU(uint8_t* sigbbits, uint8_t* sigbbitscoded, uint8_t* sigbbitscrc, c8p_mod* mod)
{
	if(mod->len > 0)
	{
		// general data packet
		// b 0-16 apep-len/4
		for(int i=0;i<17;i++)
		{
			sigbbits[i] = ((mod->len/4) >> i) & 0x01;	
		}
		// b 17-19 reserved
		memset(&sigbbits[17], 1, 3);
		// b 20-25 tail
		memset(&sigbbits[20], 0, 6);
		// compute crc 8 for service part
		genCrc8Bits(sigbbits, sigbbitscrc, 20);
	}
	else
	{
		// NDP bit pattern
		memcpy(sigbbits, VHT_NDP_SIGB_20_BITS, 26);
		memset(sigbbitscrc, 0, 8);
	}

	// ----------------------coding---------------------

	bccEncoder(sigbbits, sigbbitscoded, 26);
}

void vhtSigB20BitsGenMU(uint8_t* sigbbits0, uint8_t* sigbbitscoded0, uint8_t* sigbbitscrc0, uint8_t* sigbbits1, uint8_t* sigbbitscoded1, uint8_t* sigbbitscrc1, c8p_mod* mod)
{
	// b 0-15 apep-len/4
	for(int i=0;i<16;i++)
	{
		sigbbits0[i] = ((mod->lenMu[0]/4) >> i) & 0x01;
	}
	// b 16-19 mcs
	for(int i=0;i<4;i++)
	{
		sigbbits0[16+i] = (mod->mcsMu[0] >> i) & 0x01;
	}
	// b 20-25 tail
	memset(&sigbbits0[20], 0, 6);
	// compute crc 8 for service part
	genCrc8Bits(sigbbits0, sigbbitscrc0, 20);
	// bcc
	bccEncoder(sigbbits0, sigbbitscoded0, 26);


	// b 0-15 apep-len/4
	for(int i=0;i<16;i++)
	{
		sigbbits1[i] = ((mod->lenMu[1]/4) >> i) & 0x01;
	}
	// b 16-19 mcs
	for(int i=0;i<4;i++)
	{
		sigbbits1[16+i] = (mod->mcsMu[1] >> i) & 0x01;
	}
	// b 20-25 tail
	memset(&sigbbits1[20], 0, 6);
	// compute crc 8 for service part
	genCrc8Bits(sigbbits1, sigbbitscrc1, 20);
	// bcc
	bccEncoder(sigbbits1, sigbbitscoded1, 26);
}

void htSigBitsGen(uint8_t* sigbits, uint8_t* sigbitscoded, c8p_mod* mod)
{
	// b 0-6 mcs
	for(int i=0;i<7;i++)
	{
		sigbits[i] = (mod->mcs >> i) & 0x01;
	}
	// b 7 bw
	sigbits[7] = 0;
	// b 8-23 len
	for(int i=0;i<16;i++)
	{
		sigbits[i+8] = (mod->len >> i) & 0x01;
	}
	// b 24 smoothing
	sigbits[24] = 1;
	// b 25 no sounding
	sigbits[25] = 1;
	// b 26 reserved
	sigbits[26] = 1;
	// b 27 aggregation
	sigbits[27] = 0;
	// b 28-29 stbc
	memset(&sigbits[28], 0, 2);
	// b 30, bcc
	sigbits[30] = 0;
	// b 31 short GI
	sigbits[31] = 0;
	// b 32-33 ext ss
	memset(&sigbits[32], 0, 2);
	// b 34-41 crc 8
	genCrc8Bits(sigbits, &sigbits[34], 34);
	// 42-47 tail, all 0
	memset(&sigbits[42], 0, 6);

	// ----------------------coding---------------------

	bccEncoder(sigbits, sigbitscoded, 48);
}

