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

const gr_complex C8P_STF_F[64] = {
	gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(-1.0f, -1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(-1.0f, -1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(1.0f, 1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(1.0f, 1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(1.0f, 1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(1.0f, 1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(1.0f, 1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(-1.0f, -1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(1.0f, 1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(-1.0f, -1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(-1.0f, -1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), 
	gr_complex(1.0f, 1.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f), gr_complex(0.0f, 0.0f)/sqrtf(2.0f)
};

const gr_complex C8P_LTF_L_F[64] = {
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
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f)
};

const gr_complex C8P_LTF_NL_F[64] = {
	gr_complex(0.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f),
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f),
	gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f),
	gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), 
	gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f)
};

const gr_complex C8P_LTF_NL_F_N[64] = {
	gr_complex(0.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), 
	gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
	gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f)
};



/***************************************************/
/* signal field */
/***************************************************/
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
	if(*len > 1600)
	{
		return false;		// usually MTU 1500
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
		outMod->sumu = 1;	// mu
		// needs the position in this group, currently set 0
		// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		outMod->nSS = outSigVhtA->mu_nSTS[0];
		// still need the packet len and mcs in sig b
	}
}

void signalParserVhtB(uint8_t* inBits, c8p_mod* outMod)
{
	int tmpLen = 0;
	int tmpMcs = 0;
	for(int i=0;i<16;i++){tmpLen |= (((int)inBits[i])<<i);}
	for(int i=0;i<4;i++){tmpMcs |= (((int)inBits[i+16])<<i);}
	outMod->len = tmpLen * 4;
	modParserVht(tmpMcs, outMod);
	outMod->nSym = (outMod->len*8 + 16 + 6) / outMod->nDBPS + (((outMod->len*8 + 16 + 6) % outMod->nDBPS) != 0);
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
	uint8_t cf = 0x00;
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

const int mapDeintVhtSigB20[52] = {
	0, 13, 26, 39, 1, 14, 27, 40, 2, 15, 28, 41, 3, 16, 29, 42, 4, 17,
	30, 43, 5, 18, 31, 44, 6,19, 32, 45, 7, 20, 33, 46, 8, 21, 34, 47,
	9, 22, 35, 48, 10, 23, 36, 49, 11, 24, 37, 50, 12, 25, 38, 51};

const int mapIntelVhtSigB20[52] = {
	0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 1, 5, 9, 13, 17, 
	21, 25, 29, 33, 37, 41, 45, 49, 2, 6, 10, 14, 18, 22, 26, 30, 34, 
	38, 42, 46, 50, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51};

const int mapDeintLegacyBpsk[48] = {
    0, 16, 32, 1, 17, 33, 2, 18, 34, 3, 19, 35, 4, 20, 36, 5, 21, 37, 6, 22, 38, 7, 23, 39, 8, 24, 
    40, 9, 25, 41, 10, 26, 42, 11, 27, 43, 12, 28, 44, 13, 29, 45, 14, 30, 46, 15, 31, 47};

const int mapIntelLegacyBpsk[48] = {
	0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 1, 4, 7, 10, 13, 16, 19, 22, 25, 
	28, 31, 34, 37, 40, 43, 46, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47};

void procDeintLegacyBpsk(uint8_t* inBits, uint8_t* outBits)
{
    for(int i=0;i<48;i++)
    {
        outBits[mapDeintLegacyBpsk[i]] = inBits[i];
    }
}

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

void procSymDeintL(float* in, float* out, c8p_mod* mod)
{
	// this version follows standard
	int s = std::max(mod->nBPSCS/2, 1);
	int i, k;
	for(int j=0;j<mod->nCBPS;j++)
	{
		i = s * (j/s) + (j + (16*j)/mod->nCBPS) % s;
		k = 16 * i - (mod->nCBPS - 1) * ((16 * i) / mod->nCBPS);
		out[k] = in[j];
	}
}

void procSymDeintNL(float* in, float* out, c8p_mod* mod, int iSS_1)		// iSS_1 is: iSS - 1 in standard
{
	// this version follows standard
	int s = std::max(mod->nBPSCS/2, 1);
	int i, j, k;
	if(mod->nSS == 1)
	{
		for(int r=0; r<mod->nCBPSS; r++)
		{
			j = r;
			i = s * (j/s) + (j + (mod->nIntCol * j)/mod->nCBPSS) % s;
			k = mod->nIntCol*i - (mod->nCBPSS - 1) * (i/mod->nIntRow);
			out[k] = in[r];
		}
	}
	else
	{
		for(int r=0; r<mod->nCBPSS; r++)
		{
			j = (r + ((2*iSS_1) % 3 + 3 * (iSS_1/3)) * mod->nIntRot * mod->nBPSCS) % mod->nCBPSS;
			i = s * (j/s) + (j + (mod->nIntCol * j)/mod->nCBPSS) % s;
			k = mod->nIntCol*i - (mod->nCBPSS - 1) * (i/mod->nIntRow);
			out[k] = in[r];
		}
	}
}

void procSymDepasNL(float in[C8P_MAX_N_SS][C8P_MAX_N_CBPSS], float* out, c8p_mod* mod)
{
	// this ver only for 2 ss
	int s = std::max(mod->nBPSCS/2, 1);
	for(int i=0; i<mod->nSD; i++)
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
		mod->nSym = (mod->len*8 + 22) / mod->nDBPS + (((mod->len*8 + 22) % mod->nDBPS) != 0);
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
		std::cout<<"ieee80211 puncture, coding rate 23"<<std::endl;
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
		std::cout<<"ieee80211 puncture, coding rate 34"<<std::endl;
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
		std::cout<<"ieee80211 puncture, coding rate 56"<<std::endl;
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

void procInterLegacy(uint8_t* inBits, uint8_t* outBits, c8p_mod* mod)
{
	int s = std::max(1, mod->nBPSCS/2);
	int i, j;
	for(int k=0;k<mod->nCBPS;k++)
	{
		i = (mod->nCBPS/16) * (k % 16) + (k / 16);
		j = s * (i / s) + (i + mod->nCBPS - ((16 * i) / mod->nCBPS)) % s;
		outBits[j] = inBits[k];
	}
}

void procInterNonLegacy(uint8_t* inBits, uint8_t* outBits, c8p_mod* mod, int iss_1)
{
	int s = std::max(1, mod->nBPSCS/2);
	int i, j, r;
	for(int k=0;k<mod->nCBPSS;k++)
	{
		i = mod->nIntRow * (k % mod->nIntCol) + (k / mod->nIntCol);
		j = s * (i / s) + (i + mod->nCBPSS - ((mod->nIntCol * i) / mod->nCBPSS)) % s;
		if(mod->nSS > 1)
		{
			r = (j - ((2*iss_1)%3 + 3 * (iss_1/3)) * mod->nIntRot * mod->nBPSCS + mod->nCBPSS) % mod->nCBPSS;
		}
		else
		{
			r = j;
		}
		outBits[r] = inBits[k];
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
			// method 1
			//outChips[tmpChipIndex] |= (inBits[tmpBitIndex] << i);
			// method 2
			outChips[tmpChipIndex] <<= 1;
			outChips[tmpChipIndex] |= inBits[tmpBitIndex];

			tmpBitIndex++;
		}
		tmpChipIndex++;
	}
	
}

void procCSD(gr_complex* sig, int cycShift)
{
	float tmpStep = gr_complex(0.0f, -2.0f) * M_PI * (float)cycShift * 20.0f * 0.001f;
	for(int i=0;i<64;i++)
	{
		sig[i] = sig[i] * std::exp( tmpStep * (float)(i - 32) / 64.0f);
	}
}

void procToneScaling(gr_complex* sig, int ntf, int nss)
{
	for(int i=0;i<64;i++)
	{
		sig[i] = sig[i] / sqrtf((float)ntf * (float)nss);
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

void vhtSigABitsGenSU(uint8_t* sigabits, uint8_t* sigabitscoded, c8p_mod* mod)
{
	// b 0-1, bw
	memset(&sigabits[0], 0, 2);
	// b 2, reserved
	sigabits[2] = 1;
	// b 3, stbc
	sigabits[3] = 0;
	// b 4-9, group ID
	memset(&sigabits[4], 0, 6);
	// b 10-12 SU nSTS
	for(int i=0;i<3;i++)
	{
		sigabits[10+i] = ((mod->nSS-1) >> i) & 0x01;
	}
	// b 13-21 SU partial AID
	memset(&sigabits[13], 0, 9);
	// b 22 txop ps not allowed, set 0, allowed
	sigabits[22] = 0;
	// b 23 reserved
	sigabits[23] = 1;
	// b 24 short GI
	sigabits[24] = 0;
	// b 25 short GI disam
	sigabits[25] = 0;
	// b 26 SU coding, BCC
	sigabits[26] = 0;
	// b 27 LDPC extra
	sigabits[27] = 0;
	// b 28-31 SU mcs
	for(int i=0;i<4;i++)
	{
		sigabits[28+i] = (mod->mcs >> i) & 0x01;
	}
	// 32 beamforming
	sigabits[32] = 0;
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
	// b 0-16 apep-len/4
	for(int i=0;i<17;i++)
	{
		sigbbits[i] = ((mod->len/4) >> i) & 0x01;	
	}
	// b 17-19 reserved
	memset(&sigbbits[17], 1, 3);
	// b 20-25 tail
	memset(&sigbbits[20], 0, 5);
	// compute crc 8 for service part
	genCrc8Bits(sigbbits, sigbbitscrc, 20);

	// ----------------------coding---------------------

	bccEncoder(sigbbits, sigbbitscoded, 26);
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

