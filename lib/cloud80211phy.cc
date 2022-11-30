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

const int mapIntelLegacyQpsk[96] = {
	0, 7, 12, 19, 24, 31, 36, 43, 48, 55, 60, 67, 72, 79, 84, 91, 1, 6, 13, 18, 25, 30, 37, 42, 
	49, 54, 61, 66, 73, 78, 85, 90, 2, 9, 14, 21, 26, 33, 38, 45, 50, 57, 62, 69, 74, 81, 86, 93, 
	3, 8, 15, 20, 27, 32, 39, 44, 51, 56, 63, 68, 75, 80, 87, 92, 4, 11, 16, 23, 28, 35, 40, 47, 
	52, 59, 64, 71, 76, 83, 88, 95, 5, 10, 17, 22, 29, 34, 41, 46, 53, 58, 65, 70, 77, 82, 89, 94};

const int mapDeintLegacyQpsk[96] = {
	0, 16, 32, 48, 64, 80, 17, 1, 49, 33, 81, 65, 2, 18, 34, 50, 66, 82, 19, 3, 51, 35, 83, 67, 4, 
	20, 36, 52, 68, 84, 21, 5, 53, 37, 85, 69, 6, 22, 38, 54, 70, 86, 23, 7, 55, 39, 87, 71, 8, 24, 
	40, 56, 72, 88, 25, 9, 57, 41, 89, 73, 10, 26, 42, 58, 74, 90, 27, 11, 59, 43, 91, 75, 12, 28, 
	44, 60, 76, 92, 29, 13, 61, 45, 93, 77, 14, 30, 46, 62, 78, 94, 31, 15, 63, 47, 95, 79};

const int mapIntelLegacy16Qam[192] = {
	0, 15, 26, 37, 48, 63, 74, 85, 96, 111, 122, 133, 144, 159, 170, 181, 1, 12, 27, 38, 49, 60, 
	75, 86, 97, 108, 123, 134, 145, 156, 171, 182, 2, 13, 24, 39, 50, 61, 72, 87, 98, 109, 120, 
	135, 146, 157, 168, 183, 3, 14, 25, 36, 51, 62, 73, 84, 99, 110, 121, 132, 147, 158, 169, 180, 
	4, 19, 30, 41, 52, 67, 78, 89, 100, 115, 126, 137, 148, 163, 174, 185, 5, 16, 31, 42, 53, 64, 
	79, 90, 101, 112, 127, 138, 149, 160, 175, 186, 6, 17, 28, 43, 54, 65, 76, 91, 102, 113, 124, 
	139, 150, 161, 172, 187, 7, 18, 29, 40, 55, 66, 77, 88, 103, 114, 125, 136, 151, 162, 173, 184, 
	8, 23, 34, 45, 56, 71, 82, 93, 104, 119, 130, 141, 152, 167, 178, 189, 9, 20, 35, 46, 57, 68, 
	83, 94, 105, 116, 131, 142, 153, 164, 179, 190, 10, 21, 32, 47, 58, 69, 80, 95, 106, 117, 128, 
	143, 154, 165, 176, 191, 11, 22, 33, 44, 59, 70, 81, 92, 107, 118, 129, 140, 155, 166, 177, 188};

const int mapDeintLegacy16Qam[192] = {
	0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 17, 33, 49, 1, 81, 97, 113, 65, 145, 161, 
	177, 129, 34, 50, 2, 18, 98, 114, 66, 82, 162, 178, 130, 146, 51, 3, 19, 35, 115, 67, 83, 99, 
	179, 131, 147, 163, 4, 20, 36, 52, 68, 84, 100, 116, 132, 148, 164, 180, 21, 37, 53, 5, 85, 101, 
	117, 69, 149, 165, 181, 133, 38, 54, 6, 22, 102, 118, 70, 86, 166, 182, 134, 150, 55, 7, 23, 39, 
	119, 71, 87, 103, 183, 135, 151, 167, 8, 24, 40, 56, 72, 88, 104, 120, 136, 152, 168, 184, 25, 
	41, 57, 9, 89, 105, 121, 73, 153, 169, 185, 137, 42, 58, 10, 26, 106, 122, 74, 90, 170, 186, 
	138, 154, 59, 11, 27, 43, 123, 75, 91, 107, 187, 139, 155, 171, 12, 28, 44, 60, 76, 92, 108, 124, 
	140, 156, 172, 188, 29, 45, 61, 13, 93, 109, 125, 77, 157, 173, 189, 141, 46, 62, 14, 30, 110, 
	126, 78, 94, 174, 190, 142, 158, 63, 15, 31, 47, 127, 79, 95, 111, 191, 143, 159, 175};

const int mapIntelLegacy64Qam[288] = {
	0, 23, 40, 57, 74, 91, 108, 131, 148, 165, 182, 199, 216, 239, 256, 273, 1, 18, 41, 58, 75, 92, 
	109, 126, 149, 166, 183, 200, 217, 234, 257, 274, 2, 19, 36, 59, 76, 93, 110, 127, 144, 167, 184, 
	201, 218, 235, 252, 275, 3, 20, 37, 54, 77, 94, 111, 128, 145, 162, 185, 202, 219, 236, 253, 270, 
	4, 21, 38, 55, 72, 95, 112, 129, 146, 163, 180, 203, 220, 237, 254, 271, 5, 22, 39, 56, 73, 90, 
	113, 130, 147, 164, 181, 198, 221, 238, 255, 272, 6, 29, 46, 63, 80, 97, 114, 137, 154, 171, 188, 
	205, 222, 245, 262, 279, 7, 24, 47, 64, 81, 98, 115, 132, 155, 172, 189, 206, 223, 240, 263, 280, 
	8, 25, 42, 65, 82, 99, 116, 133, 150, 173, 190, 207, 224, 241, 258, 281, 9, 26, 43, 60, 83, 100, 
	117, 134, 151, 168, 191, 208, 225, 242, 259, 276, 10, 27, 44, 61, 78, 101, 118, 135, 152, 169, 186, 
	209, 226, 243, 260, 277, 11, 28, 45, 62, 79, 96, 119, 136, 153, 170, 187, 204, 227, 244, 261, 278, 
	12, 35, 52, 69, 86, 103, 120, 143, 160, 177, 194, 211, 228, 251, 268, 285, 13, 30, 53, 70, 87, 104, 
	121, 138, 161, 178, 195, 212, 229, 246, 269, 286, 14, 31, 48, 71, 88, 105, 122, 139, 156, 179, 196, 
	213, 230, 247, 264, 287, 15, 32, 49, 66, 89, 106, 123, 140, 157, 174, 197, 214, 231, 248, 265, 282, 
	16, 33, 50, 67, 84, 107, 124, 141, 158, 175, 192, 215, 232, 249, 266, 283, 17, 34, 51, 68, 85, 102, 
	125, 142, 159, 176, 193, 210, 233, 250, 267, 284};

const int mapDeintLegacy64Qam[288] = {
	0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 17, 33, 49, 65, 
	81, 1, 113, 129, 145, 161, 177, 97, 209, 225, 241, 257, 273, 193, 34, 50, 66, 82, 2, 18, 130, 146, 
	162, 178, 98, 114, 226, 242, 258, 274, 194, 210, 51, 67, 83, 3, 19, 35, 147, 163, 179, 99, 115, 131, 
	243, 259, 275, 195, 211, 227, 68, 84, 4, 20, 36, 52, 164, 180, 100, 116, 132, 148, 260, 276, 196, 
	212, 228, 244, 85, 5, 21, 37, 53, 69, 181, 101, 117, 133, 149, 165, 277, 197, 213, 229, 245, 261, 6, 
	22, 38, 54, 70, 86, 102, 118, 134, 150, 166, 182, 198, 214, 230, 246, 262, 278, 23, 39, 55, 71, 87, 
	7, 119, 135, 151, 167, 183, 103, 215, 231, 247, 263, 279, 199, 40, 56, 72, 88, 8, 24, 136, 152, 168, 
	184, 104, 120, 232, 248, 264, 280, 200, 216, 57, 73, 89, 9, 25, 41, 153, 169, 185, 105, 121, 137, 249, 
	265, 281, 201, 217, 233, 74, 90, 10, 26, 42, 58, 170, 186, 106, 122, 138, 154, 266, 282, 202, 218, 
	234, 250, 91, 11, 27, 43, 59, 75, 187, 107, 123, 139, 155, 171, 283, 203, 219, 235, 251, 267, 12, 28, 
	44, 60, 76, 92, 108, 124, 140, 156, 172, 188, 204, 220, 236, 252, 268, 284, 29, 45, 61, 77, 93, 13, 
	125, 141, 157, 173, 189, 109, 221, 237, 253, 269, 285, 205, 46, 62, 78, 94, 14, 30, 142, 158, 174, 
	190, 110, 126, 238, 254, 270, 286, 206, 222, 63, 79, 95, 15, 31, 47, 159, 175, 191, 111, 127, 143, 
	255, 271, 287, 207, 223, 239};


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

void procSymDeintNL2S(float* in, float* out, c8p_mod* mod)		// iSS_1 is: iSS - 1 in standard
{
	switch(mod->nCBPS)
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
		sig[i] = sig[i] / sqrtf((float)ntf * (float)nss);
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

