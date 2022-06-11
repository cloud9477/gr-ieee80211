/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Legacy Signal Field Information
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

#ifndef INCLUDED_CLOUD80211PHY_H
#define INCLUDED_CLOUD80211PHY_H

#include <gnuradio/io_signature.h>

/* legacy signal field class, use mcs 0 to 7 to represent rates from 6 to 54*/
class sigL
{
    public:
    int mcs;
    int len;
    sigL();
    ~sigL();
};

extern const gr_complex LTF_L_26_F[64];

void procDeintLegacyBpsk(uint8_t* inBits, uint8_t* outBits);
void procDeintLegacyBpsk(float* inBits, float* outBits);
void SV_Decode_Sig(float* llrv, uint8_t* decoded_bits);
bool signalParserL(uint8_t* inBits, sigL* outSigL);



#endif /* INCLUDED_IEEE80211_SIGNAL_IMPL_H */