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

#include <iostream>
#include <gnuradio/io_signature.h>

#define C8P_BW_20   0
#define C8P_BW_40   1
#define C8P_BW_80   2

/* legacy signal field class, use mcs 0 to 7 to represent rates from 6 to 54*/
class sigL
{
    public:
        int mcs;
        int len;
        sigL();
        ~sigL();
};

class sigHt
{
    public:
        int mcs;
        int len;
        int bw;
        int smooth;
        int noSound;
        int aggre;
        int stbc;
        int coding;
        int shortGi;
        int nExtSs;
        sigHt();
        ~sigHt();
};

class sigVhtA
{
    public:
        int bw;
        int stbc;
        int groupId;
        int su_nSTS;
        int su_partialAID;
        int su_coding;
        int su_mcs;
        int su_beamformed;
        int mu_coding[4];
        int mu_nSTS[4];
        int txoppsNot;
        int shortGi;
        int shortGiNsymDis;
        int ldpcExtra;
        sigVhtA();
        ~sigVhtA();
};

extern const gr_complex LTF_L_26_F[64];

void procDeintLegacyBpsk(uint8_t* inBits, uint8_t* outBits);
void procDeintLegacyBpsk(float* inBits, float* outBits);
void SV_Decode_Sig(float* llrv, uint8_t* decoded_bits, int trellisLen);
bool signalParserL(uint8_t* inBits, sigL* outSigL);
bool signalParserHt(uint8_t* inBits, sigHt* outSigHt);
bool signalParserVht(uint8_t* inBits, sigVhtA* outSigVht);
uint8_t procBitCrc8(uint8_t* inBits, int len);




#endif /* INCLUDED_IEEE80211_SIGNAL_IMPL_H */