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

#ifndef INCLUDED_CLOUD80211PHY_H
#define INCLUDED_CLOUD80211PHY_H

#include <cstring>
#include <iostream>
#include <gnuradio/io_signature.h>

#define C8P_MAX_N_LTF 4
#define C8P_MAX_N_SS 2
#define C8P_MAX_N_CBPSS 416 // 256QAM 8bit/sc * 52 = 416

#define C8P_F_L 0
#define C8P_F_HT 1
#define C8P_F_VHT 2
#define C8P_F_NL 3

#define C8P_BW_20   0
#define C8P_BW_40   1
#define C8P_BW_80   2

#define C8P_CR_12   0
#define C8P_CR_23   1
#define C8P_CR_34   2
#define C8P_CR_56   3

#define C8P_QAM_BPSK 0
#define C8P_QAM_QBPSK 1
#define C8P_QAM_QPSK 2
#define C8P_QAM_16QAM 3
#define C8P_QAM_64QAM 4
#define C8P_QAM_256QAM 5

class c8p_mod
{
    public:
        int format;     // l, ht, vht
        int mcs;
        int len;        // packet len for legacy, ht, apep-len for vht

        int nSym;
        int ampdu;
        int sumu;       // 0 for su or 1 for mu
        int nSymSamp;     // sample of a symbol
        
        int mod;        // modulation
        int cr;         // coding rate
        int nSD;        // data sub carrier
        int nSP;        // pilot sub carrier
        int nSS;        // spatial streams
        int nBPSCS;     // bit per sub carrier
        int nDBPS;      // data bit per sym
        int nCBPS;      // coded bit per sym
        int nCBPSS;     // coded bit per sym per ss
        // ht & vht
        int nIntCol;
        int nIntRow;
        int nIntRot;
        int nLTF;       // number of LTF in non-legacy part
};

class c8p_sigHt
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
};

class c8p_sigVhtA
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
};

extern const gr_complex LTF_L_26_F_COMP[64];
extern const float LTF_L_26_F_FLOAT[64];
extern const float LTF_NL_28_F_FLOAT[64];
extern const float LTF_NL_28_F_FLOAT2[64];

extern const int SV_PUNC_12[2];
extern const int SV_PUNC_23[4];
extern const int SV_PUNC_34[6];
extern const int SV_PUNC_56[10];
extern const int SV_STATE_NEXT[64][2];
extern const int SV_STATE_OUTPUT[64][2];

extern const float PILOT_P[127];
extern const float PILOT_L[4];
extern const float PILOT_HT_2_1[4];
extern const float PILOT_HT_2_2[4];
extern const float PILOT_VHT[4];
extern const uint8_t EOF_PAD_SUBFRAME[32];

extern const int mapDeintVhtSigB20[52];
void procDeintLegacyBpsk(uint8_t* inBits, uint8_t* outBits);
void procDeintLegacyBpsk(float* inBits, float* outBits);
void procIntelLegacyBpsk(uint8_t* inBits, uint8_t* outBits);
void procIntelVhtB20(uint8_t* inBits, uint8_t* outBits);
void SV_Decode_Sig(float* llrv, uint8_t* decoded_bits, int trellisLen);
void procSymQamToLlr(gr_complex* inQam, float* outLlr, c8p_mod* mod);
void procSymDeintL(float* in, float* out, c8p_mod* mod);
void procSymDeintNL(float* in, float* out, c8p_mod* mod, int iSS_1);
void procSymDepasNL(float in[C8P_MAX_N_SS][C8P_MAX_N_CBPSS], float* out, c8p_mod* mod);
int nCodedToUncoded(int nCoded, c8p_mod* mod);
int nUncodedToCoded(int nUncoded, c8p_mod* mod);

bool signalCheckLegacy(uint8_t* inBits, int* mcs, int* len, int* nDBPS);
bool signalCheckHt(uint8_t* inBits);
bool signalCheckVhtA(uint8_t* inBits);

void signalParserL(int mcs, int len, c8p_mod* outMod);
void signalParserHt(uint8_t* inBits, c8p_mod* outMod, c8p_sigHt* outSigHt);
void modParserHt(int mcs, c8p_mod* outMod);
void signalParserVhtA(uint8_t* inBits, c8p_mod* outMod, c8p_sigVhtA* outSigVhtA);
void signalParserVhtB(uint8_t* inBits, c8p_mod* outMod);
void modParserVht(int mcs, c8p_mod* outMod);

void genCrc8Bits(uint8_t* inBits, uint8_t* outBits, int len);
bool checkBitCrc8(uint8_t* inBits, int len, uint8_t* crcBits);
void bccEncoder(uint8_t* inBits, uint8_t* outBits, int len);
void scramEncoder(uint8_t* inBits, uint8_t* outBits, int len, int init);
void punctEncoder(uint8_t* inBits, uint8_t* outBits, int len, c8p_mod* mod);
void streamParser2(uint8_t* inBits, uint8_t* outBits1, uint8_t* outBits2, int len, c8p_mod* mod);
void bitsToChips(uint8_t* inBits, uint8_t* outChips, c8p_mod* mod);
void procInterLegacy(uint8_t* inBits, uint8_t* outBits, c8p_mod* mod);
void procInterNonLegacy(uint8_t* inBits, uint8_t* outBits, c8p_mod* mod, int iss_1);

void formatToModSu(c8p_mod* mod, int format, int mcs, int nss, int len);
bool formatCheck(int format, int mcs, int nss);

void legacySigBitsGen(uint8_t* sigbits, uint8_t* sigbitscoded, int mcs, int len);
void vhtSigABitsGenSU(uint8_t* sigabits, uint8_t* sigabitscoded, c8p_mod* mod);
void vhtSigB20BitsGenSU(uint8_t* sigbbits, uint8_t* sigbbitscoded, uint8_t* sigbbitscrc, c8p_mod* mod);
void htSigBitsGen(uint8_t* sigbits, uint8_t* sigbitscoded, c8p_mod* mod);

#endif /* INCLUDED_IEEE80211_SIGNAL_IMPL_H */