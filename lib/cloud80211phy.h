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
#include <math.h>

#define C8P_MAX_N_LTF 4
#define C8P_MAX_N_SS 2
#define C8P_MAX_N_CBPSS 416 // 256QAM 8bit/sc * 52 = 416

#define C8P_F_L 0
#define C8P_F_HT 1
#define C8P_F_VHT 2
#define C8P_F_VHT_MU 3
#define C8P_F_VHT_BFQ_R 10
#define C8P_F_VHT_BFQ_I 11
#define C8P_F_VHT_NDP 20

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
        int sumu;       // 0 for su or 1 for mu
        int ampdu;
        int nSym;
        int nSymSamp;   // sample of a symbol

        int nSD;        // data sub carrier
        int nSP;        // pilot sub carrier
        int nSS;        // spatial streams
        int nLTF;       // number of LTF in non-legacy part

        int mcs;
        int len;        // packet len for legacy, ht, apep-len for vht
        int mod;        // modulation
        int cr;         // coding rate
        int nBPSCS;     // bit per sub carrier
        int nDBPS;      // data bit per sym
        int nCBPS;      // coded bit per sym
        int nCBPSS;     // coded bit per sym per ss
        // ht & vht
        int nIntCol;
        int nIntRow;
        int nIntRot;

        int groupId;
        int mcsMu[4];
        int lenMu[4];        // packet len for legacy, ht, apep-len for vht
        int modMu[4];        // modulation
        int crMu[4];         // coding rate
        int nBPSCSMu[4];     // bit per sub carrier
        int nDBPSMu[4];      // data bit per sym
        int nCBPSMu[4];      // coded bit per sym
        int nCBPSSMu[4];     // coded bit per sym per ss
        // ht & vht
        int nIntColMu[4];
        int nIntRowMu[4];
        int nIntRotMu[4];
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

extern const int C8P_LEGACY_DP_SC[64];
extern const int C8P_LEGACY_D_SC[64];
extern const int FFT_26_DEMAP[64];
extern const int FFT_26_SHIFT_DEMAP[128];
extern const gr_complex LTF_L_26_F_COMP[64];
extern const float LTF_L_26_F_FLOAT[64];
extern const float LTF_NL_28_F_FLOAT[64];
extern const float LTF_NL_28_F_FLOAT2[64];

extern const int mapDeintLegacyBpsk[48];
extern const int mapDeintLegacyQpsk[96];
extern const int mapDeintLegacy16Qam[192];
extern const int mapDeintLegacy64Qam[288];
extern const int mapDeintNonlegacyBpsk[52];
extern const int mapDeintNonlegacyQpsk[104];
extern const int mapDeintNonlegacy16Qam[208];
extern const int mapDeintNonlegacy64Qam[312];
extern const int mapDeintNonlegacy256Qam[416];

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


extern const gr_complex C8P_STF_F[64];
extern const gr_complex C8P_LTF_L_F[64];
extern const gr_complex C8P_LTF_NL_F[64];
extern const gr_complex C8P_LTF_NL_F_N[64];
extern const gr_complex C8P_LTF_NL_F_VHT22[64];


void procDeintLegacyBpsk(float* inBits, float* outBits);
void procIntelLegacyBpsk(uint8_t* inBits, uint8_t* outBits);
void procIntelVhtB20(uint8_t* inBits, uint8_t* outBits);
void SV_Decode_Sig(float* llrv, uint8_t* decoded_bits, int trellisLen);
void procSymQamToLlr(gr_complex* inQam, float* outLlr, c8p_mod* mod);
void procSymDeintL2(float* in, float* out, c8p_mod* mod);
void procSymIntelL2(uint8_t* in, uint8_t* out, c8p_mod* mod);
void procSymDeintNL2SS1(float* in, float* out, c8p_mod* mod);
void procSymDeintNL2SS2(float* in, float* out, c8p_mod* mod);
void procSymIntelNL2SS1(uint8_t* in, uint8_t* out, c8p_mod* mod);
void procSymIntelNL2SS2(uint8_t* in, uint8_t* out, c8p_mod* mod);
void procSymDepasNL(float in[C8P_MAX_N_SS][C8P_MAX_N_CBPSS], float* out, c8p_mod* mod);
int nCodedToUncoded(int nCoded, c8p_mod* mod);
int nUncodedToCoded(int nUncoded, c8p_mod* mod);
void procCSD(gr_complex* sig, int cycShift);
void procToneScaling(gr_complex* sig, int ntf, int nss, int len);
void procNss2SymBfQ(gr_complex* sig0, gr_complex* sig1, gr_complex* bfQ);
void procChipsToQam(const uint8_t* inChips,  gr_complex* outQam, int qamType, int len);
void procInsertPilotsDc(gr_complex* sigIn, gr_complex* sigOut, gr_complex* pilots, int format);
void procNonDataSc(gr_complex* sigIn, gr_complex* sigOut, int format);

void signalNlDemodDecode(gr_complex *sym1, gr_complex *sym2, gr_complex *h, float *llrht, float *llrvht);
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

void formatToModSu(c8p_mod* mod, int format, int mcs, int nss, int len);
void vhtModMuToSu(c8p_mod* mod, int pos);
void vhtModSuToMu(c8p_mod* mod, int pos);
void formatToModMu(c8p_mod* mod, int mcs0, int nSS0, int len0, int mcs1, int nSS1, int len1);
bool formatCheck(int format, int mcs, int nss);

void legacySigBitsGen(uint8_t* sigbits, uint8_t* sigbitscoded, int mcs, int len);
void vhtSigABitsGen(uint8_t* sigabits, uint8_t* sigabitscoded, c8p_mod* mod);
void vhtSigB20BitsGenSU(uint8_t* sigbbits, uint8_t* sigbbitscoded, uint8_t* sigbbitscrc, c8p_mod* mod);
void vhtSigB20BitsGenMU(uint8_t* sigbbits0, uint8_t* sigbbitscoded0, uint8_t* sigbbitscrc0, uint8_t* sigbbits1, uint8_t* sigbbitscoded1, uint8_t* sigbbitscrc1, c8p_mod* mod);
void htSigBitsGen(uint8_t* sigbits, uint8_t* sigbitscoded, c8p_mod* mod);

#endif /* INCLUDED_IEEE80211_SIGNAL_IMPL_H */