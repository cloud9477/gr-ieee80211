"""
    GNU Radio IEEE 802.11a/g/n/ac 2x2
    Python tools
    Copyright (C) June 1, 2022  Zelin Yun

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import struct
import socket
import binascii
import zlib
from matplotlib import pyplot as plt
import mac80211
import phy80211header as p8h
import random
import time

class phy80211():
    def __init__(self, ifDebug = True):
        self.m = p8h.modulation()
        self.mpdu = b""
        self.ampdu = b""
        self.dataScrambler = 93         # scrambler used for scrambling, from 1 to 127
        self.ssP = [[], [], [], []]     # pilots of ss
        self.ssPPI = [0, 0, 0, 0]       # pilot polarity index of ss
        # legacy training, legacy sig
        self.ssLegacyTraining = []
        self.ssLegacySig = []
        # ht sig, ht Training, vht sig a, vht training, vht sig b
        self.ssHtSig = []
        self.ssHtTraining = []
        self.ssVhtSigA = []
        self.ssVhtTraining = []
        self.ssVhtSigB = []
        # data
        self.dataBits = []              # original data bits
        self.esCodedBits = []           # coded bits of nES encoders, legacy has only one encoder
        self.ssStreamBits = []          # parsered bits of nSS streams
        self.ssIntedBits = []           # interleaved bits of nSS streams
        self.ssSymbols = []             # qam constellations of nSS streams
        self.ssPhySig = []              # phy samples of nSS streams
        self.ssFinalSig = []            # phy samples of nSS streams for figure or GNU Radio bin file
        # vht special
        self.vhtPartialAid = 0
        self.vhtGroupId = 0             # Group ID, 0 to AP, 63 to stations
        self.vhtSigBCrcBits = []
        self.ampduMu = [b""] * 4
        self.mMu = [p8h.modulation(), p8h.modulation(), p8h.modulation(), p8h.modulation()]
        self.nUserMu = 0
        self.nSymMu = 0
        self.nSTSMu = 0
        self.vhtSigBCrcBitsMu = [[], [], [], []]
        self.bfQ = []
        # RX
        self.rxSisoSig = []
        self.rxSampNum = 0
        self.rxCfo = 0
        self.rxChanL = []
        # debug
        self.ifdb = ifDebug

    def genFromMpdu(self, mpdu, mod):
        if(isinstance(mpdu, (bytes, bytearray)) and isinstance(mod, p8h.modulation) and len(mpdu) > 0 and mod.initRes):
            if(mod.phyFormat == p8h.F.L):
                self.mpdu = mpdu
                self.m = mod
                self.m.ampdu = False
                self.m.procPktLenNonAggre(len(self.mpdu))
                self.__genLegacyTraining()
                self.__genLegacySignal()
                self.__genDataBits()
                self.__genCodedBits()
                self.__genStreamParserDataBits()
                self.__genInterleaveDataBits()
                self.__genConstellation()
                self.__genOfdmSignal()
            elif(mod.phyFormat == p8h.F.HT):
                self.mpdu = mpdu
                self.m = mod
                self.m.ampdu = False
                self.m.procPktLenNonAggre(len(self.mpdu))
                self.__genLegacyTraining()
                self.__genLegacySignal()
                self.__genHtSignal()
                self.__genNonLegacyTraining()
                self.__genDataBits()
                self.__genCodedBits()
                self.__genStreamParserDataBits()
                self.__genInterleaveDataBits()
                self.__genConstellation()
                self.__genOfdmSignal()
            else:
                print("cloud phy80211, genFromMpdu input format error, %s is not supported" % mod.phyFormat)
                return
        else:
            print("cloud phy80211, genFromMpdu input param error")
            return

    def genFromAmpdu(self, ampdu, mod, vhtPartialAid = 0, vhtGroupId = 0):
        if(isinstance(ampdu, (bytes, bytearray)) and isinstance(mod, p8h.modulation) and mod.initRes):
            if(mod.phyFormat == p8h.F.HT):
                self.ampdu = ampdu
                self.m = mod
                self.m.ampdu = True
                self.m.procPktLenAggre(len(self.ampdu))
                self.__genLegacyTraining()
                self.__genLegacySignal()
                self.__genHtSignal()
                self.__genNonLegacyTraining()
                self.__genDataBits()
                self.__genCodedBits()
                self.__genStreamParserDataBits()
                self.__genInterleaveDataBits()
                self.__genConstellation()
                self.__genOfdmSignal()
            elif(mod.phyFormat == p8h.F.VHT):
                if(vhtPartialAid >= 0 and (vhtGroupId == 0 or vhtGroupId == 63)):
                    # single user packet either goes to AP or to station
                    # 0: sta to AP, 63: AP to sta
                    if(len(ampdu) > 0):
                        if self.ifdb: print("cloud phy80211, genVht type general packet")
                        self.ampdu = ampdu
                        self.m = mod
                        self.m.ampdu = True
                        self.m.procPktLenAggre(len(self.ampdu))
                        self.vhtPartialAid = vhtPartialAid
                        self.vhtGroupId = vhtGroupId
                        self.m.mu = False
                        self.__genLegacyTraining()
                        self.__genLegacySignal()
                        self.__genVhtSignalA()
                        self.__genNonLegacyTraining()
                        self.__genVhtSignalB()
                        self.__genDataBits()
                        self.__genCodedBits()
                        self.__genStreamParserDataBits()
                        self.__genInterleaveDataBits()
                        self.__genConstellation()
                        self.__genOfdmSignal()
                    else:
                        # NDP
                        if self.ifdb: print("cloud phy80211, genVht type NDP packet")
                        self.ampdu = b""
                        self.m = mod
                        self.m.ampdu = True
                        self.m.procPktLenAggre(0)
                        self.vhtPartialAid = vhtPartialAid
                        self.vhtGroupId = vhtGroupId
                        self.m.mu = False
                        self.__genLegacyTraining()
                        self.__genLegacySignal()
                        self.__genVhtSignalA()
                        self.__genNonLegacyTraining()
                        self.__genVhtSignalB()
                        self.__genOfdmSignalNdp()
            else:
                print("cloud phy80211, genFromAmpdu input format error, %s is not supported" % mod.phyFormat)
                return
        else:
            print("cloud phy80211, genFromAmpdu input param error")
            return

    def genAmpduMu(self, nUser = 2, bfQ = [], groupId = 1, ampdu0=b"", mod0 = p8h.modulation(), ampdu1=b"", mod1 = p8h.modulation(), ampdu2=b"", mod2 = p8h.modulation(), ampdu3=b"", mod3 = p8h.modulation()):
        # get input params
        self.ampduMu = [ampdu0, ampdu1, ampdu2, ampdu3]
        self.mMu = [mod0, mod1, mod2, mod3]
        self.nUserMu = nUser
        if self.ifdb: print("VHT MU number of users:", self.nUserMu)
        # prepare each mod
        self.nSymMu = 0
        self.nSTSMu = 0
        for u in range(0, nUser):
            self.mMu[u].procPktLenAggre(len(self.ampduMu[u]))
            if self.ifdb: print("user %d symbol number %d" % (u, self.mMu[u].nSym))
            self.nSymMu = max(self.nSymMu, self.mMu[u].nSym)
            self.nSTSMu += self.mMu[u].nSTS
        self.m = p8h.modulation(phyFormat=p8h.F.VHT, mcs=0, bw=mod0.bw, nSTS=self.nSTSMu, shortGi=mod0.sgi)
        self.m.procPktLenAggreMu(0, self.nSymMu)
        self.m.mu = True
        if self.ifdb: print("VHT MU mod nSTS %d, nSS %d, nLTF %d" % (self.m.nSTS, self.m.nSS, self.m.nLtf))
        self.bfQ = bfQ
        self.vhtGroupId = groupId       # multiple user, 1 to 62
        self.__genLegacyTraining()
        self.__genLegacySignal()
        self.__genVhtSignalA()
        self.__genNonLegacyTraining()
        self.__genVhtSignalBMu()
        self.ssSymbolsMu = []
        for u in range(0, self.nUserMu):
            self.ampdu = self.ampduMu[u]
            self.m = self.mMu[u]
            self.m.procPktLenAggreMu(len(self.ampdu), self.nSymMu)
            self.vhtSigBCrcBits = self.vhtSigBCrcBitsMu[u]
            self.__genDataBits()
            self.__genCodedBits()
            self.__genStreamParserDataBits()
            self.__genInterleaveDataBits()
            self.__genConstellation()
            for each in self.ssSymbols:
                self.ssSymbolsMu.append(each)
        self.m = p8h.modulation(phyFormat=p8h.F.VHT, mcs=0, bw=mod0.bw, nSTS=self.nSTSMu, shortGi=mod0.sgi)
        self.m.procPktLenAggreMu(0, self.nSymMu)
        self.m.mu = True
        self.__genOfdmSignalMu()

    def __genLegacyTraining(self):
        self.ssLegacyTraining = []
        for ssItr in range(0, self.m.nSS):
            tmpStf = p8h.procToneScaling(p8h.procFftMod(
                p8h.procLegacyCSD(p8h.procNonDataSC(p8h.C_STF_L[self.m.bw.value]), self.m.nSS, ssItr, self.m.spr)),
                                         p8h.C_SCALENTF_STF_L[self.m.bw.value], self.m.nSS)
            tmpLtf = p8h.procToneScaling(p8h.procFftMod(
                p8h.procLegacyCSD(p8h.procNonDataSC(p8h.C_LTF_L[self.m.bw.value]), self.m.nSS, ssItr, self.m.spr)),
                                         p8h.C_SCALENTF_LTF_L[self.m.bw.value], self.m.nSS)
            tmpStf = tmpStf[int(len(tmpStf)/2):] + tmpStf + tmpStf
            tmpLtf = tmpLtf[int(len(tmpLtf)/2):] + tmpLtf + tmpLtf
            self.ssLegacyTraining.append(p8h.procConcat2Symbol(tmpStf, tmpLtf))
        if self.ifdb: print("cloud phy80211, legacy training sample len %d" % (len(self.ssLegacyTraining[0])))

    def __genLegacySignal(self):
        # ieee 80211 2016 ofdm phy, sec 17.3.4
        self.ssLegacySig = []
        tmpHeaderBits = []
        # legacy rate
        if(self.m.phyFormat == p8h.F.L):
            tmpHeaderBits += p8h.C_LEGACY_RATE_BIT[self.m.mcs]      # legacy use mcs
        else:
            tmpHeaderBits += p8h.C_LEGACY_RATE_BIT[0]               # ht vht use 6M
        # legacy reserved
        tmpHeaderBits.append(0)
        # add length bits
        for i in range(0, 12):
            tmpHeaderBits.append((self.m.legacyLen >> i) & (1))
        # add parity bit
        tmpHeaderBits.append(sum(tmpHeaderBits)%2)
        # add 6 tail bits
        tmpHeaderBits += [0] * 6
        if self.ifdb: print("legacy len:", self.m.legacyLen)
        if self.ifdb: print("legacy sig bit length:", len(tmpHeaderBits))
        if self.ifdb: print(tmpHeaderBits)
        # convolution of sig bits, no scrambling for header
        tmpHeaderCodedBits = p8h.procBcc(tmpHeaderBits, p8h.CR.CR12)
        # interleave of header bits
        if self.ifdb: print("legacy sig coded bits: %d" % (len(tmpHeaderCodedBits)))
        if self.ifdb: print(tmpHeaderCodedBits)
        # interleave of header using standard method
        tmpHeaderInterleaveBits = p8h.procInterleaveSigL(tmpHeaderCodedBits)
        if self.ifdb: print("legacy sig interleaved bits %s" % (len(tmpHeaderInterleaveBits)))
        if self.ifdb: print(tmpHeaderInterleaveBits)
        tmpHeaderQam = []
        for each in tmpHeaderInterleaveBits:
            tmpHeaderQam.append(p8h.C_QAM_MODU_TAB[p8h.M.BPSK.value][int(each)])
        # add pilot
        tmpHeaderQam = p8h.procPilotInsert(tmpHeaderQam, [1, 1, 1, -1])
        # add non data sc
        tmpHeaderQam = p8h.procNonDataSC(p8h.procDcInsert(tmpHeaderQam))
        if (self.m.bw == p8h.BW.BW20):
            tmpHeaderQam = tmpHeaderQam
        elif (self.m.bw == p8h.BW.BW40):
            tmpHeaderQam = tmpHeaderQam * 2
        else:   # 80
            tmpHeaderQam = tmpHeaderQam * 4
        for ssItr in range(0, self.m.nSS):
            self.ssLegacySig.append(p8h.procGi(p8h.procToneScaling(p8h.procFftMod(p8h.procLegacyCSD(tmpHeaderQam, self.m.nSS, ssItr, self.m.spr)), p8h.C_SCALENTF_SIG_L[self.m.bw.value], self.m.nSS)))
        if self.ifdb: print("legacy sig sample len %d" % (len(self.ssLegacySig[0])))

    def __genHtSignal(self):
        self.ssHtSig = []
        tmpHtSigBits = []
        # mcs
        for i in range(0, 7):
            tmpHtSigBits.append((self.m.mcs >> i) & (1))
        # bw
        if(self.m.bw == p8h.BW.BW20):
            tmpHtSigBits.append(0)
        elif(self.m.bw == p8h.BW.BW40):
            tmpHtSigBits.append(1)
        else:
            print("cloud phy80211, phy80211 genHtSignal input bw error")
        # len
        for i in range(0, 16):
            tmpHtSigBits.append((self.m.psduLen >> i) & (1))
        # channel smoothing
        tmpHtSigBits.append(1)
        # not sounding
        tmpHtSigBits.append(1)
        # reserved
        tmpHtSigBits.append(1)
        # aggregation
        if(self.m.ampdu):
            tmpHtSigBits.append(1)
        else:
            tmpHtSigBits.append(0)
        # stbc not supported
        tmpHtSigBits += [0, 0]
        # only bcc is supported
        tmpHtSigBits.append(0)
        # short GI
        if(self.m.sgi):
            tmpHtSigBits.append(1)
        else:
            tmpHtSigBits.append(0)
        # no ess
        tmpHtSigBits += [0, 0]
        # crc 8
        tmpHtSigBits += p8h.genBitBitCrc8(tmpHtSigBits)
        # tail
        tmpHtSigBits += [0] * 6
        if self.ifdb: print("HT signal bits: %d" % (len(tmpHtSigBits)))
        if self.ifdb: print(tmpHtSigBits)
        # coding
        tmpHtSigCodedBits = p8h.procBcc(tmpHtSigBits, p8h.CR.CR12)
        if self.ifdb: print("HT coded bits: %d" % (len(tmpHtSigCodedBits)))
        if self.ifdb: print(tmpHtSigCodedBits)
        # interleave
        tmpHtSigIntedBits = p8h.procInterleaveSigL(tmpHtSigCodedBits[0:48]) + p8h.procInterleaveSigL(tmpHtSigCodedBits[48:96])
        if self.ifdb: print("HT interleaved bits: %d" % (len(tmpHtSigIntedBits)))
        if self.ifdb: print(tmpHtSigIntedBits)
        # bpsk modulation
        tmpSig1Qam = [p8h.C_QAM_MODU_TAB[p8h.M.QBPSK.value][each] for each in tmpHtSigIntedBits[0:48]]
        tmpSig2Qam = [p8h.C_QAM_MODU_TAB[p8h.M.QBPSK.value][each] for each in tmpHtSigIntedBits[48:96]]
        # insert pilot and non data sc
        tmpSig1Qam = p8h.procNonDataSC(p8h.procDcInsert(p8h.procPilotInsert(tmpSig1Qam, [1,1,1,-1])))
        tmpSig2Qam = p8h.procNonDataSC(p8h.procDcInsert(p8h.procPilotInsert(tmpSig2Qam, [1,1,1,-1])))
        # higher bw
        if (self.m.bw == p8h.BW.BW40):
            tmpSig1Qam = tmpSig1Qam * 2
            tmpSig2Qam = tmpSig2Qam * 2
        for ssItr in range(0, self.m.nSS):
            self.ssHtSig.append(
                p8h.procConcat2Symbol(
                p8h.procGi(p8h.procToneScaling(p8h.procFftMod(p8h.procLegacyCSD(tmpSig1Qam, self.m.nSS, ssItr, self.m.spr)),p8h.C_SCALENTF_SIG_HT[self.m.bw.value], self.m.nSS)),
                p8h.procGi(p8h.procToneScaling(p8h.procFftMod(p8h.procLegacyCSD(tmpSig2Qam, self.m.nSS, ssItr, self.m.spr)),p8h.C_SCALENTF_SIG_HT[self.m.bw.value], self.m.nSS))))

    def __genVhtSignalA(self):
        self.ssVhtSigA = []
        tmpVhtSigABits = []
        # b 0 1, bw
        for i in range(0, 2):
            tmpVhtSigABits.append((self.m.bw.value >> i) & (1))
        # b 2, reserved
        tmpVhtSigABits.append(1)
        # b 3, STBC not supported
        tmpVhtSigABits.append(0)
        # b 4 9, group id
        for i in range(0, 6):
            tmpVhtSigABits.append((self.vhtGroupId >> i) & (1))
        if(self.m.mu):
            # multiple user mimo
            # b 10 21, user nSTS, 3 bits per user
            for u in range(0, self.nUserMu):
                for i in range(0, 3):
                    tmpVhtSigABits.append((self.mMu[u].nSTS >> i) & (1))
            for u in range(0, 4 - self.nUserMu):
                for i in range(0, 3):
                    tmpVhtSigABits.append(0)
        else:
            # single user
            # b 10 12, SU nSTS
            for i in range(0, 3):
                tmpVhtSigABits.append(((self.m.nSTS - 1) >> i) & (1))
            # b 13 21, Partial AID
            for i in range(0, 9):
                tmpVhtSigABits.append((self.vhtPartialAid >> i) & (1))
        # b 22 txop ps not allowed, set 0, allowed
        tmpVhtSigABits.append(0)
        # b 23 reserved
        tmpVhtSigABits.append(1)
        # b 0 short GI
        if(self.m.sgi):
            tmpVhtSigABits.append(1)
        else:
            tmpVhtSigABits.append(0)
        # b 1 short GI disam
        if(self.m.sgi and (self.m.nSym%10) == 9):
            tmpVhtSigABits.append(1)
        else:
            tmpVhtSigABits.append(0)
        # b 2 coding
        if (self.m.mu):
            # MU user 0 coding, BCC
            tmpVhtSigABits.append(0)
        else:
            # SU coding, BCC
            tmpVhtSigABits.append(0)
        # b 3 LDPC extra
        tmpVhtSigABits.append(0)
        # b 4 8
        if (self.m.mu):
            # b 4 6, MU 1 3 coding
            # if user in pos, then bcc
            for u in range(1, self.nUserMu):
                tmpVhtSigABits.append(0)
            # others reserved
            for u in range(0, 4 - self.nUserMu):
                tmpVhtSigABits.append(1)
            # b 7, MU reserved
            tmpVhtSigABits.append(1)
            # b 8, beamformed reserved
            tmpVhtSigABits.append(1)
        else:
            # b 4 7, MCS
            for i in range(0, 4):
                tmpVhtSigABits.append((self.m.mcs >> i) & (1))
            # b 8, beamformed
            tmpVhtSigABits.append(0)
        # b 9, reserved
        tmpVhtSigABits.append(1)
        # b 10 17 crc
        tmpVhtSigABits += p8h.genBitBitCrc8(tmpVhtSigABits)
        # b 18 23 tail
        tmpVhtSigABits += [0] * 6
        if self.ifdb: print("VHT sig A bits: %d" % (len(tmpVhtSigABits)))
        if self.ifdb: print(tmpVhtSigABits)
        tmpCodedBits = p8h.procBcc(tmpVhtSigABits, p8h.CR.CR12)
        if self.ifdb: print("VHT sig A coded bits: %d" % len(tmpCodedBits))
        if self.ifdb: print(tmpCodedBits)
        tmpIntedBits = p8h.procInterleaveSigL(tmpCodedBits[0:48]) + p8h.procInterleaveSigL(tmpCodedBits[48:96])
        if self.ifdb: print("VHT sig A interleaved bits: %d" % len(tmpIntedBits))
        if self.ifdb: print(tmpIntedBits)
        # bpsk modulation
        tmpSig1Qam = [p8h.C_QAM_MODU_TAB[p8h.M.BPSK.value][each] for each in tmpIntedBits[0:48]]
        tmpSig2Qam = [p8h.C_QAM_MODU_TAB[p8h.M.QBPSK.value][each] for each in tmpIntedBits[48:96]]
        # insert pilot and non data sc
        tmpSig1Qam = p8h.procNonDataSC(p8h.procDcInsert(p8h.procPilotInsert(tmpSig1Qam, [1,1,1,-1])))
        tmpSig2Qam = p8h.procNonDataSC(p8h.procDcInsert(p8h.procPilotInsert(tmpSig2Qam, [1,1,1,-1])))
        # higher bw
        if (self.m.bw == p8h.BW.BW40):
            tmpSig1Qam = tmpSig1Qam * 2
            tmpSig2Qam = tmpSig2Qam * 2
        elif (self.m.bw == p8h.BW.BW80):
            tmpSig1Qam = tmpSig1Qam * 4
            tmpSig2Qam = tmpSig2Qam * 4
        for ssItr in range(0, self.m.nSS):
            self.ssVhtSigA.append(
                p8h.procConcat2Symbol(
                p8h.procGi(p8h.procToneScaling(p8h.procFftMod(p8h.procLegacyCSD(tmpSig1Qam, self.m.nSS, ssItr, self.m.spr)),p8h.C_SCALENTF_SIG_VHT_A[self.m.bw.value], self.m.nSS)),
                p8h.procGi(p8h.procToneScaling(p8h.procFftMod(p8h.procLegacyCSD(tmpSig2Qam, self.m.nSS, ssItr, self.m.spr)),p8h.C_SCALENTF_SIG_VHT_A[self.m.bw.value], self.m.nSS))))

    

    def __genNonLegacyTraining(self):
        tmpSsNonLegacyTraining = []
        for i in range(0, self.m.nSS):
            tmpSsNonLegacyTraining.append([])
        # short training field, consider the beamforming spatial mapping
        tmpStf = []
        for ssItr in range(0, self.m.nSS):
            tmpStf.append(p8h.procCSD(p8h.procNonDataSC(p8h.C_STF_VHT[self.m.bw.value]), self.m.nSS, ssItr, self.m.spr))
        if(self.m.phyFormat == p8h.F.VHT and self.m.mu):
            tmpStf = p8h.procSpatialMapping(tmpStf, self.bfQ)
        for ssItr in range(0, self.m.nSS):
            tmpSsNonLegacyTraining[ssItr] = p8h.procGi(p8h.procToneScaling(p8h.procFftMod(tmpStf[ssItr]), p8h.C_SCALENTF_STF_VHT[self.m.bw.value], self.m.nSS))
        # long training field, consider the vht different polarity setting and beamforming spatial mapping
        for ltfIter in range(0, self.m.nLtf):
            # LTF is processed symbol by symbol
            tmpLtfSs = []
            for ssItr in range(0, self.m.nSS):
                if(self.m.phyFormat == p8h.F.VHT):
                    """
                        vht uses different polarity in LTF for data sc and pilot sc
                    """
                    tmpVhtLtf = []
                    for j in range(0, len(p8h.C_LTF_VHT[self.m.bw.value])):
                        if (self.m.bw == p8h.BW.BW20 and (j - 28) in [-21, -7, 7, 21]):
                            tmpVhtLtf.append(p8h.C_LTF_VHT[self.m.bw.value][j] * p8h.C_R_LTF_VHT_4[ltfIter])
                        elif(self.m.bw == p8h.BW.BW40 and (j - 58) in [-53, -25, -11, 11, 25, 53]):
                            tmpVhtLtf.append(p8h.C_LTF_VHT[self.m.bw.value][j] * p8h.C_R_LTF_VHT_4[ltfIter])
                        elif (self.m.bw == p8h.BW.BW80 and (j - 122) in [-103, -75, -39, -11, 11, 39, 75, 103]):
                            tmpVhtLtf.append(p8h.C_LTF_VHT[self.m.bw.value][j] * p8h.C_R_LTF_VHT_4[ltfIter])
                        else:   # ht or vht non pilot sub carriers
                            tmpVhtLtf.append(p8h.C_LTF_VHT[self.m.bw.value][j] * p8h.C_P_LTF_VHT_4[ssItr][ltfIter])
                    if self.ifdb: print("ltf n ", ltfIter, ", ss ", ssItr, p8h.procNonDataSC(tmpVhtLtf))
                    tmpLtfSs.append(p8h.procCSD(p8h.procNonDataSC(tmpVhtLtf), self.m.nSS, ssItr, self.m.spr))
                else:
                    tmpVhtLtf = []
                    for eachSc in p8h.C_LTF_HT[self.m.bw.value]:
                        tmpVhtLtf.append(eachSc * p8h.C_P_LTF_VHT_4[ssItr][ltfIter])
                    tmpLtfSs.append(p8h.procCSD(p8h.procNonDataSC(tmpVhtLtf), self.m.nSS, ssItr, self.m.spr))
            # spatial mapping for beamforming
            if(self.m.phyFormat == p8h.F.VHT and self.m.mu):
                tmpLtfSs = p8h.procSpatialMapping(tmpLtfSs, self.bfQ)
            for ssItr in range(0, self.m.nSS):
                tmpSsNonLegacyTraining[ssItr] = p8h.procConcat2Symbol(
                    tmpSsNonLegacyTraining[ssItr],
                    p8h.procGi(p8h.procToneScaling(p8h.procFftMod(tmpLtfSs[ssItr]), p8h.C_SCALENTF_LTF_VHT[self.m.bw.value], self.m.nSS)))
        if(self.m.phyFormat == p8h.F.VHT):
            self.ssVhtTraining = tmpSsNonLegacyTraining
        else:
            self.ssHtTraining = tmpSsNonLegacyTraining
        for ssIter in range(0, self.m.nSS):
            if self.ifdb: print("non legacy training ss %d, sample len %d" % (ssIter, len(tmpSsNonLegacyTraining[ssIter])))
            if self.ifdb: print(tmpSsNonLegacyTraining[ssIter])

    def __genVhtSignalB(self):
        self.ssVhtSigB = []
        tmpVhtSigBBits = []
        # bits for length
        # compute APEP Len first, single user, use mpdu byte number as APEP len
        tmpSigBLen = int(np.ceil(self.m.ampduLen/4))
        if self.ifdb: print("sig b len: %d" % tmpSigBLen)
        if(self.m.bw == p8h.BW.BW20):
            tmpLenBitN = 17
            tmpReservedBitN = 3
            tmpSigBMod = p8h.modulation(phyFormat=p8h.F.VHT, mcs=0, bw=p8h.BW.BW20, nSTS=1, shortGi=False)
        elif(self.m.bw == p8h.BW.BW40):
            tmpLenBitN = 19
            tmpReservedBitN = 2
            tmpSigBMod = p8h.modulation(phyFormat=p8h.F.VHT, mcs=0, bw=p8h.BW.BW40, nSTS=1, shortGi=False)
        else:   # bw 80
            tmpLenBitN = 21
            tmpReservedBitN = 2
            tmpSigBMod = p8h.modulation(phyFormat=p8h.F.VHT, mcs=0, bw=p8h.BW.BW80, nSTS=1, shortGi=False)
        tmpSigBMod.nSym = 1 # for interleave
        if(self.m.ampduLen > 0):
            for i in range(0, tmpLenBitN):
                tmpVhtSigBBits.append((tmpSigBLen >> i) & (1))
            # bits for reserved
            tmpVhtSigBBits = tmpVhtSigBBits + [1] * tmpReservedBitN
            # crc 8 for sig b used in data
            self.vhtSigBCrcBits = p8h.genBitBitCrc8(tmpVhtSigBBits)
        else:
            tmpVhtSigBBits = p8h.C_NDP_SIG_B[self.m.bw.value]
        # bits for tail
        tmpVhtSigBBits = tmpVhtSigBBits + [0] * 6
        if(self.m.bw == p8h.BW.BW40):
            tmpVhtSigBBits = tmpVhtSigBBits * 2
        elif(self.m.bw == p8h.BW.BW80):
            tmpVhtSigBBits = tmpVhtSigBBits * 2 + [0]
        if self.ifdb: print("vht sig b bits: %d" % len(tmpVhtSigBBits))
        if self.ifdb: print(tmpVhtSigBBits)
        # convolution
        tmpVhtSigBCodedBits = p8h.procBcc(tmpVhtSigBBits, p8h.CR.CR12)
        if self.ifdb: print("vht sig b convolved bits: %d" % (len(tmpVhtSigBCodedBits)))
        if self.ifdb: print(tmpVhtSigBCodedBits)
        # no segment parse, interleave
        tmpIntedBits = p8h.procInterleaveNonLegacy([tmpVhtSigBCodedBits], tmpSigBMod)[0]
        if self.ifdb: print("VHT sig b interleaved bits: %d" % len(tmpIntedBits))
        if self.ifdb: print(tmpIntedBits)
        # modulation
        for ssItr in range(0, self.m.nSS):
            tmpSigQam = [p8h.C_QAM_MODU_TAB[p8h.M.BPSK.value][each] for each in tmpIntedBits]
            # map constellations to user specific P_VHT_LTF, actually only flip the BPSK when nSS is 4, 7 or 8
            if(self.m.nSS in [4, 7, 8]):
                tmpSigQam = [each * p8h.C_P_SIG_B_NSTS478[ssItr] for each in tmpSigQam]
            tmpSigQam = p8h.procNonDataSC(p8h.procDcInsert(p8h.procPilotInsert(tmpSigQam, p8h.C_PILOT_VHT[self.m.bw.value])))
            self.ssVhtSigB.append(p8h.procGi(p8h.procToneScaling(
                p8h.procFftMod(p8h.procCSD(tmpSigQam, self.m.nSS, ssItr, self.m.spr)),
                p8h.C_SCALENTF_SIG_VHT_B[self.m.bw.value], self.m.nSS)))
        for ssIter in range(0, self.m.nSS):
            if self.ifdb: print("ss %d VHT signal B: %d" % (ssIter, len(self.ssVhtSigB[ssIter])))
            if self.ifdb: print(self.ssVhtSigB[ssIter])

    def __genVhtSignalBMu(self):
        self.ssVhtSigB = []
        self.vhtSigBCrcBitsMu = []
        tmpSsSigQam = []
        for u in range(0, self.nUserMu):
            tmpVhtSigBBits = []
            # compute APEP Len first, single user, use mpdu byte number as APEP len
            tmpSigBLen = int(len(self.ampduMu[u])/4)
            if self.ifdb: print("user %d sig b len: %d" % (u, tmpSigBLen))
            if(self.m.bw == p8h.BW.BW20):
                tmpLenBitN = 16
                tmpSigBMod = p8h.modulation(phyFormat=p8h.F.VHT, mcs=0, bw=p8h.BW.BW20, nSTS=1, shortGi=False)
            elif(self.m.bw == p8h.BW.BW40):
                tmpLenBitN = 17
                tmpSigBMod = p8h.modulation(phyFormat=p8h.F.VHT, mcs=0, bw=p8h.BW.BW40, nSTS=1, shortGi=False)
            else:   # bw 80
                tmpLenBitN = 19
                tmpSigBMod = p8h.modulation(phyFormat=p8h.F.VHT, mcs=0, bw=p8h.BW.BW80, nSTS=1, shortGi=False)
            tmpSigBMod.nSym = 1 # for interleave
            tmpMcsBitN = 4
            # bits for length
            for i in range(0, tmpLenBitN):
                tmpVhtSigBBits.append((tmpSigBLen >> i) & (1))
            # bits for mcs
            for i in range(0, tmpMcsBitN):
                tmpVhtSigBBits.append((self.mMu[u].mcs >> i) & (1))
            # crc 8 for sig b used in data
            self.vhtSigBCrcBitsMu.append(p8h.genBitBitCrc8(tmpVhtSigBBits))
            if self.ifdb: print("vht sig b crc bits:", self.vhtSigBCrcBitsMu[u])
            # bits for tail
            tmpVhtSigBBits = tmpVhtSigBBits + [0] * 6
            if(self.m.bw == p8h.BW.BW40):
                tmpVhtSigBBits = tmpVhtSigBBits * 2
            elif(self.m.bw == p8h.BW.BW80):
                tmpVhtSigBBits = tmpVhtSigBBits * 2 + [0]
            if self.ifdb: print("vht sig b bits: %d" % len(tmpVhtSigBBits))
            if self.ifdb: print(tmpVhtSigBBits)
            # convolution
            tmpVhtSigBCodedBits = p8h.procBcc(tmpVhtSigBBits, p8h.CR.CR12)
            if self.ifdb: print("vht sig b convolved bits: %d" % (len(tmpVhtSigBCodedBits)))
            if self.ifdb: print(tmpVhtSigBCodedBits)
            # no segment parse, interleave
            tmpIntedBits = p8h.procInterleaveNonLegacy([tmpVhtSigBCodedBits], tmpSigBMod)[0]
            if self.ifdb: print("VHT sig b interleaved bits: %d" % len(tmpIntedBits))
            if self.ifdb: print(tmpIntedBits)
            # modulation
            for ssItr in range(0, self.mMu[u].nSS):
                tmpSigQam = [p8h.C_QAM_MODU_TAB[p8h.M.BPSK.value][each] for each in tmpIntedBits]
                tmpSsSigQam.append(tmpSigQam)
        for ssItr in range(0, self.m.nSS):
            """
            - map constellations to user specific P_VHT_LTF, actually only flip the BPSK when nSS is 4, 7 or 8
            - beacause the P_VHT_LTF first column has -1
            """
            if(self.m.nSS in [4, 7, 8]):
                tmpSsSigQam[ssItr] = [each * p8h.C_P_SIG_B_NSTS478[ssItr] for each in tmpSsSigQam[ssItr]]
            tmpSsSigQam[ssItr] = p8h.procDcInsert(p8h.procPilotInsert(tmpSsSigQam[ssItr], p8h.C_PILOT_VHT[self.m.bw.value]))
            tmpSsSigQam[ssItr] = p8h.procCSD(p8h.procNonDataSC(tmpSsSigQam[ssItr]), self.m.nSS, ssItr, self.m.spr)
        # add spatial mapping
        tmpSsSigQamQ = p8h.procSpatialMapping(tmpSsSigQam, self.bfQ)
        for ssItr in range(0, self.m.nSS):
            self.ssVhtSigB.append(p8h.procGi(p8h.procToneScaling(
                    p8h.procFftMod(tmpSsSigQamQ[ssItr]),
                    p8h.C_SCALENTF_SIG_VHT_B[self.m.bw.value], self.m.nSS)))
        for each in self.ssVhtSigB:
            if self.ifdb: print("VHT signal B: %d" % len(each))
            if self.ifdb: print(each)

    def __genDataBits(self):
        self.dataBits = []
        if(self.m.phyFormat == p8h.F.VHT):
            """ 
            - psdu is multiple of 8 bits, exclude service 16, include ampdu, inlcude eof pad n*4, include pad octects 0-3 * 8, exclude the pad bits 0-7, exclude tail bits nES*6 
            - input of bcc includes service, psdu and pad bits, after coding the tails are added
            - vht is scrambled first then add tail bits and then coded
            """
            if self.ifdb: print("vht data apep len: %d, psdu len: %d, pad eof: %d, pad octets: %d, pad bits: %d, totoal bits: %d" % (self.m.ampduLen, self.m.psduLen, self.m.nPadEof, self.m.nPadOctet, self.m.nPadBits, (self.m.nSym * self.m.nDBPS)))
            # convert MAC data
            tmpAmpduBits = []
            for each in self.ampdu:
                for i in range(0,8):
                    tmpAmpduBits.append((each>>i) & (1))
            # service bits, 7 scrambler init, 1 reserved, sig b crc
            tmpServiceBits = [0] * 8 + self.vhtSigBCrcBits
            if self.ifdb: print("vht service bits: ", tmpServiceBits)
            # to do the eof padding
            tmpPsduBits = tmpAmpduBits + p8h.C_VHT_EOF * self.m.nPadEof + [0] * 8 * self.m.nPadOctet
            self.dataBits = tmpServiceBits + tmpPsduBits + [0] * self.m.nPadBits
        else:
            """
            - legacy and ht, just service, psdu is mpdu, tail bits nES*6 and padded bits 
            - legacy and ht, first scramble all and reset the tail and then coded
            """
            # convert MAC data
            
            if(self.m.ampdu):
                tmpAmpduBits = []
                for each in self.ampdu:
                    for i in range(0,8):
                        tmpAmpduBits.append((each>>i) & (1))
                tmpPsduBits = tmpAmpduBits
            else:
                tmpMpduBits = []
                for each in self.mpdu:
                    for i in range(0,8):
                        tmpMpduBits.append((each>>i) & (1))
                tmpPsduBits = tmpMpduBits
            # service bits
            tmpServiceBits = [0] * 16
            self.dataBits = tmpServiceBits + tmpPsduBits + [0] * 6 * self.m.nES + [0] * self.m.nPadBits
        if self.ifdb: print("data bits: %d" % len(self.dataBits))
        if self.ifdb: print(self.dataBits)

    def __genCodedBits(self):
        self.esCodedBits = []
        tmpScrambledBits = p8h.procScramble(self.dataBits, self.dataScrambler)
        if self.ifdb: print("scrambled bits: %d" % len(tmpScrambledBits))
        if self.ifdb: print(tmpScrambledBits)
        if(self.m.phyFormat == p8h.F.VHT):
            for i in range(0, self.m.nES):
                # divide scrambled bits for bcc coders, since tail is not added yet, minus 6
                tmpDividedBits = [tmpScrambledBits[each] for each in range((0+i), int(self.m.nDBPS * self.m.nSym / self.m.nES - 6), self.m.nES)]
                # add tail bits to it
                tmpDividedBits = tmpDividedBits + [0] * 6
                # coding and puncturing
                self.esCodedBits.append(p8h.procBcc(tmpDividedBits, self.m.cr))
        else:
            # legacy and ht, reset the tail bits
            for i in range(0, self.m.nES * 6):
                tmpScrambledBits[16 + self.m.psduLen*8 + i] = 0
            if self.ifdb: print("scrambled bits after reset tail: %d" % len(tmpScrambledBits))
            if self.ifdb: print(tmpScrambledBits)
            for i in range(0, self.m.nES):
                # divide scrambled bits for bcc coders, for HT bits are divided alternatively
                tmpDividedBits = [tmpScrambledBits[each] for each in range((0+i), int(self.m.nDBPS * self.m.nSym / self.m.nES), self.m.nES)]
                # coding and puncturing
                self.esCodedBits.append(p8h.procBcc(tmpDividedBits, self.m.cr))
        for each in self.esCodedBits:
            if self.ifdb: print("coded bits: %d" % len(each))
            if self.ifdb: print(each)

    def __genStreamParserDataBits(self):
        """
            each round get S bits from one encoder, then assign the S bits to the streams
            this works for HT and VHT usual conditions
            for vht 20/40/80 4x4, nBlock * nES * S is smaller than nCBPSS
        """
        self.ssStreamBits = []
        if(self.m.phyFormat == p8h.F.L):
            self.ssStreamBits = self.esCodedBits
        else:
            for i in range(0, self.m.nSS):
                self.ssStreamBits.append([0] * self.m.nSym * self.m.nCBPSS)
            s = int(max(1, self.m.nBPSCS/2))
            cs = self.m.nSS * s     # cs is the capital S used in standard
            for isym in range(0, self.m.nSym):
                for iss in range(0, self.m.nSS):
                    for k in range(0, int(self.m.nCBPSS)):
                        j = int(np.floor(k/s)) % self.m.nES
                        i = (iss) * s + cs * int(np.floor(k/(self.m.nES * s))) + int(k % s)
                        self.ssStreamBits[iss][k + int(isym * self.m.nCBPSS)] = self.esCodedBits[j][i + int(isym * self.m.nCBPS)]
        for each in self.ssStreamBits:
            if self.ifdb: print("stream parser bits:", len(each))
            if self.ifdb: print(each)

    def __genInterleaveDataBits(self):
        self.ssIntedBits = []
        for i in range(0, self.m.nSS):
            self.ssIntedBits.append([])
        if(self.m.phyFormat == p8h.F.L):
            self.ssIntedBits = p8h.procInterleaveLegacy(self.ssStreamBits, self.m)
        else:
            self.ssIntedBits = p8h.procInterleaveNonLegacy(self.ssStreamBits, self.m)
        for each in self.ssIntedBits:
            if self.ifdb: print("interleaved stream bits:", len(each))
            if self.ifdb: print(each)

    def __genConstellation(self):
        self.ssSymbols = []
        for i in range(0, self.m.nSS):
            self.ssSymbols.append([])
        tmpCarrierNumStream = int(self.m.nSD * self.m.nSym)
        for ssItr in range(0, self.m.nSS):
            for i in range(0, tmpCarrierNumStream):
                tmpCarrierChip = 0
                for j in range(0, self.m.nBPSCS):
                    tmpCarrierChip += self.ssIntedBits[ssItr][i * self.m.nBPSCS + j] * (2 ** j)
                self.ssSymbols[ssItr].append(p8h.C_QAM_MODU_TAB[self.m.mod.value][tmpCarrierChip])
            if self.ifdb: print(ssItr, ssItr, ssItr)
        for each in self.ssSymbols:
            if self.ifdb: print("constellation data", len(each))

    def __genOfdmSignalMu(self):
        self.ssPhySig = []
        for i in range(0, self.m.nSS):
            self.ssPhySig.append([])
        for ssItr in range(0, self.m.nSS):
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssLegacyTraining[ssItr], self.ssLegacySig[ssItr])
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], self.ssVhtSigA[ssItr])
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], self.ssVhtTraining[ssItr])
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], self.ssVhtSigB[ssItr])

        tmpPilot = p8h.C_PILOT_VHT[self.m.bw.value]
        tmpPilotPIdx = 4
        for symIter in range(0, self.m.nSym):
            tmpSsDataSig = []
            for ssItr in range(0, self.m.nSS):
                tmpPilotAdded = p8h.procPilotInsert(self.ssSymbolsMu[ssItr][int(symIter*self.m.nSD): int((symIter+1)*self.m.nSD)], [each * p8h.C_PILOT_PS[tmpPilotPIdx] for each in tmpPilot])
                tmpSsDataSig.append(p8h.procCSD(p8h.procNonDataSC(p8h.procDcInsert(tmpPilotAdded)), self.m.nSS, ssItr, self.m.spr))
            tmpSsDataSigQ = p8h.procSpatialMapping(tmpSsDataSig, self.bfQ)
            for ssItr in range(0, self.m.nSS):
                self.ssPhySig[ssItr] = p8h.procConcat2Symbol(
                    self.ssPhySig[ssItr],
                    p8h.procGi(p8h.procToneScaling(p8h.procFftMod(tmpSsDataSigQ[ssItr]), p8h.C_SCALENTF_DATA_VHT[self.m.bw.value], self.m.nSS)))
            tmpPilotPIdx = (tmpPilotPIdx + 1) % 127
            tmpPilot = tmpPilot[1:] + [tmpPilot[0]]

    def __genOfdmSignal(self):
        self.ssPhySig = []
        for i in range(0, self.m.nSS):
            self.ssPhySig.append([])
        for ssItr in range(0, self.m.nSS):
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssLegacyTraining[ssItr], self.ssLegacySig[ssItr])
            if(self.m.phyFormat == p8h.F.VHT):
                self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], self.ssVhtSigA[ssItr])
                self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], self.ssVhtTraining[ssItr])
                self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], self.ssVhtSigB[ssItr])
                tmpPilot = p8h.C_PILOT_VHT[self.m.bw.value]
                tmpPilotPIdx = 4
                tmpDataScaleFactor = p8h.C_SCALENTF_DATA_VHT[self.m.bw.value]
            elif(self.m.phyFormat == p8h.F.HT):
                self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], self.ssHtSig[ssItr])
                self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], self.ssHtTraining[ssItr])
                tmpPilot = p8h.C_PILOT_HT[self.m.bw.value][self.m.nSS - 1][ssItr]
                tmpPilotPIdx = 3
                # print("cloud phy80211, gen ofdm get ht pilot: ", tmpPilot)
                tmpDataScaleFactor = p8h.C_SCALENTF_DATA_HT[self.m.bw.value]
            else:
                tmpPilot = p8h.C_PILOT_L
                tmpPilotPIdx = 1
                tmpDataScaleFactor = 52
            for symIter in range(0, self.m.nSym):
                tmpPilotAdded = p8h.procPilotInsert(self.ssSymbols[ssItr][int(symIter*self.m.nSD): int((symIter+1)*self.m.nSD)], [each * p8h.C_PILOT_PS[tmpPilotPIdx] for each in tmpPilot])
                self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], p8h.procGi(p8h.procToneScaling(
                p8h.procFftMod(p8h.procCSD(p8h.procNonDataSC(p8h.procDcInsert(tmpPilotAdded)), self.m.nSS, ssItr, self.m.spr)),
                tmpDataScaleFactor, self.m.nSS)))
                tmpPilotPIdx = (tmpPilotPIdx + 1) % 127
                if(not(self.m.phyFormat == p8h.F.L)):
                    tmpPilot = tmpPilot[1:] + [tmpPilot[0]]

    def __genOfdmSignalNdp(self):
        self.ssPhySig = []
        for i in range(0, self.m.nSS):
            self.ssPhySig.append([])
        for ssItr in range(0, self.m.nSS):
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssLegacyTraining[ssItr], self.ssLegacySig[ssItr])
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], self.ssVhtSigA[ssItr])
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], self.ssVhtTraining[ssItr])
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], self.ssVhtSigB[ssItr])

    def __genSignalWithCfo(self, inSig, cfoHz):
        self.cfohz = cfoHz
        tmpRadStep = cfoHz * 2.0 * np.pi / 20000000.0
        outSig = []
        for i in range(0, len(inSig)):
            outSig.append(inSig[i] * (np.cos(i * tmpRadStep) + np.sin(i * tmpRadStep) * 1j))
        return outSig

    def __genSignalWithAmp(self, inSig, m):
        return [each * m for each in inSig]

    def __procRxLegacyStfTrigger(self, inSig):
        if(len(inSig) > 32):
            tmpPlateau = 0
            for i in range(0, len(inSig) - 32):
                if(p8h.procCorrelation(inSig[i:i+16], inSig[i+16:i+32]) > 0.4):
                    tmpPlateau += 1
                    if(tmpPlateau > 20):
                        return i
                else:
                    tmpPlateau = 0
            return -1

    def __procRxLegacyStfCoarseCfoEst(self, inSig, nRad):
        if(len(inSig) >= (nRad + 16)):
            tmpMultiAvg = np.mean([inSig[i] * np.conj(inSig[i + 16]) for i in range(0, nRad)])
            return np.arctan2(np.imag(tmpMultiAvg), np.real(tmpMultiAvg)) / 16 * 20000000 / 2 / np.pi

    def __procRxLegacyLtfSync(self, inSig, coarseCfo):
        if(len(inSig) >= 240):
            tmpAutoCorre = []
            for i in range(0, 110):
                tmpAutoCorre.append(p8h.procCorrelation(inSig[i:i+64], inSig[i+64:i+128]))
            maxValue = max(tmpAutoCorre)
            maxIndex = tmpAutoCorre.index(maxValue)
            leftCorre = tmpAutoCorre[0:maxIndex]
            rightCorre = tmpAutoCorre[maxIndex:]
            if(len(leftCorre) and len(rightCorre)):
                leftIndex = min(range(len(leftCorre)), key = lambda i: abs(leftCorre[i]-maxValue*0.8))
                rightIndex = min(range(len(rightCorre)), key = lambda i: abs(rightCorre[i]-maxValue*0.8))
                midIndex = int((leftIndex + rightIndex + maxIndex)/2)
                tmpRadStep = coarseCfo * 2 * np.pi / 20000000
                tmpSig = [inSig[midIndex + i] * complex(np.cos(tmpRadStep*i), np.sin(tmpRadStep*i)) for i in range(0, 128)]
                tmpMultiAvg = np.mean([tmpSig[i] * np.conj(tmpSig[i + 64]) for i in range(0, 64)])
                return midIndex, (np.arctan2(np.imag(tmpMultiAvg), np.real(tmpMultiAvg)) / 64 * 20000000 / 2 / np.pi)
        return -1, 0

    def __procRxLegacyChanEst(self, inSig):
        if(len(inSig) >= 128):
            tmpLtf1 = p8h.procFftDemod(inSig[0:64])
            tmpLtf2 = p8h.procFftDemod(inSig[64:128])
            tmpLtfOri = p8h.procNonDataSC(p8h.C_LTF_L[p8h.BW.BW20.value])
            self.rxChanL = []
            for i in range(0, 64):
                self.rxChanL.append((tmpLtf1[i] + tmpLtf2[i]) / tmpLtfOri[i] / 2.0)

    def __procRxLegacySigDemod(self, inSig):
        if(len(inSig) >= 64):
            tmpSigFreq = p8h.procFftDemod(inSig[0:64])
            tmpSigFreq = [tmpSigFreq[i] / self.rxChanL[i] for i in range(0, 64)]
            tmpSigFreq = p8h.procRmDcNonDataSc(tmpSigFreq, p8h.F.L)
            tmpSigLlr = p8h.procRemovePilots(tmpSigFreq)
            

    def procSisoRx(self, inSig):
        if(isinstance(inSig, list) and len(inSig) > 480):
            self.rxSisoSig = inSig
            self.rxSampNum = len(inSig)
            print(self.rxSampNum)
            procIndex = 0
            while(self.rxSampNum > (procIndex + 480)):
                # find beginning of stf
                tmpTriggerIndex = self.__procRxLegacyStfTrigger(self.rxSisoSig[procIndex:])
                if(tmpTriggerIndex < 0):
                    procIndex += 80
                    continue
                stfIndex = procIndex + tmpTriggerIndex
                # estimate cfo with stf
                triggerCfo = self.__procRxLegacyStfCoarseCfoEst(self.rxSisoSig[stfIndex:], 64)
                # find beginning of ltf, mid of 80% shoulders, should be mid of GI, 
                tmpSyncIndex, syncCfo = self.__procRxLegacyLtfSync(self.rxSisoSig[stfIndex+80:stfIndex+320], triggerCfo)
                if(tmpSyncIndex < 0):
                    procIndex += 80
                    continue
                # total cfo is sum of stf and ltf cfo
                self.rxCfo = triggerCfo + syncCfo
                ltfIndex = stfIndex + 80 + tmpSyncIndex + 10
                # estimate legacy channel
                tmpCfoRadStep = self.rxCfo * 2 * np.pi / 20000000
                tmpLtfSig = [self.rxSisoSig[ltfIndex+i] * complex(np.cos(tmpCfoRadStep*i), np.sin(tmpCfoRadStep*i)) for i in range(0, 208)]
                self.__procRxLegacyChanEst(tmpLtfSig[0:128])
                # demod legacy signal part
                legacySigIndex = ltfIndex+144
                self.__procRxLegacySigDemod(tmpLtfSig[144:])
                print("cloud phy80211, procSisoRx, stf: %d, cfo: %f, ltf: %d" % (stfIndex, self.rxCfo, ltfIndex))
                procIndex = stfIndex + 80


    def genFinalSig(self, multiplier = 1.0, cfoHz = 0.0, num = 1, gap = True, gapLen = 10000):
        if not self.ssPhySig:
            print("cloud phy80211, genFinalSig phy sig is empty")
            return
        self.ssFinalSig = []
        if(num < 1):
            print("cloud phy80211, genFinalSig input param error")
            return
        for ssItr in range(0, self.m.nSS):
            tmpSig = self.__genSignalWithAmp(self.ssPhySig[ssItr], multiplier)
            tmpSig = self.__genSignalWithCfo(tmpSig, cfoHz)
            if(gap):
                tmpSig = ([0] * gapLen + tmpSig) * num + [0] * gapLen
            else:
                tmpSig = tmpSig * num
            self.ssFinalSig.append(tmpSig)
        return self.ssFinalSig
        

    def genSigBinFile(self, ssSig, fileAddr="", draw = False):
        if(len(fileAddr)<1):
            print("cloud phy80211, genSigBinFile file address not given")
            return
        if not ssSig:
            print("cloud phy80211, genSigBinFile input sig empty")
            return
        print("write signal into bin file")
        for ssItr in range(0, len(ssSig)):
            tmpFilePathStr = fileAddr + "_" + str(len(ssSig)) + "x" + str(len(ssSig)) + "_" + str(ssItr) + ".bin"
            binF = open(tmpFilePathStr, "wb")
            tmpSig = ssSig[ssItr]
            print("%d sample number %d" % (ssItr, len(tmpSig)))
            for i in range(0, len(tmpSig)):
               binF.write(struct.pack("f", np.real(tmpSig[i])))
               binF.write(struct.pack("f", np.imag(tmpSig[i])))
            binF.close()
            print("written in " + tmpFilePathStr)
            if(draw):
                plt.figure(100 + ssItr)
                plt.ylim([-1, 1])
                plt.plot(np.real(tmpSig))
                plt.plot(np.imag(tmpSig))
        if(draw):
            plt.show()
    
    def genMultiSigBinFile(self, ssSigList, fileAddr="", draw = False):
        if(len(fileAddr)<1):
            print("cloud phy80211, genSigBinFile file address not given")
            return
        if not ssSigList:
            print("cloud phy80211, genSigBinFile input sig empty")
            return
        print("write multiple signal into bin file")
        tmpSsNum = len(ssSigList[0])
        ssSigConcate = []
        for ssItr in range(0, tmpSsNum):
            ssSigConcate.append([])
        for eachSsSig in ssSigList:
            if(len(eachSsSig) != tmpSsNum):
                print("cloud phy80211, genMultiSigBinFile input sssig item len error")
                return
            for ssItr in range(0, tmpSsNum):
                ssSigConcate[ssItr] += eachSsSig[ssItr]
        for ssItr in range(0, tmpSsNum):
            tmpFilePathStr = fileAddr + "_" + str(tmpSsNum) + "x" + str(tmpSsNum) + "_" + str(ssItr) + ".bin"
            binF = open(tmpFilePathStr, "wb")
            tmpSig = ssSigConcate[ssItr]
            print("%d sample number %d" % (ssItr, len(tmpSig)))
            for i in range(0, len(tmpSig)):
               binF.write(struct.pack("f", np.real(tmpSig[i])))
               binF.write(struct.pack("f", np.imag(tmpSig[i])))
            binF.close()
            print("written in " + tmpFilePathStr)
            if(draw):
                plt.figure(100 + ssItr)
                plt.ylim([-1, 1])
                plt.plot(np.real(tmpSig))
                plt.plot(np.imag(tmpSig))
        if(draw):
            plt.show()
    
    def genSigOwTextFile(self, ssSig, fileAddr="", draw = False):
        if(len(fileAddr)<1):
            print("cloud phy80211, genSigBinFile file address not given")
            return
        if not ssSig:
            print("cloud phy80211, genSigBinFile input sig empty")
            return
        print("write signal into bin file")
        for ssItr in range(0, len(ssSig)):
            tmpFilePathStr = fileAddr + "_" + str(len(ssSig)) + "x" + str(len(ssSig)) + "_" + str(ssItr) + ".txt"
            textF = open(tmpFilePathStr, "w")
            tmpSig = ssSig[ssItr]
            print("stream %d sample number %d" % (ssItr, len(tmpSig)))
            for i in range(0, len(tmpSig)):
               tmpSig[i] = int(np.real(tmpSig[i])) + int(np.imag(tmpSig[i])) * 1j
            for i in range(0, len(tmpSig)):
               tmpStr = str(int(np.real(tmpSig[i]))) + " " + str(int(np.imag(tmpSig[i]))) + "\n"
               textF.write(tmpStr)
            textF.close()
            print("written in " + tmpFilePathStr)
            if(draw):
                plt.figure(100 + ssItr)
                plt.plot(np.real(tmpSig))
                plt.plot(np.imag(tmpSig))
        if(draw):
            plt.show()

def genPktGrData(mpdu, mod):
    if(isinstance(mpdu, (bytes, bytearray)) and isinstance(mod, p8h.modulation) and len(mpdu) < 4096):
        tmpBytes = b""
        tmpBytes += struct.pack('<B', mod.phyFormat.value)
        tmpBytes += struct.pack('<B', mod.mcs)
        tmpBytes += struct.pack('<B', mod.nSTS)
        tmpBytes += struct.pack('<H', len(mpdu))
        tmpBytes += mpdu
        return tmpBytes
    else:
        print("cloud 80211phy, genPktGrData input param error")
        return b""

def genPktGrDataMu(mpdu0, mod0, mpdu1, mod1, groupId):
    if(
        isinstance(mpdu0, (bytes, bytearray)) and 
        isinstance(mpdu1, (bytes, bytearray)) and 
        isinstance(mod0, p8h.modulation) and 
        isinstance(mod1, p8h.modulation) and 
        len(mpdu0) < 4096 and len(mpdu1) < 4096 and
        groupId > 0 and groupId < 63):
        tmpBytes = b""
        tmpBytes += struct.pack('<B', p8h.GR_F.MU.value)
        tmpBytes += struct.pack('<B', mod0.mcs)
        tmpBytes += struct.pack('<B', 1)    # nSTS 1
        tmpBytes += struct.pack('<H', len(mpdu0))
        tmpBytes += struct.pack('<B', mod1.mcs)
        tmpBytes += struct.pack('<B', 1)    # nSTS 1
        tmpBytes += struct.pack('<H', len(mpdu1))
        tmpBytes += struct.pack('<B', groupId)
        tmpBytes += mpdu0
        tmpBytes += mpdu1
        return tmpBytes
    else:
        print("cloud 80211phy, genPktGrDataMu input param error")
        return b""

def genPktGrBfQ(bfQ):
    if(isinstance(bfQ, list)):
        tmpBytesR = b"" + struct.pack('<B', p8h.GR_F.QR.value)
        tmpBytesI = b"" + struct.pack('<B', p8h.GR_F.QI.value)
        for each in bfQ:
            for i in range(0, 2):
                for j in range(0, 2):
                    tmpBytesR += struct.pack('<f', np.real(each[i][j]))
                    tmpBytesI += struct.pack('<f', np.imag(each[i][j]))
        return tmpBytesR, tmpBytesI
    else:
        print("cloud 80211phy, genPktGrData input param error")
        return b"", b""

def genPktGrNdp():
    return b"\x02\x00\x02\x00\x00"  # vht packet mcs 0 nSTS 2 len 0

def procVhtPhiQuanti(phi, bphi):
    tmpK = list(range(0, 2**bphi))
    tmpShift = np.pi / (2**(bphi))
    tmpPhi = [each * np.pi / (2**(bphi-1)) + tmpShift for each in tmpK]
    # print(tmpPhi)
    return min(range(len(tmpPhi)), key=lambda i: abs(tmpPhi[i]-phi))

def procVhtPsiQuanti(psi, bpsi):
    tmpK = list(range(0, 2**bpsi))
    tmpShift = np.pi / (2**(bpsi+2))
    tmpPsi = [each * np.pi / (2**(bpsi+1)) + tmpShift for each in tmpK]
    # print(tmpPsi)
    return min(range(len(tmpPsi)), key=lambda i: abs(tmpPsi[i]-psi))

def procVhtVCompress(v, codeBookInfo = 0, ifDebug = False):
    resValue = []       # value and type
    resType = []
    # Phi for Di, Psi for Gli, feedback type is MU-MIMO for VHT PPDU
    if(isinstance(v, np.ndarray) and isinstance(ifDebug, bool) and isinstance(codeBookInfo, int)):
        [m, n] = v.shape    # m is row, n is col
        if(ifDebug):
            print("V, get V %d row %d col" % (m, n))
            print(v)
        if(m > 0 and n > 0 and m >= n):
            if(codeBookInfo):
                nBitPhi = 9
                nBitPsi = 7
            else:
                nBitPhi = 7
                nBitPsi = 5
            if(ifDebug):
                print("quantization codebook %d, Phi %d bits, Psi %d bits" % (codeBookInfo, nBitPhi, nBitPsi))
            dt = np.zeros((n, n), dtype=complex)
            for j in range(0, n):
                tmpTheta = np.arctan2(np.imag(v[m-1][j]), np.real(v[m-1][j]))
                tmpValue = np.exp(tmpTheta * 1.0j)
                dt[j][j] = tmpValue
            if(ifDebug):
                print("Dt, get D tilde")
                print(dt)
            dtH = dt.conjugate().T
            if(ifDebug):
                print("DtH, get D tilde hermitian")
                print(dtH)
            vdtH = np.matmul(v, dtH)
            if(ifDebug):
                vdtHRes = np.matmul(v, dtH)
            if(ifDebug):
                print("VDtH, get V dot D tilde hermitian, which is also the matrix with all real number on the last row, to be decomposed with givens rotation")
                print(vdtH)
            for j in range(0, n):
                vdtH[m-1][j] = np.real(vdtH[m-1][j])
            if(ifDebug):
                print("VDtH, get V dot D tilde hermitian, remove residual imag for last row")
                print(vdtH)
            
            glidiHvdtH_name = "VDtH"
            if(ifDebug):
                vtRes = np.identity(m)
            # each loop we compute Di and following Gli(s), and keep updating the vdtH to be GliDiHVDtH
            for grIter in range(0, min(m-1, n)):      # gr for givens rotation
                i = grIter + 1
                if(ifDebug):
                    print("givens rotation loop round %d" % (i))
                di = np.zeros((m, m), dtype=complex)
                for j in range(0, i-1):
                    di[j][j] = 1
                diPhi = []
                for j in range(i, m):
                    tmpTheta = np.arctan2(np.imag(vdtH[j-1][i-1]), np.real(vdtH[j-1][i-1]))
                    diPhi.append(tmpTheta)
                    if(ifDebug):
                        print("D%d Phi:%f" % (i, tmpTheta))
                    tmpValue = np.exp(tmpTheta * 1.0j)
                    di[j-1][j-1] = tmpValue
                if(len(diPhi)):
                    diPhi = np.unwrap(diPhi)
                    if(diPhi[0] < 0):
                        diPhi += np.pi * 2
                    for each in diPhi:
                        resValue.append(procVhtPhiQuanti(each, nBitPhi))
                        resType.append(0)
                    if(ifDebug):
                        print("D%d Unwrapped Phi:" % (i), diPhi)
                        print("D%d Unwrapped Phi Quantized:" % (i), [procVhtPhiQuanti(each, nBitPhi) for each in diPhi])
                di[m-1][m-1] = 1
                if(ifDebug):
                    print("D%d, get Di here" % (i))
                    print(di)
                if(ifDebug):                
                    vtRes = np.matmul(vtRes, di) # compute the final Vt to compare
                vdtH = np.matmul(di.conjugate().T, vdtH)    # now vdtH is from VDtH to DiHVDtH
                if(ifDebug):
                    glidiHvdtH_name = "D%dH" % (i) + glidiHvdtH_name
                    print(glidiHvdtH_name + ", get D%d hermitian dot V dot D tilde hermitian" % (i))
                    print(vdtH)
                # remove residual imag of the column
                for l in range(i, m):
                    vdtH[l-1][i-1] = np.real(vdtH[l-1][i-1])
                if(ifDebug):
                    print(glidiHvdtH_name + ", the %dth column should be all real" % (i))
                    print(vdtH)
                for l in range(i+1, m+1):
                    if(ifDebug):
                        print("compute givens rotation G%d%d" % (l, i))
                    gli = np.zeros((m, m), dtype=complex)
                    x1 = np.real(vdtH[i-1][i-1])
                    x2 = np.real(vdtH[l-1][i-1])
                    y = np.sqrt(x1*x1 + x2*x2)
                    gliPsi = np.arccos(x1 / y)
                    resValue.append(procVhtPsiQuanti(gliPsi, nBitPsi))
                    resType.append(1)
                    if(ifDebug):
                        print("x1:%f, x2:%f, y:%f, GliPsi:%f, Quantized GliPsi:%d" % (x1, x2, y, gliPsi, procVhtPsiQuanti(gliPsi, nBitPsi)))
                    if(gliPsi < 0 or gliPsi > np.pi/2):
                        print("Gli psi value error!!!!!!!!!!!!!!!!!!!!")
                    for j in range(0, m):
                        gli[j][j] = 1
                    gli[i-1][i-1] = np.cos(gliPsi)
                    gli[l-1][i-1] = -np.sin(gliPsi)
                    gli[i-1][l-1] = np.sin(gliPsi)
                    gli[l-1][l-1] = np.cos(gliPsi)
                    if(ifDebug):
                        print("G%d%d" % (l, i))
                        print(gli)
                    vdtH = np.matmul(gli, vdtH)
                    gliT = gli.T
                    if(ifDebug):
                        vtRes = np.matmul(vtRes, gliT) # compute the final Vt to compare
                    if(ifDebug):
                        glidiHvdtH_name = "G%d%d" % (l, i) + glidiHvdtH_name
                        print(glidiHvdtH_name + ", now the %d%d location shoule be zero" % (l, i))
                        print(vdtH)
                    vdtH[l-1][i-1] = 0
                    if(ifDebug):
                        print(glidiHvdtH_name + ", remove %d%d residual error" % (l, i))
                        print(vdtH)
            if(ifDebug):
                vIt = np.zeros((m, n), dtype=complex) # I tilde for V tilde
                for j in range(0, min(m, n)):
                    vIt[j][j] = 1
                vtRes = np.matmul(vtRes, vIt)
                print("compare the VDtH and decompsed results")
                print(vdtHRes)
                print(vtRes)
                print(resValue)
                print(resType)
    return resValue, resType

if __name__ == "__main__":
    pass