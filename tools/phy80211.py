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
    def __init__(self):
        self.h = p8h.phy80211header()
        self.m = p8h.phy80211mod()
        self.f = p8h.phy80211format('l', 0, self.h.BW_20, 1, False)

        self.nSym = 0       # number of symbols in data field
        self.psduLen = 0
        self.apepLen = 0
        self.dataScrambler = 93  # scrambler used for scrambling, from 1 to 127
        self.vhtPartialAid = 0
        self.vhtGroupId = 0
        self.vhtPadN = 0
        self.vhtSigBCrcBits = []
        # ----------------------------------------------------------
        self.mpduMu = []
        self.fMu = []
        self.mMu = []
        self.ifMu = 0
        self.nMuUser = 0

        self.nSymMu = 0
        self.nSSMu = 0
        self.vhtSigBCrcBitsMu = []
        self.ssSymbolsMu = []

        self.bfQ = []
        #-----------------------------------------------------------
        self.ssP = [[], [], [], []]     # pilots of ss
        self.ssPPI = [0, 0, 0, 0]      # pilot polarity index of ss
        # L-STF & L-LTF, L-SIG
        self.ssLegacyPreamble = []
        self.ssLegacySig = []
        # HT-SIG, HT-STF- HT-LTF
        self.ssHtSig = []
        self.ssHtPreamble = []
        # VHT-SIG, VHT-STF, VHT-LTF
        self.ssVhtSigA = []
        self.ssVhtPreamble = []
        self.ssVhtSigB = []
        # VHT data
        self.dataBits = []  # original data bits
        self.dataScrambledBits = []  # scrambled data bits
        self.dataConvolvedBits = []  # convolved data bits, use convolutional coding
        self.dataPuncturedBits = []  # punctured bits according to rate
        self.ssStreamParserBits = []
        self.ssInterleaveBits = []    # interleave bits to sub carriers
        self.ssSymbols = []          # bits map to constellation
        self.ssPhySig = []          # sig sample of phy
        self.ssFinalSig = []        # sig sample to be used

    def genLegacy(self, mpdu, phyFormat):
        self.mpdu = mpdu
        self.f = phyFormat
        self.m = p8h.getMod(self.f)
        self.psduLen = len(self.mpdu)

    def genHt(self, mpdu, phyFormat):
        self.mpdu = mpdu
        self.f = phyFormat
        self.m = p8h.getMod(self.f)
        self.psduLen = len(self.mpdu)

    def genVht(self, mpdu, phyFormat, partialAid = 0, groupId = 0):
        self.mpdu = mpdu
        self.f = phyFormat
        self.m = p8h.getMod(self.f)
        self.vhtPartialAid = partialAid
        self.vhtGroupId = groupId       # Group ID, 0 to AP, 63 to stations
        self.apepLen = len(self.mpdu)
        print("VHT physical APEP len:", self.apepLen)
        self.ifMu = 0
        if(self.apepLen > 0):
            self.__genLegacyTraining()
            self.__genLegacySignal()
            self.__genVhtSignalA()
            self.__genVhtTraining()
            self.__genVhtSignalB()
            self.__genDataBits()
            self.__genScrambledDataBits()
            self.__genBccDataBits()
            self.__genPuncturedDataBits()
            self.__genStreamParserDataBits()
            self.__genInterleaveDataBits()
            self.__genConstellation()
            self.__genOfdmSignal()
        else:
            # NDP
            self.__genLegacyTraining()
            self.__genLegacySignal()
            self.__genVhtSignalA()
            self.__genVhtTraining()
            self.__genVhtSignalB()
            self.__genOfdmSignalNdp()
        return self.ssPhySig

    def genVhtMu(self, mpduMu, phyFormatMu, bfQ = [], groupId = 0):
        if(len(mpduMu) == len(phyFormatMu)):
            pass
        else:
            return
        self.ifMu = 1
        self.nMuUser = len(mpduMu)
        print("VHT MU number of users:", self.nMuUser)
        self.mpduMu = mpduMu
        self.fMu = phyFormatMu
        self.mMu = []
        for each in self.fMu:
            self.mMu.append(p8h.getMod(each))
        self.vhtGroupId = groupId       # multiple user, 1 to 62
        print("VHT MU nSym of users:", [each.nSym for each in self.mMu])
        self.nSymMu = max([each.nSym for each in self.mMu])
        for i in range(0, self.nMuUser):
            self.mMu[i].nSym = self.nSymMu

        self.m = p8h.getMod(self.fMu[0])
        self.nSSMu = sum([each.nSS for each in self.mMu])
        self.m.nSS = self.nSSMu
        self.f.type = 'vht'
        self.f.nSTS = sum([each.nSTS for each in self.fMu])
        self.m.nLtf = self.h.LTF_VHT_N[self.m.nSS]
        print("VHT MU nSTS, nSS, nLTF:", self.f.nSTS, self.m.nSS, self.m.nLtf)
        self.bfQ = bfQ

        self.__genLegacyTraining()
        self.__genLegacySignal()
        self.__genVhtSignalA()
        self.__genVhtTraining()
        self.__genVhtSignalBMu()

        self.ssSymbolsMu = []
        for n in range(0, self.nMuUser):
            self.mpdu = self.mpduMu[n]
            self.apepLen = len(self.mpdu)
            self.f = self.fMu[n]
            self.m = p8h.getMod(self.f)
            self.vhtSigBCrcBits = self.vhtSigBCrcBitsMu[n]
            self.__genDataBits()
            self.__genScrambledDataBits()
            self.__genBccDataBits()
            self.__genPuncturedDataBits()
            self.__genStreamParserDataBits()
            self.__genInterleaveDataBits()
            self.__genConstellation()
            for each in self.ssSymbols:
                self.ssSymbolsMu.append(each)

        self.m = p8h.getMod(self.fMu[0])
        self.nSSMu = sum([each.nSS for each in self.mMu])
        self.m.nSS = self.nSSMu
        self.f.type = 'vht'
        self.f.nSTS = sum([each.nSTS for each in self.fMu])
        self.__genOfdmSignalMu()

    def __genLegacyTraining(self):
        self.ssLegacyPreamble = []
        for ssItr in range(0, self.m.nSS):
            tmpStf = p8h.procToneScaling(p8h.procIDFT(
                p8h.procLegacyCSD(p8h.procNonDataSC(self.h.STF_L[self.f.bw]), self.m.nSS, ssItr, self.m.spr)),
                                         self.h.SCALENTF_STF_L[self.f.bw], self.m.nSS)
            tmpLtf = p8h.procToneScaling(p8h.procIDFT(
                p8h.procLegacyCSD(p8h.procNonDataSC(self.h.LTF_L[self.f.bw]), self.m.nSS, ssItr, self.m.spr)),
                                         self.h.SCALENTF_LTF_L[self.f.bw], self.m.nSS)
            tmpStf = tmpStf[int(len(tmpStf)/2):] + tmpStf + tmpStf
            tmpLtf = tmpLtf[int(len(tmpLtf)/2):] + tmpLtf + tmpLtf
            self.ssLegacyPreamble.append(p8h.procConcat2Symbol(tmpStf, tmpLtf))
        print("legacy training sample len %d" % (len(self.ssLegacyPreamble[0])))

    def __genLegacySignal(self):
        self.ssLegacySig = []
        tmpHeaderBits = []
        # legacy rate
        for i in range(0, 4):
            tmpHeaderBits.append((self.m.legacyRate>>i) & (1))
        # legacy reserved
        tmpHeaderBits.append(0)
        # legacy length
        if(self.f.type == 'l'):
            tmpHeaderLength = self.psduLen
        else:
            tLegacyPreamble = 8 + 8
            tLegacySig = 4
            tHtSig = 8
            tHtPreamble = 4 + self.m.nLtf * 4
            tVhtSigA = 8
            tVhtPreamble = 4 + self.m.nLtf * 4
            tVhtSigB = 4
            tSymL = 4
            txTime = 0
            if(self.f.type == 'ht'):
                self.nSym = self.m.nSym
                txTime = tLegacyPreamble + tLegacySig + tHtSig + tHtPreamble + self.nSym * tSymL
            elif(self.f.type == 'vht'):
                self.nSym = self.m.nSym
                txTime = tLegacyPreamble + tLegacySig + tVhtSigA + tVhtPreamble + tVhtSigB + self.nSym * tSymL
            print(txTime)
            tmpHeaderLength = int(np.ceil((txTime - 20) / 4)) * 3 - 3
        # add length bits
        for i in range(0, 12):
            tmpHeaderBits.append((tmpHeaderLength >> i) & (1))
        # add parity bit
        tmpHeaderBits.append(sum(tmpHeaderBits)%2)
        # add 6 tail bits
        tmpHeaderBits += [0] * 6
        print("legacy length:", tmpHeaderLength)
        print("legacy sig bit length:", len(tmpHeaderBits))
        print(tmpHeaderBits)
        # convolution of header bits, no scrambling for header
        tmpHeaderConvolvedBits = [0] * 48
        tmpState = 0
        for i in range(0, len(tmpHeaderBits)):
            tmpState = ((tmpState << 1) & 0x7e) | tmpHeaderBits[i]
            tmpHeaderConvolvedBits[i*2] = (bin(tmpState & 0o155).count("1")) % 2
            tmpHeaderConvolvedBits[i * 2 + 1] = (bin(tmpState & 0o117).count("1")) % 2
        # interleave of header bits from wime
        print("legacy sig convolved bits: %d" % (len(tmpHeaderConvolvedBits)))
        print(tmpHeaderConvolvedBits)
        # interleave of header using standard method
        tmpHeaderInterleaveBits = [0] * len(tmpHeaderConvolvedBits)
        s = 1
        # k is original index
        for k in range(0, 48):
            i = int((48/16) * (k % 16) + np.floor(k/16))
            j = int(s * np.float(i/s) + (int(i + 48 - np.floor(16 * i / 48)) % s))
            tmpHeaderInterleaveBits[j] = tmpHeaderConvolvedBits[k]
        print("legacy sig interleaved bits %s" % (len(tmpHeaderInterleaveBits)))
        print(tmpHeaderInterleaveBits)
        tmpHeaderQam = []
        for each in tmpHeaderInterleaveBits:
            tmpHeaderQam.append(self.h.QAM_MODU_TAB[self.h.QAM_BPSK][int(each)])
        # add pilot
        tmpHeaderQam = tmpHeaderQam[0:5] + [1] + tmpHeaderQam[5:18] + [1] + tmpHeaderQam[18:30] + [1] + tmpHeaderQam[30:43] + [-1] + tmpHeaderQam[43:48]
        # add DC
        tmpHeaderQam = p8h.procNonDataSC(p8h.procDcInsert(tmpHeaderQam))
        if (self.f.bw == self.h.BW_20):
            tmpHeaderQam = tmpHeaderQam
        elif (self.f.bw == self.h.BW_40):
            tmpHeaderQam = tmpHeaderQam * 2
        elif (self.f.bw == self.h.BW_80):
            tmpHeaderQam = tmpHeaderQam * 4
        else:
            print("len error")
        for ssItr in range(0, self.m.nSS):
            self.ssLegacySig.append(p8h.procGi(p8h.procToneScaling(p8h.procIDFT(p8h.procLegacyCSD(tmpHeaderQam, self.m.nSS, ssItr, self.m.spr)), self.h.SCALENTF_SIG_L[self.f.bw], self.m.nSS)))
        print("legacy sig sample len %d" % (len(self.ssLegacySig[0])))

    def __genVhtSignalA(self):
        self.ssVhtSigA = []
        tmpVhtSigABits = []
        # b 0 1, bw
        for i in range(0, 2):
            tmpVhtSigABits.append((self.f.bw >> i) & (1))
        # b 2, reserved
        tmpVhtSigABits.append(1)
        # b 3, STBC
        tmpVhtSigABits.append(0)
        # b 4 9, group id
        for i in range(0, 6):
            tmpVhtSigABits.append((self.vhtGroupId >> i) & (1))
        if(self.ifMu):
            # multiple user mimo
            # b 10 21, user nSTS, 3 bits per user
            for u in range(0, self.nMuUser):
                print("user ", u, ", nSTS ", self.fMu[u].nSTS)
                for i in range(0, 3):
                    tmpVhtSigABits.append((self.fMu[u].nSTS >> i) & (1))
            for u in range(0, 4 - self.nMuUser):
                for i in range(0, 3):
                    tmpVhtSigABits.append(0)
        else:
            # single user
            # b 10 12, SU nSTS
            for i in range(0, 3):
                tmpVhtSigABits.append(((self.f.nSTS - 1) >> i) & (1))
            # b 13 21, Partial AID
            for i in range(0, 9):
                tmpVhtSigABits.append((self.vhtPartialAid >> i) & (1))     # matlab use 275 as partial AID
        # b 22 txop ps not allowed, set 0, allowed
        tmpVhtSigABits.append(0)
        # b 23 reserved
        tmpVhtSigABits.append(1)

        # b 0 short GI, 0
        tmpVhtSigABits.append(0)
        # b 1 short GI disam
        tmpVhtSigABits.append(0)
        if (self.ifMu):
            # b 2 MU 0 coding, BCC
            tmpVhtSigABits.append(0)
        else:
            # b 2 SU coding, BCC
            tmpVhtSigABits.append(0)
        # b 3 LDPC extra
        tmpVhtSigABits.append(0)
        if (self.ifMu):
            # b 4 6, MU 1 3 coding
            # if user in pos, then bcc
            for u in range(1, self.nMuUser):
                tmpVhtSigABits.append(0)
            # others reserved
            for u in range(0, 4 - self.nMuUser):
                tmpVhtSigABits.append(1)
            # b 7, MU reserved
            tmpVhtSigABits.append(1)
            # b 8, beamformed reserved
            tmpVhtSigABits.append(1)
        else:
            # b 4 7, MCS
            for i in range(0, 4):
                tmpVhtSigABits.append((self.f.mcs >> i) & (1))
            # b 8, beamformed
            tmpVhtSigABits.append(0)
        # b 9, reserved
        tmpVhtSigABits.append(1)
        # b 10 17 crc
        tmpVhtSigABits += p8h.getBitCrc8(tmpVhtSigABits)
        # b 18 23 tail
        tmpVhtSigABits += [0] * 6
        print("VHT signal A bits: %d" % (len(tmpVhtSigABits)))
        print(tmpVhtSigABits[0:24])
        print(tmpVhtSigABits[24:48])
        tmpConvolvedBits = [0]*96
        tmpState = 0
        for i in range(0, len(tmpVhtSigABits)):
            tmpState = ((tmpState << 1) & 0x7e) | tmpVhtSigABits[i]
            tmpConvolvedBits[i*2] = (bin(tmpState & 0o155).count("1")) % 2
            tmpConvolvedBits[i * 2 + 1] = (bin(tmpState & 0o117).count("1")) % 2
        tmpInterleavedBits = [0]*96
        tmpSeq1 = [0] * 48
        tmpSeq2 = [0] * 48
        s = 1
        for i in range(0, 48):
            tmpSeq1[i] = int(s * int(i / s) + ((i + np.floor(16.0 * i / 48)) % s))
            tmpSeq2[i] = int(16 * i - (48 - 1) * np.floor(16 * i / 48))
        for i in range(0, 2):
            for j in range(0, 48):
                tmpInterleavedBits[int(i * 48 + j)] = tmpConvolvedBits[
                    int(i * 48 + tmpSeq2[int(tmpSeq1[j])])]
        print("VHT sig A interleaved")
        print(tmpInterleavedBits)
        tmpSig1Qam = [self.h.QAM_MODU_TAB[self.h.QAM_BPSK][each] for each in tmpInterleavedBits[0:48]]
        tmpSig2Qam = [self.h.QAM_MODU_TAB[self.h.QAM_QBPSK][each] for each in tmpInterleavedBits[48:96]]
        # insert pilot and DC
        tmpSig1Qam = p8h.procDcInsert(p8h.procPilotInsert(tmpSig1Qam, [1,1,1,-1]))
        tmpSig2Qam = p8h.procDcInsert(p8h.procPilotInsert(tmpSig2Qam, [1,1,1,-1]))
        # copy to other sc for higher bandwidth
        if (self.f.bw == self.h.BW_40):
            tmpSig1Qam = tmpSig1Qam * 2
            tmpSig2Qam = tmpSig2Qam * 2
        elif (self.f.bw == self.h.BW_80):
            tmpSig1Qam = tmpSig1Qam * 4
            tmpSig2Qam = tmpSig2Qam * 4
        print("VHT signal A sig after DC sample len %d %d" % (len(tmpSig1Qam), len(tmpSig2Qam)))
        for ssItr in range(0, self.m.nSS):
            self.ssVhtSigA.append(
                p8h.procConcat2Symbol(
                p8h.procGi(p8h.procToneScaling(p8h.procIDFT(p8h.procLegacyCSD(p8h.procNonDataSC(tmpSig1Qam), self.m.nSS, ssItr, self.m.spr)),self.h.SCALENTF_SIG_VHT_A[self.f.bw], self.m.nSS)),
                p8h.procGi(p8h.procToneScaling(p8h.procIDFT(p8h.procLegacyCSD(p8h.procNonDataSC(tmpSig2Qam), self.m.nSS, ssItr, self.m.spr)),self.h.SCALENTF_SIG_VHT_A[self.f.bw], self.m.nSS))
                ))
        print("VHT signal A sample len %d" % len(self.ssVhtSigA[0]))

    # def __genVhtTraining(self):
    #     self.ssVhtPreamble = []
    #     for ssItr in range(0, self.m.nSS):
    #         tmpVhtStf = p8h.procCSD(p8h.procNonDataSC(self.h.STF_VHT[self.f.bw]), self.m.nSS, ssItr, self.m.spr)
    #
    #         tmpVhtPreamble = p8h.procGi(p8h.procToneScaling(p8h.procIDFT(tmpVhtStf),self.h.SCALENTF_STF_VHT[self.f.bw], self.m.nSS))
    #         if (self.m.nLtf <= 4):
    #             for i in range(0, self.m.nLtf):
    #                 # VHT LTF P and R !!!!!!!!
    #                 tmpVhtLtf = []
    #                 for j in range(0, len(self.h.LTF_VHT[self.f.bw])):
    #                     if (self.f.bw == self.h.BW_20 and (j - 28) in [-21, -7, 7, 21]):
    #                         tmpVhtLtf.append(self.h.LTF_VHT[self.f.bw][j] * self.h.R_LTF_VHT_4[i])
    #                     elif(self.f.bw == self.h.BW_40 and (j - 58) in [-53, -25, -11, 11, 25, 53]):
    #                         tmpVhtLtf.append(self.h.LTF_VHT[self.f.bw][j] * self.h.R_LTF_VHT_4[i])
    #                     elif (self.f.bw == self.h.BW_80 and (j - 122) in [-103, -75, -39, -11, 11, 39, 75, 103]):
    #                         tmpVhtLtf.append(self.h.LTF_VHT[self.f.bw][j] * self.h.R_LTF_VHT_4[i])
    #                     else:
    #                         tmpVhtLtf.append(self.h.LTF_VHT[self.f.bw][j] * self.h.P_LTF_VHT_4[ssItr][i])
    #                 tmpVhtLtf = p8h.procGi(p8h.procToneScaling(p8h.procIDFT(p8h.procCSD(
    #                     p8h.procNonDataSC(tmpVhtLtf), self.m.nSS, ssItr, self.m.spr)),
    #                                      self.h.SCALENTF_LTF_VHT[self.f.bw], self.m.nSS))
    #                 tmpVhtPreamble = p8h.procConcat2Symbol(tmpVhtPreamble, tmpVhtLtf)
    #         self.ssVhtPreamble.append(tmpVhtPreamble)
    #     print("VHT training sample len %d" % len(self.ssVhtPreamble[0]))

    def procSpatialMapping(self, inSigSs, inQ):
        tmpNSs = len(inSigSs)
        print("spatial mapping, fft len:", len(inQ))
        outSigSs = []
        for j in range(0, tmpNSs):
            outSigSs.append([])
        for k in range(0, len(inQ)):
            tmpX = []
            for j in range(0, tmpNSs):
                tmpX.append([inSigSs[j][k]])
            tmpX = np.array(tmpX)
            tmpQX = np.matmul(inQ[k], tmpX)
            for j in range(0, tmpNSs):
                outSigSs[j].append(tmpQX[j][0])
        return outSigSs


    def __genVhtTraining(self):
        self.ssVhtPreamble = []
        for ssItr in range(0, self.m.nSS):
            self.ssVhtPreamble.append([])
        tmpVhtStf = []
        for ssItr in range(0, self.m.nSS):
            tmpVhtStf.append(p8h.procCSD(p8h.procNonDataSC(self.h.STF_VHT[self.f.bw]), self.m.nSS, ssItr, self.m.spr))
        if(self.ifMu):
            tmpVhtStfQ = self.procSpatialMapping(tmpVhtStf, self.bfQ)
        else:
            tmpVhtStfQ = tmpVhtStf
        for ssItr in range(0, self.m.nSS):
            self.ssVhtPreamble[ssItr] = p8h.procGi(
                p8h.procToneScaling(p8h.procIDFT(tmpVhtStfQ[ssItr]), self.h.SCALENTF_STF_VHT[self.f.bw], self.m.nSS))

        for ltfIter in range(0, self.m.nLtf):
            tmpVhtLtfSs = []
            for ssItr in range(0, self.m.nSS):
                tmpVhtLtf = []
                for j in range(0, len(self.h.LTF_VHT[self.f.bw])):
                    if (self.f.bw == self.h.BW_20 and (j - 28) in [-21, -7, 7, 21]):
                        tmpVhtLtf.append(self.h.LTF_VHT[self.f.bw][j] * self.h.R_LTF_VHT_4[ltfIter])
                    elif(self.f.bw == self.h.BW_40 and (j - 58) in [-53, -25, -11, 11, 25, 53]):
                        tmpVhtLtf.append(self.h.LTF_VHT[self.f.bw][j] * self.h.R_LTF_VHT_4[ltfIter])
                    elif (self.f.bw == self.h.BW_80 and (j - 122) in [-103, -75, -39, -11, 11, 39, 75, 103]):
                        tmpVhtLtf.append(self.h.LTF_VHT[self.f.bw][j] * self.h.R_LTF_VHT_4[ltfIter])
                    else:
                        tmpVhtLtf.append(self.h.LTF_VHT[self.f.bw][j] * self.h.P_LTF_VHT_4[ssItr][ltfIter])
                print("ltf n ", ltfIter, ", ss ", ssItr, p8h.procNonDataSC(tmpVhtLtf))
                tmpVhtLtfSs.append(p8h.procCSD(p8h.procNonDataSC(tmpVhtLtf), self.m.nSS, ssItr, self.m.spr))

            if (self.ifMu):
                tmpVhtLtfSsQ = self.procSpatialMapping(tmpVhtLtfSs, self.bfQ)
            else:
                tmpVhtLtfSsQ = tmpVhtLtfSs

            for ssItr in range(0, self.m.nSS):
                # for k in range(0, 64):
                #     print(tmpVhtLtfSsQ[ssItr][k])
                self.ssVhtPreamble[ssItr] = p8h.procConcat2Symbol(
                    self.ssVhtPreamble[ssItr],
                    p8h.procGi(p8h.procToneScaling(p8h.procIDFT(tmpVhtLtfSsQ[ssItr]), self.h.SCALENTF_LTF_VHT[self.f.bw], self.m.nSS)))

        for each in self.ssVhtPreamble:
            print("VHT training sample len %d" % len(each))

    def __genVhtSignalB(self):
        self.ssVhtSigB = []
        tmpVhtSigBBits = []
        # bits for length
        # compute APEP Len first, single user, use mpdu byte number as APEP len
        tmpSigBLen = int(np.ceil(self.apepLen/4))
        print("sig b len: %d" % tmpSigBLen)
        tmpLenBitN = 17
        tmpReservedBitN = 3
        tmpSigBnBPSCS = 1
        tmpSigBnCBPS = 52
        tmpSigBnCBPSSI = 52
        tmpSigBnIntlevCol = 13
        tmpSigBnIntlevRow = 4 * tmpSigBnBPSCS
        if(self.f.bw == self.h.BW_40):
            tmpLenBitN = 19
            tmpReservedBitN = 2
            tmpSigBnBPSCS = 1
            tmpSigBnCBPS = 108
            tmpSigBnCBPSSI = 108
            tmpSigBnIntlevCol = 18
            tmpSigBnIntlevRow = 6 * tmpSigBnBPSCS
        elif(self.f.bw == self.h.BW_80):
            tmpLenBitN = 21
            tmpReservedBitN = 2
            tmpSigBnBPSCS = 1
            tmpSigBnCBPS = 234
            tmpSigBnCBPSSI = 234
            tmpSigBnIntlevCol = 26
            tmpSigBnIntlevRow = 9 * tmpSigBnBPSCS
        if(self.apepLen > 0):
            for i in range(0, tmpLenBitN):
                tmpVhtSigBBits.append((tmpSigBLen >> i) & (1))
            # bits for reserved
            tmpVhtSigBBits = tmpVhtSigBBits + [1] * tmpReservedBitN
            # crc 8 for sig b used in data
            self.vhtSigBCrcBits = p8h.getBitCrc8(tmpVhtSigBBits)
        else:
            tmpVhtSigBBits = self.h.NDP_SIG_B[self.f.bw]
        # bits for tail
        tmpVhtSigBBits = tmpVhtSigBBits + [0] * 6
        if(self.f.bw == self.h.BW_40):
            tmpVhtSigBBits = tmpVhtSigBBits * 2
        elif(self.f.bw == self.h.BW_80):
            tmpVhtSigBBits = tmpVhtSigBBits * 2 + [0]
        print("vht sig b bits: %d" % len(tmpVhtSigBBits))
        print(tmpVhtSigBBits)
        # convolution
        tmpHeaderConvolvedBits = [0] * tmpSigBnCBPS
        tmpState = 0
        for i in range(0, len(tmpVhtSigBBits)):
            tmpState = ((tmpState << 1) & 0x7e) | tmpVhtSigBBits[i]
            tmpHeaderConvolvedBits[i * 2] = (bin(tmpState & 0o155).count("1")) % 2
            tmpHeaderConvolvedBits[i * 2 + 1] = (bin(tmpState & 0o117).count("1")) % 2
        print("vht sig b convolved bits: %d" % (len(tmpHeaderConvolvedBits)))
        print(tmpHeaderConvolvedBits)
        # no segment parse, interleave
        s = 1
        tmpInterleavedBits = [0] * len(tmpHeaderConvolvedBits)
        for k in range(0, tmpSigBnCBPSSI):
            i = tmpSigBnIntlevRow * (k % tmpSigBnIntlevCol) + int(np.floor(k / tmpSigBnIntlevCol))
            j = s * int(np.floor(i / s)) + (i + tmpSigBnCBPSSI - int(np.floor(tmpSigBnIntlevCol * i / tmpSigBnCBPSSI))) % s
            r = j
            tmpInterleavedBits[r] = tmpHeaderConvolvedBits[k]
        print("VHT sig b interleaved")
        print(tmpInterleavedBits)
        # modulation
        for ssItr in range(0, self.m.nSS):
            tmpSigQam = [self.h.QAM_MODU_TAB[self.h.QAM_BPSK][each] for each in tmpInterleavedBits]
            # map constellations to user specific P_VHT_LTF, actually only flip the BPSK when nSS is 4, 7 or 8
            if(self.m.nSS in [4, 7, 8]):
                tmpSigQam = [each * self.h.P_SIG_B_NSTS478[ssItr] for each in tmpSigQam]
            tmpSigQam = p8h.procDcInsert(p8h.procPilotInsert(tmpSigQam, self.h.PILOT_VHT[self.f.bw]))
            self.ssVhtSigB.append(p8h.procGi(p8h.procToneScaling(
                p8h.procIDFT(p8h.procCSD(p8h.procNonDataSC(tmpSigQam), self.m.nSS, ssItr, self.m.spr)),
                self.h.SCALENTF_SIG_VHT_B[self.f.bw], self.m.nSS)))
        print("VHT signal B: %d" % len(self.ssVhtSigB[0]))

    def __genVhtSignalBMu(self):
        self.ssVhtSigB = []
        self.vhtSigBCrcBitsMu = []
        tmpSsSigQam = []
        for n in range(0, self.nMuUser):
            tmpVhtSigBBits = []
            # compute APEP Len first, single user, use mpdu byte number as APEP len
            tmpSigBLen = int(len(self.mpduMu[n])/4)
            print("sig b len: %d" % tmpSigBLen)
            tmpLenBitN = 16
            tmpMcsBitN = 4
            tmpSigBnBPSCS = 1
            tmpSigBnCBPS = 52
            tmpSigBnCBPSSI = 52
            tmpSigBnIntlevCol = 13
            tmpSigBnIntlevRow = 4 * tmpSigBnBPSCS
            # bits for length
            for i in range(0, tmpLenBitN):
                tmpVhtSigBBits.append((tmpSigBLen >> i) & (1))
            # bits for mcs
            for i in range(0, tmpMcsBitN):
                tmpVhtSigBBits.append((self.fMu[n].mcs >> i) & (1))
            # crc 8 for sig b used in data
            self.vhtSigBCrcBitsMu.append(p8h.getBitCrc8(tmpVhtSigBBits))
            print("vht sig b crc bits:", self.vhtSigBCrcBitsMu[n])
            # bits for tail
            tmpVhtSigBBits = tmpVhtSigBBits + [0] * 6
            print("vht sig b bits: %d" % len(tmpVhtSigBBits))
            print(tmpVhtSigBBits)
            # convolution
            tmpHeaderConvolvedBits = [0] * tmpSigBnCBPS
            tmpState = 0
            for i in range(0, len(tmpVhtSigBBits)):
                tmpState = ((tmpState << 1) & 0x7e) | tmpVhtSigBBits[i]
                tmpHeaderConvolvedBits[i * 2] = (bin(tmpState & 0o155).count("1")) % 2
                tmpHeaderConvolvedBits[i * 2 + 1] = (bin(tmpState & 0o117).count("1")) % 2
            print("vht sig b convolved bits: %d" % (len(tmpHeaderConvolvedBits)))
            print(tmpHeaderConvolvedBits)
            # no segment parse, interleave
            s = 1
            tmpInterleavedBits = [0] * len(tmpHeaderConvolvedBits)
            for k in range(0, tmpSigBnCBPSSI):
                i = tmpSigBnIntlevRow * (k % tmpSigBnIntlevCol) + int(np.floor(k / tmpSigBnIntlevCol))
                j = s * int(np.floor(i / s)) + (i + tmpSigBnCBPSSI - int(np.floor(tmpSigBnIntlevCol * i / tmpSigBnCBPSSI))) % s
                r = j
                tmpInterleavedBits[r] = tmpHeaderConvolvedBits[k]
            print("VHT sig b interleaved")
            print(tmpInterleavedBits)
            # modulation
            for ssItr in range(0, self.mMu[n].nSS):
                tmpSigQam = [self.h.QAM_MODU_TAB[self.h.QAM_BPSK][each] for each in tmpInterleavedBits]
                tmpSsSigQam.append(tmpSigQam)
        for ssItr in range(0, self.m.nSS):
            # map constellations to user specific P_VHT_LTF, actually only flip the BPSK when nSS is 4, 7 or 8
            # if(self.m.nSS in [4, 7, 8]):
            #     tmpSsSigQam[ssItr] = [each * self.h.P_SIG_B_NSTS478[ssItr] for each in tmpSsSigQam[ssItr]]
            tmpSsSigQam[ssItr] = p8h.procDcInsert(p8h.procPilotInsert(tmpSsSigQam[ssItr], self.h.PILOT_VHT[self.f.bw]))
            tmpSsSigQam[ssItr] = p8h.procCSD(p8h.procNonDataSC(tmpSsSigQam[ssItr]), self.m.nSS, ssItr, self.m.spr)
        # add spatial mapping
        tmpSsSigQamQ = self.procSpatialMapping(tmpSsSigQam, self.bfQ)
        for ssItr in range(0, self.m.nSS):
            self.ssVhtSigB.append(p8h.procGi(p8h.procToneScaling(
                    p8h.procIDFT(tmpSsSigQamQ[ssItr]),
                    self.h.SCALENTF_SIG_VHT_B[self.f.bw], self.m.nSS)))
        for each in self.ssVhtSigB:
            print("VHT signal B: %d" % len(each))
            print(each)

    def __genDataBits(self):
        # structure is: service, psdu, pad, tail is added after scrambling
        self.dataBits = []
        tmpDataBitStrForM = ""
        tmpPsduBitStrForM = ""
        # service bits, scrambler init
        self.dataBits += [0] * 7
        # service bits, reserved
        self.dataBits += [0]
        # service bits, sig b crc
        self.dataBits += self.vhtSigBCrcBits
        print("data service bit len: %d" % len(self.dataBits))
        print(self.dataBits)
        nService = 16
        nTail = 6
        self.nPad = int(self.nSym * self.m.nDBPS - 8*self.m.psduLen - nService - nTail * self.m.nES)
        print("data pad bits number: %d" % self.nPad)
        # convert MAC data
        tmpPsdu = []
        for each in self.mpdu:
            for i in range(0,8):
                tmpPsdu.append((each>>i) & (1))
                tmpDataBitStrForM += str((each>>i) & (1))
                tmpDataBitStrForM += " "
        print("vht data apep len:", self.m.pktLen, ", psdu len:", self.m.psduLen, ", mpdu len:", len(self.mpdu))
        tmpPsdu = tmpPsdu + tmpPsdu[0:int(self.m.psduLen * 8 - self.m.pktLen * 8)]
        for each in tmpPsdu:
            tmpPsduBitStrForM += str(each)
            tmpPsduBitStrForM += " "
        # add 6 tail bits
        print("pure data bits for matlab", len(self.mpdu) * 8)
        print(tmpDataBitStrForM)
        print("psdu data bits for matlab", len(tmpPsdu))
        print(tmpPsduBitStrForM)
        self.dataBits = self.dataBits + tmpPsdu
        # add pad zero
        for i in range(0, self.nPad):
            self.dataBits.append(0)
        print("vht data bits (not include tail bits):", len(self.dataBits))
        print(self.dataBits)

    def __genScrambledDataBits(self):
        self.dataScrambledBits = []
        tmpScrambler = self.dataScrambler
        # scrambling
        for i in range(0, len(self.dataBits)):
            tmpFeedback = int(not not(tmpScrambler & 64)) ^ int(not not(tmpScrambler & 8))
            self.dataScrambledBits.append(tmpFeedback ^ self.dataBits[i])
            tmpScrambler = ((tmpScrambler << 1) & 0x7e) | tmpFeedback
        # no tail bits reset for vht, tail bits added in next part depending on nES
        print("scrambled data bits:", len(self.dataScrambledBits), ", scrambler: ", self.dataScrambler)
        print(self.dataScrambledBits)

    def __genBccDataBits(self):
        # separate bits into coders
        self.dataConvolvedBits = []
        nTail = 6
        for i in range(0, self.m.nES):
            self.dataConvolvedBits.append([])
            # divide bits into bcc coders
            tmpDividedBits = [self.dataScrambledBits[each] for each in range((0+i), int(self.m.nDBPS * self.nSym / self.m.nES - nTail), self.m.nES)]
            tmpDividedBits = tmpDividedBits + [0] * nTail
            # coding
            tmpState = 0
            self.dataConvolvedBits[i] = [0]*(len(tmpDividedBits)*2)
            for j in range(0, len(tmpDividedBits)):
                tmpState = ((tmpState << 1) & 0x7e) | tmpDividedBits[j]
                self.dataConvolvedBits[i][j*2] = (bin(tmpState & 0o155).count("1")) % 2
                self.dataConvolvedBits[i][j * 2 + 1] = (bin(tmpState & 0o117).count("1")) % 2
        print("convolutional data bits:", len(self.dataConvolvedBits[0]))
        print(self.dataConvolvedBits[0])

    def __genPuncturedDataBits(self):
        tmpIndex = 0
        self.dataPuncturedBits = []
        for i in range(0, self.m.nES):
            self.dataPuncturedBits.append([])
            if(self.m.cr == self.h.CR_12):
                for each in self.dataConvolvedBits[i]:
                    self.dataPuncturedBits[i].append(each)
            elif(self.m.cr == self.h.CR_23):
                for each in self.dataConvolvedBits[i]:
                    if((tmpIndex%4) != 3):
                        self.dataPuncturedBits[i].append(each)
                    tmpIndex += 1
            elif(self.m.cr == self.h.CR_34):
                for each in self.dataConvolvedBits[i]:
                    if ((tmpIndex % 6) in [3, 4]):
                        pass
                    else:
                        self.dataPuncturedBits[i].append(each)
                    tmpIndex += 1
            elif (self.m.cr == self.h.CR_56):
                for each in self.dataConvolvedBits[i]:
                    if ((tmpIndex % 10) in [3, 4, 7, 8]):
                        pass
                    else:
                        self.dataPuncturedBits[i].append(each)
                    tmpIndex += 1
        print("punctured data bits", len(self.dataPuncturedBits[0]))
        print(self.dataPuncturedBits[0])

    def __genStreamParserDataBits(self):
        # each round, get S
        self.ssStreamParserBits = []
        for i in range(0, self.m.nSS):
            self.ssStreamParserBits.append([0] * self.nSym * self.m.nCBPSS)
        s = int(max(1, self.m.nBPSCS/2))
        S = self.m.nSS * s
        nBlock = int(np.floor(self.m.nCBPS/(self.m.nES * S)))
        M = int((self.m.nCBPS - nBlock * self.m.nES * S)/(s * self.m.nES))
        # 20/40/80 4x4, nBlock * nES * S is smaller than nCBPSS
        for sIter in range(0, self.nSym):
            for iss in range(0, self.m.nSS):
                for k in range(0, int(self.m.nCBPSS)):
                    if(k < int(nBlock * self.m.nES * s)):
                        j = int(np.floor(k/s)) % self.m.nES
                        i = (iss) * s + S * int(np.floor(k/(self.m.nES * s))) + int(k % s)
                    else:
                        k_ = int(k - nBlock * self.m.nES * s)
                        L = int(np.floor(k_/s)) * self.m.nSS + (iss) # if iss start from 1, then it is iss - 1
                        j = int(np.floor(L/M))
                        i = int(L % M) * s + nBlock * S + int(k % s)
                    # print(iss, k, i)
                    self.ssStreamParserBits[iss][k + int(sIter * self.m.nCBPSS)] = self.dataPuncturedBits[j][i + int(sIter * self.m.nCBPS)]
        for each in self.ssStreamParserBits:
            print("stream parser bits:", len(each))
            print(each)

    def __genInterleaveDataBits(self):
        # stream parser
        self.ssInterleaveBits = []
        for i in range(0, self.m.nSS):
            self.ssInterleaveBits.append([])
        s = int(max(1, self.m.nBPSCS/2))
        for ssItr in range(0, self.m.nSS):
            self.ssInterleaveBits[ssItr] = [0] * len(self.ssStreamParserBits[ssItr])
            for symPtr in range(0, self.nSym):
                for k in range(0, self.m.nCBPSS):
                    i = self.m.nIntlevRow * (k % self.m.nIntlevCol) + int(np.floor(k/self.m.nIntlevCol))
                    j = s * int(np.floor(i/s)) + (i + self.m.nCBPSS - int(np.floor(self.m.nIntlevCol * i / self.m.nCBPSS))) % s
                    r = j
                    if(self.m.nSS >=2):
                        r = int((j-(((ssItr)*2)%3 + 3*int(np.floor((ssItr)/3)))*self.m.nIntlevRot*self.m.nBPSCS)) % self.m.nCBPSS
                    self.ssInterleaveBits[ssItr][r + symPtr * self.m.nCBPSS] = self.ssStreamParserBits[ssItr][k + symPtr * self.m.nCBPSS]
        for each in self.ssInterleaveBits:
            print("interleaved stream bits:", len(each))
            print(each)

    def __genConstellation(self):
        self.ssSymbols = []
        for i in range(0, self.m.nSS):
            self.ssSymbols.append([])
        tmpCarrierNumStream = int(self.m.nCBPSS * self.nSym/self.m.nBPSCS)
        for ssItr in range(0, self.m.nSS):
            for i in range(0, tmpCarrierNumStream):
                tmpCarrierChip = 0
                for j in range(0, self.m.nBPSCS):
                    tmpCarrierChip += self.ssInterleaveBits[ssItr][i * self.m.nBPSCS + j] * (2 ** j)
                self.ssSymbols[ssItr].append(self.h.QAM_MODU_TAB[self.m.mod][tmpCarrierChip])
        for ssItr in range(0, self.m.nSS):
            print("constellation data", len(self.ssSymbols[ssItr]))
            print(self.ssSymbols[ssItr])

    def __genOfdmSignalMu(self):
        self.ssPhySig = []
        for i in range(0, self.m.nSS):
            self.ssPhySig.append([])
        for ssItr in range(0, self.m.nSS):
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssLegacyPreamble[ssItr], self.ssLegacySig[ssItr])
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], self.ssVhtSigA[ssItr])
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], self.ssVhtPreamble[ssItr])
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], self.ssVhtSigB[ssItr])

        tmpPilot = self.h.PILOT_VHT[self.f.bw]
        tmpPilotPIdx = 4
        for i in range(0, self.nSym):
            tmpSsDataSig = []
            for ssItr in range(0, self.m.nSS):
                tmpPilotAdded = p8h.procPilotInsert(self.ssSymbolsMu[ssItr][int(i * self.m.nSD): int((i + 1) * self.m.nSD)],
                                                    [each * self.h.PILOT_PS[tmpPilotPIdx] for each in tmpPilot])
                tmpPilotAdded = tmpPilotAdded[0:int(len(tmpPilotAdded) / 2)] + [0] + tmpPilotAdded[int(len(tmpPilotAdded) / 2):]
                tmpSsDataSig.append(p8h.procCSD(p8h.procNonDataSC(tmpPilotAdded), self.m.nSS, ssItr, self.m.spr))
            tmpSsDataSigQ = self.procSpatialMapping(tmpSsDataSig, self.bfQ)
            for ssItr in range(0, self.m.nSS):
                self.ssPhySig[ssItr] = p8h.procConcat2Symbol(
                    self.ssPhySig[ssItr],
                    p8h.procGi(p8h.procToneScaling(p8h.procIDFT(tmpSsDataSigQ[ssItr]), self.h.SCALENTF_DATA_VHT[self.f.bw], self.m.nSS)))
            tmpPilotPIdx = (tmpPilotPIdx + 1) % 127
            tmpPilot = tmpPilot[1:] + [tmpPilot[0]]

    def __genOfdmSignal(self):
        self.ssPhySig = []
        for i in range(0, self.m.nSS):
            self.ssPhySig.append([])
        for ssItr in range(0, self.m.nSS):
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssLegacyPreamble[ssItr], self.ssLegacySig[ssItr])
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], self.ssVhtSigA[ssItr])
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], self.ssVhtPreamble[ssItr])
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], self.ssVhtSigB[ssItr])
            tmpPilot = self.h.PILOT_L
            tmpPilotPIdx = 1
            if(self.f.type == 'ht'):
                tmpPilot = self.h.PILOT_HT[self.f.bw][self.m.nSS][ssItr]
                tmpPilotPIdx = 3
            elif(self.f.type == 'vht'):
                tmpPilot = self.h.PILOT_VHT[self.f.bw]
                tmpPilotPIdx = 4
            for i in range(0, self.nSym):
                tmpPilotAdded = p8h.procPilotInsert(self.ssSymbols[ssItr][int(i*self.m.nSD): int((i+1)*self.m.nSD)], [each * self.h.PILOT_PS[tmpPilotPIdx] for each in tmpPilot])
                tmpPilotAdded = tmpPilotAdded[0:int(len(tmpPilotAdded)/2)] + [0] + tmpPilotAdded[int(len(tmpPilotAdded)/2):]
                self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], p8h.procGi(p8h.procToneScaling(
                p8h.procIDFT(p8h.procCSD(p8h.procNonDataSC(tmpPilotAdded), self.m.nSS, ssItr, self.m.spr)),
                self.h.SCALENTF_DATA_VHT[self.f.bw], self.m.nSS)))
                tmpPilotPIdx = (tmpPilotPIdx + 1) % 127
                tmpPilot = tmpPilot[1:] + [tmpPilot[0]]


    def __genOfdmSignalNdp(self):
        self.ssPhySig = []
        for i in range(0, self.m.nSS):
            self.ssPhySig.append([])
        for ssItr in range(0, self.m.nSS):
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssLegacyPreamble[ssItr], self.ssLegacySig[ssItr])
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], self.ssVhtSigA[ssItr])
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], self.ssVhtPreamble[ssItr])
            self.ssPhySig[ssItr] = p8h.procConcat2Symbol(self.ssPhySig[ssItr], self.ssVhtSigB[ssItr])

    def __genSignalWithCfo(self, inSig, cfoHz):
        tmpRadStep = cfoHz * 2.0 * np.pi / 20000000.0
        outSig = []
        for i in range(0, len(inSig)):
            outSig.append(inSig[i] * (np.cos(i * tmpRadStep) + np.sin(i * tmpRadStep) * 1j))
        return outSig

    def __genSignalWithAmp(self, inSig, m):
        return [each * m for each in inSig]

    def __procAddCfo(self, cfoHz):
        for ssItr in range(0, self.m.nSS):
            self.ssPhySig[ssItr] = self.__genSignalWithCfo(self.ssPhySig[ssItr], cfoHz)
    
    def __procAddAmp(self, multiplier):
        for ssItr in range(0, self.m.nSS):
            self.ssPhySig[ssItr] = [each * multiplier for each in self.ssPhySig[ssItr]]

    def genFinalSig(self, multiplier = 1.0, cfoHz = 0.0, num = 1, gap = True, gapLen = 10000):
        self.ssFinalSig = []
        if(num < 1):
            print("gen final sig num error")
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
        print("write signal into bin file")
        if(len(fileAddr)<1):
            print("error: file address not given")
            return

        for ssItr in range(0, len(ssSig)):
            binF = open(fileAddr + "_" + str(self.f.nSTS) + "x" + str(self.f.nSTS) + "_" + str(ssItr) + ".bin", "wb")
            tmpSig = ssSig[ssItr]
            print("%d sample number %d" % (ssItr, len(tmpSig)))
            for i in range(0, len(tmpSig)):
               binF.write(struct.pack("f", np.real(tmpSig[i])))
               binF.write(struct.pack("f", np.imag(tmpSig[i])))
            binF.close()
            print("written in " + (fileAddr + "_" + str(ssItr)))
            if(draw):
                plt.figure(100 + ssItr)
                plt.plot(np.real(tmpSig))
                plt.plot(np.imag(tmpSig))
        if(draw):
            plt.show()

def genMac80211UdpMPDU(udpPayload):
    udpIns = mac80211.udp("10.10.0.6",  # sour ip
                          "10.10.0.1",  # dest ip
                          39379,  # sour port
                          8889,  # dest port
                          bytearray(udpPayload, 'utf-8'))  # bytes payload
    udpPacket = udpIns.genPacket()
    print("udp packet")
    print(udpPacket.hex())
    ipv4Ins = mac80211.ipv4(43778,  # identification
                            64,  # TTL
                            "10.10.0.6",
                            "10.10.0.1",
                            udpPacket)
    ipv4Packet = ipv4Ins.genPacket()
    print("ipv4 packet")
    print(ipv4Packet.hex())
    llcIns = mac80211.llc()
    llcPacket = llcIns.genPacket() + ipv4Packet
    print("llc packet")
    print(llcPacket.hex())
    mac80211nIns = mac80211.mac80211(2,  # type
                                     8,  # sub type, 8 = QoS Data
                                     1,  # to DS, station to AP
                                     0,  # from DS
                                     0,  # retry
                                     0,  # protected
                                     'f4:69:d5:80:0f:a0',  # dest add
                                     '00:c0:ca:b1:5b:e1',  # sour add
                                     'f4:69:d5:80:0f:a0',  # recv add
                                     2704,  # sequence
                                     llcPacket, True)
    mac80211Packet = mac80211nIns.genPacket()
    print("mac packet: ", len(mac80211Packet))
    print(mac80211Packet.hex())
    return mac80211Packet

udpPayload  = "123456789012345678901234567890"
udpPayload500  = "123456789012345678901234567890abcdefghijklmnopqrst" * 10
udpPayload1 = "This is packet for station 001"
udpPayload2 = "This is packet for station 002"

if __name__ == "__main__":
    h = p8h.phy80211header()
    phy80211Ins = phy80211()




    pkt = genMac80211UdpMPDU(udpPayload500)
    phy80211Ins.genVht(pkt, p8h.phy80211format('vht', mcs=0, bw=h.BW_20, nSTS=2, pktLen=len(pkt), shortGi=False))
    ssFinal = phy80211Ins.genFinalSig(100.0, 311233, 2000, True, 10000)
    phy80211Ins.genSigBinFile(ssFinal, "/home/cloud/sdr/sig80211VhtGenCfo100", False)




    # mcsSigFinal = [[]]
    # pkt = genMac80211UdpMPDU(udpPayload500)
    # for mcsIter in range(0, 9):
    #     phy80211Ins.genVht(pkt, p8h.phy80211format('vht', mcs=8-mcsIter, bw=h.BW_20, nSTS=1, pktLen=len(pkt), shortGi=False))
    #     # 100 for 1.5 power in LTF, and 20 for max under 1
    #     ssFinal = phy80211Ins.genFinalSig(100.0, 311233, 100, True, 10000)
    #     mcsSigFinal[0] += ssFinal[0]
    # phy80211Ins.genSigBinFile(mcsSigFinal, "/home/cloud/sdr/sig80211VhtGenCfoMcs100", False)











    # NDP
    # phyFormat = p8h.phy80211format('vht', mcs = 0, bw = h.BW_20, nSTS = 2, shortGi = False)
    # sigg = phy80211Ins.genVht(b'', phyFormat, partialAid=0)

    # # mu-mimo
    # ndpRawDataR1 = p8h.readBinFileFromMatDouble("/home/cloud/sdr/gr-ieee80211/tools/ndp1r.bin")
    # ndpRawDataI1 = p8h.readBinFileFromMatDouble("/home/cloud/sdr/gr-ieee80211/tools/ndp1i.bin")
    # ndpRawDataR2 = p8h.readBinFileFromMatDouble("/home/cloud/sdr/gr-ieee80211/tools/ndp2r.bin")
    # ndpRawDataI2 = p8h.readBinFileFromMatDouble("/home/cloud/sdr/gr-ieee80211/tools/ndp2i.bin")
    # ndpRx1 = []
    # ndpRx2 = []
    # for i in range(0, len(ndpRawDataR1)):
    #     ndpRx1.append(ndpRawDataR1[i] + ndpRawDataI1[i] * 1j)
    #     ndpRx2.append(ndpRawDataR2[i] + ndpRawDataI2[i] * 1j)
    # ltfIndex = 640

    # # plt.figure(91)
    # # plt.plot(np.real(ndpRx1[ltfIndex + 16: ltfIndex + 80]))
    # # plt.plot(np.imag(ndpRx1[ltfIndex + 16: ltfIndex + 80]))
    # # plt.figure(92)
    # # plt.plot(np.real(ndpRx1[ltfIndex + 16+80: ltfIndex + 80+80]))
    # # plt.plot(np.imag(ndpRx1[ltfIndex + 16+80: ltfIndex + 80+80]))
    # # plt.figure(93)
    # # plt.plot(np.real(ndpRx2[ltfIndex + 16: ltfIndex + 80]))
    # # plt.plot(np.imag(ndpRx2[ltfIndex + 16: ltfIndex + 80]))
    # # plt.figure(94)
    # # plt.plot(np.real(ndpRx2[ltfIndex + 16+80: ltfIndex + 80+80]))
    # # plt.plot(np.imag(ndpRx2[ltfIndex + 16+80: ltfIndex + 80+80]))

    # # get symbols
    # nScDataPilot = 56
    # nSts = 2
    # nRx = 1
    # ltfSym = []
    # ltfSym.append(p8h.procRemovePilots(p8h.procFftDemod(ndpRx1[ltfIndex + 16: ltfIndex + 80], nScDataPilot, nSts)))
    # ltfSym.append(p8h.procRemovePilots(p8h.procFftDemod(ndpRx1[ltfIndex + 16 + 80: ltfIndex + 80 + 80], nScDataPilot, nSts)))
    # # compute feedback
    # vFb1 = p8h.procVhtChannelFeedback(ltfSym, nSts, nRx)
    # print("feedback v 1")
    # for each in vFb1:
    #     print(each)
    # ltfSym = []
    # ltfSym.append(p8h.procRemovePilots(p8h.procFftDemod(ndpRx2[ltfIndex + 16: ltfIndex + 80], nScDataPilot, nSts)))
    # ltfSym.append(p8h.procRemovePilots(p8h.procFftDemod(ndpRx2[ltfIndex + 16 + 80: ltfIndex + 80 + 80], nScDataPilot, nSts)))
    # plt.figure(121)
    # plt.plot(np.real(ndpRx2[ltfIndex + 16: ltfIndex + 80]))
    # plt.plot(np.imag(ndpRx2[ltfIndex + 16: ltfIndex + 80]))
    # plt.figure(122)
    # plt.plot(np.real(ndpRx2[ltfIndex + 16 + 80: ltfIndex + 80 + 80]))
    # plt.plot(np.imag(ndpRx2[ltfIndex + 16 + 80: ltfIndex + 80 + 80]))
    # # compute feedback
    # vFb2 = p8h.procVhtChannelFeedback(ltfSym, nSts, nRx)
    # print("feedback v 2")
    # for each in vFb2:
    #     print(each)
    # # combine the channel together
    # bfH = []
    # for k in range(0, nScDataPilot):
    #     print("bfH", k)
    #     bfH.append(np.concatenate((vFb1[k], vFb2[k]), axis=1))
    #     print(bfH[k])
    # # plt.figure(111)
    # # plt.plot(np.real([each[0][0] for each in bfH]))
    # # plt.plot(np.imag([each[0][0] for each in bfH]))
    # # plt.figure(112)
    # # plt.plot(np.real([each[0][1] for each in bfH]))
    # # plt.plot(np.imag([each[0][1] for each in bfH]))
    # # plt.figure(113)
    # # plt.plot(np.real([each[1][0] for each in bfH]))
    # # plt.plot(np.imag([each[1][0] for each in bfH]))
    # # plt.figure(114)
    # # plt.plot(np.real([each[1][1] for each in bfH]))
    # # plt.plot(np.imag([each[1][1] for each in bfH]))
    # plt.figure(111)
    # plt.plot(np.real([each[0][0] for each in vFb1]))
    # plt.plot(np.imag([each[0][0] for each in vFb1]))
    # plt.figure(112)
    # plt.plot(np.real([each[1][0] for each in vFb1]))
    # plt.plot(np.imag([each[1][0] for each in vFb1]))
    # plt.figure(113)
    # plt.plot(np.real([each[0][0] for each in vFb2]))
    # plt.plot(np.imag([each[0][0] for each in vFb2]))
    # plt.figure(114)
    # plt.plot(np.real([each[1][0] for each in vFb2]))
    # plt.plot(np.imag([each[1][0] for each in vFb2]))
    # # compute spatial matrix Q, ZF
    # bfQTmp = []
    # for k in range(0, nScDataPilot):
    #     print("bfQ", k)
    #     bfQTmp.append(np.matmul(bfH[k], np.linalg.inv(np.matmul(bfH[k].conjugate().T, bfH[k]))))
    #     print(bfQTmp[k])
    # # normalize Q
    # bfQForFftNormd = []
    # for k in range(0, nScDataPilot):
    #     bfQForFftNormd.append(bfQTmp[k] / np.linalg.norm(bfQTmp[k]) * np.sqrt(nSts))
    #     print("bfQNormd", k)
    #     print(bfQForFftNormd[k])
    # # map Q to FFT non-zero sub carriers
    # bfQForFftNormdForFft = [np.ones_like(bfQForFftNormd[0])] * 3 + bfQForFftNormd[0:28] + [
    #     np.ones_like(bfQForFftNormd[0])] + bfQForFftNormd[28:56] + [np.ones_like(bfQForFftNormd[0])] * 4

    # # plt.figure(101)
    # # plt.plot(np.real([each[0][0] for each in bfQForFftNormdForFft]))
    # # plt.plot(np.imag([each[0][0] for each in bfQForFftNormdForFft]))
    # # plt.figure(102)
    # # plt.plot(np.real([each[0][1] for each in bfQForFftNormdForFft]))
    # # plt.plot(np.imag([each[0][1] for each in bfQForFftNormdForFft]))
    # # plt.figure(103)
    # # plt.plot(np.real([each[1][0] for each in bfQForFftNormdForFft]))
    # # plt.plot(np.imag([each[1][0] for each in bfQForFftNormdForFft]))
    # # plt.figure(104)
    # # plt.plot(np.real([each[1][1] for each in bfQForFftNormdForFft]))
    # # plt.plot(np.imag([each[1][1] for each in bfQForFftNormdForFft]))

    # pkt1 = genMac80211UdpMPDU(udpPayload1)
    # pkt2 = genMac80211UdpMPDU(udpPayload2)
    # print("pkt 1 byte numbers:", len(pkt1))
    # print([int(each) for each in pkt1])
    # print("pkt 2 byte numbers:", len(pkt2))
    # print([int(each) for each in pkt2])
    # sigg = phy80211Ins.genVhtMu([pkt1, pkt2], [p8h.phy80211format('vht', mcs=0, bw=h.BW_20, nSTS=1, pktLen=len(pkt1), shortGi=False), p8h.phy80211format('vht', mcs=0, bw=h.BW_20, nSTS=1, pktLen=len(pkt2), shortGi=False)], bfQ = bfQForFftNormdForFft, groupId=2)
    # # phy80211Ins.genSigBinFile("sig80211VhtGenMu", False)

    # plt.show()







