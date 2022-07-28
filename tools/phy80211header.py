import numpy as np
from matplotlib import pyplot as plt
import struct

"""
|______-_________________________-_________________________-_____|______-_________________________-_________________________-_____|______-_________________________-_________________________-_____|______-_________________________-_________________________-_____|   20
  0x6 -26                        0                         26 0x5 
|______-_________________________________________________________|-_________________________________________________________-_____|______-_________________________________________________________|-_________________________________________________________-_____|   40
  0x6 -58                                                         0                                                        58  0x5
|______-_________________________________________________________|________________________________________________________________|-_______________________________________________________________|__________________________________________________________-_____|   80
  0x6 -122                                                                                                                         0                                                                                                                        122  0x5

ver 1.0
support up to 4x4 80M
"""

def getBitCrc8(bitsIn):
    c = [1] * 8
    for b in bitsIn:
        next_c = [0] * 8
        next_c[0] = b ^ c[7]
        next_c[1] = b ^ c[7] ^ c[0]
        next_c[2] = b ^ c[7] ^ c[1]
        next_c[3] = c[2]
        next_c[4] = c[3]
        next_c[5] = c[4]
        next_c[6] = c[5]
        next_c[7] = c[6]
        c = next_c
    return [1 - b for b in c[::-1]]

def procConcat2Symbol(sa, sb):
    sb[0] = sb[0] * 0.5 + sa[len(sa) - 1] * 0.5
    return sa + sb

def procTonePhase(inQam):
    # input QAM with DC
    h = phy80211header()
    if (len(inQam) == 57):
        return inQam
    elif (len(inQam) == 117):
        return [inQam[i] * h.TONE_ROTATION_40[i] for i in range(0, 117)]
    elif (len(inQam) == 245):
        return [inQam[i] * h.TONE_ROTATION_80[i] for i in range(0, 245)]
    else:
        print("cloud 80211 procTonePhase: input length error")

def procPilotInsert(inQam, p):
    if(len(inQam) == 48 and len(p) == 4):
        return (inQam[0:5] + [p[0]] + inQam[5:18] + [p[1]] + inQam[18:30] + [p[2]] + inQam[30:43] + [p[3]] + inQam[43:48])
    elif(len(inQam) == 52 and len(p) == 4):
        return (inQam[0:7] + [p[0]] + inQam[7:20] + [p[1]] + inQam[20:32] + [p[2]] + inQam[32:45] + [p[3]] + inQam[45:52])
    elif(len(inQam) == 110 and len(p) == 6):
        return (inQam[0:5] + [p[0]] + inQam[5:20] + [p[1]] + inQam[20:45] + [p[2]] + inQam[45:65] + [p[3]] + inQam[65:78] + [p[4]] + inQam[78:105] + [p[5]] + inQam[105:110])
    elif(len(inQam) == 236 and len(p) == 8):
        return (inQam[0:19] + [p[0]] + inQam[19:46] + [p[1]] + inQam[46:81] + [p[2]] + inQam[81:108] + [p[3]] + inQam[108:128] + [p[4]] + inQam[128:155] + [p[5]] + inQam[155:190] + [p[6]] + inQam[190:217] + [p[7]] + inQam[217:236])
    else:
        print("cloud 80211 procPilotInsert: input length error qam %d pilots %d" % (len(inQam), len(p)))
        return []

def procDcInsert(inQam):
    if(len(inQam) == 52):
        return (inQam[:26] + [0] + inQam[26:])
    elif(len(inQam) == 56):
        return (inQam[:28] + [0] + inQam[28:])
    elif(len(inQam) == 114):
        return (inQam[:57] + [0]*3 + inQam[57:])
    elif(len(inQam) == 242):
        return (inQam[:121] + [0]*3 + inQam[121:])
    else:
        print("cloud 80211 procDcInsert: input length error")
        return []


def procNonDataSCOld(inQam):
    # input QAM has DC 0
    if (len(inQam) == 53):
        return (inQam[26:53] + [0] * 11 + inQam[0:26])
    elif (len(inQam) == 57):
        return (inQam[28:57] + [0] * 7 + inQam[0:28])
    elif (len(inQam) == 117):
        return (inQam[58:117] + [0] * 11 + inQam[0:58])
    elif (len(inQam) == 245):
        return (inQam[122:245] + [0] * 11 + inQam[0:122])
    else:
        print("cloud 80211 procNonDataSC: input length error")
        return []

def procNonDataSC(inQam):
    # input QAM has DC 0
    if (len(inQam) in [53, 117, 245]):
        return ([0] * 6 + inQam + [0] * 5)
    elif (len(inQam) == 57):
        return ([0] * 4 + inQam + [0] * 3)
    else:
        print("cloud 80211 procNonDataSC: input length error")
        return []

def procLegacyCSD(inQam, nSs, iSs, spr):
    h = phy80211header()
    tmpPhase = -1.0j * 2 * np.pi * h.CYCLIC_SHIFT_L[nSs-1][iSs] * spr * 0.001
    return [(inQam[i] * np.exp(tmpPhase * ((i - int(len(inQam)/2)) / len(inQam)))) for i in range(0, len(inQam))]

def procCSD(inQam, nSs, iSs, spr):
    h = phy80211header()
    tmpPhase = -1.0j * 2 * np.pi * h.CYCLIC_SHIFT_NL[nSs-1][iSs] * spr * 0.001
    return [(inQam[i] * np.exp(tmpPhase * ((i - int(len(inQam)/2)) / len(inQam)))) for i in range(0, len(inQam))]

def procIDFTOld(inQam):
    return list(np.fft.ifft(inQam))

def procIDFT(inQam):
    # fft shift
    if(len(inQam) in [64, 128, 256]):
        return list(np.fft.ifft((inQam[int(len(inQam)/2):] + inQam[:int(len(inQam)/2)])))
    else:
        print("cloud 80211 procIDFT: input length error")
        return inQam

def procToneScaling(inSig, inNtf, nSs):
    return [each/np.sqrt(inNtf * nSs) for each in inSig]

def procGi(inSig):
    if(len(inSig) == 64):
        return inSig[48:64] + inSig
    elif(len(inSig) == 128):
        return inSig[96:128] + inSig
    elif(len(inSig) == 256):
        return inSig[192:256] + inSig
    else:
        print("cloud 80211 procGi: input length error")
        return inSig

def getMod(f):
    h = phy80211header()
    m = phy80211mod()
    if(f.type == 'l'):
        m.nSD = 48
        m.nSP = 4
        m.nSS = 1
        m.nES = 1
        m.spr = 20
        if (f.mcs == 0):
            m.mod = h.QAM_BPSK
            m.legacyRate = 11
            m.nBPSCS = 1
            m.cr = h.CR_12
        elif (f.mcs == 1):
            m.mod = h.QAM_BPSK
            m.legacyRate = 15
            m.nBPSCS = 1
            m.cr = h.CR_34
        elif (f.mcs == 2):
            m.mod = h.QAM_QPSK
            m.legacyRate = 10
            m.nBPSCS = 2
            m.cr = h.CR_12
        elif (f.mcs == 3):
            m.mod = h.QAM_QPSK
            m.legacyRate = 14
            m.nBPSCS = 2
            m.cr = h.CR_34
        elif (f.mcs == 4):
            m.mod = h.QAM_16QAM
            m.legacyRate = 9
            m.nBPSCS = 4
            m.cr = h.CR_12
        elif (f.mcs == 5):
            m.mod = h.QAM_16QAM
            m.legacyRate = 13
            m.nBPSCS = 4
            m.cr = h.CR_34
        elif (f.mcs == 6):
            m.mod = h.QAM_64QAM
            m.legacyRate = 8
            m.nBPSCS = 6
            m.cr = h.CR_23
        elif (f.mcs == 7):
            m.mod = h.QAM_64QAM
            m.legacyRate = 12
            m.nBPSCS = 6
            m.cr = h.CR_34
        else:
            print("cloud 80211 get mod: mcs error")
        m.nCBPSS = int(m.nSD * m.nBPSCS)
        m.nCBPS = int(m.nCBPSS * m.nSS)
        if (m.cr == h.CR_12):
            m.nDBPS = int(m.nCBPS / 2)
        elif (m.cr == h.CR_23):
            m.nDBPS = int(m.nCBPS / 3 * 2)
        elif (m.cr == h.CR_34):
            m.nDBPS = int(m.nCBPS / 4 * 3)
        else:
            print("error")
            return
        m.dr = m.nDBPS / 4
        m.drs = 0
        m.pktLen = f.pktLen
        m.nSym = int(np.ceil((m.pktLen*8 + 22)/m.nDBPS))
        m.psduLen = m.pktLen

    elif(f.type == 'ht'):
        m.legacyRate = 11
        m.nSS = int(np.floor(f.mcs / 8)) + 1
        if ((f.mcs % 8) == 0):
            m.mod = h.QAM_BPSK
            m.cr = h.CR_12
            m.nBPSCS = 1
        elif ((f.mcs % 8) == 1):
            m.mod = h.QAM_QPSK
            m.cr = h.CR_12
            m.nBPSCS = 2
        elif ((f.mcs % 8) == 2):
            m.mod = h.QAM_QPSK
            m.cr = h.CR_34
            m.nBPSCS = 2
        elif ((f.mcs % 8) == 3):
            m.mod = h.QAM_16QAM
            m.cr = h.CR_12
            m.nBPSCS = 4
        elif ((f.mcs % 8) == 4):
            m.mod = h.QAM_16QAM
            m.cr = h.CR_34
            m.nBPSCS = 4
        elif ((f.mcs % 8) == 5):
            m.mod = h.QAM_64QAM
            m.cr = h.CR_23
            m.nBPSCS = 6
        elif ((f.mcs % 8) == 6):
            m.mod = h.QAM_64QAM
            m.cr = h.CR_34
            m.nBPSCS = 6
        elif ((f.mcs % 8) == 7):
            m.mod = h.QAM_64QAM
            m.cr = h.CR_56
            m.nBPSCS = 6
        else:
            print("error")
            return
        if(f.bw == h.BW_20):
            m.spr = 20
            m.nSD = 52
            m.nSP = 4
            m.nIntlevCol = 13
            m.nIntlevRow = int(4 * m.nBPSCS)
            m.nIntlevRot = 11
        elif(f.bw == h.BW_40):
            m.spr = 40
            m.nSD = 108
            m.nSP = 6
            m.nIntlevCol = 18
            m.nIntlevRow = int(6 * m.nBPSCS)
            m.nIntlevRot = 29
        else:
            print("error")
            return
        m.nCBPSS = int(m.nSD * m.nBPSCS)
        m.nCBPS = int(m.nCBPSS * m.nSS)
        if(m.cr == h.CR_12):
            m.nDBPS = int(m.nCBPS / 2)
        elif (m.cr == h.CR_23):
            m.nDBPS = int(m.nCBPS / 3 * 2)
        elif (m.cr == h.CR_34):
            m.nDBPS = int(m.nCBPS / 4 * 3)
        elif (m.cr == h.CR_56):
            m.nDBPS = int(m.nCBPS / 6 * 5)
        else:
            print("error")
            return
        m.dr = m.nDBPS / 4.0
        m.drs = m.nDBPS / 3.6
        if(m.drs < 300.001):
            m.nES = 1
        elif(m.drs < 600.001):
            m.nES = 2
        else:
            print("not supported yet")
            return None
        m.nLtf = h.LTF_HT_N[m.nSS]
        m.pktLen = f.pktLen
        m.nSym = int(np.ceil((m.pktLen * 8 + 16 + 6 * m.nES) / m.nDBPS))
        m.psduLen = m.pktLen


    elif (f.type == 'vht'):
        m.legacyRate = 11
        m.nSS = f.nSTS
        if (f.mcs == 0):
            m.mod = h.QAM_BPSK
            m.cr = h.CR_12
            m.nBPSCS = 1
        elif (f.mcs == 1):
            m.mod = h.QAM_QPSK
            m.cr = h.CR_12
            m.nBPSCS = 2
        elif (f.mcs == 2):
            m.mod = h.QAM_QPSK
            m.cr = h.CR_34
            m.nBPSCS = 2
        elif (f.mcs == 3):
            m.mod = h.QAM_16QAM
            m.cr = h.CR_12
            m.nBPSCS = 4
        elif (f.mcs == 4):
            m.mod = h.QAM_16QAM
            m.cr = h.CR_34
            m.nBPSCS = 4
        elif (f.mcs == 5):
            m.mod = h.QAM_64QAM
            m.cr = h.CR_23
            m.nBPSCS = 6
        elif (f.mcs == 6):
            m.mod = h.QAM_64QAM
            m.cr = h.CR_34
            m.nBPSCS = 6
        elif (f.mcs == 7):
            m.mod = h.QAM_64QAM
            m.cr = h.CR_56
            m.nBPSCS = 6
        elif (f.mcs == 8):
            m.mod = h.QAM_256QAM
            m.cr = h.CR_34
            m.nBPSCS = 8
        elif(f.mcs == 9):
            m.mod = h.QAM_256QAM
            m.cr = h.CR_56
            m.nBPSCS = 8
        else:
            print("cloud 80211 get mod: mcs error")
            return None
        if (f.bw == h.BW_20):
            if(f.mcs == 9 and m.nSS in [1, 2, 4, 5, 7, 8]):
                print("cloud 80211 get mod: mcs error")
                return None
            m.spr = 20
            m.nSD = 52
            m.nSP = 4
            m.nIntlevCol = 13
            m.nIntlevRow = int(4 * m.nBPSCS)
            if(m.nSS <= 4):
                m.nIntlevRot = 11
            else:
                m.nIntlevRot = 6
        elif(f.bw == h.BW_40):
            m.spr = 40
            m.nSD = 108
            m.nSP = 6
            m.nIntlevCol = 18
            m.nIntlevRow = int(6 * m.nBPSCS)
            if (m.nSS <= 4):
                m.nIntlevRot = 29
            else:
                m.nIntlevRot = 13
        elif(f.bw == h.BW_80):
            if ((f.mcs == 6 and m.nSS in [3, 7]) or (f.mcs == 9 and m.nSS == 6)):
                print("cloud 80211 get mod: mcs error")
                return None
            m.spr = 80
            m.nSD = 234
            m.nSP = 8
            m.nIntlevCol = 26
            m.nIntlevRow = int(9 * m.nBPSCS)
            if (m.nSS <= 4):
                m.nIntlevRot = 58
            else:
                m.nIntlevRot = 28
        else:
            print("error")
            return None
        m.nCBPSS = int(m.nSD * m.nBPSCS)
        m.nCBPSSI = m.nCBPSS
        m.nCBPS = int(m.nCBPSS * m.nSS)
        if (m.cr == h.CR_12):
            m.nDBPS = int(m.nCBPS / 2)
        elif (m.cr == h.CR_23):
            m.nDBPS = int(m.nCBPS / 3 * 2)
        elif (m.cr == h.CR_34):
            m.nDBPS = int(m.nCBPS / 4 * 3)
        elif (m.cr == h.CR_56):
            m.nDBPS = int(m.nCBPS / 6 * 5)
        else:
            print("error")
            return None
        m.dr = m.nDBPS / 4.0
        m.drs = m.nDBPS / 3.6
        if(m.drs < 600.001):
            m.nES = 1
        elif(m.drs < 1200.001):
            m.nES = 2
        elif(m.drs < 1800.001):
            m.nES = 3
        else:
            print("not supported yet")
            return None
        m.nLtf = h.LTF_VHT_N[m.nSS]
        m.pktLen = f.pktLen # for VHT, this is apep len
        mSTBC = 1   # this ver not supporting STBC
        if(m.pktLen > 0):
            m.nSym = mSTBC * int(np.ceil((8*m.pktLen + 16 + 6 * m.nES)/(mSTBC * m.nDBPS)))
            m.psduLen = int(np.floor((m.nSym * m.nDBPS - 16 - 6 * m.nES) / 8))
        else:
            m.nSym = 0
            m.psduLen = 0

    else:
        print("type error")
        return None

    print("SPR %d, legacy rate %d, SS %d, mcs %d, mod %s, CR %s, BPSCS %d, SD %d, SP %d, CBPS %d, DBPS %d, ES %d, DR %d, DRS %d" % (
        m.spr, m.legacyRate, m.nSS, f.mcs, h.QAM_MODU_STR[m.mod], h.CR_STR[m.cr], m.nBPSCS, m.nSD, m.nSP, m.nCBPS, m.nDBPS, m.nES, m.dr, m.drs))
    return m

class phy80211header:
    # 20M, 40M, 80M, up to 4 streams, a/g/n/ac
    def __init__(self):
        # bandwidth
        self.BW_20 = 0
        self.BW_40 = 1
        self.BW_80 = 2
        # coding rate
        self.CR_12 = 0
        self.CR_23 = 1
        self.CR_34 = 2
        self.CR_56 = 3
        # qam index
        self.QAM_BPSK = 0
        self.QAM_QBPSK = 1
        self.QAM_QPSK = 2
        self.QAM_16QAM = 3
        self.QAM_64QAM = 4
        self.QAM_256QAM = 5
        # qam constellation
        self.QAM_MODU_TAB = [
            [(-1 + 0j), (1 + 0j)],      # 0 bpsk
            [(0 - 1j), (0 + 1j)],       # 1 q-bpsk
            [each * np.sqrt(1 / 2) for each in [(-1-1.0j), (1-1.0j), (-1+1.0j), (1+1.0j)]],     # 2 qpsk
            [each * np.sqrt(1 / 10) for each in [                                               # 3 16 qam
                (-3 - 3j), (3 - 3j), (-1 - 3j), (1 - 3j), (-3 + 3j), (3 + 3j), (-1 + 3j), (1 + 3j),
                (-3 - 1j), (3 - 1j), (-1 - 1j), (1 - 1j), (-3 + 1j), (3 + 1j), (-1 + 1j), (1 + 1j)]],
            [each * np.sqrt(1 / 42) for each in [
                (-7 - 7j), (7 - 7j), (-1 - 7j), (1 - 7j), (-5 - 7j), (5 - 7j), (-3 - 7j), (3 - 7j),     # 4 64qam
                (-7 + 7j), (7 + 7j), (-1 + 7j), (1 + 7j), (-5 + 7j), (5 + 7j), (-3 + 7j), (3 + 7j),
                (-7 - 1j), (7 - 1j), (-1 - 1j), (1 - 1j), (-5 - 1j), (5 - 1j), (-3 - 1j), (3 - 1j),
                (-7 + 1j), (7 + 1j), (-1 + 1j), (1 + 1j), (-5 + 1j), (5 + 1j), (-3 + 1j), (3 + 1j),
                (-7 - 5j), (7 - 5j), (-1 - 5j), (1 - 5j), (-5 - 5j), (5 - 5j), (-3 - 5j), (3 - 5j),
                (-7 + 5j), (7 + 5j), (-1 + 5j), (1 + 5j), (-5 + 5j), (5 + 5j), (-3 + 5j), (3 + 5j),
                (-7 - 3j), (7 - 3j), (-1 - 3j), (1 - 3j), (-5 - 3j), (5 - 3j), (-3 - 3j), (3 - 3j),
                (-7 + 3j), (7 + 3j), (-1 + 3j), (1 + 3j), (-5 + 3j), (5 + 3j), (-3 + 3j), (3 + 3j)]],
            [0] * 256]
        tmp256qam = [-15, 15, -1, 1, -9, 9, -7, 7, -13, 13, -3, 3, -11, 11, -5, 5]                         # 5 256qam
        for i in range(0, 256):
            self.QAM_MODU_TAB[self.QAM_256QAM][i] = np.sqrt(1 / 170) * (
                tmp256qam[i % 16] + tmp256qam[int(np.floor(i / 16))] * 1.0j)
        # string for print
        self.QAM_MODU_STR = ['BPSK','Q-BPSK', 'QPSK', '16QAM', '64QAM', '256QAM']
        self.CR_STR = ['1/2', '2/3', '3/4', '5/6']
        # training field, all include DC 0
        # legacy stf
        self.STF_L_26 = [each * np.sqrt(1 / 2) for each in
                         [0, 0, 1 + 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, -1 - 1j, 0, 0, 0,
                          1 + 1j, 0, 0, 0, 0, 0, 0, 0, -1 - 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0,
                          0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0, 0]]
        self.STF_L_58 = self.STF_L_26 + [0] * 11 + self.STF_L_26
        self.STF_L_122 = self.STF_L_58 + [0] * 11 + self.STF_L_58
        self.STF_L = [self.STF_L_26, self.STF_L_58, self.STF_L_122]
        # ht stf
        self.STF_HT_28 = [0, 0] + self.STF_L_26 + [0, 0]
        self.STF_HT_58 = self.STF_L_26 + [0] * 11 + self.STF_L_26
        self.STF_HT = [self.STF_HT_28, self.STF_HT_58]
        # vht stf
        self.STF_VHT_28 = self.STF_HT_28
        self.STF_VHT_58 = self.STF_HT_58
        self.STF_VHT_122 = self.STF_VHT_58 + [0] * 11 + self.STF_VHT_58
        self.STF_VHT = [self.STF_VHT_28, self.STF_VHT_58, self.STF_VHT_122]
        # time domain, use for trigger
        self.STF_L_26_T = procIDFT(procNonDataSC(self.STF_L_26))
        self.STF_L_26_T_SEQ = procIDFT(procNonDataSC(self.STF_L_26))[0:16]
        self.STF_L_26_T_SEQ_CONJ = list(np.conj(self.STF_L_26_T_SEQ))
        # legacy ltf
        self.__LTF_L = [1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1]
        self.__LTF_R = [1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1]
        self.LTF_L_26 = self.__LTF_L + [0] + self.__LTF_R
        self.LTF_L_58 = self.LTF_L_26 + [0] * 11 + self.LTF_L_26
        self.LTF_L_122 = self.LTF_L_58 + [0] * 11 + self.LTF_L_58
        self.LTF_L = [self.LTF_L_26, self.LTF_L_58, self.LTF_L_122]
        # ht ltf
        self.LTF_HT_28 = [1, 1] + self.__LTF_L + [0] + self.__LTF_R + [-1, -1]
        self.LTF_HT_58 = self.__LTF_L + [1] + self.__LTF_R + [-1, -1, -1, 1, 0, 0, 0, -1, 1, 1, -1] + \
                         self.__LTF_L + [1] + self.__LTF_R
        self.LTF_HT = [self.LTF_HT_28, self.LTF_HT_58]
        # vht ltf
        self.LTF_VHT_28 = self.LTF_HT_28
        self.LTF_VHT_58 = self.LTF_HT_58
        self.LTF_VHT_122 = self.__LTF_L + [1] + self.__LTF_R + [-1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1] + \
                           self.__LTF_L + [1] + self.__LTF_R + [1, -1, 1, -1, 0, 0, 0, 1, -1, -1, 1] + \
                           self.__LTF_L + [1] + self.__LTF_R + [-1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1] +\
                           self.__LTF_L + [1] + self.__LTF_R
        self.LTF_VHT = [self.LTF_VHT_28, self.LTF_VHT_58, self.LTF_VHT_122]
        # non legacy ltf number
        self.LTF_HT_N = [0, 1, 2, 4, 4]
        self.LTF_VHT_N = [0, 1, 2, 4, 4]
        # LTF polarity of ss
        self.P_LTF_HT_4 = [
            [1, -1, 1, 1],
            [1, 1, -1, 1],
            [1, 1, 1, -1],
            [-1, 1, 1, 1]]
        # for data sub carriers, times P_VHT-LTF, for pilot sub carriers, times R_VHT-LTF, which is first row of P_VHT-LTF
        self.P_LTF_VHT_4 = [
            [1, -1, 1, 1],
            [1, 1, -1, 1],
            [1, 1, 1, -1],
            [-1, 1, 1, 1]]
        self.R_LTF_VHT_4 = [1, -1, 1, 1]
        # polarity of sig b constellations
        self.P_SIG_B_NSTS478 = [1, 1, 1, -1, 1, 1, 1, -1]
        # scale factor N_Tone_Field, use with BW index
        self.SCALENTF_STF_L = [12, 24, 48]
        self.SCALENTF_LTF_L = [52, 104, 208]
        self.SCALENTF_SIG_L = [52, 104, 208]
        self.SCALENTF_SIG_HT = [52, 104, 0]
        self.SCALENTF_STF_HT = [12, 24, 0]
        self.SCALENTF_LTF_HT = [56, 114, 0]
        self.SCALENTF_DATA_HT = [56, 114, 0]
        self.SCALENTF_SIG_VHT_A = [52, 104, 208]
        self.SCALENTF_STF_VHT = [12, 24, 48]
        self.SCALENTF_LTF_VHT = [56, 114, 242]
        self.SCALENTF_SIG_VHT_B = [56, 114, 242]
        self.SCALENTF_DATA_VHT = [56, 114, 242]
        # pilot
        self.PILOT_L = [ 1,  1,  1, -1]
        self.PILOT_HT = [
            # 20M
            [
                # sts 1
                [[ 1,  1,  1, -1]],
                # sts 2
                [[ 1,  1, -1, -1],
                 [ 1, -1, -1,  1]],
                # sts 3
                [[ 1,  1, -1, -1],
                 [ 1, -1,  1, -1],
                 [-1,  1,  1, -1]],
                # sts 4
                [[ 1,  1,  1, -1],
                 [ 1,  1, -1,  1],
                 [ 1, -1,  1,  1],
                 [-1,  1,  1,  1]]
            ],
            # 40M
            [
                # sts 1
                [[ 1,  1,  1, -1, -1,  1]],
                # sts 2
                [[ 1,  1, -1, -1, -1, -1],
                 [ 1,  1,  1, -1,  1,  1]],
                # sts 3
                [[ 1,  1, -1, -1, -1, -1],
                 [ 1,  1,  1, -1,  1,  1],
                 [ 1, -1,  1, -1, -1,  1]],
                # sts 4
                [[ 1,  1, -1, -1, -1, -1],
                 [ 1,  1,  1, -1,  1,  1],
                 [ 1, -1,  1, -1, -1,  1],
                 [-1,  1,  1,  1, -1,  1]]
            ]
        ]
        self.PILOT_VHT = [
            # 20M
            [ 1,  1,  1, -1],
            # 40M
            [ 1,  1,  1, -1, -1,  1],
            # 80M
            [ 1,  1,  1, -1, -1,  1,  1,  1]
        ]
        # pilot polarity sequence
        self.PILOT_PS = [
            1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1,
            1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1,
            1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1,
            -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1,
            -1, -1, -1]
        # cylic shift
        # Legacy part, include lstf, lltf, lsig, htsig and vhtsiga
        self.CYCLIC_SHIFT_L = [
            [0,    0,    0,    0],
            [0, -200,    0,    0],
            [0, -100, -200,    0],
            [0,  -50, -100, -150]
        ]
        # non-legacy part, htdata, vhtsigb, vhtdata
        self.CYCLIC_SHIFT_NL = [
            [0,    0,    0,    0],
            [0, -400,    0,    0],
            [0, -400, -200,    0],
            [0, -400, -200, -600]
        ]
        # tone rotation
        self.TONE_ROTATION_20 = [1] * 57  #
        self.TONE_ROTATION_40 = [1] * 58 + [1j] * 59  # 117 = 58 + 59
        self.TONE_ROTATION_80 = [1] * 58 + [-1] * 187  # 245 is -122 to 122
        # NDP sig b bits
        self.NDP_SIG_B_20 = [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        self.NDP_SIG_B_40 = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1]
        self.NDP_SIG_B_80 = [0 ,1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
        self.NDP_SIG_B = [self.NDP_SIG_B_20, self.NDP_SIG_B_40, self.NDP_SIG_B_80]

class phy80211format:
    def __init__(self, type, mcs, bw, nSTS, pktLen, shortGi = False):
        h = phy80211header()
        if(type == 'l'):
            if(mcs>=0 and mcs<=7 and bw==h.BW_20 and nSTS == 1):
                pass
            else:
                print("cloud 80211 phy format: mcs or bw or nSTS error")
                return
        elif(type == 'ht'):
            if (mcs >= 0 and mcs <= 31 and bw in [h.BW_20, h.BW_40] and nSTS in [1, 2, 3, 4]):
                pass
            else:
                print("cloud 80211 phy format: mcs or bw or nSTS error")
                return
        elif(type == 'vht'):
            if (mcs >= 0 and mcs <= 9 and bw in [h.BW_20, h.BW_40, h.BW_80] and nSTS in [1, 2, 3, 4]):
                pass
            else:
                print("cloud 80211 phy format: mcs or bw or nSTS error")
                return
        else:
            print("cloud 80211 phy format: type error, chose in l, ht, vht")
            return
        self.type = type
        self.mcs = mcs  # l:0-7, ht:0-31, vht:0-9
        self.bw = bw
        self.nSTS = nSTS
        self.pktLen = pktLen
        if (type == 'l'):
            self.sgi = False
        else:
            if(shortGi):
                self.sgi = True
            else:
                self.sgi = False
        bwStr = ['20', '40', '80']
        print("cloud 80211 phy format: type %s, mcs %d, bw %s, nSTS %d, short GI %r" % (self.type, self.mcs, bwStr[self.bw], self.nSTS, self.sgi))

class phy80211mod:
    def __init__(self):
        self.nSym = 0
        self.pktLen = 0
        self.psduLen = 0

        self.legacyRate = 11    # rate for legacy packet or legacy sig
        self.mod = 0        # modulation
        self.cr = 0         # coding rate
        self.spr = 20       # sampling rate in MHz
        self.nBPSCS = 1     # bits per single carrier for spatial stream
        self.nSD = 48       # number of data sub carriers
        self.nSP = 4        # number of pilot sub carriers
        self.nSS = 1        # number of spatial streams
        self.nCBPS = 48     # coded bits per symbol
        self.nDBPS = 24     # data bits per symbol
        self.nCBPSS = 48    # coded bits per symbol per spatial-stream symbol
        self.nCBPSSI = 48   # coded bits per symbol per spatial-stream symbol per interleaver
        self.dr = 6.5       # data rate Mbps
        self.drs = 7.2      # data rate short gi Mbps
        self.nES = 1        # number of BCC encoder, only used when the data rate is >= 600M
        self.nIntlevCol = 0 # used for interleave
        self.nIntlevRow = 0 # used for interleave
        self.nIntlevRot = 0 # used for interleave
        self.nLtf = 1       # non legacy LTF number

def procFftDemod(inSig, nScDataPilot, nSts):
    # check len
    if(len(inSig) in [64, 128, 256]):
        pass
    else:
        #print("cloud 80211 procFftDemod input len error")
        return [0] * len(inSig)
    # fft
    tmpF = list(np.fft.fft(inSig))
    # fft shift
    if(len(inSig) == 64):
        tmpF = tmpF[32:64] + tmpF[0:32]
        if(nScDataPilot == 56):
            # remove zero sc for non-legacy 20mhz
            tmpF = tmpF[4:32] + tmpF[33:61]
        else:
            # remove zero sc for legacy 20mhz
            tmpF = tmpF[6:32] + tmpF[33:59]
    elif(len(inSig) == 128):
        pass
    elif(len(inSig) == 256):
        pass
    else:
        tmpF = tmpF[32:64] + tmpF[0:32]
        if (nScDataPilot == 56):
            tmpF = tmpF[4:32] + tmpF[33:61]
        else:
            tmpF = tmpF[6:32] + tmpF[33:59]
    # Scale by number of active tones and FFT length
    for i in range(0, nScDataPilot):
        tmpF[i] = tmpF[i] * np.sqrt(nScDataPilot) / len(inSig)
    # Denormalization
    for i in range(0, nScDataPilot):
        tmpF[i] = tmpF[i] * np.sqrt(nSts)  # tx STS
    return tmpF

def procRemovePilots(inSig):
    if(len(inSig) == 52):
        pass
    elif(len(inSig) == 56):
        return inSig[0:7] + inSig[8:21] + inSig[22:34] + inSig[35:48] + inSig[49:56]

def procVhtDataChanEst(ltfSym, nScData, nSts, nRx):
    # ltfSym is 2d array with nLtf * nSc, nLtf is like rxAnt0Ltf0, rxAnt0Ltf1, rxAnt1Ltf0, rxAnt1Ltf1
    con_ltf_20 = [1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1,
                  -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1]
    #print("ltf con len:", len(con_ltf_20))
    con_P = [[1, -1, 1, 1],
             [1, 1, -1, 1],
             [1, 1, 1, -1],
             [-1, 1, 1, 1]]
    con_nLtf = [0, 1, 2, 4, 4]
    if(nSts == 1):
        pass
        return
    # prepare, P, Puse, ltf
    nLtf = con_nLtf[nSts]
    P = []
    for i in range(0, nSts):
        P.append([])
        for j in range(0, nLtf):
            P[i].append(con_P[i][j])
    P = np.array(P)
    #print("P", P)
    Puse = P.conj().T
    #print("Puse", Puse)
    rxLtf = []
    #print(ltfSym)
    for k in range(0, nScData):
        tmpM = []
        for i in range(0, nRx):
            tmpM.append([])
            for j in range(0, nLtf):
                tmpM[i].append(ltfSym[i*nRx + j][k])
        rxLtf.append(np.array(tmpM))
    #print(rxLtf)
    # eatimation, H = rxLTF * PH / (ltf * nLtf)
    est = []
    for k in range(0, nScData):
        est.append(np.matmul(rxLtf[k], Puse) / con_ltf_20[k] / nLtf)
        #print(est[k])

    """
    |----------------- > nSTS
    |
    |
    v
    nRX
    
    each row belongs to a rx, it shows the channel of each STS to the rx
    """

    return est

def procVhtPilotChanIntpo(dataEst, nScData, nSts, nRx):
    con_nLtf = [0, 1, 2, 4, 4]
    nLtf = con_nLtf[nSts]
    if(nScData == 52):
        nScDataPilot = nScData + 4
    else:
        nScDataPilot = nScData + 4
    h = phy80211header()
    kDP = list(range(-28, 0)) + list(range(1, 29))
    kD = kDP[0:7] + kDP[8:21] + kDP[22:34] + kDP[35:48] + kDP[49:56]
    # first, undo the CSD
    dataEstNoCsd = []
    #print(" undo the CSD ")
    for k in range(0, nScData):
        tmpCsd = np.array(h.CYCLIC_SHIFT_NL[nSts - 1][0:nSts])
        tmpCsdCompensate = tmpCsd * 20 * 0.001 * -1
        tmpCsdCompensate = np.exp(-2j * np.pi * tmpCsdCompensate * kD[k] / 64)
        tmpChannel = dataEst[k]
        for i in range(0, nRx):
            tmpChannel[i] = tmpChannel[i] * tmpCsdCompensate
        dataEstNoCsd.append(tmpChannel)
    # convert to phase and mag
    tmpEstAbs = []
    tmpEstPhase = []
    for k in range(0, nScData):
        tmpEstAbs.append(np.abs(dataEstNoCsd[k]))
        tmpEstPhase.append(np.arctan2(np.imag(dataEstNoCsd[k]), np.real(dataEstNoCsd[k])))
    # pilot position interpolation
    tmpEstAbs = tmpEstAbs[0:7] + [(tmpEstAbs[6] + tmpEstAbs[7]) / 2] + tmpEstAbs[7:20] +\
                [(tmpEstAbs[19] + tmpEstAbs[20]) / 2] + tmpEstAbs[20:32] + [(tmpEstAbs[31] + tmpEstAbs[32]) / 2] +\
                tmpEstAbs[32:45] + [(tmpEstAbs[44] + tmpEstAbs[45]) / 2] + tmpEstAbs[45:52]
    tmpEstPhase = tmpEstPhase[0:7] + [(tmpEstPhase[6] + tmpEstPhase[7]) / 2] + tmpEstPhase[7:20] + \
                [(tmpEstPhase[19] + tmpEstPhase[20]) / 2] + tmpEstPhase[20:32] + [(tmpEstPhase[31] + tmpEstPhase[32]) / 2] + \
                tmpEstPhase[32:45] + [(tmpEstPhase[44] + tmpEstPhase[45]) / 2] + tmpEstPhase[45:52]
    # recover from phase and mag
    #print(" recover from phase and mag ")
    dataPilotEstNoCsd = []
    for k in range(0, nScDataPilot):
        dataPilotEstNoCsd.append(tmpEstAbs[k] * np.cos(tmpEstPhase[k]) + tmpEstAbs[k] * np.sin(tmpEstPhase[k]) * 1j)
        #print(dataPilotEstNoCsd[k])
    # re apply csd after interpolation
    #print(" redo the CSD ")
    dataPilotEst = []
    for k in range(0, nScDataPilot):
        tmpCsd = np.array(h.CYCLIC_SHIFT_NL[nSts - 1][0:nSts])
        tmpCsdCompensate = tmpCsd * 20 * 0.001
        tmpCsdCompensate = np.exp(-2j * np.pi * tmpCsdCompensate * kDP[k] / 64)
        tmpChannel = dataPilotEstNoCsd[k]
        for i in range(0, nRx):
            tmpChannel[i] = tmpChannel[i] * tmpCsdCompensate
        dataPilotEst.append(tmpChannel)
        #print(dataPilotEst[k])
    return dataPilotEst

def procVhtChannelFeedback(ltfSym, nSts, nRx):
    # estimate channel
    nScData = 52
    nScDataPilot = 56
    # nSts = 2
    # nRx = 1
    dataChanEst = procVhtDataChanEst(ltfSym, nScData, nSts, nRx)
    dataPilotChanEst = procVhtPilotChanIntpo(dataChanEst, nScData, nSts, nRx)
    # for beamforming feed back, remove CSD
    h = phy80211header()
    kDP = list(range(-28, 0)) + list(range(1, 29))
    dataPilotChanEstNoCsd = []
    #print("for beamforming feed back, remove CSD")
    for k in range(0, nScDataPilot):
        tmpCsd = np.array(h.CYCLIC_SHIFT_NL[nSts - 1][0:nSts])
        tmpCsdCompensate = tmpCsd * 20 * 0.001 * -1
        tmpCsdCompensate = np.exp(-2j * np.pi * tmpCsdCompensate * kDP[k] / 64)
        tmpChannel = dataPilotChanEst[k]
        for i in range(0, nRx):
            tmpChannel[i] = tmpChannel[i] * tmpCsdCompensate
        dataPilotChanEstNoCsd.append(tmpChannel)
        #print(dataPilotChanEstNoCsd[k])
    # SVD, get the V
    #print("get SVD, ")
    vDataPilot = []
    for k in range(0, nScDataPilot):
        """If X is m-by-n with m >= n, then it is equivalent to SVD(X,0).
        For m < n, only the first m columns of V are computed and S is m-by-m."""
        u, s, vh = np.linalg.svd(dataPilotChanEstNoCsd[k], full_matrices=False)
        v = vh.conjugate().T * -1
        vDataPilot.append(v)
    return vDataPilot

def procFftMod(inSig):
    if(len(inSig) == 64):
        tmpSig = inSig[32:64] + inSig[0:32]
        tmpSig = list(np.fft.ifft(tmpSig))
        return tmpSig[48:64] + tmpSig


def readBinFileFromMatDouble(addr):
    return np.fromfile(addr, '<f8')

def procComplexToGrBin(inSig, addr):
    fWaveBin = open(addr, 'wb')
    for each in inSig:
        fWaveBin.write(struct.pack('f', np.real(each)))
        fWaveBin.write(struct.pack('f', np.imag(each)))
    fWaveBin.close()

def procBfQMatrixToBytes(inBfQ, nSts):
    print("convert mat Q")
    tmpBytesR = b''
    tmpBytesI = b''
    for each in inBfQ:
        for i in range(0, nSts):
            for j in range(0, nSts):
                tmpBytesR += struct.pack('<f', np.real(each[i][j]))
                tmpBytesI += struct.pack('<f', np.imag(each[i][j]))
    return [tmpBytesR, tmpBytesI]


if __name__ == "__main__":
    muR1 = readBinFileFromMatDouble("ndp1r.bin")
    muI1 = readBinFileFromMatDouble("ndp1i.bin")
    muR2 = readBinFileFromMatDouble("ndp2r.bin")
    muI2 = readBinFileFromMatDouble("ndp2i.bin")
    muRx1 = []
    muRx2 = []
    for i in range(0, len(muR1)):
        muRx1.append(muR1[i] + muI1[i] * 1j)
        muRx2.append(muR2[i] + muI2[i] * 1j)
    muRx1 = [0] * 1000 + muRx1 + [0] * 1000
    muRx2 = [0] * 1000 + muRx2 + [0] * 1000
    plt.figure(40)
    plt.plot(np.real(muRx1))
    plt.plot(np.imag(muRx1))
    plt.figure(41)
    plt.plot(np.real(muRx2))
    plt.plot(np.imag(muRx2))
    procComplexToGrBin(muRx1, "sig80211VhtGenNdp_0.bin")
    procComplexToGrBin(muRx2, "sig80211VhtGenNdp_1.bin")

    # ndpRawDataR1 = np.loadtxt("ndp1r.txt")
    # ndpRawDataI1 = np.loadtxt("ndp1i.txt")
    # ndpRawDataR2 = np.loadtxt("ndp2r.txt")
    # ndpRawDataI2 = np.loadtxt("ndp2i.txt")
    ndpRawDataR1 = readBinFileFromMatDouble("ndp1r.bin")
    ndpRawDataI1 = readBinFileFromMatDouble("ndp1i.bin")
    ndpRawDataR2 = readBinFileFromMatDouble("ndp2r.bin")
    ndpRawDataI2 = readBinFileFromMatDouble("ndp2i.bin")
    rx1 = []
    rx2 = []
    for i in range(0, len(ndpRawDataR1)):
        rx1.append(ndpRawDataR1[i] + ndpRawDataI1[i] * 1j)
        rx2.append(ndpRawDataR2[i] + ndpRawDataI2[i] * 1j)
    plt.figure(50)
    plt.plot(np.real(rx1))
    plt.plot(np.imag(rx1))
    plt.figure(51)
    plt.plot(np.real(rx2))
    plt.plot(np.imag(rx2))
    ltfIndex = 640

    # get symbols
    nScDataPilot = 56
    nSts = 2
    nRx = 1
    ltfSym = []
    ltfSym.append(procRemovePilots(procFftDemod(rx1[ltfIndex + 16: ltfIndex + 80], nScDataPilot, nSts)))
    ltfSym.append(procRemovePilots(procFftDemod(rx1[ltfIndex + 16 + 80: ltfIndex + 80 + 80], nScDataPilot, nSts)))
    # compute feedback
    vFb1 = procVhtChannelFeedback(ltfSym, nSts, nRx)
    print("feedback v 1")
    print(vFb1)
    ltfSym = []
    ltfSym.append(procRemovePilots(procFftDemod(rx2[ltfIndex + 16: ltfIndex + 80], nScDataPilot, nSts)))
    ltfSym.append(procRemovePilots(procFftDemod(rx2[ltfIndex + 16 + 80: ltfIndex + 80 + 80], nScDataPilot, nSts)))
    # compute feedback
    vFb2 = procVhtChannelFeedback(ltfSym, nSts, nRx)
    print("feedback v 2")
    print(vFb2)
    # combine the channel together
    bfH = []
    for k in range(0, nScDataPilot):
        print(k)
        bfH.append(np.concatenate((vFb1[k],vFb2[k]),axis=1))
        print(bfH[k])
    # compute spatial matrix Q, ZF
    bfQ = []
    for k in range(0, nScDataPilot):
        print("bfQ", k)
        bfQ.append(np.matmul(bfH[k], np.linalg.inv(np.matmul(bfH[k].conjugate().T, bfH[k]))))
        print(bfQ[k])
    # normalize Q
    bfQForFftNormd = []
    for k in range(0, nScDataPilot):
        bfQForFftNormd.append(bfQ[k] / np.linalg.norm(bfQ[k]) * np.sqrt(nSts))
        print(bfQForFftNormd[k])
    # map Q to FFT non-zero sub carriers
    bfQForFftNormdForFft = [np.ones_like(bfQForFftNormd[0])] * 3 + bfQForFftNormd[0:28] + [np.ones_like(bfQForFftNormd[0])] + bfQForFftNormd[28:56] +[np.ones_like(bfQForFftNormd[0])] * 4
    print("--------------------------------")
    for each in bfQForFftNormdForFft:
        print(each)
    bfQByteStrList = procBfQMatrixToBytes(bfQForFftNormdForFft, 2)
    bfQByteNumR = [int(each) for each in bfQByteStrList[0]]
    print(len(bfQByteNumR))
    print(bfQByteNumR)
    bfQByteNumI = [int(each) for each in bfQByteStrList[1]]
    print(len(bfQByteNumI))
    print(bfQByteNumI)
    # vector of X, Y = H*Q*X, put QX in FFT
    # sigB1 = [0, 0, 0, 0,
    #          1, 1, -1, 1,
    #          1, 1, -1, 1,
    #          1, -1, -1, -1,
    #          1, 1, -1, -1,
    #          -1, -1, -1, -1,
    #          -1, 1, -1, 1,
    #          -1, 1, 1, -1, 0,
    #          -1, -1, -1, -1,
    #          -1, 1, 1, -1,
    #          1, 1, 1, -1,
    #          1, 1, -1, 1,
    #          1, -1, -1, 1,
    #          -1, -1, 1, -1,
    #          -1, -1, 1, -1, 0, 0, 0]
    # sigB2 = [0, 0, 0, 0,
    #          -1, -1, -1, 1,
    #          -1, -1, -1, 1,
    #          1, -1, -1, -1,
    #          1, -1, 1, -1,
    #          1, 1, 1, -1,
    #          1, 1, 1, 1,
    #          -1, -1, -1, 1, 0,
    #          -1, -1, 1, 1,
    #          -1, 1, 1, 1,
    #          -1, -1, -1, 1,
    #          -1, -1, 1, -1,
    #          1, 1, 1, -1,
    #          -1, 1, 1, -1,
    #          1, 1, -1, -1, 0, 0, 0]
    # print(len(sigB1), len(sigB2))
    # tmpCsd = -400
    # for k in range(0, 64):
    #     tmpCsdCompensate = tmpCsd * 20 * 0.001
    #     tmpCsdCompensate = np.exp(-2j * np.pi * tmpCsdCompensate * (k-32) / 64)
    #     sigB2[k] = sigB2[k] * tmpCsdCompensate
    # print(sigB2)
    # # gen symbol
    # sigBX = []
    # for i in range(0, 64):
    #     sigBX.append(np.array([[sigB1[i]],[sigB2[i]]]))
    #     print(sigBX[i])
    # # times spatial Q
    # print(len(bfQForFftNormdForFft))
    # sigBXwiQ = []
    # for i in range(0, 64):
    #     sigBXwiQ.append(np.matmul(bfQForFftNormdForFft[i], sigBX[i]))
    #     print(sigBXwiQ[i])

    plt.show()


