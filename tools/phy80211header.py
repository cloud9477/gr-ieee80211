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
from enum import Enum
from matplotlib import pyplot as plt

"""
|______-_________________________-_________________________-_____|______-_________________________-_________________________-_____|______-_________________________-_________________________-_____|______-_________________________-_________________________-_____|   20
  0x6 -26                        0                         26 0x5 
|______-_________________________________________________________|-_________________________________________________________-_____|______-_________________________________________________________|-_________________________________________________________-_____|   40
  0x6 -58                                                         0                                                        58  0x5
|______-_________________________________________________________|________________________________________________________________|-_______________________________________________________________|__________________________________________________________-_____|   80
  0x6 -122                                                                                                                         0                                                                                                                        122  0x5

ver 1.0
support up to 4x4 80M (20M, 40M, 80M)
"""

class GR_F(Enum):
    L = 0
    HT = 1
    VHT = 2
    MU = 3
    QR = 10
    QI = 11
    NDP = 20

class F(Enum):
    L = 0
    HT = 1
    VHT = 2

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class M(Enum):
    BPSK = 0
    QBPSK = 1
    QPSK = 2
    QAM16 = 3
    QAM64 = 4
    QAM256 = 5
    QAM1024 = 6

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class CR(Enum):
    CR12 = 0
    CR23 = 1
    CR34 = 2
    CR56 = 3

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class BW(Enum):
    BW20 = 0
    BW40 = 1
    BW80 = 2

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

def genBitBitCrc8(bitsIn):
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

class modulation:
    def __init__(self, phyFormat=F.L, mcs=0, bw=BW.BW20, nSTS=1, shortGi=False):
        self.initRes = True
        if (not isinstance(phyFormat, F)):
            print("cloud phy80211 header, mod format error")
            self.initRes = False
            return
        self.phyFormat = phyFormat  # packet format
        self.mu = False  # if mu mimo
        self.ampdu = False  # if ampdu

        self.nSym = 0  # symbol number
        self.mpduLen = 0  # mpdu len, input mpdu byte number
        self.ampduLen = 0  # ampdu len, input ampdu byte number
        self.psduLen = 0  # psdu len, in vht psdu len is different from mpdu len due to padding
        self.legacyLen = 0  # legacy len, only for legacy
        self.txTime = 0 # packet transmission time
        self.nPadEof = 0    # vht padding
        self.nPadOctet = 0  # vht padding
        self.nPadBits = 0   # vht padding

        self.mcs = int(mcs)  # mcs
        if (not isinstance(bw, BW)):
            print("cloud phy80211 header, mod bw error")
            self.initRes = False
            return
        self.bw = bw
        self.nSTS = int(nSTS)  # STS number, 1 to 4
        self.sgi = bool(shortGi)  # short GI
        self.mod = 0  # modulation
        self.cr = 0  # coding rate
        self.spr = 20  # sampling rate in MHz

        self.nBPSCS = 1  # bits per single carrier for spatial stream
        self.nSD = 48  # number of data sub carriers
        self.nSP = 4  # number of pilot sub carriers
        self.nSS = 1  # number of spatial streams
        self.nCBPS = 48  # coded bits per symbol
        self.nDBPS = 24  # data bits per symbol
        self.nCBPSS = 48  # coded bits per symbol per spatial-stream symbol
        self.nCBPSSI = 48  # coded bits per symbol per spatial-stream symbol per interleaver
        self.dr = 6  # data rate Mbps
        self.drs = 7.2  # data rate short gi Mbps
        self.nES = 1  # number of BCC encoder, only used when the data rate is >= 600M
        self.nIntlevCol = 0  # used for interleave
        self.nIntlevRow = 0  # used for interleave
        self.nIntlevRot = 0  # used for interleave
        self.nLtf = 1  # non legacy LTF number
        if(self.nSTS > 1):
            self.nLtf = int(np.ceil(self.nSTS/2)) * 2

        if (self.phyFormat == F.L):
            if (self.mcs < 0 or self.mcs > 7 or self.bw != BW.BW20 or self.nSTS != 1 or self.sgi):
                self.initRes = False
                print("cloud phy80211 header, mod init legacy param error")
                return

            self.mu = False
            self.ampdu = False
            self.nSD = 48
            self.nSP = 4
            self.nSS = 1
            self.nES = 1
            self.spr = 20

            if (self.mcs == 0):
                self.mod = M.BPSK
                self.nBPSCS = 1
                self.cr = CR.CR12
            elif (self.mcs == 1):
                self.mod = M.BPSK
                self.nBPSCS = 1
                self.cr = CR.CR34
            elif (self.mcs == 2):
                self.mod = M.QPSK
                self.nBPSCS = 2
                self.cr = CR.CR12
            elif (self.mcs == 3):
                self.mod = M.QPSK
                self.nBPSCS = 2
                self.cr = CR.CR34
            elif (self.mcs == 4):
                self.mod = M.QAM16
                self.nBPSCS = 4
                self.cr = CR.CR12
            elif (self.mcs == 5):
                self.mod = M.QAM16
                self.nBPSCS = 4
                self.cr = CR.CR34
            elif (self.mcs == 6):
                self.mod = M.QAM64
                self.nBPSCS = 6
                self.cr = CR.CR23
            else:  # 7
                self.mod = M.QAM64
                self.nBPSCS = 6
                self.cr = CR.CR34

            self.nCBPSSI = int(self.nSD * self.nBPSCS)
            self.nCBPSS = int(self.nSD * self.nBPSCS)
            self.nCBPS = int(self.nCBPSS * self.nSS)

            if (self.cr == CR.CR12):
                self.nDBPS = int(self.nCBPS / 2)
            elif (self.cr == CR.CR23):
                self.nDBPS = int(self.nCBPS / 3 * 2)
            elif (self.cr == CR.CR34):
                self.nDBPS = int(self.nCBPS / 4 * 3)
            else:
                self.nDBPS = int(self.nCBPS / 6 * 5)
            self.dr = self.nDBPS / 4
            self.drs = 0

        elif (self.phyFormat == F.HT):
            self.nSS = int(np.floor(self.mcs / 8)) + 1
            if (self.mcs < 0 or self.mcs > 31 or self.nSS != self.nSTS or self.nSS < 1 or self.nSS > 4 or (
            not self.bw in [BW.BW20, BW.BW40])):
                self.initRes = False
                print("cloud phy80211 header, mod init ht param error")
                return

            if ((self.mcs % 8) == 0):
                self.mod = M.BPSK
                self.cr = CR.CR12
                self.nBPSCS = 1
            elif ((self.mcs % 8) == 1):
                self.mod = M.QPSK
                self.cr = CR.CR12
                self.nBPSCS = 2
            elif ((self.mcs % 8) == 2):
                self.mod = M.QPSK
                self.cr = CR.CR34
                self.nBPSCS = 2
            elif ((self.mcs % 8) == 3):
                self.mod = M.QAM16
                self.cr = CR.CR12
                self.nBPSCS = 4
            elif ((self.mcs % 8) == 4):
                self.mod = M.QAM16
                self.cr = CR.CR34
                self.nBPSCS = 4
            elif ((self.mcs % 8) == 5):
                self.mod = M.QAM64
                self.cr = CR.CR23
                self.nBPSCS = 6
            elif ((self.mcs % 8) == 6):
                self.mod = M.QAM64
                self.cr = CR.CR34
                self.nBPSCS = 6
            else:
                self.mod = M.QAM64
                self.cr = CR.CR56
                self.nBPSCS = 6

            if (self.bw == BW.BW20):
                self.spr = 20
                self.nSD = 52
                self.nSP = 4
                self.nIntlevCol = 13
                self.nIntlevRow = int(4 * self.nBPSCS)
                self.nIntlevRot = 11
            else:
                self.spr = 40
                self.nSD = 108
                self.nSP = 6
                self.nIntlevCol = 18
                self.nIntlevRow = int(6 * self.nBPSCS)
                self.nIntlevRot = 29

            self.nCBPSS = int(self.nSD * self.nBPSCS)
            self.nCBPS = int(self.nCBPSS * self.nSS)
            if (self.cr == CR.CR12):
                self.nDBPS = int(self.nCBPS / 2)
            elif (self.cr == CR.CR23):
                self.nDBPS = int(self.nCBPS / 3 * 2)
            elif (self.cr == CR.CR34):
                self.nDBPS = int(self.nCBPS / 4 * 3)
            else:  # 56
                self.nDBPS = int(self.nCBPS / 6 * 5)

            self.dr = self.nDBPS / 4.0
            self.drs = self.nDBPS / 3.6
            if (self.drs < 300.1):
                self.nES = 1
            else:
                self.nES = 2
        elif (self.phyFormat == F.VHT):
            self.nSS = self.nSTS
            if (self.mcs < 0 or self.mcs > 9 or self.nSS < 1 or self.nSS > 4 or (
            not self.bw in [BW.BW20, BW.BW40, BW.BW80])):
                self.initRes = False
                print("cloud phy80211 header, mod init vht param error")
                return

            if (self.mcs == 0):
                self.mod = M.BPSK
                self.cr = CR.CR12
                self.nBPSCS = 1
            elif (self.mcs == 1):
                self.mod = M.QPSK
                self.cr = CR.CR12
                self.nBPSCS = 2
            elif (self.mcs == 2):
                self.mod = M.QPSK
                self.cr = CR.CR34
                self.nBPSCS = 2
            elif (self.mcs == 3):
                self.mod = M.QAM16
                self.cr = CR.CR12
                self.nBPSCS = 4
            elif (self.mcs == 4):
                self.mod = M.QAM16
                self.cr = CR.CR34
                self.nBPSCS = 4
            elif (self.mcs == 5):
                self.mod = M.QAM64
                self.cr = CR.CR23
                self.nBPSCS = 6
            elif (self.mcs == 6):
                self.mod = M.QAM64
                self.cr = CR.CR34
                self.nBPSCS = 6
            elif (self.mcs == 7):
                self.mod = M.QAM64
                self.cr = CR.CR56
                self.nBPSCS = 6
            elif (self.mcs == 8):
                self.mod = M.QAM256
                self.cr = CR.CR34
                self.nBPSCS = 8
            else:
                self.mod = M.QAM256
                self.cr = CR.CR56
                self.nBPSCS = 8

            if (self.bw == BW.BW20):
                if (self.mcs == 9 and self.nSS in [1, 2, 4]):
                    self.initRes = False
                    print("cloud phy80211 header, mod init vht 20M mcs error")
                    return
                self.spr = 20
                self.nSD = 52
                self.nSP = 4
                self.nIntlevCol = 13
                self.nIntlevRow = int(4 * self.nBPSCS)
                self.nIntlevRot = 11
            elif (self.bw == BW.BW40):
                self.spr = 40
                self.nSD = 108
                self.nSP = 6
                self.nIntlevCol = 18
                self.nIntlevRow = int(6 * self.nBPSCS)
                self.nIntlevRot = 29
            else:  # 80M
                if (self.mcs == 6 and self.nSS == 3):
                    print("cloud phy80211 header, mod init vht 80M mcs error")
                    return
                self.spr = 80
                self.nSD = 234
                self.nSP = 8
                self.nIntlevCol = 26
                self.nIntlevRow = int(9 * self.nBPSCS)
                self.nIntlevRot = 58

            self.nCBPSS = int(self.nSD * self.nBPSCS)
            self.nCBPSSI = self.nCBPSS
            self.nCBPS = int(self.nCBPSS * self.nSS)

            if (self.cr == CR.CR12):
                self.nDBPS = int(self.nCBPS / 2)
            elif (self.cr == CR.CR23):
                self.nDBPS = int(self.nCBPS / 3 * 2)
            elif (self.cr == CR.CR34):
                self.nDBPS = int(self.nCBPS / 4 * 3)
            else:  # CR56
                self.nDBPS = int(self.nCBPS / 6 * 5)

            self.dr = self.nDBPS / 4.0
            self.drs = self.nDBPS / 3.6
            if (self.drs < 600.1):
                self.nES = 1
            elif (self.drs < 1200.1):
                self.nES = 2
            else:
                self.nES = 3
        else:
            self.initRes = False
            print("cloud phy80211 header, mod init format error")
        # print("SPR %d, SS %d, LTF %d, mcs %d, mod %s, CR %s, BPSCS %d, SD %d, SP %d, CBPS %d, DBPS %d, ES %d, DR %d, DRS %d" % (self.spr, self.nSS, self.nLtf, self.mcs, self.mod, self.cr, self.nBPSCS, self.nSD, self.nSP, self.nCBPS, self.nDBPS, self.nES, self.dr, self.drs))

    def procPktLenNonAggre(self, mpduLen):
        nService = 16
        nTail = 6
        mSTBC = 1
        if (self.phyFormat == F.L):
            self.mpduLen = mpduLen  # mpdu len
            self.ampduLen = 0  # ampdu len
            self.psduLen = mpduLen  # psdu len
            self.nSym = int(np.ceil((self.psduLen * 8 + nService + nTail) / self.nDBPS))
            self.nPadEof = 0
            self.nPadOctet = 0
            self.nPadBits = int(self.nSym * self.nDBPS - 8 * self.psduLen - nService - nTail)
            # tx time in us, sum of legacy preamble, legacy sig, symbol
            self.txTime = 20 + self.nSym * 4
            self.legacyLen = mpduLen
        elif (self.phyFormat == F.HT):
            self.mpduLen = mpduLen  # mpdu len
            self.ampduLen = 0  # ampdu len
            self.psduLen = mpduLen  # psdu len
            self.nSym = mSTBC * int(np.ceil((self.psduLen * 8 + nService + nTail * self.nES) / (self.nDBPS * mSTBC)))
            self.nPadEof = 0
            self.nPadOctet = 0
            self.nPadBits = int(self.nSym * self.nDBPS - 8 * self.psduLen - nService - nTail * self.nES)
            # tx time in us, sum of 
            # legacy preamble + legacy sig = 20, ht sig = 8, ht preamble = 4 + 4n, symbol
            if(self.sgi):
                self.txTime = int( 20 + 8 + 4 + self.nLtf * 4 + int(np.ceil((self.nSym * 3.6)/4)) * 4 )
            else:
                self.txTime = int( 20 + 8 + 4 + self.nLtf * 4 + self.nSym * 4 )
            self.legacyLen = int((self.txTime - 20)/4) * 3 - 3
        else:  # VHT
            print("cloud phy80211 header, mod gen pkt len non aggre only l and ht")
        # print("mpdu len: %d, psdu len: %d, nSym: %d, txTime: %d" % (self.mpduLen, self.psduLen, self.nSym, self.txTime))

    def procPktLenAggre(self, ampduLen):
        nService = 16
        nTail = 6
        mSTBC = 1
        self.mpduLen = 0  # mpdu len
        if (self.phyFormat == F.HT):
            self.ampduLen = ampduLen  # ampdu len
            self.psduLen = ampduLen  # psdu len
            self.nSym = mSTBC * int(np.ceil((self.psduLen * 8 + nService + nTail * self.nES) / (self.nDBPS * mSTBC)))
            self.nPadEof = 0
            self.nPadOctet = 0
            self.nPadBits = int(self.nSym * self.nDBPS - 8 * self.psduLen - nService - nTail * self.nES)
            # tx time in us, sum of 
            # legacy preamble + legacy sig = 20, ht sig = 8, ht preamble = 4 + 4n, symbol
            if(self.sgi):
                self.txTime = int( 20 + 8 + 4 + self.nLtf * 4 + int(np.ceil((self.nSym * 3.6)/4)) * 4 )
            else:
                self.txTime = int( 20 + 8 + 4 + self.nLtf * 4 + self.nSym * 4 )
            self.legacyLen = int((self.txTime - 20)/4) * 3 - 3
        elif (self.phyFormat == F.VHT):
            if(ampduLen > 0):
                self.ampduLen = ampduLen  # ampdu len
                self.nSym = mSTBC * int(np.ceil((8 * self.ampduLen + nService + nTail * self.nES) / (mSTBC * self.nDBPS)))
                self.psduLen = int(np.floor((self.nSym * self.nDBPS - nService - nTail * self.nES) / 8))
                self.nPadEof = int(np.floor((self.psduLen - self.ampduLen) / 4))
                self.nPadOctet = int(self.psduLen - self.ampduLen - self.nPadEof * 4)
                self.nPadBits = int(self.nSym * self.nDBPS - 8*self.psduLen - nService - nTail * self.nES)
                # tx time in us, sum of 
                # legacy preamble, legacy sig, vht sig a, vht preamble, vht sig b, symbol
                if(self.sgi):
                    self.txTime = int( 20 + 8 + 4 + self.nLtf * 4 + 4 + int(np.ceil((self.nSym * 3.6)/4)) * 4 )
                else:
                    self.txTime = int( 20 + 8 + 4 + self.nLtf * 4 + 4 + self.nSym * 4 )
                self.legacyLen = int((self.txTime - 20)/4) * 3 - 3
            else:
                # NDP
                self.ampduLen = 0
                self.nSym = 0
                self.psduLen = 0
                self.nPadEof = 0
                self.nPadOctet = 0
                self.nPadBits = 0
                self.txTime = int( 20 + 8 + 4 + self.nLtf * 4 + 4)
                self.legacyLen = int((self.txTime - 20)/4) * 3 - 3
        else:
            print("cloud phy80211 header, mod gen pkt len aggre only ht and vht")
        # print("mpdu len: %d, ampdu len:%d, psdu len: %d, nSym: %d, txTime: %d" % (self.mpduLen, self.ampduLen, self.psduLen, self.nSym, self.txTime))
    
    def procPktLenAggreMu(self, ampduLen, nSymMu):
        nService = 16
        nTail = 6
        self.mpduLen = 0  # mpdu len
        self.ampduLen = ampduLen  # ampdu len
        self.nSym = nSymMu
        self.psduLen = int(np.floor((self.nSym * self.nDBPS - nService - nTail * self.nES) / 8))
        self.nPadEof = int(np.floor((self.psduLen - self.ampduLen) / 4))
        self.nPadOctet = int(self.psduLen - self.ampduLen - self.nPadEof * 4)
        self.nPadBits = int(self.nSym * self.nDBPS - 8*self.psduLen - nService - nTail * self.nES)
        # tx time in us, sum of 
        # legacy preamble, legacy sig, vht sig a, vht preamble, vht sig b, symbol
        if(self.sgi):
            self.txTime = int( 20 + 8 + 4 + self.nLtf * 4 + 4 + int(np.ceil((self.nSym * 3.6)/4)) * 4 )
        else:
            self.txTime = int( 20 + 8 + 4 + self.nLtf * 4 + 4 + self.nSym * 4 )
        self.legacyLen = int((self.txTime - 20)/4) * 3 - 3

C_QAM_MODU_TAB = [
    [(-1 + 0j), (1 + 0j)],      # 0 bpsk
    [(0 - 1j), (0 + 1j)],       # 1 q-bpsk
    [each * np.sqrt(1 / 2) for each in [(-1 - 1j), (1 - 1j), (-1 + 1j), (1 + 1j)]],     # 2 qpsk
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
    [each * np.sqrt(1/170) for each in [
        (-15 - 15j), (15 - 15j), (-1  - 15j), (1  - 15j), (-9  - 15j), (9  - 15j), (-7  - 15j), (7  - 15j),
        (-13 - 15j), (13 - 15j), (-3  - 15j), (3  - 15j), (-11 - 15j), (11 - 15j), (-5  - 15j), (5  - 15j),
        (-15 + 15j), (15 + 15j), (-1  + 15j), (1  + 15j), (-9  + 15j), (9  + 15j), (-7  + 15j), (7  + 15j),
        (-13 + 15j), (13 + 15j), (-3  + 15j), (3  + 15j), (-11 + 15j), (11 + 15j), (-5  + 15j), (5  + 15j),
        (-15 -  1j), (15 -  1j), (-1  -  1j), (1  -  1j), (-9  -  1j), (9  -  1j), (-7  -  1j), (7  -  1j),
        (-13 -  1j), (13 -  1j), (-3  -  1j), (3  -  1j), (-11 -  1j), (11 -  1j), (-5  -  1j), (5  -  1j),
        (-15 +  1j), (15 +  1j), (-1  +  1j), (1  +  1j), (-9  +  1j), (9  +  1j), (-7  +  1j), (7  +  1j),
        (-13 +  1j), (13 +  1j), (-3  +  1j), (3  +  1j), (-11 +  1j), (11 +  1j), (-5  +  1j), (5  +  1j),
        (-15 -  9j), (15 -  9j), (-1  -  9j), (1  -  9j), (-9  -  9j), (9  -  9j), (-7  -  9j), (7  -  9j),
        (-13 -  9j), (13 -  9j), (-3  -  9j), (3  -  9j), (-11 -  9j), (11 -  9j), (-5  -  9j), (5  -  9j),
        (-15 +  9j), (15 +  9j), (-1  +  9j), (1  +  9j), (-9  +  9j), (9  +  9j), (-7  +  9j), (7  +  9j),
        (-13 +  9j), (13 +  9j), (-3  +  9j), (3  +  9j), (-11 +  9j), (11 +  9j), (-5  +  9j), (5  +  9j),
        (-15 -  7j), (15 -  7j), (-1  -  7j), (1  -  7j), (-9  -  7j), (9  -  7j), (-7  -  7j), (7  -  7j),
        (-13 -  7j), (13 -  7j), (-3  -  7j), (3  -  7j), (-11 -  7j), (11 -  7j), (-5  -  7j), (5  -  7j),
        (-15 +  7j), (15 +  7j), (-1  +  7j), (1  +  7j), (-9  +  7j), (9  +  7j), (-7  +  7j), (7  +  7j),
        (-13 +  7j), (13 +  7j), (-3  +  7j), (3  +  7j), (-11 +  7j), (11 +  7j), (-5  +  7j), (5  +  7j),
        (-15 - 13j), (15 - 13j), (-1  - 13j), (1  - 13j), (-9  - 13j), (9  - 13j), (-7  - 13j), (7  - 13j),
        (-13 - 13j), (13 - 13j), (-3  - 13j), (3  - 13j), (-11 - 13j), (11 - 13j), (-5  - 13j), (5  - 13j),
        (-15 + 13j), (15 + 13j), (-1  + 13j), (1  + 13j), (-9  + 13j), (9  + 13j), (-7  + 13j), (7  + 13j),
        (-13 + 13j), (13 + 13j), (-3  + 13j), (3  + 13j), (-11 + 13j), (11 + 13j), (-5  + 13j), (5  + 13j),
        (-15 -  3j), (15 -  3j), (-1  -  3j), (1  -  3j), (-9  -  3j), (9  -  3j), (-7  -  3j), (7  -  3j),
        (-13 -  3j), (13 -  3j), (-3  -  3j), (3  -  3j), (-11 -  3j), (11 -  3j), (-5  -  3j), (5  -  3j),
        (-15 +  3j), (15 +  3j), (-1  +  3j), (1  +  3j), (-9  +  3j), (9  +  3j), (-7  +  3j), (7  +  3j),
        (-13 +  3j), (13 +  3j), (-3  +  3j), (3  +  3j), (-11 +  3j), (11 +  3j), (-5  +  3j), (5  +  3j),
        (-15 - 11j), (15 - 11j), (-1  - 11j), (1  - 11j), (-9  - 11j), (9  - 11j), (-7  - 11j), (7  - 11j),
        (-13 - 11j), (13 - 11j), (-3  - 11j), (3  - 11j), (-11 - 11j), (11 - 11j), (-5  - 11j), (5  - 11j),
        (-15 + 11j), (15 + 11j), (-1  + 11j), (1  + 11j), (-9  + 11j), (9  + 11j), (-7  + 11j), (7  + 11j),
        (-13 + 11j), (13 + 11j), (-3  + 11j), (3  + 11j), (-11 + 11j), (11 + 11j), (-5  + 11j), (5  + 11j),
        (-15 -  5j), (15 -  5j), (-1  -  5j), (1  -  5j), (-9  -  5j), (9  -  5j), (-7  -  5j), (7  -  5j),
        (-13 -  5j), (13 -  5j), (-3  -  5j), (3  -  5j), (-11 -  5j), (11 -  5j), (-5  -  5j), (5  -  5j),
        (-15 +  5j), (15 +  5j), (-1  +  5j), (1  +  5j), (-9  +  5j), (9  +  5j), (-7  +  5j), (7  +  5j),
        (-13 +  5j), (13 +  5j), (-3  +  5j), (3  +  5j), (-11 +  5j), (11 +  5j), (-5  +  5j), (5  +  5j)
    ]]]


# legacy rate
C_LEGACY_RATE_BIT = [
    [1, 1, 0, 1],
    [1 ,1, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 1, 1],
    [0, 0, 0, 1],
    [0, 0, 1, 1]
]
# legacy stf
C_STF_L_26 = [each * np.sqrt(1 / 2) for each in
                    [0, 0, 1 + 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, -1 - 1j, 0, 0, 0,
                    1 + 1j, 0, 0, 0, 0, 0, 0, 0, -1 - 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0,
                    0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0, 0]]
C_STF_L_58 = C_STF_L_26 + [0] * 11 + C_STF_L_26
C_STF_L_122 = C_STF_L_58 + [0] * 11 + C_STF_L_58
C_STF_L = [C_STF_L_26, C_STF_L_58, C_STF_L_122]
# ht stf
C_STF_HT_28 = [0, 0] + C_STF_L_26 + [0, 0]
C_STF_HT_58 = C_STF_L_26 + [0] * 11 + C_STF_L_26
C_STF_HT = [C_STF_HT_28, C_STF_HT_58]
# vht stf
C_STF_VHT_28 = C_STF_HT_28
C_STF_VHT_58 = C_STF_HT_58
C_STF_VHT_122 = C_STF_VHT_58 + [0] * 11 + C_STF_VHT_58
C_STF_VHT = [C_STF_VHT_28, C_STF_VHT_58, C_STF_VHT_122]
# legacy ltf
C___LTF_L = [1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1]
C___LTF_R = [1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1]
C_LTF_L_26 = C___LTF_L + [0] + C___LTF_R
C_LTF_L_58 = C_LTF_L_26 + [0] * 11 + C_LTF_L_26
C_LTF_L_122 = C_LTF_L_58 + [0] * 11 + C_LTF_L_58
C_LTF_L = [C_LTF_L_26, C_LTF_L_58, C_LTF_L_122]
# ht ltf
C_LTF_HT_28 = [1, 1] + C___LTF_L + [0] + C___LTF_R + [-1, -1]
C_LTF_HT_58 = C___LTF_L + [1] + C___LTF_R + [-1, -1, -1, 1, 0, 0, 0, -1, 1, 1, -1] + \
                    C___LTF_L + [1] + C___LTF_R
C_LTF_HT = [C_LTF_HT_28, C_LTF_HT_58]
# vht ltf
C_LTF_VHT_28 = C_LTF_HT_28
C_LTF_VHT_58 = C_LTF_HT_58
C_LTF_VHT_122 = C___LTF_L + [1] + C___LTF_R + [-1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1] + \
                    C___LTF_L + [1] + C___LTF_R + [1, -1, 1, -1, 0, 0, 0, 1, -1, -1, 1] + \
                    C___LTF_L + [1] + C___LTF_R + [-1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1] +\
                    C___LTF_L + [1] + C___LTF_R
C_LTF_VHT = [C_LTF_VHT_28, C_LTF_VHT_58, C_LTF_VHT_122]
# non legacy ltf number
C_LTF_HT_N = [0, 1, 2, 4, 4]
C_LTF_VHT_N = [0, 1, 2, 4, 4]
# LTF polarity of ss
C_P_LTF_HT_4 = [
    [1, -1, 1, 1],
    [1, 1, -1, 1],
    [1, 1, 1, -1],
    [-1, 1, 1, 1]]
# for data sub carriers, times P_VHT-LTF, for pilot sub carriers, times R_VHT-LTF, which is first row of P_VHT-LTF
C_P_LTF_VHT_4 = [
    [1, -1, 1, 1],
    [1, 1, -1, 1],
    [1, 1, 1, -1],
    [-1, 1, 1, 1]]
C_R_LTF_VHT_4 = [1, -1, 1, 1]
# polarity of sig b constellations
C_P_SIG_B_NSTS478 = [1, 1, 1, -1, 1, 1, 1, -1]
# scale factor N_Tone_Field, use with BW index
C_SCALENTF_STF_L = [12, 24, 48]
C_SCALENTF_LTF_L = [52, 104, 208]
C_SCALENTF_SIG_L = [52, 104, 208]
C_SCALENTF_SIG_HT = [52, 104]
C_SCALENTF_STF_HT = [12, 24]
C_SCALENTF_LTF_HT = [56, 114]
C_SCALENTF_DATA_HT = [56, 114]
C_SCALENTF_SIG_VHT_A = [52, 104, 208]
C_SCALENTF_STF_VHT = [12, 24, 48]
C_SCALENTF_LTF_VHT = [56, 114, 242]
C_SCALENTF_SIG_VHT_B = [56, 114, 242]
C_SCALENTF_DATA_VHT = [56, 114, 242]
# pilot
C_PILOT_L = [ 1,  1,  1, -1]
C_PILOT_HT = [
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
C_PILOT_VHT = [
    # 20M
    [ 1,  1,  1, -1],
    # 40M
    [ 1,  1,  1, -1, -1,  1],
    # 80M
    [ 1,  1,  1, -1, -1,  1,  1,  1]
]
# pilot polarity sequence
C_PILOT_PS = [
    1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1,
    1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1,
    1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1,
    -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1,
    -1, -1, -1]
# cylic shift
# Legacy part, include lstf, lltf, lsig, htsig and vhtsiga
C_CYCLIC_SHIFT_L = [
    [0,    0,    0,    0],
    [0, -200,    0,    0],
    [0, -100, -200,    0],
    [0,  -50, -100, -150]
]
# non-legacy part, htdata, vhtsigb, vhtdata
C_CYCLIC_SHIFT_NL = [
    [0,    0,    0,    0],
    [0, -400,    0,    0],
    [0, -400, -200,    0],
    [0, -400, -200, -600]
]
# tone rotation
C_TONE_ROTATION_20 = [1] * 57  #
C_TONE_ROTATION_40 = [1] * 58 + [1j] * 59  # 117 = 58 + 59
C_TONE_ROTATION_80 = [1] * 58 + [-1] * 187  # 245 is -122 to 122
# NDP sig b bits
C_NDP_SIG_B_20 = [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
C_NDP_SIG_B_40 = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1]
C_NDP_SIG_B_80 = [0 ,1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
C_NDP_SIG_B = [C_NDP_SIG_B_20, C_NDP_SIG_B_40, C_NDP_SIG_B_80]

# eof bit is 1, reserved bit 0, length is 0, crc 8, signature 0x4e
C_VHT_EOF = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + genBitBitCrc8([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) + [0, 1, 1, 1, 0, 0, 1, 0]

def procBcc(inBits, cr):
    if(isinstance(inBits, list) or isinstance(cr, CR)):
        # binary convolutional coding, ieee 802.11 2016 ofdm sec 17.3.5.6
        tmpCodedBits = [0] * (len(inBits)*2)
        tmpState = 0
        for j in range(0, len(inBits)):
            tmpState = ((tmpState << 1) & 0x7e) | inBits[j]
            tmpCodedBits[j * 2] = (bin(tmpState & 0o155).count("1")) % 2
            tmpCodedBits[j * 2 + 1] = (bin(tmpState & 0o117).count("1")) % 2
        tmpIndex = 0
        tmpPunctBits = []
        if(cr == CR.CR12):
            tmpPunctBits = tmpCodedBits
        elif(cr == CR.CR23):
            for each in tmpCodedBits:
                if((tmpIndex%4) != 3):
                    tmpPunctBits.append(each)
                tmpIndex += 1
        elif(cr == CR.CR34):
            for each in tmpCodedBits:
                if ((tmpIndex % 6) in [3, 4]):
                    pass
                else:
                    tmpPunctBits.append(each)
                tmpIndex += 1
        elif (cr == CR.CR56):
            for each in tmpCodedBits:
                if ((tmpIndex % 10) in [3, 4, 7, 8]):
                    pass
                else:
                    tmpPunctBits.append(each)
                tmpIndex += 1
        return tmpPunctBits
    else:
        print("cloud phy80211 header, bcc input error")
        return []

def procScramble(inBits, scrambler):
    tmpScrambleBits = []
    for i in range(0, len(inBits)):
        tmpFeedback = int(not not(scrambler & 64)) ^ int(not not(scrambler & 8))
        tmpScrambleBits.append(tmpFeedback ^ inBits[i])
        scrambler = ((scrambler << 1) & 0x7e) | tmpFeedback
    return tmpScrambleBits

def procInterleaveSigL(inBits):
    if(isinstance(inBits, list)):
        tmpIntedBits = [0] * len(inBits)
        s = 1
        for k in range(0, 48):
            i = int((48/16) * (k % 16) + np.floor(k/16))
            j = int(s * int(np.floor(i/s)) + (int(i + 48 - np.floor(16 * i / 48)) % s))
            tmpIntedBits[j] = inBits[k]
        return tmpIntedBits
    else:
        print("cloud phy80211 header, interleave sig legacy input error")
        return []

def procStreamParserNonLegacy(inEsBits, mod):
    if(isinstance(inEsBits, list) and isinstance(mod, modulation)):
        if(len(inEsBits) == mod.nES):
            tmpSsStreamBits = []
            for i in range(0, mod.nSS):
                    tmpSsStreamBits.append([0] * mod.nSym * mod.nCBPSS)
            s = int(max(1, mod.nBPSCS/2))
            cs = mod.nSS * s     # cs is the capital S used in standard
            for isym in range(0, mod.nSym):
                for iss in range(0, mod.nSS):
                    for k in range(0, int(mod.nCBPSS)):
                        j = int(np.floor(k/s)) % mod.nES
                        i = (iss) * s + cs * int(np.floor(k/(mod.nES * s))) + int(k % s)
                        tmpSsStreamBits[iss][k + int(isym * mod.nCBPSS)] = inEsBits[j][i + int(isym * mod.nCBPS)]
            return tmpSsStreamBits
    return []

def procInterleaveLegacy(inSsBits, mod):
    if(isinstance(inSsBits, list) and isinstance(mod, modulation) and len(inSsBits) == 1 and mod.nSS == 1):
        if(len(inSsBits[0]) != mod.nSym*mod.nCBPSS):
            print("cloud phy80211 header, interleave legacy input bits len error")
            return []
        # follow the IEEE 802.11-2016 OFDM section 17.3.5.7
        s = int(max(1, mod.nBPSCS/2))
        tmpSsIntedBits = [[0] * len(inSsBits[0])]
        for symPtr in range(0, mod.nSym):
            for k in range(0, mod.nCBPS):
                i = int((mod.nCBPS/16) * (k % 16) + np.floor(k/16))
                j = int(s * int(np.floor(i/s)) + (int(i + mod.nCBPS - np.floor(16 * i / mod.nCBPS)) % s))
                tmpSsIntedBits[0][j + symPtr * mod.nCBPSS] = inSsBits[0][k + symPtr * mod.nCBPSS]
        return tmpSsIntedBits
    print("cloud phy80211 header, interleave legacy input error")
    return []

def procInterleaveNonLegacy(inSsBits, mod):
    if(isinstance(inSsBits, list) and isinstance(mod, modulation) and len(inSsBits) == mod.nSS):
        for eachSsBits in inSsBits:
            if(len(eachSsBits) != mod.nSym*mod.nCBPSS):
                print("cloud phy80211 header, interleave non legacy input bits len error")
                return []
        # follow the IEEE 802.11-2016 HT and VHT, section 19.3.11.8 and 21.3.10.8
        tmpSsIntedBits = []
        for i in range(0, mod.nSS):
            tmpSsIntedBits.append([])
        s = int(max(1, mod.nBPSCS/2))
        for ssItr in range(0, mod.nSS):
            tmpSsIntedBits[ssItr] = [0] * len(inSsBits[ssItr])
            for symPtr in range(0, mod.nSym):
                for k in range(0, mod.nCBPSS):
                    i = mod.nIntlevRow * (k % mod.nIntlevCol) + int(np.floor(k/mod.nIntlevCol))
                    j = s * int(np.floor(i/s)) + (i + mod.nCBPSS - int(np.floor(mod.nIntlevCol * i / mod.nCBPSS))) % s
                    r = j
                    if(mod.nSS >=2):
                        r = int((j-(((ssItr)*2)%3 + 3*int(np.floor((ssItr)/3)))*mod.nIntlevRot*mod.nBPSCS)) % mod.nCBPSS
                    tmpSsIntedBits[ssItr][r + symPtr * mod.nCBPSS] = inSsBits[ssItr][k + symPtr * mod.nCBPSS]
        return tmpSsIntedBits
    else:
        print("cloud phy80211 header, interleave non legacy input error")
        return []

def procConcat2Symbol(sa, sb):
    if(isinstance(sa, list) and isinstance(sb, list)):
        if(len(sa) > 0 and len(sb) > 0):
            sb[0] = sb[0] * 0.5
            sa[len(sa) - 1] = sa[len(sa) - 1] * 0.5
            return sa + sb
        print("cloud phy80211 header, procConcat2Symbol: input len error, sa %d, sb %d" % (len(sa), len(sb)))
    print("cloud phy80211 header, procConcat2Symbol: input type error")

def procTonePhase(inQam):
    # input QAM with DC
    if (len(inQam) == 57):
        return inQam
    elif (len(inQam) == 117):
        return [inQam[i] * C_TONE_ROTATION_40[i] for i in range(0, 117)]
    elif (len(inQam) == 245):
        return [inQam[i] * C_TONE_ROTATION_80[i] for i in range(0, 245)]
    else:
        print("cloud phy80211 header, procTonePhase: input length error %d" % (len(inQam)))

def procPilotInsert(inQam, p):
    if(len(inQam) == 48 and len(p) == 4):   # 20
        return (inQam[0:5] + [p[0]] + inQam[5:18] + [p[1]] + inQam[18:30] + [p[2]] + inQam[30:43] + [p[3]] + inQam[43:48])
    elif(len(inQam) == 52 and len(p) == 4): # 20
        return (inQam[0:7] + [p[0]] + inQam[7:20] + [p[1]] + inQam[20:32] + [p[2]] + inQam[32:45] + [p[3]] + inQam[45:52])
    elif(len(inQam) == 110 and len(p) == 6):    # 40
        return (inQam[0:5] + [p[0]] + inQam[5:20] + [p[1]] + inQam[20:45] + [p[2]] + inQam[45:65] + [p[3]] + inQam[65:78] + [p[4]] + inQam[78:105] + [p[5]] + inQam[105:110])
    elif(len(inQam) == 236 and len(p) == 8):    # 80
        return (inQam[0:19] + [p[0]] + inQam[19:46] + [p[1]] + inQam[46:81] + [p[2]] + inQam[81:108] + [p[3]] + inQam[108:128] + [p[4]] + inQam[128:155] + [p[5]] + inQam[155:190] + [p[6]] + inQam[190:217] + [p[7]] + inQam[217:236])
    else:
        print("cloud phy80211 header, procPilotInsert: input length error qam %d pilots %d" % (len(inQam), len(p)))
        return []

def procDcInsert(inQam):
    if(len(inQam) == 52):       # 20
        return (inQam[:26] + [0] + inQam[26:])
    elif(len(inQam) == 56):     # 20
        return (inQam[:28] + [0] + inQam[28:])
    elif(len(inQam) == 114):    # 40
        return (inQam[:57] + [0]*3 + inQam[57:])
    elif(len(inQam) == 242):    # 80
        return (inQam[:121] + [0]*3 + inQam[121:])
    else:
        print("cloud phy80211 header, procDcInsert: input length error %d" % (len(inQam)))
        return []

def procNonDataSC(inQam):
    # input QAM has DC 0
    if (len(inQam) in [53, 117, 245]):
        return ([0] * 6 + inQam + [0] * 5)
    elif (len(inQam) == 57):
        return ([0] * 4 + inQam + [0] * 3)
    else:
        print("cloud phy80211 header, procNonDataSC: input length error %d" % (len(inQam)))
        return []

def procLegacyCSD(inQam, nSs, iSs, spr):
    tmpPhase = -1.0j * 2 * np.pi * C_CYCLIC_SHIFT_L[nSs-1][iSs] * spr * 0.001
    return [(inQam[i] * np.exp(tmpPhase * ((i - int(len(inQam)/2)) / len(inQam)))) for i in range(0, len(inQam))]

def procCSD(inQam, nSs, iSs, spr):
    tmpPhase = -1.0j * 2 * np.pi * C_CYCLIC_SHIFT_NL[nSs-1][iSs] * spr * 0.001
    return [(inQam[i] * np.exp(tmpPhase * ((i - int(len(inQam)/2)) / len(inQam)))) for i in range(0, len(inQam))]

def procFftMod(inQam):
    # ifft shift
    if(len(inQam) in [64, 128, 256]):
        return list(np.fft.ifft((inQam[int(len(inQam)/2):] + inQam[:int(len(inQam)/2)])))
    else:
        print("cloud phy80211 header, procFftMod: input length error %d" % (len(inQam)))
        return []

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
        print("cloud phy80211 header, procGi: input length error %d" % (len(inSig)))
        return []

def procFftDemod(inSig):
    # check len
    if(isinstance(inSig, list) and len(inSig) in [64, 128, 256]):
        # fft shift
        tmpF = list(np.fft.fft(inSig))
        if(len(inSig) == 64):       # 20
            tmpF = tmpF[32:64] + tmpF[0:32]
        elif(len(inSig) == 128):    # 40
            tmpF = tmpF[64:128] + tmpF[0:64]
        else:                       # 80
            tmpF = tmpF[128:256] + tmpF[0:128]
        return tmpF
    else:
        print("cloud phy80211 header, procFftDemod: input length error %d" % (len(inSig)))
        return []

def procRmDcNonDataSc(inSig, phyFormat):
    if(isinstance(inSig, list) and isinstance(phyFormat, F)):
        if(len(inSig) in [64, 128, 256]):
            # fft shift
            if(len(inSig) == 64):       # 20
                if(phyFormat == F.L):
                    return inSig[6:32] + inSig[33:59]
                else:
                    return inSig[4:32] + inSig[33:61]
            elif(len(inSig) == 128):    # 40
                return inSig[6:64] + inSig[64:123]
            else:                       # 80
                return inSig[6:127] + inSig[129:251]
        else:
            print("cloud phy80211 header, procRmDcNonDataSc: input len error %d" % (len(inSig)))
            return []
    else:
        print("cloud phy80211 header, procRmDcNonDataSc: input type error")
        return []

def procToneDescaling(inSig, inNtf, nSs):
    return [each * np.sqrt(inNtf * nSs) / len(inSig) for each in inSig]

def procRemovePilots(inSig):
    if(len(inSig) == 52):   # 20
        return inSig[0:5] + inSig[6:19] + inSig[20:32] + inSig[33:46] + inSig[47:52]
    elif(len(inSig) == 56): # 20
        return inSig[0:7] + inSig[8:21] + inSig[22:34] + inSig[35:48] + inSig[49:56]
    elif(len(inSig) == 116):    # 40
        return inSig[0:5] + inSig[6:21] + inSig[22:47] + inSig[48:68] + inSig[69:82] + inSig[83:110] + inSig[111:116]
    elif(len(inSig) == 244):    # 80
        return inSig[0:19] + inSig[20:47] + inSig[48:83] + inSig[84:111] + inSig[112:132] + inSig[133:160] + inSig[161:196] + inSig[197:224] + inSig[225:244]
    else:
        print("cloud phy80211 header, procRemovePilots: input length error %d " % (len(inSig)))
        return []

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
    kDP = list(range(-28, 0)) + list(range(1, 29))
    kD = kDP[0:7] + kDP[8:21] + kDP[22:34] + kDP[35:48] + kDP[49:56]
    # first, undo the CSD
    dataEstNoCsd = []
    #print(" undo the CSD ")
    for k in range(0, nScData):
        tmpCsd = np.array(C_CYCLIC_SHIFT_NL[nSts - 1][0:nSts])
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
        tmpCsd = np.array(C_CYCLIC_SHIFT_NL[nSts - 1][0:nSts])
        tmpCsdCompensate = tmpCsd * 20 * 0.001
        tmpCsdCompensate = np.exp(-2j * np.pi * tmpCsdCompensate * kDP[k] / 64)
        tmpChannel = dataPilotEstNoCsd[k]
        for i in range(0, nRx):
            tmpChannel[i] = tmpChannel[i] * tmpCsdCompensate
        dataPilotEst.append(tmpChannel)
        #print(dataPilotEst[k])
    return dataPilotEst

def procVhtChannelFeedback(ltfSym, bw, nSts, nRx):
    if(isinstance(bw, BW)):
        # estimate channel
        if(bw == BW.BW20):
            nScData = 52
            nScDataPilot = 56
        elif(bw == BW.BW40):
            nScData = 108
            nScDataPilot = 114
        else:   # 80
            nScData = 234
            nScDataPilot = 242
        dataChanEst = procVhtDataChanEst(ltfSym, nScData, nSts, nRx)
        dataPilotChanEst = procVhtPilotChanIntpo(dataChanEst, nScData, nSts, nRx)
        # for beamforming feed back, remove CSD
        kDP = list(range(-28, 0)) + list(range(1, 29))
        dataPilotChanEstNoCsd = []
        #print("for beamforming feed back, remove CSD")
        for k in range(0, nScDataPilot):
            tmpCsd = np.array(C_CYCLIC_SHIFT_NL[nSts - 1][0:nSts])
            tmpCsdCompensate = tmpCsd * 20 * 0.001 * -1
            tmpCsdCompensate = np.exp(-2j * np.pi * tmpCsdCompensate * kDP[k] / 64)
            tmpChannel = dataPilotChanEst[k]
            for i in range(0, nRx):
                tmpChannel[i] = tmpChannel[i] * tmpCsdCompensate
            dataPilotChanEstNoCsd.append(tmpChannel)
        # SVD, get the V
        vDataPilot = []
        for k in range(0, nScDataPilot):
            """
            If X is m-by-n with m >= n, then it is equivalent to SVD(X,0).
            For m < n, only the first m columns of V are computed and S is m-by-m.
            """
            u, s, vh = np.linalg.svd(dataPilotChanEstNoCsd[k], full_matrices=False)
            v = vh.conjugate().T * -1
            vDataPilot.append(v)
        return vDataPilot
    else:
        print("cloud phy80211header, procVhtChannelFeedback input param error")
        return []

def procSpatialMapping(inSigSs, inQ):
        tmpnSS = len(inSigSs)
        # print("spatial mapping, nSS %d, nSC %d, bfQ len %d, bfQ element len %d" % (tmpnSS, len(inSigSs[0]), len(inQ), len(inQ[0])))
        outSigSs = []
        for j in range(0, tmpnSS):
            outSigSs.append([])
        for k in range(0, len(inQ)):
            tmpX = []
            for j in range(0, tmpnSS):
                tmpX.append([inSigSs[j][k]])
            tmpX = np.array(tmpX)
            tmpQX = np.matmul(inQ[k], tmpX)
            for j in range(0, tmpnSS):
                outSigSs[j].append(tmpQX[j][0])
        return outSigSs

def genNoiseAmpWithSnrDb(sigAmp, snrDbList):
    if(isinstance(sigAmp, (int, float)) and isinstance(snrDbList, list)):
        return [sigAmp/(10**(each/20)) for each in snrDbList]

if __name__ == "__main__":
    print(genNoiseAmpWithSnrDb(0.26125001, [20, 22, 24, 26, 28, 30]))
    pass



