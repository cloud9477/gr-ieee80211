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

import phy80211header as p8h
from matplotlib import pyplot as plt
import numpy as np
import time
import struct
from enum import Enum

# compressed feedback V angle type
class CVA_TPYE(Enum):
    PHI = 0
    PSI = 1

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

# number of bit for each angle to quantize for V of 2 rows, 3 rows and 4 rows
C_VHT_CB0_R2_ANGLE_BIT_N = [7, 5]
C_VHT_CB0_R3_ANGLE_BIT_N = [7, 7, 5, 5, 7, 5]
C_VHT_CB0_R4_ANGLE_BIT_N = [7, 7, 7, 5, 5, 5, 7, 7, 5, 5, 7, 5]
C_VHT_CB1_R2_ANGLE_BIT_N = [9, 7]
C_VHT_CB1_R3_ANGLE_BIT_N = [9, 9, 7, 7, 9, 7]
C_VHT_CB1_R4_ANGLE_BIT_N = [9, 9, 9, 7, 7, 7, 9, 9, 7, 7, 9, 7]

C_VHT_BFFB_ANGLE_TYPE = [
    [], [],
    [CVA_TPYE.PHI, CVA_TPYE.PSI],
    [CVA_TPYE.PHI, CVA_TPYE.PHI, CVA_TPYE.PSI, CVA_TPYE.PSI, CVA_TPYE.PHI, CVA_TPYE.PSI],
    [CVA_TPYE.PHI, CVA_TPYE.PHI, CVA_TPYE.PHI, CVA_TPYE.PSI, CVA_TPYE.PSI, CVA_TPYE.PSI, CVA_TPYE.PHI, CVA_TPYE.PHI, CVA_TPYE.PSI, CVA_TPYE.PSI, CVA_TPYE.PHI, CVA_TPYE.PSI]
]

C_VHT_BFFB_ANGLE_NUM = [
    [],     # Nr = 0
    [],     # Nr = 1
    [0, 2, 2],      # Nr = 2, Nc = 0, 1, 2
    [0, 4, 6, 6],   # Nr = 3, Nc = 0, 1, 2, 3
    [0, 6, 10, 12, 12]  # Nr = 4, Nc = 0, 1, ,2 ,3 ,4
]

# IEEE80211-2020 Table 9-76—Subcarrier indices for which a compressed beamforming feedback matrix is sent back
C_VHT_BFFB_SCIDX_20 = [
    [], [-28, -27, -26, -25, -24, -23, -22, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -6, -5, -4, -3, -2, -1, 
    1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28],
    [-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
    [], [-28, -24, -20, -16, -12, -8, -4, -1, 1, 4, 8, 12, 16, 20, 24, 28]
]

# IEEE80211-2020 Table 9-79—Number of subcarriers and subcarrier mapping
C_MU_EXBF_SCIDX_20 = [
    [], [-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
    [-28, -24, -20, -16, -12, -8, -4, -1, 1, 4, 8, 12, 16, 20, 24, 28], [], [-28, -20, -12, -4, -1, 1, 4, 12, 20, 28]
]

class FC_TPYE(Enum):
    MGMT = 0
    CTRL = 1
    DATA = 2
    EXT = 3

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class FC_SUBTPYE_MGMT(Enum):
    ASSOREQ = 0
    ASSORES = 1
    REASSOREQ = 2
    REASSORES = 3
    PROBEREQ = 4
    PROBERES = 5
    TIMINGAD = 6
    RESERVED7 = 7
    BEACON = 8
    ATIM = 9
    DISASSO = 10
    AUTH = 11
    DEAUTH = 12
    ACT = 13
    ACTNOACK = 14
    RESERVED15 = 15

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class FC_SUBTPYE_CTRL(Enum):
    RESERVED0 = 0
    RESERVED1 = 1
    RESERVED2 = 2
    RESERVED3 = 3
    BFREPOPOLL = 4
    VHTNDPANNO = 5
    FRAMEEXT = 6
    WRAPPER = 7
    BLOCKACKREQ = 8
    BLOCKACK = 9
    PSPOLL = 10
    RTS = 11
    CTS = 12
    ACK = 13
    CFEND = 14
    CFENDCFACK = 15

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class FC_SUBTPYE_DATA(Enum):
    DATA = 0
    DATACFACK = 1
    DATACFPOLL = 2
    DATACFACKCFPOLL = 3
    NULL = 4
    CFACK = 5
    CFPOLL = 6
    CFACKCFPOLL = 7
    QOSDATA = 8
    QOSDATACFACK = 9
    QOSDATACFPOLL = 10
    QOSDATACFACKCFPOLL = 11
    QOSNULL = 12
    RESERVED13 = 13
    QOSCFPOLL = 14
    QOSCFACKCFPOLL = 15

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class FC_SUBTPYE_EXT(Enum):
    DMGBEACON = 0
    RESERVED1 = 1
    RESERVED2 = 2
    RESERVED3 = 3
    RESERVED4 = 4
    RESERVED5 = 5
    RESERVED6 = 6
    RESERVED7 = 7
    RESERVED8 = 8
    RESERVED9 = 9
    RESERVED10 = 10
    RESERVED11 = 11
    RESERVED12 = 12
    RESERVED13 = 13
    RESERVED14 = 14
    RESERVED15 = 15

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class MGMT_ELE(Enum):
    SSID = 0
    SUPOTRATE = 1
    DSSSPARAM = 3
    TIM = 5
    COUNTRY = 7
    BSSLOAD = 11
    HTCAP = 45
    RSN = 48
    HTOPS = 61
    ANTENNA = 64
    RMENABLED = 70
    EXTCAP = 127
    VHTCAP = 191
    VHTOPS = 192
    TXPOWER = 195
    VENDOR = 221

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class MGMT_ACT_CAT(Enum):
    SPECTRUM_MGMT = 0
    QOS = 1
    RESERVED2 = 2
    BLOCK_ACK = 3
    PUBLIC = 4
    RADIO_MEA = 5
    FAST_BSS_TRANS = 6
    HT = 7
    SA_QUERY = 8
    PROCTECT_DUAL_PUB_ACT = 9
    WNM = 10
    UNPROTECT_WNM = 11
    TDLS = 12
    MESH = 13
    MULTIHOP = 14
    SELF_PROTECTED = 15
    DMG = 16
    WIFI_ALLIANCE = 17
    FAST_SESSION_TRANS = 18
    ROBUST_AV_STREAM = 19
    UNPROTECT_DMG = 20
    VHT = 21
    UNPROTECT_S1G = 22
    S1G = 23
    FLOW_CTRL = 24
    CTRL_RESP_MCS_NEGOTIATION = 25
    FILS = 26
    CDMG = 27
    CMMG = 28
    GLK = 29

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

# some define
C_FC_SUBTYPE_MGMT_STR = ["Association Request", "Association Response", "Reassociation Request", "Reassociation Response", "Probe Request", "Probe Response", "Timing Advertisement", "Reserved", "Beacon", "ATIM", "Disassociation", "Authentication", "Deauthentication", "Action", "Action No Ack", "Reserved"]
C_FC_SUBTYPE_CTRL_STR = ["Reserved", "Reserved", "Reserved", "Reserved", "Beamforming Report Poll", "VHT NDP Announcement", "Control Frame Extension", "Control Wrapper", "Block Ack Request (BlockAckReq)", "Block Ack (BlockAck)", "PS-Poll", "RTS", "CTS", "Ack", "CF-End", "CF-End +CF-Ack"]
C_FC_SUBTYPE_DATA_STR = ["Data", "Data +CF-Ack", "Data +CF-Poll", "Data +CF-Ack +CF-Poll", "Null (no data)", "CF-Ack (no data)", "CF-Poll (no data)", "CF-Ack +CF-Poll (no data)", "QoS Data", "QoS Data +CF-Ack", "QoS Data +CF-Poll", "QoS Data +CF-Ack +CF-Poll", "QoS Null (no data)", "Reserved", "QoS CF-Poll (no data)", "QoS CF-Ack +CF-Poll (no data)"]
C_FC_SUBTYPE_EXT_STR = ["DMG Beacon", "Reserved", "Reserved", "Reserved", "Reserved","Reserved", "Reserved", "Reserved", "Reserved","Reserved", "Reserved", "Reserved", "Reserved", "Reserved", "Reserved", "Reserved"]

class frameControl:
    def __init__(self, fc):
        self.fcValue = fc
        self.protocalVer = fc & 3
        self.type = FC_TPYE((fc >> 2) & 3)
        if(self.type == FC_TPYE.MGMT):
            self.subType = FC_SUBTPYE_MGMT((fc >> 4) & 15)
        elif(self.type == FC_TPYE.CTRL):
            self.subType = FC_SUBTPYE_CTRL((fc >> 4) & 15)
        elif(self.type == FC_TPYE.DATA):
            self.subType = FC_SUBTPYE_DATA((fc >> 4) & 15)
        elif(self.type == FC_TPYE.EXT):
            self.subType = FC_SUBTPYE_EXT((fc >> 4) & 15)
        else:
            self.subType = str((fc >> 4) & 15)
        self.toDs = (fc >> 8) & 1
        self.fromDs = (fc >> 9) & 1
        self.moreFrag = (fc >> 10) & 1
        self.retry = (fc >> 11) & 1
        self.powerMgmt = (fc >> 12) & 1
        self.moreData = (fc >> 13) & 1
        self.protectFrame = (fc >> 14) & 1
        self.htcOrder = (fc >> 15) & 1      # in non qos data set 1 for order, in qos data set 1 for htc
    
    def printInfo(self):
        print("cloud mac80211header, value %s, FC Info protocol:%d, type:%s, sub type:%s, to DS:%d, from DS:%d, more frag:%d, retry:%d" % (hex(self.fcValue), self.protocalVer, self.type, self.subType, self.toDs, self.fromDs, self.moreFrag, self.retry))

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

def procVhtPhiDequanti(phiquan, bphi):
    tmpK = list(range(0, 2**bphi))
    tmpShift = np.pi / (2**(bphi))
    if(phiquan >= 0 and phiquan < len(tmpK)):
        tmpPhi = [each * np.pi / (2**(bphi-1)) + tmpShift for each in tmpK]
        # print(tmpPhi)
        return tmpPhi[phiquan]
    return 0.0

def procVhtPsiDequanti(psiquan, bpsi):
    tmpK = list(range(0, 2**bpsi))
    tmpShift = np.pi / (2**(bpsi+2))
    if(psiquan >= 0 and psiquan < len(tmpK)):
        tmpPsi = [each * np.pi / (2**(bpsi+1)) + tmpShift for each in tmpK]
        # print(tmpPsi)
        return tmpPsi[psiquan]
    return 0.0

# IEEE802.11-2020 section 19.3.12.3.6 Compressed beamforming feedback matrix
def procVhtVCompressDebug(v, codebook = 0, ifDebug = True):
    resValue = []       # value and type
    resType = []
    # Phi for Di, Psi for Gli, feedback type is MU-MIMO for VHT PPDU
    if(isinstance(v, np.ndarray) and isinstance(ifDebug, bool) and isinstance(codebook, int)):
        [m, n] = v.shape    # m is row, n is col
        if(ifDebug):
            print("V, get V %d row %d col" % (m, n))
            print(v)
        if(m > 0 and n > 0 and m >= n):
            if(codebook):
                nBitPhi = 9
                nBitPsi = 7
            else:
                nBitPhi = 7
                nBitPsi = 5
            if(ifDebug):
                print("quantization codebook %d, Phi %d bits, Psi %d bits" % (codebook, nBitPhi, nBitPsi))
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
                        resType.append(CVA_TPYE.PHI)
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
                    resType.append(CVA_TPYE.PSI)
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
                print("compare the VDtH and decompsed Vt results")
                print(vdtHRes)
                print(vtRes)
                print(resValue)
                print(resType)
    return resValue, resType

def procVhtVCompressDebugVt(v):
    # Phi for Di, Psi for Gli, feedback type is MU-MIMO for VHT PPDU
    if(isinstance(v, np.ndarray)):
        [m, n] = v.shape    # m is row, n is col
        if(m > 0 and n > 0 and m >= n):
            dt = np.zeros((n, n), dtype=complex)
            for j in range(0, n):
                tmpTheta = np.arctan2(np.imag(v[m-1][j]), np.real(v[m-1][j]))
                tmpValue = np.exp(tmpTheta * 1.0j)
                dt[j][j] = tmpValue
            dtH = dt.conjugate().T
            vdtHRes = np.matmul(v, dtH)
            return vdtHRes

def procVhtVCompress(v, codebook = 0):
    resValue = []       # value and type
    resType = []
    resOri = []
    # Phi for Di, Psi for Gli, feedback type is MU-MIMO for VHT PPDU
    if(isinstance(v, np.ndarray) and isinstance(codebook, int)):
        [m, n] = v.shape    # m is row, n is col
        if(m > 0 and n > 0 and m >= n):
            if(codebook):
                nBitPhi = 9
                nBitPsi = 7
            else:
                nBitPhi = 7
                nBitPsi = 5
            dt = np.zeros((n, n), dtype=complex)
            for j in range(0, n):
                tmpTheta = np.arctan2(np.imag(v[m-1][j]), np.real(v[m-1][j]))
                tmpValue = np.exp(tmpTheta * 1.0j)
                dt[j][j] = tmpValue
            dtH = dt.conjugate().T
            vdtH = np.matmul(v, dtH)
            for j in range(0, n):
                vdtH[m-1][j] = np.real(vdtH[m-1][j])
            # each loop we compute Di and following Gli(s), and keep updating the vdtH to be GliDiHVDtH
            for grIter in range(0, min(m-1, n)):      # gr for givens rotation
                i = grIter + 1
                di = np.zeros((m, m), dtype=complex)
                for j in range(0, i-1):
                    di[j][j] = 1
                diPhi = []
                for j in range(i, m):
                    tmpTheta = np.arctan2(np.imag(vdtH[j-1][i-1]), np.real(vdtH[j-1][i-1]))
                    diPhi.append(tmpTheta)
                    tmpValue = np.exp(tmpTheta * 1.0j)
                    di[j-1][j-1] = tmpValue
                if(len(diPhi)):
                    diPhi = np.unwrap(diPhi)
                    if(diPhi[0] < 0):
                        diPhi += np.pi * 2
                    for each in diPhi:
                        resValue.append(procVhtPhiQuanti(each, nBitPhi))
                        resType.append(CVA_TPYE.PHI)
                        resOri.append(each)
                di[m-1][m-1] = 1
                vdtH = np.matmul(di.conjugate().T, vdtH)    # now vdtH is from VDtH to DiHVDtH
                # remove residual imag of the column
                for l in range(i, m):
                    vdtH[l-1][i-1] = np.real(vdtH[l-1][i-1])
                for l in range(i+1, m+1):
                    gli = np.zeros((m, m), dtype=complex)
                    x1 = np.real(vdtH[i-1][i-1])
                    x2 = np.real(vdtH[l-1][i-1])
                    y = np.sqrt(x1*x1 + x2*x2)
                    gliPsi = np.arccos(x1 / y)
                    resValue.append(procVhtPsiQuanti(gliPsi, nBitPsi))
                    resType.append(CVA_TPYE.PSI)
                    resOri.append(gliPsi)
                    if(gliPsi < 0 or gliPsi > np.pi/2):
                        print("Gli psi value error")
                    for j in range(0, m):
                        gli[j][j] = 1
                    gli[i-1][i-1] = np.cos(gliPsi)
                    gli[l-1][i-1] = -np.sin(gliPsi)
                    gli[i-1][l-1] = np.sin(gliPsi)
                    gli[l-1][l-1] = np.cos(gliPsi)
                    vdtH = np.matmul(gli, vdtH)
                    vdtH[l-1][i-1] = 0
    # print(resValue)
    # print(resType)
    # print(resOri, ",")
    return resValue, resType

def procVhtVRecover(nr, nc, quanAngle, codebook):
    if(isinstance(quanAngle, list)):
        if(len(quanAngle) == C_VHT_BFFB_ANGLE_NUM[nr][nc]):
            if(codebook):
                nBitPhi = 9
                nBitPsi = 7
            else:
                nBitPhi = 7
                nBitPsi = 5
            angleIter = 0
            vtRes = np.identity(nr)
            for grIter in range(0, min(nr-1, nc)):      # gr for givens rotation
                i = grIter + 1
                di = np.zeros((nr, nr), dtype=complex)
                for j in range(0, i-1):
                    di[j][j] = 1
                for j in range(i, nr):
                    tmpPhi = procVhtPhiDequanti(quanAngle[angleIter], nBitPhi)
                    angleIter += 1
                    tmpValue = np.exp(tmpPhi * 1.0j)
                    di[j-1][j-1] = tmpValue
                di[nr-1][nr-1] = 1
                vtRes = np.matmul(vtRes, di)
                for l in range(i+1, nr+1):
                    gli = np.zeros((nr, nr), dtype=complex)
                    for j in range(0, nr):
                        gli[j][j] = 1
                    tmpPsi = procVhtPsiDequanti(quanAngle[angleIter], nBitPsi)
                    angleIter += 1
                    gli[i-1][i-1] = np.cos(tmpPsi)
                    gli[l-1][i-1] = -np.sin(tmpPsi)
                    gli[i-1][l-1] = np.sin(tmpPsi)
                    gli[l-1][l-1] = np.cos(tmpPsi)
                    vtRes = np.matmul(vtRes, gli.T) # compute the final Vt to compare
            vIt = np.zeros((nr, nc), dtype=complex) # I tilde for V tilde
            for j in range(0, min(nr, nc)):
                vIt[j][j] = 1
            vtRes = np.matmul(vtRes, vIt)
            # print(vtRes)
            return vtRes

# interpolation of grouped channel feedback
def procVhtVIntpoV1(vD, group):
    if(isinstance(vD, list) and group in [1, 2, 4]):
        if(group == 1):
            # only interpolate pilots
            return vD[0:7] + [(vD[6] + vD[7])/2] + vD[7:20] + [(vD[19] + vD[20])/2] + vD[20:32] + [(vD[31] + vD[32])/2] + vD[32:45] + [(vD[44] + vD[45])/2] + vD[45:52]

"""tx gen functions ------------------------------------------------------------------------------------------"""
# vDP: input channel fb V of data and pilot subcarriers sorted by channel index from -Ns to Ns
# snrNc: input snr of received streams of RX number, should be the column of V
def genVhtCompressedBfReport(vDP, snrNC, group, codebook):
    tmpReportBytes = b""
    if(isinstance(vDP, list) and isinstance(snrNC, list) and group in [1, 2, 4] and codebook in [0, 1]):
        # print("cloud mac80211header vht compressed bf report, vDP len %d, group: %d, codebook %d" % (len(vDP), group, codebook))
        tmpReportBits = []
        if(codebook):
            nBitPhi = 9
            nBitPsi = 7
        else:
            nBitPhi = 7
            nBitPsi = 5
        if(len(vDP) == 56):
            for i in range(0, len(snrNC)):
                tmpSnrTab = [0.25 * each for each in range(-40, 216)]
                tmpSnrDPQuantized = np.byte(min(range(256), key=lambda k: abs(tmpSnrTab[k]-snrNC[i])) - 128)
                for j in range(0, 8):
                    tmpReportBits.append((tmpSnrDPQuantized >> j) & 1)
            for i in range(-28, 0):
                if(i in C_VHT_BFFB_SCIDX_20[group]):
                    tmpAngle, tmpType = procVhtVCompress(vDP[i+28], codebook)
                    for j in range(0, len(tmpAngle)):
                        if(tmpType[j] == CVA_TPYE.PHI):
                            for k in range(0, nBitPhi):
                                tmpReportBits.append((tmpAngle[j] >> k) & 1)
                        else:
                            for k in range(0, nBitPsi):
                                tmpReportBits.append((tmpAngle[j] >> k) & 1)
            for i in range(1, 29):
                if(i in C_VHT_BFFB_SCIDX_20[group]):
                    tmpAngle, tmpType = procVhtVCompress(vDP[i+27], codebook)
                    for j in range(0, len(tmpAngle)):
                        if(tmpType[j] == CVA_TPYE.PHI):
                            for k in range(0, nBitPhi):
                                tmpReportBits.append((tmpAngle[j] >> k) & 1)
                        else:
                            for k in range(0, nBitPsi):
                                tmpReportBits.append((tmpAngle[j] >> k) & 1)
        elif(len(vDP) == 114):
            pass
        elif(len(vDP) == 242):
            pass
        else:
            return []
        tmpNPadBits = int(np.ceil(len(tmpReportBits) / 8)) * 8 - len(tmpReportBits)
        tmpReportBits += [0] * tmpNPadBits
        print("genVhtCompressedBfReport npadbits %d, bits num %d" % (tmpNPadBits, len(tmpReportBits)))
        for i in range(int(len(tmpReportBits)/8)):
            tmpByte = 0
            for j in range(0, 8):
                tmpByte |= (tmpReportBits[i*8+j] << j)
            tmpReportBytes += struct.pack('<B', tmpByte)
    return tmpReportBytes

# IEEE80211-2020 section 9.6.22
# vDP: input channel fb V of data and pilot subcarriers sorted by channel index from -Ns to Ns
# snrNc: input snr of received streams of RX number, should be the column of V
def genMgmtActVhtCompressBf(vDP, group, codebook, fbType, token):
    tmpVhtCompressBfPkt = b"\x00"   # VHT action value 0 to be compressed bf
    if(isinstance(vDP, list) and all(isinstance(each, np.ndarray) for each in vDP)):
        print("cloud mac80211header, mgmt action no ack: vht compressed bf, vDP len %d, group: %d, codebook %d, fbType %d, token %d" % (len(vDP), group, codebook, fbType, token))
        if(len(vDP) == 56):
            tmpChanBw = 0
        elif(len(vDP) == 114):
            tmpChanBw = 1
        elif(len(vDP) == 242):
            tmpChanBw = 2
        else:
            tmpChanBw = 3
        if(group == 1):
            tmpGroupValue = 0
        elif(group == 2):
            tmpGroupValue = 1
        else:
            tmpGroupValue = 2
        tmpVhtMimoCtrl = 0
        [nr, nc] = vDP[0].shape
        tmpVhtMimoCtrl = (nc - 1)
        tmpVhtMimoCtrl += ((nr - 1) << 3)
        tmpVhtMimoCtrl += (tmpChanBw << 6)
        tmpVhtMimoCtrl += (tmpGroupValue << 8)
        tmpVhtMimoCtrl += (codebook << 10)
        tmpVhtMimoCtrl += (fbType << 11)
        tmpVhtMimoCtrl += (0 << 12)         # only one feedback packet
        tmpVhtMimoCtrl += (1 << 15)         # only one feedback packet
        tmpVhtMimoCtrl += (token << 18)
        print("mimo control value %d" % (tmpVhtMimoCtrl))
        tmpVhtCompressBfPkt += struct.pack('<L', tmpVhtMimoCtrl)[0:3]
        # print("cloud mac80211header, vht mimo control field bytes")
        # print(tmpVhtCompressBfPkt.hex())
        # add compressed bf report field
        tmpVhtCompressBfPkt += genVhtCompressedBfReport(vDP, [0] * nc, group, codebook)
        # add MU Exclusive bf report field
        if(len(vDP) == 56):
            if(group == 1):
                tmpMuExItem = 30 * nc
            elif(group == 1):
                tmpMuExItem = 16 * nc
            else:
                tmpMuExItem = 10 * nc
        else:
            tmpMuExItem = 0
        tmpVhtCompressBfPkt += b"\x00" * int(tmpMuExItem/2)     # MU exclusive part not implemented yet
    return tmpVhtCompressBfPkt

"""rx parser functions ------------------------------------------------------------------------------------------"""

# use major type and sub type to avoid keyword: type
def rxPacketTypeCheck(pkt, mType, sType):
    if(isinstance(pkt, (bytes, bytearray)) and isinstance(mType, FC_TPYE) and isinstance(sType, (FC_SUBTPYE_MGMT, FC_SUBTPYE_CTRL, FC_SUBTPYE_DATA, FC_SUBTPYE_EXT))):
        pktLen = len(pkt)
        procdLen = 0
        # fc
        if((procdLen + 2) <= pktLen):
            hdr_fc = frameControl(struct.unpack('<H', pkt[0:2])[0])
            procdLen += 2
            if(hdr_fc.type == mType and hdr_fc.subType == sType):
                return True
    return False


def mgmtVhtActCompressBfParser(pkt):
    if(isinstance(pkt, (bytes, bytearray))):
        vhtMimoCtrl = struct.unpack('<L', pkt[0:3]+b"\x00")[0]
        print("mgmtVhtActCompressBfParser mimo ctrl value %d" % vhtMimoCtrl)
        nc = (vhtMimoCtrl & 7) + 1
        nr = ((vhtMimoCtrl >> 3) & 7) + 1
        bw = ((vhtMimoCtrl >> 6) & 3)
        group = 2 ** ((vhtMimoCtrl >> 8) & 3)
        codebook = ((vhtMimoCtrl >> 10) & 1)
        fbType = ((vhtMimoCtrl >> 11) & 1)
        token = ((vhtMimoCtrl >> 18) & 63)
        print("cloud mac80211header, vht compressed bf parser, nc %d, nr %d, bw %d, group: %d, codebook %d, fbType %d, token %d" % (nc, nr, bw, group, codebook, fbType, token))
        if(codebook):
            nBitPhi = 9
            nBitPsi = 7
        else:
            nBitPhi = 7
            nBitPsi = 5
        tmpAngleByteNum = int(np.ceil(len(C_VHT_BFFB_SCIDX_20[group]) * C_VHT_BFFB_ANGLE_NUM[nr][nc] * (nBitPhi + nBitPsi) / 2)/8)
        tmpAngleBits = []
        for i in range(0, tmpAngleByteNum):
            for j in range(0, 8):
                tmpAngleBits.append((int(pkt[i+3+nc]) >> j) & 1)
        tmpV = []
        tmpIter = 0
        for i in range(0, len(C_VHT_BFFB_SCIDX_20[group])):
            tmpAngle = [0] * C_VHT_BFFB_ANGLE_NUM[nr][nc]
            for j in range(0, C_VHT_BFFB_ANGLE_NUM[nr][nc]):
                tmpQuanAngle = 0
                if(C_VHT_BFFB_ANGLE_TYPE[nr][j] == CVA_TPYE.PHI):
                    for k in range(0, nBitPhi):
                        tmpQuanAngle |= (tmpAngleBits[tmpIter] << k)
                        tmpIter += 1
                    tmpAngle[j] = tmpQuanAngle
                else:
                    for k in range(0, nBitPsi):
                        tmpQuanAngle |= (tmpAngleBits[tmpIter] << k)
                        tmpIter += 1
                    tmpAngle[j] = tmpQuanAngle
            tmpV.append(procVhtVRecover(nr, nc, tmpAngle, codebook))
        return procVhtVIntpoV1(tmpV, group)
    return []


def mgmtElementParser(inbytes):
    if(isinstance(inbytes, (bytes, bytearray)) and len(inbytes) > 0):
        elementIter = 0
        tmpMgmtElements = []
        inPutLen = len(inbytes)
        while((elementIter+2) < inPutLen):  # one byte type, one byte len
            # print("Element at %d, ID %d, Len %d" % (elementIter, inbytes[elementIter], inbytes[elementIter+1]))
            if(MGMT_ELE.has_value(inbytes[elementIter])):
                tmpElement = MGMT_ELE(inbytes[elementIter])
                elementIter += 1
                tmpLen = inbytes[elementIter]
                elementIter += 1
                if((elementIter+tmpLen) < inPutLen):
                    if(tmpElement == MGMT_ELE.SSID):
                        print(inbytes[elementIter:elementIter+tmpLen].hex())
                        tmpStr = inbytes[elementIter:elementIter+tmpLen].decode("utf-8")
                        tmpMgmtElements.append("SSID: " + tmpStr)
                    elif(tmpElement == MGMT_ELE.SUPOTRATE):
                        tmpStr = ""
                        for i in range(0, tmpLen):
                            if(inbytes[elementIter+i] >= 0x80):
                                tmpStr += str((inbytes[elementIter+i]-0x80)*500/1000)
                                tmpStr += "Mbps(Basic) "
                            else:
                                tmpStr += str(inbytes[elementIter+i]*500/1000)
                                tmpStr += "Mbps "
                        tmpMgmtElements.append("Suppoprted Rates: " + tmpStr)
                    elif(tmpElement == MGMT_ELE.DSSSPARAM):
                        tmpStr = str(inbytes[elementIter])
                        tmpMgmtElements.append("DS Channel: " + tmpStr)
                    elif(tmpElement == MGMT_ELE.TIM):
                        tmpMgmtElements.append("TIM: To be added")
                    elif(tmpElement == MGMT_ELE.COUNTRY):
                        tmpStr = inbytes[elementIter:elementIter+3].decode("utf-8")
                        tmpMgmtElements.append("Country: " + tmpStr)
                    elif(tmpElement == MGMT_ELE.BSSLOAD):
                        tmpStaCount = struct.unpack('<H', inbytes[elementIter:elementIter+2])[0]
                        tmpChanUtil = int(inbytes[elementIter+2])
                        tmpAvailAdmCap = struct.unpack('<H', inbytes[elementIter+3:elementIter+5])[0]
                        tmpMgmtElements.append("BSS Load, Station Count: " + str(tmpStaCount) + ", Channel Utilization: " + str(tmpChanUtil) + ", Available Admission Capacity: " + str(tmpAvailAdmCap))
                    elif(tmpElement == MGMT_ELE.RSN):
                        tmpMgmtElements.append("RSN: To be added")
                    elif(tmpElement == MGMT_ELE.HTCAP):
                        tmpHtCapInfo = struct.unpack('<H', inbytes[elementIter:elementIter+2])[0]
                        tmpMgmtElements.append("HT CAP Info, LDPC %d, Chan Width %d, GF %d, SGI20 %d, SGI40 %d, TXSTBC %d" % (
                            (tmpHtCapInfo >> 0) & 1,
                            (tmpHtCapInfo >> 1) & 1,
                            (tmpHtCapInfo >> 4) & 1,
                            (tmpHtCapInfo >> 5) & 1,
                            (tmpHtCapInfo >> 6) & 1,
                            (tmpHtCapInfo >> 7) & 1
                        ))
                        tmpMcsBits = []
                        for i in range(0, 10):
                            for j in range(0, 8):
                                tmpMcsBits.append((inbytes[elementIter+3+i] >> j) & 1)
                        tmpMcsStr = "HT CAP MCS 0-31: "
                        for i in range(0, 32):
                            tmpMcsStr += str(tmpMcsBits[i])
                        tmpMgmtElements.append(tmpMcsStr)
                        tmpMcsStr = "HT CAP MCS 32-76: "
                        for i in range(32, 77):
                            tmpMcsStr += str(tmpMcsBits[i])
                        tmpMgmtElements.append(tmpMcsStr)
                    elif(tmpElement == MGMT_ELE.HTOPS):
                        tmpMgmtElements.append("HT Operation: To be added")
                    elif(tmpElement == MGMT_ELE.ANTENNA):
                        tmpMgmtElements.append("Antenna: %d" % inbytes[elementIter])
                    elif(tmpElement == MGMT_ELE.RMENABLED):
                        tmpMgmtElements.append("RM Enable Cap: To be added")
                    elif(tmpElement == MGMT_ELE.EXTCAP):
                        tmpMgmtElements.append("Ext Cap: To be added")
                    elif(tmpElement == MGMT_ELE.VHTCAP):
                        tmpVhtCapInfo = struct.unpack('<I', inbytes[elementIter:elementIter+4])[0]
                        tmpMgmtElements.append("VHT CAP Info, MAX MPDU Len %d, RX LDPC %d, TX STBC %d, RX STBC %d, Sounding Dim %d" % (
                            (tmpVhtCapInfo >> 0) & 3,
                            (tmpVhtCapInfo >> 4) & 1,
                            (tmpVhtCapInfo >> 7) & 1,
                            (tmpVhtCapInfo >> 8) & 7,
                            (tmpVhtCapInfo >> 16) & 7
                        ))
                    elif(tmpElement == MGMT_ELE.VHTOPS):
                        tmpMgmtElements.append("VHT Operation: To be added")
                    elif(tmpElement == MGMT_ELE.TXPOWER):
                        tmpMgmtElements.append("TX Power: Local Max Tx Pwr Count %d, Local Max Tx Pwr 20M %d" % (inbytes[elementIter] & 3, inbytes[elementIter+1]))
                    elif(tmpElement == MGMT_ELE.VENDOR):
                        tmpMgmtElements.append("Vendor: To be added")
                    else:
                        pass
                elementIter += tmpLen
            else:
                tmpElementByte = inbytes[elementIter]
                elementIter += 1
                tmpLen = inbytes[elementIter]
                elementIter += 1
                elementIter += tmpLen
        return tmpMgmtElements
    print("cloud mac80211header, mgmtParser input type error")
    return []


def pktParser(pkt):
    if(isinstance(pkt, (bytes, bytearray))):
        pktLen = len(pkt)
        procdLen = 0
        # fc
        if((procdLen + 2) <= pktLen):
            hdr_fc = frameControl(struct.unpack('<H', pkt[0:2])[0])
            hdr_fc.printInfo()
            procdLen += 2
        else:
            return
        # duration
        if((procdLen + 2) <= pktLen):
            hdr_duration = struct.unpack('<H', pkt[2:4])[0]
            procdLen += 2
            print("Packet duration %d us" % hdr_duration)
        else:
            return
        # check type
        if(hdr_fc.type == FC_TPYE.MGMT):
            if((procdLen + 18) <= pktLen):
                hdr_addRx = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen:procdLen+6])
                hdr_addTx = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen+6:procdLen+12])
                hdr_addDest = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen+12:procdLen+18])
                procdLen += 18
                print("Management to %s from %s dest %s" % (hdr_addRx, hdr_addTx, hdr_addDest))
            if((procdLen + 2) <= pktLen):
                hdr_seqCtrl= struct.unpack('<H', pkt[procdLen:procdLen+2])[0]
                procdLen += 2
                print("Management sequence control %d" % hdr_seqCtrl)
            if(hdr_fc.subType == FC_SUBTPYE_MGMT.BEACON):
                # Timestamp, Beacon Interval, Cap
                if((procdLen + 12) <= pktLen):
                    beacon_timestamp = struct.unpack('<Q', pkt[procdLen:procdLen+8])[0]
                    beacon_interval = struct.unpack('<H', pkt[procdLen+8:procdLen+10])[0]
                    beacon_cap = struct.unpack('<H', pkt[procdLen+10:procdLen+12])[0]
                    procdLen += 12
                    print("Management Beacon Timestamp %d, Interval %d, Cap %d" % (beacon_timestamp, beacon_interval, beacon_cap))
                # Elements
                if(procdLen <= pktLen):
                    beaconElements = mgmtElementParser(pkt[procdLen:])
                    for each in beaconElements:
                        print(each)
            elif(hdr_fc.subType == FC_SUBTPYE_MGMT.PROBEREQ):
                print("Management Probe Request")
            elif(hdr_fc.subType == FC_SUBTPYE_MGMT.PROBERES):
                print("Management Probe Response")
            else:
                print("Management subtype, not supported yet")
        elif(hdr_fc.type == FC_TPYE.CTRL):
            if(hdr_fc.subType == FC_SUBTPYE_CTRL.ACK):
                if((procdLen + 6) <= pktLen):
                    hdr_addRx = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen:procdLen+6])
                    procdLen += 6
                    print("ACK to %s" % hdr_addRx)
            elif(hdr_fc.subType == FC_SUBTPYE_CTRL.BLOCKACK):
                if((procdLen + 12) <= pktLen):
                    hdr_addRx = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen:procdLen+6])
                    hdr_addTx = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen+6:procdLen+12])
                    procdLen += 12
                    print("BLOCK ACK to %s from %s" % (hdr_addRx, hdr_addTx))
                    # details to be added
            elif(hdr_fc.subType == FC_SUBTPYE_CTRL.RTS):
                if((procdLen + 12) <= pktLen):
                    hdr_addRx = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen:procdLen+6])
                    hdr_addTx = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen+6:procdLen+12])
                    procdLen += 12
                    print("RTS to %s from %s" % (hdr_addRx, hdr_addTx))
            elif(hdr_fc.subType == FC_SUBTPYE_CTRL.CTS):
                if((procdLen + 6) <= pktLen):
                    hdr_addRx = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen:procdLen+6])
                    procdLen += 6
                    print("CTS to %s" % hdr_addRx)
            else:
                print("Control subtype, not supported yet")
        elif(hdr_fc.type == FC_TPYE.DATA):
            if((procdLen + 18) <= pktLen):
                hdr_addRx = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen:procdLen+6])
                hdr_addTx = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen+6:procdLen+12])
                hdr_addDest = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen+12:procdLen+18])
                procdLen += 18
                print("Data to %s from %s dest %s" % (hdr_addRx, hdr_addTx, hdr_addDest))
            if((procdLen + 2) <= pktLen):
                hdr_seqCtrl= struct.unpack('<H', pkt[procdLen:procdLen+2])[0]
                procdLen += 2
                print("Data sequence control %d" % hdr_seqCtrl)
            if(hdr_fc.subType == FC_SUBTPYE_DATA.DATA):
                print("Data")
                if(procdLen <= pktLen):
                    print(pkt[procdLen:].hex())
            elif(hdr_fc.subType == FC_SUBTPYE_DATA.QOSDATA):
                print("QoS Data")
                if((procdLen+2) <= pktLen):
                    print("QoS control " + pkt[procdLen:procdLen+2].hex())
                    procdLen += 2
                if(procdLen <= pktLen):
                    print(pkt[procdLen:].hex())
            elif(hdr_fc.subType == FC_SUBTPYE_DATA.QOSNULL):
                print("QoS Null Data")
                if((procdLen+2) <= pktLen):
                    print("QoS control " + pkt[procdLen:procdLen+2].hex())
                    procdLen += 2
            else:
                print("Data subtype, not supported yet")
        else:
            print("cloud mac80211header, type not supported yet")


if __name__ == "__main__":
    pass
    # chan0 = []
    # fWaveComp = open("/home/cloud/sdr/gr-ieee80211/tools/cmu_chan0.bin", 'rb')
    # for i in range(0,128):
    #     tmpR = struct.unpack('f', fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1))[0]
    #     tmpI = struct.unpack('f', fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1))[0]
    #     chan0.append(tmpR + tmpI * 1j)
    # fWaveComp.close()
    # nTx = 2
    # nRx = 1
    # # compute feedback
    # ltfSym = []
    # ltfSym.append(p8h.procRemovePilots(p8h.procToneDescaling(p8h.procRmDcNonDataSc(p8h.procFftDemod(chan0[0:64]), p8h.F.VHT), p8h.C_SCALENTF_LTF_VHT[p8h.BW.BW20.value], nTx)))
    # ltfSym.append(p8h.procRemovePilots(p8h.procToneDescaling(p8h.procRmDcNonDataSc(p8h.procFftDemod(chan0[64:128]), p8h.F.VHT), p8h.C_SCALENTF_LTF_VHT[p8h.BW.BW20.value], nTx)))
    # vFb1 = p8h.procVhtChannelFeedback(ltfSym, p8h.BW.BW20, nTx, nRx)
    # for i in range(0, len(vFb1)):
    #     tmpAngle, tmpType = procVhtVCompressDebug(vFb1[i], 1, True)
    #     print(procVhtVRecover(2, 1, tmpAngle, 1))
    #     print(procVhtVCompressDebugVt(vFb1[i]))
    #     print("------------------------------------------")
    #     print("")
