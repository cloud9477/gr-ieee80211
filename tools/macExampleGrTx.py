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

import socket
import mac80211
import phy80211
import phy80211header as p8h
import mac80211header as m8h
import time
import numpy as np
import struct

def genMac80211UdpMPDU(udpPayload):
    udpIns = mac80211.udp("10.10.0.6",  # sour ip
                        "10.10.0.1",  # dest ip
                        39379,  # sour port
                        8889)  # dest port
    udpPacket = udpIns.genPacket(bytearray(udpPayload, 'utf-8'))
    print("udp packet")
    print(udpPacket.hex())
    ipv4Ins = mac80211.ipv4(43778,  # identification
                            64,  # TTL
                            "10.10.0.6",
                            "10.10.0.1")
    ipv4Packet = ipv4Ins.genPacket(udpPacket)
    print("ipv4 packet")
    print(ipv4Packet.hex())
    llcIns = mac80211.llc()
    llcPacket = llcIns.genPacket(ipv4Packet)
    print("llc packet")
    print(llcPacket.hex())
    
    mac80211Ins = mac80211.mac80211(2,  # type
                                    0,  # sub type, 8 = QoS Data, 0 = Data
                                    1,  # to DS, station to AP
                                    0,  # from DS
                                    0,  # retry
                                    0,  # protected
                                    'f4:69:d5:80:0f:a0',  # dest add
                                    '00:c0:ca:b1:5b:e1',  # sour add
                                    'f4:69:d5:80:0f:a0',  # recv add
                                    2704)  # sequence
    mac80211Packet = mac80211Ins.genPacket(llcPacket)
    print("mac packet: ", len(mac80211Packet))
    print(mac80211Packet.hex())
    return mac80211Packet

def genMac80211UdpAmpduVht(udpPayloads):
    if(isinstance(udpPayloads, list)):
        macPkts = []
        for eachUdpPayload in udpPayloads:
            udpIns = mac80211.udp("10.10.0.6",  # sour ip
                                "10.10.0.1",  # dest ip
                                39379,  # sour port
                                8889)  # dest port
            udpPacket = udpIns.genPacket(bytearray(eachUdpPayload, 'utf-8'))
            print("udp packet")
            print(udpPacket.hex())
            ipv4Ins = mac80211.ipv4(43778,  # identification
                                    64,  # TTL
                                    "10.10.0.6",
                                    "10.10.0.1")
            ipv4Packet = ipv4Ins.genPacket(udpPacket)
            print("ipv4 packet")
            print(ipv4Packet.hex())
            llcIns = mac80211.llc()
            llcPacket = llcIns.genPacket(ipv4Packet)
            print("llc packet")
            print(llcPacket.hex())
            
            mac80211Ins = mac80211.mac80211(2,  # type
                                            8,  # sub type, 8 = QoS Data, 0 = Data
                                            1,  # to DS, station to AP
                                            0,  # from DS
                                            0,  # retry
                                            0,  # protected
                                            'f4:69:d5:80:0f:a0',  # dest add
                                            '00:c0:ca:b1:5b:e1',  # sour add
                                            'f4:69:d5:80:0f:a0',  # recv add
                                            2704)  # sequence
            mac80211Packet = mac80211Ins.genPacket(llcPacket)
            print("mac packet: ", len(mac80211Packet))
            print(mac80211Packet.hex())
            macPkts.append(mac80211Packet)
        macAmpduVht = mac80211.genAmpduVHT(macPkts)
        print("vht ampdu packet")
        print(macAmpduVht.hex())
        return macAmpduVht
    else:
        print("genMac80211UdpAmpduVht udpPakcets is not list type")
        return b""

def genMac80211UdpAmpduHt(udpPayloads):
    if(isinstance(udpPayloads, list)):
        macPkts = []
        for eachUdpPayload in udpPayloads:
            udpIns = mac80211.udp("10.10.0.6",  # sour ip
                                "10.10.0.1",  # dest ip
                                39379,  # sour port
                                8889)  # dest port
            udpPacket = udpIns.genPacket(bytearray(eachUdpPayload, 'utf-8'))
            print("udp packet")
            print(udpPacket.hex())
            ipv4Ins = mac80211.ipv4(43778,  # identification
                                    64,  # TTL
                                    "10.10.0.6",
                                    "10.10.0.1")
            ipv4Packet = ipv4Ins.genPacket(udpPacket)
            print("ipv4 packet")
            print(ipv4Packet.hex())
            llcIns = mac80211.llc()
            llcPacket = llcIns.genPacket(ipv4Packet)
            print("llc packet")
            print(llcPacket.hex())
            
            mac80211Ins = mac80211.mac80211(2,  # type
                                            8,  # sub type, 8 = QoS Data, 0 = Data
                                            1,  # to DS, station to AP
                                            0,  # from DS
                                            0,  # retry
                                            0,  # protected
                                            'f4:69:d5:80:0f:a0',  # dest add
                                            '00:c0:ca:b1:5b:e1',  # sour add
                                            'f4:69:d5:80:0f:a0',  # recv add
                                            2704)  # sequence
            mac80211Packet = mac80211Ins.genPacket(llcPacket)
            print("mac packet: ", len(mac80211Packet))
            print(mac80211Packet.hex())
            macPkts.append(mac80211Packet)
        macAmpduVht = mac80211.genAmpduHT(macPkts)
        print("ht ampdu packet")
        print(macAmpduVht.hex())
        return macAmpduVht
    else:
        print("genMac80211UdpAmpduHt udpPakcets is not list type")
        return b""

print("cloud mac80211 example starts")

phyRxIp = "127.0.0.1"
phyRxPort = 9527
phyRxAddr = (phyRxIp, phyRxPort)
phyTxIp = "127.0.0.1"
phyTxPort = 9528
phyTxAddr = (phyTxIp, phyTxPort)
grSocket = socket.socket(family = socket.AF_INET, type = socket.SOCK_DGRAM)
grSocket.bind(phyRxAddr)

udpPayload  = "123456789012345678901234567890"
pkt = genMac80211UdpMPDU(udpPayload)
pkts = genMac80211UdpAmpduVht([udpPayload])

"""packets of different formats or MCS SISO """
for mcsIter in range(0, 8):
    grPkt = phy80211.genPktGrData(pkt, p8h.modulation(phyFormat=p8h.F.L, mcs = mcsIter, bw=p8h.BW.BW20, nSTS=1, shortGi=False))
    grSocket.sendto(grPkt, phyTxAddr)
    time.sleep(0.5)
for mcsIter in range(0, 8):
    grPkt = phy80211.genPktGrData(pkt, p8h.modulation(phyFormat=p8h.F.HT, mcs = mcsIter, bw=p8h.BW.BW20, nSTS=1, shortGi=False))
    grSocket.sendto(grPkt, phyTxAddr)
    time.sleep(0.5)
for mcsIter in range(0, 9):
    grPkt = phy80211.genPktGrData(pkts, p8h.modulation(phyFormat=p8h.F.VHT, mcs = mcsIter, bw=p8h.BW.BW20, nSTS=1, shortGi=False))
    grSocket.sendto(grPkt, phyTxAddr)
    time.sleep(0.5)

"""packets of different formats or MCS MIMO """
for mcsIter in range(0, 8):
    grPkt = phy80211.genPktGrData(pkt, p8h.modulation(phyFormat=p8h.F.HT, mcs = mcsIter+8, bw=p8h.BW.BW20, nSTS=2, shortGi=False))
    grSocket.sendto(grPkt, phyTxAddr)
    time.sleep(0.5)
for mcsIter in range(0, 9):
    grPkt = phy80211.genPktGrData(pkts, p8h.modulation(phyFormat=p8h.F.VHT, mcs = mcsIter, bw=p8h.BW.BW20, nSTS=2, shortGi=False))
    grSocket.sendto(grPkt, phyTxAddr)
    time.sleep(0.5)

"""packet of NDP 2x2 """
# grSocket.sendto(phy80211.genPktGrNdp(), phyTxAddr)

"""packet of multi-user mimo """
# # read channel bin file
# # they are the vht long training field received at each station
# chan0 = []
# chan1 = []
# fWaveComp = open("/home/cloud/sdr/gr-ieee80211/tools/cmu_chan0.bin", 'rb')
# for i in range(0,128):
#     tmpR = struct.unpack('f', fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1))[0]
#     tmpI = struct.unpack('f', fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1))[0]
#     chan0.append(tmpR + tmpI * 1j)
# fWaveComp.close()
# fWaveComp = open("/home/cloud/sdr/gr-ieee80211/tools/cmu_chan1.bin", 'rb')
# for i in range(0, 128):
#     tmpR = struct.unpack('f', fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1))[0]
#     tmpI = struct.unpack('f', fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1))[0]
#     chan1.append(tmpR + tmpI * 1j)
# fWaveComp.close()

# nSts = 2
# nRx = 1
# # compute feedback
# ltfSym = []
# ltfSym.append(p8h.procRemovePilots(p8h.procToneDescaling(p8h.procRmDcNonDataSc(p8h.procFftDemod(chan0[0:64]), p8h.F.VHT), p8h.C_SCALENTF_LTF_VHT[p8h.BW.BW20.value], nSts)))
# ltfSym.append(p8h.procRemovePilots(p8h.procToneDescaling(p8h.procRmDcNonDataSc(p8h.procFftDemod(chan0[64:128]), p8h.F.VHT), p8h.C_SCALENTF_LTF_VHT[p8h.BW.BW20.value], nSts)))
# vFb1 = p8h.procVhtChannelFeedback(ltfSym, p8h.BW.BW20, nSts, nRx)
# print("feedback v 1")
# for each in vFb1:
#     print(each)
# ltfSym = []
# ltfSym.append(p8h.procRemovePilots(p8h.procToneDescaling(p8h.procRmDcNonDataSc(p8h.procFftDemod(chan1[0:64]), p8h.F.VHT), p8h.C_SCALENTF_LTF_VHT[p8h.BW.BW20.value], nSts)))
# ltfSym.append(p8h.procRemovePilots(p8h.procToneDescaling(p8h.procRmDcNonDataSc(p8h.procFftDemod(chan1[64:128]), p8h.F.VHT), p8h.C_SCALENTF_LTF_VHT[p8h.BW.BW20.value], nSts)))
# # compute feedback
# vFb2 = p8h.procVhtChannelFeedback(ltfSym, p8h.BW.BW20, nSts, nRx)
# print("feedback v 2")
# for each in vFb2:
#     print(each)
# # combine the channel together
# bfH = []
# for k in range(0, len(vFb1)):
#     print("bfH", k)
#     bfH.append(np.concatenate((vFb1[k], vFb2[k]), axis=1))
#     print(bfH[k])
# # compute spatial matrix Q, ZF
# bfQ = []
# for k in range(0, len(vFb1)):
#     print("bfQ", k)
#     bfQ.append(np.matmul(bfH[k], np.linalg.inv(np.matmul(bfH[k].conjugate().T, bfH[k]))))
#     print(bfQ[k])
# # normalize Q
# bfQNormd = []
# for k in range(0, len(vFb1)):
#     bfQNormd.append(bfQ[k] / np.linalg.norm(bfQ[k]) * np.sqrt(nSts))
#     print("bfQNormd", k)
#     print(bfQNormd[k])
# # map Q to FFT non-zero sub carriers
# bfQForFft = [np.ones_like(bfQNormd[0])] * 3 + bfQNormd[0:28] + [
#     np.ones_like(bfQNormd[0])] + bfQNormd[28:56] + [np.ones_like(bfQNormd[0])] * 4
# bfQPktForGr0, bfQPktForGr1 = phy80211.genPktGrBfQ(bfQForFft)
# grSocket.sendto(bfQPktForGr0, phyTxAddr)
# time.sleep(0.5)
# grSocket.sendto(bfQPktForGr1, phyTxAddr)
# time.sleep(0.5)
# pkt0 = genMac80211UdpAmpduVht(["1234567 packet for station 000"])
# pkt1 = genMac80211UdpAmpduVht(["7654321 packet for station 111"])
# grMuPkt = phy80211.genPktGrDataMu(pkt0, p8h.modulation(p8h.F.VHT, 0, p8h.BW.BW20, 1, False), pkt1, p8h.modulation(p8h.F.VHT, 0, p8h.BW.BW20, 1, False), 2)
# print("gr pkt len %d" % len(grMuPkt))
# grSocket.sendto(grMuPkt, phyTxAddr)



        






    







