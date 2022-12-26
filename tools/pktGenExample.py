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
import random
import time
import mac80211
import phy80211header as p8h
import phy80211

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

if __name__ == "__main__":
    udpPayload  = "123456789012345678901234567890"
    udpPayload1 = "This is packet for station 001"
    udpPayload2 = "This is packet for station 002"
    udpPayload500  = "123456789012345678901234567890abcdefghijklmnopqrst" * 10

    phy80211Ins = phy80211.phy80211()
    """multiple packets of different formats concatenate SISO """
    ssMultiList = []
    # pkt = genMac80211UdpMPDU(udpPayload)
    # for mcsIter in range(0, 8):
    #     phy80211Ins.genFromMpdu(pkt, p8h.modulation(phyFormat=p8h.F.L, mcs=mcsIter, bw=p8h.BW.BW20, nSTS=1, shortGi=False))
    #     ssFinal = phy80211Ins.genFinalSig(multiplier = 18.0, cfoHz = 0.0, num = 10, gap = True, gapLen = 10000)
    #     ssMultiList.append(ssFinal)
    # for mcsIter in range(0, 8):
    #     phy80211Ins.genFromMpdu(pkt, p8h.modulation(phyFormat=p8h.F.HT, mcs=mcsIter, bw=p8h.BW.BW20, nSTS=1, shortGi=False))
    #     ssFinal = phy80211Ins.genFinalSig(multiplier = 18.0, cfoHz = 0.0, num = 10, gap = True, gapLen = 10000)
    #     ssMultiList.append(ssFinal)
    pkts = genMac80211UdpAmpduVht([udpPayload])
    for mcsIter in range(0, 9):
        phy80211Ins.genFromAmpdu(pkts, p8h.modulation(phyFormat=p8h.F.VHT, mcs=mcsIter, bw=p8h.BW.BW20, nSTS=1, shortGi=False), vhtPartialAid=0, vhtGroupId=0)
        ssFinal = phy80211Ins.genFinalSig(multiplier = 18.0, cfoHz = 0.0, num = 1, gap = True, gapLen = 10000)
        ssMultiList.append(ssFinal)
    phy80211Ins.genMultiSigBinFile(ssMultiList, "/home/cloud/sdr/sig80211GenMultipleSiso", True)

    """multiple packets of different formats concatenate MIMO """
    # ssMultiList = []
    # pkt = genMac80211UdpMPDU(udpPayload)
    # for mcsIter in range(8, 16):
    #     phy80211Ins.genFromMpdu(pkt, p8h.modulation(phyFormat=p8h.F.HT, mcs=mcsIter, bw=p8h.BW.BW20, nSTS=2, shortGi=False))
    #     ssFinal = phy80211Ins.genFinalSig(multiplier = 18.0, cfoHz = 0.0, num = 10, gap = True, gapLen = 10000)
    #     ssMultiList.append(ssFinal)
    # pkts = genMac80211UdpAmpduVht([udpPayload])
    # for mcsIter in range(0, 9):
    #     phy80211Ins.genFromAmpdu(pkts, p8h.modulation(phyFormat=p8h.F.VHT, mcs=mcsIter, bw=p8h.BW.BW20, nSTS=2, shortGi=False))
    #     ssFinal = phy80211Ins.genFinalSig(multiplier = 18.0, cfoHz = 0.0, num = 10, gap = True, gapLen = 10000)
    #     ssMultiList.append(ssFinal)
    # phy80211Ins.genMultiSigBinFile(ssMultiList, "/home/cloud/sdr/sig80211GenMultipleMimo", True)

    """multiple packets of different formats concatenate SISO AMPDU"""
    # ssMultiList = []
    # pkts = genMac80211UdpAmpduHt([udpPayload, udpPayload, udpPayload])
    # for mcsIter in range(0, 8):
    #     phy80211Ins.genFromAmpdu(pkts, p8h.modulation(phyFormat=p8h.F.HT, mcs=mcsIter, bw=p8h.BW.BW20, nSTS=1, shortGi=False))
    #     ssFinal = phy80211Ins.genFinalSig(multiplier = 18.0, cfoHz = 0.0, num = 10, gap = True, gapLen = 10000)
    #     ssMultiList.append(ssFinal)
    # pkts = genMac80211UdpAmpduVht([udpPayload, udpPayload, udpPayload])
    # for mcsIter in range(0, 9):
    #     phy80211Ins.genFromAmpdu(pkts, p8h.modulation(phyFormat=p8h.F.VHT, mcs=mcsIter, bw=p8h.BW.BW20, nSTS=1, shortGi=False))
    #     ssFinal = phy80211Ins.genFinalSig(multiplier = 18.0, cfoHz = 0.0, num = 10, gap = True, gapLen = 10000)
    #     ssMultiList.append(ssFinal)
    # phy80211Ins.genMultiSigBinFile(ssMultiList, "/home/cloud/sdr/sig80211GenMultipleSisoAmpdu", True)

    """VHT NDP 2x2"""
    # phy80211Ins.genFromAmpdu(b"", p8h.modulation(phyFormat=p8h.F.VHT, mcs=0, bw=p8h.BW.BW20, nSTS=2, shortGi=False), vhtPartialAid=0, vhtGroupId=0)
    # ssFinal = phy80211Ins.genFinalSig(multiplier = 18.0, cfoHz = 0.0, num = 10, gap = True, gapLen = 10000)
    # phy80211Ins.genSigBinFile(ssFinal, "/home/cloud/sdr/sig80211GenVhtNdp", True)
