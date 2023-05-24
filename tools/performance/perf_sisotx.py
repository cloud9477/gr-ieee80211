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
from matplotlib import pyplot as plt
import numpy as np
import struct
import time
import sys
import os
sys.path.append(os.path.join(sys.path[0], '../'))
import mac80211
import phy80211
import phy80211header as p8h
import mac80211header as m8h


def genMac80211UdpMPDU(udpPayload):
    udpIns = mac80211.udp("10.10.0.6",  # sour ip
                        "10.10.0.1",  # dest ip
                        39379,  # sour port
                        8889)  # dest port
    if(isinstance(udpPayload, str)):
        udpPacket = udpIns.genPacket(bytearray(udpPayload, 'utf-8'))
    elif(isinstance(udpPayload, (bytes, bytearray))):
        udpPacket = udpIns.genPacket(udpPayload)
    else:
        udpPacket = b""
        print("genMac80211UdpAmpduVht packet element is not str or bytes")
        return b""
    ipv4Ins = mac80211.ipv4(43778,  # identification
                            64,  # TTL
                            "10.10.0.6",
                            "10.10.0.1")
    ipv4Packet = ipv4Ins.genPacket(udpPacket)
    llcIns = mac80211.llc()
    llcPacket = llcIns.genPacket(ipv4Packet)
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
    return mac80211Packet

def genMac80211UdpAmpduVht(udpPayloads):
    if(isinstance(udpPayloads, list)):
        macPkts = []
        for eachUdpPayload in udpPayloads:
            udpIns = mac80211.udp("10.10.0.6",  # sour ip
                                "10.10.0.1",  # dest ip
                                39379,  # sour port
                                8889)  # dest port
            if(isinstance(eachUdpPayload, str)):
                udpPacket = udpIns.genPacket(bytearray(eachUdpPayload, 'utf-8'))
            elif(isinstance(eachUdpPayload, (bytes, bytearray))):
                udpPacket = udpIns.genPacket(eachUdpPayload)
            else:
                udpPacket = b""
                print("genMac80211UdpAmpduVht packet element is not str or bytes")
                return b""
            ipv4Ins = mac80211.ipv4(43778,  # identification
                                    64,  # TTL
                                    "10.10.0.6",
                                    "10.10.0.1")
            ipv4Packet = ipv4Ins.genPacket(udpPacket)
            llcIns = mac80211.llc()
            llcPacket = llcIns.genPacket(ipv4Packet)
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
            macPkts.append(mac80211Packet)
        macAmpduVht = mac80211.genAmpduVHT(macPkts)
        return macAmpduVht
    else:
        print("genMac80211UdpAmpduVht udpPakcets is not list type")
        return b""

if __name__ == "__main__":
    print("cloud mac80211 example starts")

    pyToolPath = os.path.dirname(__file__)
    phyRxIp = "127.0.0.1"
    phyRxPort = 9527
    phyRxAddr = (phyRxIp, phyRxPort)
    phyTxIp = "127.0.0.1"
    phyTxPort = 9528
    phyTxAddr = (phyTxIp, phyTxPort)
    grSocket = socket.socket(family = socket.AF_INET, type = socket.SOCK_DGRAM)
    grSocket.bind(phyRxAddr)

    perfSnrList = list(np.arange(0, 30, 1))
    perfSigAmp = 0.18750000
    udpPayload  = bytearray(os.urandom(48))
    perfPktNum = 100
    os.system('pkill -f gr_sisotx.py')

    for snrIter in range(0, len(perfSnrList)):
        print("current snrIter %d of %d" % (snrIter, len(perfSnrList)))
        tmpNoiseAmp = np.sqrt((perfSigAmp**2)/(10.0**(perfSnrList[snrIter]/10.0)))
        os.system("python3 " + os.path.join(pyToolPath, "./gr_sisotx.py ") + str(tmpNoiseAmp) + " > " + os.path.join(pyToolPath, "../../tmp/tmpSisoTxPerf.txt") + " &")
        pkts = genMac80211UdpAmpduVht([udpPayload + bytes(str(snrIter).zfill(2), 'utf-8')])
        time.sleep(10)

        # sometimes usrp sink report error for the first packet
        pkt = genMac80211UdpMPDU(bytearray(os.urandom(50)))
        grPkt = phy80211.genPktGrData(pkt, p8h.modulation(phyFormat=p8h.F.L, mcs = 0, bw=p8h.BW.BW20, nSTS=1, shortGi=False))
        grSocket.sendto(grPkt, phyTxAddr)
        time.sleep(0.5)

        for mcsIter in range(0, 9):
            print("current mcsIter %d" % (mcsIter))
            grPkt = phy80211.genPktGrData(pkts, p8h.modulation(phyFormat=p8h.F.VHT, mcs = mcsIter, bw=p8h.BW.BW20, nSTS=1, shortGi=False))
            for i in range(0, perfPktNum):
                grSocket.sendto(grPkt, phyTxAddr)
                time.sleep(0.5)
        time.sleep(1)
        os.system('pkill -f gr_sisotx.py')

