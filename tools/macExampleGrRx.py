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
import phy80211header as p8h
import mac80211header as m8h

# device info
staID = 0
print("cloud mac80211 example starts")

phyRxIp = "127.0.0.1"
phyRxPort = 9527
phyRxSocket = socket.socket(family = socket.AF_INET, type = socket.SOCK_DGRAM)
phyRxSocket.bind((phyRxIp, phyRxPort))

packetSeq = 0
while(True):
    rxMsg = phyRxSocket.recvfrom(65535)     # lo max mtu
    rxPkt = rxMsg[0]
    rxAddr = rxMsg[1]
    tmpPktType = int(rxPkt[0])
    tmpPktLen = int(rxPkt[1]) + int(rxPkt[2]) * 256
    tmpPkt = rxPkt[3:(3+tmpPktLen)]
    # print(len(rxPkt), tmpPktLen+3, tmpPktLen, tmpPktType, packetSeq, rxAddr)
    if(tmpPktType == 20):
        print("cloud mac80211 NDP received, channel info saved")
        # tmpChanPkt = rxPkt[3:1024 + 3]
        # print("write chan into file")
        # fWaveBin = open("/home/cloud/sdr/cmu_chan.bin", 'wb')
        # fWaveBin.write(tmpChanPkt)
        # fWaveBin.close()
    elif(tmpPktType in [0, 1, 2]):
        # self defined phy format
        if(tmpPktType == 0):
            print("cloud mac80211 received legacy packet")
        elif(tmpPktType == 1):
            print("cloud mac80211 received HT packet")
        elif(tmpPktType == 2):
            print("cloud mac80211 received VHT packet")
        m8h.pktParser(tmpPkt[:-4])
    else:
        print("cloud mac80211 packet type error")
    print("--------------------------------------------------------")

        






    







