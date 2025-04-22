from matplotlib import pyplot as plt
import multiprocessing
import numpy as np
import struct
import socket
import time
import sys
import os
sys.path.append(os.path.join(sys.path[0], '../'))
import mac80211
import phy80211header as p8h
import phy80211

"""
Cloud Multi-User Mimo AP (CMU)
"""

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

"""
1. Send NDP by GR
2. Fetch channel info
3. Generate Q (Zero-Forcing)
4. Write Q to GR
5. Send MU-MIMO by GR
"""

def macRxStationChannel(s, q):
    if(isinstance(s, type(socket.socket)) and isinstance(q, type(multiprocessing.Queue))):
        rxCount = 0
        while(rxCount != 3):
            rxMsg = s.recvfrom(65535)
            rxPkt = rxMsg[0]
            rxAddr = rxMsg[1]
            tmpPktType = int(rxPkt[0])
            tmpPktLen = int(rxPkt[1]) + int(rxPkt[2]) * 256
            tmpPkt = rxPkt[3:(3+tmpPktLen)]
            print(len(rxPkt), rxAddr, tmpPktType, tmpPktLen)
            if(tmpPktType == 2):
                if(mac80211.procCheckCrc32(tmpPkt[:-4], tmpPkt[-4:])):
                    tmpType = int(tmpPkt[0])
                    if(tmpType == 136):
                        # 0x88, QoS data
                        tmpMacPaylaod = tmpPkt[26:-4]
                        if(len(tmpMacPaylaod) > 10):
                            pre5BStr = str(tmpMacPaylaod[0:10], 'UTF-8')
                            if(pre5BStr == "cloudchan0" and (rxCount & 1)==0):
                                q.put(tmpMacPaylaod)
                                rxCount = rxCount | 1
                            if(pre5BStr == "cloudchan1" and (rxCount & 2)==0):
                                q.put(tmpMacPaylaod)
                                rxCount = rxCount | 2
                            

if __name__ == "__main__":
    # init the socket with gr
    phyRxIp = "127.0.0.1"
    phyRxPort = 9527 #9527
    phyRxAddr = (phyRxIp, phyRxPort)
    phyTxIp = "127.0.0.1"
    phyTxPort = 9528  #9528
    phyTxAddr = (phyTxIp, phyTxPort)
    grSocket = socket.socket(family = socket.AF_INET, type = socket.SOCK_DGRAM)
    grSocket.bind(phyRxAddr)

    # send ndp packet
    grNdpPkt = phy80211.genPktGrNdp()
    grSocket.sendto(grNdpPkt, phyTxAddr)

    chan0B = []
    chan1B = []
    rxCount = 0
    grSocket.settimeout(5.0)
    for mcsIter in range(0,9):
        while(rxCount != 3):
            print("go to receiving")
            try:
                rxMsg = grSocket.recvfrom(65535)
                rxPkt = rxMsg[0]
                rxAddr = rxMsg[1]
                tmpPktType = int(rxPkt[0])
                tmpPktLen = int(rxPkt[1]) + int(rxPkt[2]) * 256
                tmpPkt = rxPkt[3:]
                print(len(rxPkt), rxAddr, tmpPktType, tmpPktLen)
                if(tmpPktType == 2):
                    #if(mac80211.procCheckCrc32(tmpPkt)):
                    tmpType = int(tmpPkt[0])
                    if(tmpType == 136):
                        # 0x88, QoS data
                        tmpMacPaylaod = tmpPkt[26:-4]       # llc
                        tmpMacPaylaod = tmpMacPaylaod[8:]   # ip
                        tmpMacPaylaod = tmpMacPaylaod[20:]  # udp
                        tmpMacPaylaod = tmpMacPaylaod[8:]   # udp payload
                        if(len(tmpMacPaylaod) > 10):
                            pre5BStr = str(tmpMacPaylaod[0:10], 'UTF-8')
                            if(pre5BStr == "cloudchan0" and (rxCount & 1)==0):
                                print(pre5BStr)
                                chan0B = tmpMacPaylaod[10:906]
                                rxCount = rxCount | 1
                            if(pre5BStr == "cloudchan1" and (rxCount & 2)==0):
                                print(pre5BStr)
                                chan1B = tmpMacPaylaod[10:906]
                                rxCount = rxCount | 2
            except:
                print("send ndp again")
                grSocket.sendto(grNdpPkt, phyTxAddr)

        # extract feedback
        if len(chan0B) == 896 and len(chan1B) == 896: 
            chan0 = []
            chan1 = []
            vFb1 = []
            vFb2 = []
            nLenVfb = len(chan0B)//16
            nByteVfb = len(chan0B)//4
            chan0 = struct.unpack(f'{nByteVfb}f',chan0B)
            for i in range(nLenVfb):
                H11 = chan0[i*4 ] + chan0[i*4 +1] * 1j
                H21 = chan0[i*4+2] + chan0[i*4+3] * 1j
                tmpvFb1 = [np.array([[H11, H21]])]
                vFb1.append(tmpvFb1)

            chan1 = struct.unpack(f'{nByteVfb}f',chan1B)
            for i in range(nLenVfb):
                H12 = chan1[i*4 ] + chan1[i*4 +1] * 1j
                H22 = chan1[i*4+2] + chan1[i*4+3] * 1j
                tmpvFb2 = [np.array([[H12, H22]])]
                vFb2.append(tmpvFb2)

            # combine the channel together
            bfH = []
            for k in range(0, len(vFb1)):
                print("bfH", k)
                bfH.append(np.concatenate((vFb1[k], vFb2[k]), axis=1))
                print(bfH[k])

            # compute spatial matrix Q, ZF
            # and normalization
            bfQ = []
            for k in range(0, len(vFb1)):
                divid1 = np.sqrt((abs(bfH[k][0][0][1]))**2+(abs(bfH[k][0][0][0]))**2)
                divid2 = np.sqrt((abs(bfH[k][0][1][1]))**2+(abs(bfH[k][0][1][0]))**2) 
                bfQ.append([[-1*bfH[k][0][1][1]/divid2, -1*bfH[k][0][0][1]/divid1],[ bfH[k][0][1][0]/divid2 , bfH[k][0][0][0]/divid1 ]])
            
            bfQForFft = [np.ones_like(bfQ[0])] * 4 + bfQ[0:28] + [
                np.ones_like(bfQ[0])] + bfQ[28:56] + [np.ones_like(bfQ[0])] * 3

            bfQPktForGr = phy80211.genPktGrBfQ(bfQForFft)
            grSocket.sendto(bfQPktForGr, phyTxAddr)
            time.sleep(0.001)
            pkt0 = genMac80211UdpAmpduVht([" station 0 "+" mcs "+str(mcsIter)])
            pkt1 = genMac80211UdpAmpduVht([" station 1 "+" mcs "+str(mcsIter)])
            grMuPkt = phy80211.genPktGrDataMu(pkt0, p8h.modulation(p8h.F.VHT, mcsIter, p8h.BW.BW20, 1, False), pkt1, p8h.modulation(p8h.F.VHT, mcsIter, p8h.BW.BW20, 1, False), 2)
            for i in range(0, 10): #pkt send each  ndp
                grSocket.sendto(grMuPkt, phyTxAddr)
                time.sleep(0.001)
              
            # for each mcs sounding again    
            chan0B = []
            chan1B = []
            rxCount = 0

            print(f"mcs {mcsIter} done sending")

