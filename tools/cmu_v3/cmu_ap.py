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
    phyRxPort = 9527
    phyRxAddr = (phyRxIp, phyRxPort)
    phyTxIp = "127.0.0.1"
    phyTxPort = 9528
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
                            chan0B = tmpMacPaylaod[10:1034]
                            rxCount = rxCount | 1
                        if(pre5BStr == "cloudchan1" and (rxCount & 2)==0):
                            print(pre5BStr)
                            chan1B = tmpMacPaylaod[10:1034]
                            rxCount = rxCount | 2
        except:
            print("send ndp again")
            grSocket.sendto(grNdpPkt, phyTxAddr)

    if(len(chan0B) >= 1024 and len(chan1B) >= 1024):
        chan0 = []
        chan1 = []
        for i in range(0, 128):
            tmpR = struct.unpack('f', chan0B[i*8:i*8+4])[0]
            tmpI = struct.unpack('f', chan0B[i*8+4:i*8+8])[0]
            chan0.append(tmpR + tmpI * 1j)
            tmpR = struct.unpack('f', chan1B[i*8:i*8+4])[0]
            tmpI = struct.unpack('f', chan1B[i*8+4:i*8+8])[0]
            chan1.append(tmpR + tmpI * 1j)
        print(len(chan0))
        print(chan0)
        print(len(chan1))
        print(chan1)
        nSts = 2
        nRx = 1
        # compute feedback
        ltfSym = []
        ltfSym.append(p8h.procRemovePilots(p8h.procToneDescaling(p8h.procRmDcNonDataSc(p8h.procFftDemod(chan0[0:64]), p8h.F.VHT), p8h.C_SCALENTF_LTF_VHT[p8h.BW.BW20.value], nSts)))
        ltfSym.append(p8h.procRemovePilots(p8h.procToneDescaling(p8h.procRmDcNonDataSc(p8h.procFftDemod(chan0[64:128]), p8h.F.VHT), p8h.C_SCALENTF_LTF_VHT[p8h.BW.BW20.value], nSts)))
        vFb1 = p8h.procVhtChannelFeedback(ltfSym, p8h.BW.BW20, nSts, nRx)
        print("feedback v 1")
        for each in vFb1:
            print(each)
        ltfSym = []
        ltfSym.append(p8h.procRemovePilots(p8h.procToneDescaling(p8h.procRmDcNonDataSc(p8h.procFftDemod(chan1[0:64]), p8h.F.VHT), p8h.C_SCALENTF_LTF_VHT[p8h.BW.BW20.value], nSts)))
        ltfSym.append(p8h.procRemovePilots(p8h.procToneDescaling(p8h.procRmDcNonDataSc(p8h.procFftDemod(chan1[64:128]), p8h.F.VHT), p8h.C_SCALENTF_LTF_VHT[p8h.BW.BW20.value], nSts)))
        # compute feedback
        vFb2 = p8h.procVhtChannelFeedback(ltfSym, p8h.BW.BW20, nSts, nRx)
        print("feedback v 2")
        for each in vFb2:
            print(each)
        # combine the channel together
        bfH = []
        for k in range(0, len(vFb1)):
            print("bfH", k)
            bfH.append(np.concatenate((vFb1[k], vFb2[k]), axis=1))
            print(bfH[k])
        # compute spatial matrix Q, ZF
        bfQ = []
        for k in range(0, len(vFb1)):
            print("bfQ", k)
            bfQ.append(np.matmul(bfH[k], np.linalg.inv(np.matmul(bfH[k].conjugate().T, bfH[k]))))
            print(bfQ[k])
        # normalize Q
        bfQNormd = []
        for k in range(0, len(vFb1)):
            bfQNormd.append(bfQ[k] / np.linalg.norm(bfQ[k]) * np.sqrt(nSts))
            print("bfQNormd", k)
            print(bfQNormd[k])
        # map Q to FFT non-zero sub carriers
        bfQForFft = [np.ones_like(bfQNormd[0])] * 3 + bfQNormd[0:28] + [
            np.ones_like(bfQNormd[0])] + bfQNormd[28:56] + [np.ones_like(bfQNormd[0])] * 4

        bfQPktForGr = phy80211.genPktGrBfQ(bfQForFft)
        grSocket.sendto(bfQPktForGr, phyTxAddr)
        pkt0 = genMac80211UdpAmpduVht(["1234567 packet for station 000"])
        pkt1 = genMac80211UdpAmpduVht(["7654321 packet for station 111"])
        grMuPkt = phy80211.genPktGrDataMu(pkt0, p8h.modulation(p8h.F.VHT, 0, p8h.BW.BW20, 1, False), pkt1, p8h.modulation(p8h.F.VHT, 0, p8h.BW.BW20, 1, False), 2)
        print("gr pkt len %d" % len(grMuPkt))
        for i in range(0, 1000):
            grSocket.sendto(grMuPkt, phyTxAddr)
            time.sleep(0.01)

