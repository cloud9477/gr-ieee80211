from matplotlib import pyplot as plt
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
    time.sleep(1.0)
    grSocket.sendto(grNdpPkt, phyTxAddr)
    time.sleep(1.0)
    # wait for some time
    print("GR NDP Pakcet Sent")
    
    # fetch the channel info through ethernet, use the tool "pscp" with user name and passwd, you could replace with other methods
    os.system("pscp -pw 7777 -r cloud@192.168.10.68:/home/cloud/sdr/cmu_chan0.bin /home/cloud/sdr/")
    os.system("pscp -pw 7777 -r cloud@192.168.10.50:/home/cloud/sdr/cmu_chan1.bin /home/cloud/sdr/")

    # physical instance
    phy80211Ins = phy80211.phy80211()
    # read channel bin file
    # they are the vht long training field received at each station
    chan0 = []
    chan1 = []
    fWaveComp = open("/home/cloud/sdr/cmu_chan0.bin", 'rb')
    for i in range(0,128):
        tmpR = struct.unpack('f', fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1))[0]
        tmpI = struct.unpack('f', fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1))[0]
        chan0.append(tmpR + tmpI * 1j)
    fWaveComp.close()
    fWaveComp = open("/home/cloud/sdr/cmu_chan1.bin", 'rb')
    for i in range(0, 128):
        tmpR = struct.unpack('f', fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1))[0]
        tmpI = struct.unpack('f', fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1))[0]
        chan1.append(tmpR + tmpI * 1j)
    fWaveComp.close()

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
    for i in range(0, 10):
        grSocket.sendto(grMuPkt, phyTxAddr)
        time.sleep(0.01)

    # plt.figure(11)
    # plt.plot(np.real([each[0][0] for each in bfQForFft]))
    # plt.plot(np.imag([each[0][0] for each in bfQForFft]))
    # plt.figure(12)
    # plt.plot(np.real([each[0][1] for each in bfQForFft]))
    # plt.plot(np.imag([each[0][1] for each in bfQForFft]))
    # plt.figure(13)
    # plt.plot(np.real([each[1][0] for each in bfQForFft]))
    # plt.plot(np.imag([each[1][0] for each in bfQForFft]))
    # plt.figure(14)
    # plt.plot(np.real([each[1][1] for each in bfQForFft]))
    # plt.plot(np.imag([each[1][1] for each in bfQForFft]))

    """ genrate the mu-mimo pakcet by python by not gr """
    # phy80211Ins.genAmpduMu(nUser = 2, bfQ = bfQForFft, groupId = 2, ampdu0=pkt0, mod0=p8h.modulation(p8h.F.VHT, 0, p8h.BW.BW20, 1, False), ampdu1=pkt1, mod1=p8h.modulation(p8h.F.VHT, 0, p8h.BW.BW20, 1, False))
    # ssFinal = phy80211Ins.genFinalSig(multiplier = 18.0, cfoHz = 0.0, num = 1, gap = False, gapLen = 10000)
    # phy80211Ins.genSigBinFile(ssFinal, "/home/cloud/sdr/cmu_mu", True)

