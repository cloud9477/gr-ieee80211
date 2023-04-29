import numpy as np
import struct
from matplotlib import pyplot as plt
import sys
import os
sys.path.append(os.path.join(sys.path[0], '../'))
import mac80211
import mac80211header as m8h
import phy80211header as p8h
import phy80211
import time

def genMac80211UdpMPDU(udpPayload):
    udpIns = mac80211.udp("10.10.0.6",  # sour ip
                        "10.10.0.1",  # dest ip
                        39379,  # sour port
                        8889)  # dest port
    udpPacket = udpIns.genPacket(bytearray(udpPayload, 'utf-8'))
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
            udpPacket = udpIns.genPacket(bytearray(eachUdpPayload, 'utf-8'))
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
    udpPayload200  = "123456789012345678901234567890abcdefghijklmnopqrst" * 4
    perfPktNum = 10
    perfSnrList = list(np.arange(0, 31, 1))
    perfSigAmp = 0.18750000
    perfSampNum = 0
    perfRes = []
    phy80211Ins = phy80211.phy80211(ifDebug=False)

    ssMultiList = []
    pkt = genMac80211UdpMPDU(udpPayload200)
    for mcsIter in range(0, 8):
        phy80211Ins.genFromMpdu(pkt, p8h.modulation(phyFormat=p8h.F.L, mcs=mcsIter, bw=p8h.BW.BW20, nSTS=1, shortGi=False))
        ssFinal = phy80211Ins.genFinalSig(multiplier = 12.0, cfoHz = 0.0, num = perfPktNum, gap = True, gapLen = 1600)
        ssMultiList.append(ssFinal)
    # for mcsIter in range(0, 8):
    #     phy80211Ins.genFromMpdu(pkt, p8h.modulation(phyFormat=p8h.F.HT, mcs=mcsIter, bw=p8h.BW.BW20, nSTS=1, shortGi=False))
    #     ssFinal = phy80211Ins.genFinalSig(multiplier = 12.0, cfoHz = 0.0, num = perfPktNum, gap = True, gapLen = 1600)
    #     ssMultiList.append(ssFinal)
    # pkts = genMac80211UdpAmpduVht([udpPayload200])
    # for mcsIter in range(0, 9):
    #     phy80211Ins.genFromAmpdu(pkts, p8h.modulation(phyFormat=p8h.F.VHT, mcs=mcsIter, bw=p8h.BW.BW20, nSTS=1, shortGi=False), vhtPartialAid=0, vhtGroupId=0)
    #     ssFinal = phy80211Ins.genFinalSig(multiplier = 12.0, cfoHz = 0.0, num = perfPktNum, gap = True, gapLen = 1600)
    #     ssMultiList.append(ssFinal)
    phy80211Ins.genMultiSigBinFile(ssMultiList, "/home/cloud/sdr/sig80211GenMultipleSiso", False)

    
    for snrIter in range(0, len(perfSnrList)):
        tmpNoiseAmp = np.sqrt((perfSigAmp**2)/(10.0**(perfSnrList[snrIter]/10.0)))
        os.system("python3 /home/cloud/sdr/gr-ieee80211/tools/performance/gr_siso.py " + str(tmpNoiseAmp) + " > /home/cloud/sdr/tmpSiso.txt &")
        tmpPreSize = 0
        tmpCurSize = 0
        while(True):
            time.sleep(1)
            tmpCurSize = os.path.getsize("/home/cloud/sdr/tmpSiso.txt")
            print("cur size %d, pre size %d" % (tmpCurSize, tmpPreSize))
            if(tmpPreSize == tmpCurSize):
                print("finish")
                break
            else:
                tmpPreSize = tmpCurSize
                print("continue")
        os.system('pkill -f gr_siso.py')

        resLine = open("/home/cloud/sdr/tmpSiso.txt").readlines()[-1]
        resItems = resLine.split(",")
        tmpRes = []
        for i in range(3, 3+len(resItems)-6):
            tmpRes.append(int(resItems[i].split(":")[1]))
        perfRes.append(tmpRes)
    
    print(perfRes)


    