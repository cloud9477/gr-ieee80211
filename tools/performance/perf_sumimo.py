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
# from presiso import presiso  # grc-generated hier_block


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

def testSnrPdrSuMimo(pktFormat, nMcs, listSnr, ampSig):
    pyToolPath = os.path.dirname(__file__)
    tmpPerfRes = []
    for snrIter in range(0, len(listSnr)):
        print("current snrIter %d of %d" % (snrIter, len(listSnr)))
        tmpNoiseAmp = np.sqrt((ampSig**2)/(10.0**(listSnr[snrIter]/10.0)))
        os.system("python3 " + os.path.join(pyToolPath, "./gr_sumimo.py ") + str(tmpNoiseAmp) + " > " + os.path.join(pyToolPath, "../../tmp/tmpSuMimoPerf.txt") + " &")
        tmpPreSize = 0
        tmpCurSize = 0
        while(True):
            time.sleep(1)
            tmpCurSize = os.path.getsize(os.path.join(pyToolPath, "../../tmp/tmpSuMimoPerf.txt"))
            if(tmpPreSize == tmpCurSize):
                break
            else:
                tmpPreSize = tmpCurSize
        os.system('pkill -f gr_sumimo.py')

        resFile = open(os.path.join(pyToolPath, "../../tmp/tmpSuMimoPerf.txt")).readlines()
        resFile.reverse()
        resLine = ""
        for each in resFile:
            if("crc32" in each and pktFormat in each):
                resLine = each
                break
        
        if(len(resLine)):
            resItems = resLine.split(",")
            tmpRes = []
            for i in range(3, 3+nMcs):
                tmpRes.append(int(resItems[i].split(":")[1]))
            tmpPerfRes.append(tmpRes)
        else:
            tmpPerfRes.append([0] * nMcs)
    return tmpPerfRes

if __name__ == "__main__":
    pyToolPath = os.path.dirname(__file__)
    udpPayload200  = "123456789012345678901234567890abcdefghijklmnopqrst" * 4
    udpPayload500R = bytearray(os.urandom(500))
    perfPktNum = 100
    perfSnrList = list(np.arange(0, 31, 1))
    perfSigAmp = 0.18750000
    phy80211Ins = phy80211.phy80211(ifDebug=False)

    pkt = genMac80211UdpMPDU(udpPayload500R)
    pkts = genMac80211UdpAmpduVht([udpPayload500R])

    ssMultiList = []
    for mcsIter in range(8, 16):
        phy80211Ins.genFromMpdu(pkt, p8h.modulation(phyFormat=p8h.F.HT, mcs=mcsIter, bw=p8h.BW.BW20, nSTS=2, shortGi=False))
        ssFinal = phy80211Ins.genFinalSig(multiplier = 12.0 * np.sqrt(2), cfoHz = 0.0, num = perfPktNum, gap = True, gapLen = 1200)
        ssMultiList.append(ssFinal)
    phy80211Ins.genMultiSigBinFile(ssMultiList, os.path.join(pyToolPath, "../../tmp/sig80211GenMultipleMimo"), False)
    htPerfRes = testSnrPdrSuMimo("ht", 8, perfSnrList, perfSigAmp)
    
    ssMultiList = []
    for mcsIter in range(0, 9):
        phy80211Ins.genFromAmpdu(pkts, p8h.modulation(phyFormat=p8h.F.VHT, mcs=mcsIter, bw=p8h.BW.BW20, nSTS=2, shortGi=False))
        ssFinal = phy80211Ins.genFinalSig(multiplier = 12.0 * np.sqrt(2), cfoHz = 0.0, num = perfPktNum, gap = True, gapLen = 1200)
        ssMultiList.append(ssFinal)
    phy80211Ins.genMultiSigBinFile(ssMultiList, os.path.join(pyToolPath, "../../tmp/sig80211GenMultipleMimo"), False)
    vhtPerfRes = testSnrPdrSuMimo("vht", 10, perfSnrList, perfSigAmp)

    print(htPerfRes)
    print(vhtPerfRes)

    widths = [8]
    heights = [4,4]
    pltFig = plt.figure(figsize=(8,8))
    spec = pltFig.add_gridspec(ncols=1, nrows=2, width_ratios=widths,
                            height_ratios=heights, wspace=0.3, hspace=0.4)
    bx = pltFig.add_subplot(spec[0, 0])
    for i in range(0, 8):
        bx.plot([each[i] for each in htPerfRes])
    cx = pltFig.add_subplot(spec[1, 0])
    for i in range(0, 9):
        cx.plot([each[i] for each in vhtPerfRes])
    plt.show()



    