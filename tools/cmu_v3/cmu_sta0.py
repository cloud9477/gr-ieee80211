import socket
import time
import sys
import os
sys.path.append(os.path.join(sys.path[0], '../'))
import mac80211
import phy80211
import phy80211header as p8h

def genMac80211QosDataAmpduVht(macPayloads):
    if(isinstance(macPayloads, (bytes, bytearray))):
        macPkts = []
        for eachMacPayload in macPayloads:
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
            mac80211Packet = mac80211Ins.genPacket(eachMacPayload)
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

if __name__ == "__main__":
    # device info
    staID = 0
    print("cloud80211 Cloud MU-MIMO V3 station starts, station ID:" + str(staID))

    staMacIp = "127.0.0.1"
    staMacPort = 9527
    staPhyIp = "127.0.0.1"
    staPhyPort = 9528
    staMacSocket = socket.socket(family = socket.AF_INET, type = socket.SOCK_DGRAM)
    staMacSocket.bind((staMacIp, staMacPort))

    while(True):
        rxMsg = staMacSocket.recvfrom(1500)
        rxPkt = rxMsg[0]
        rxAddr = rxMsg[1]
        tmpPktType = int(rxPkt[0])
        tmpPktLen = int(rxPkt[1]) + int(rxPkt[2]) * 256
        tmpPkt = rxPkt[3:(3+tmpPktLen)]
        print(len(rxPkt), rxAddr, tmpPktType, tmpPktLen)
        if(tmpPktType == 20):
            if(tmpPktLen == 1024):
                print("station NDP channel info recvd, gen channel packet")
                tmpChanPkt = ("cloudchan"+str(staID)).encode('utf-8') + rxPkt[3:1024 + 3]
                grPkt = phy80211.genPktGrData(tmpChanPkt, p8h.modulation(p8h.F.VHT, 0, p8h.BW.BW20, 1, False))
                time.sleep(staID)
                staMacSocket.sendto(grPkt, (staPhyIp, staPhyPort))
                print("station channel packet sent")
