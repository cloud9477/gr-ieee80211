import socket
import time
import sys
import os
sys.path.append(os.path.join(sys.path[0], '../'))
import mac80211
import phy80211
import phy80211header as p8h

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
                print("not support, return")
                return []
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

if __name__ == "__main__":
    # device info
    staID = 1
    print("cloud80211 Cloud MU-MIMO V3 station starts, station ID:" + str(staID))

    staMacIp = "127.0.0.1"
    staMacPort = 9527
    staPhyIp = "127.0.0.1"
    staPhyPort = 9528
    staMacSocket = socket.socket(family = socket.AF_INET, type = socket.SOCK_DGRAM)
    staMacSocket.bind((staMacIp, staMacPort))
    count = 0

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
                count += 1
                print("station NDP channel info recvd, gen channel packet %d" % (count))
                tmpChanPkt = ("cloudchan"+str(staID)).encode('utf-8') + rxPkt[3:1024 + 3]
                print(tmpChanPkt)
                pkts = genMac80211UdpAmpduVht([tmpChanPkt])
                grPkt = phy80211.genPktGrData(pkts, p8h.modulation(phyFormat=p8h.F.VHT, mcs = 1, bw=p8h.BW.BW20, nSTS=1, shortGi=False))
                time.sleep(staID * 0.001)
                staMacSocket.sendto(grPkt, (staPhyIp, staPhyPort))
                print("station channel packet sent")
        elif(tmpPktType == 2):
            print("received VHT packet")
            print(tmpPkt.hex())
            if(mac80211.procCheckCrc32(tmpPkt)):
                tmpType = int(tmpPkt[0])
                if(tmpType == 136):
                    # 0x88, QoS data
                    tmpMacPaylaod = tmpPkt[26:-4]
                    if(int(tmpMacPaylaod[0]) == 170 and int(tmpMacPaylaod[1]) == 170):
                        print("udp data packet")
                        print(tmpMacPaylaod.hex())
                        print(tmpMacPaylaod)
                    else:
                        print("other packets")
                else:
                    print("data packet")
                    tmpMacPaylaod = tmpPkt[24:-4]
                    print(tmpMacPaylaod.hex())
                