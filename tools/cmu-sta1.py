import socket
import mac80211
import phy80211header as p8h
# from matplotlib import pyplot as plt
import numpy as np
import time
import struct

# device info
staID = 1
print("cloud80211 pyMac starts, station ID:" + str(staID))

staMacIp = "127.0.0.1"
staMacPort = 9527
staPhyIp = "127.0.0.1"
staPhyPort = 9528
staMacSocket = socket.socket(family = socket.AF_INET, type = socket.SOCK_DGRAM)
staMacSocket.bind((staMacIp, staMacPort))

packetSeq = 0
while(True):
    rxMsg = staMacSocket.recvfrom(1500)
    rxPkt = rxMsg[0]
    rxAddr = rxMsg[1]
    tmpPktType = int(rxPkt[0])
    tmpPktLen = int(rxPkt[1]) + int(rxPkt[2]) * 256
    tmpPkt = rxPkt[3:(3+tmpPktLen)]
    print(len(rxPkt), rxAddr, packetSeq, tmpPktType, tmpPktLen)
    if(tmpPktType == 20):
        if(tmpPktLen == 1024):
            print("station NDP channel info recvd")
            tmpChanPkt = rxPkt[3:1024 + 3]
            print("write chan into file" + str(staID))
            fWaveBin = open("/home/cloud/sdr/cmu-chan"+str(staID)+".bin", 'wb')
            fWaveBin.write(tmpChanPkt)
            fWaveBin.close()
            print("write chan into file done!!!!!!!!")

    elif(tmpPktType == 0):
        print("received legacy packet")
        print(tmpPkt.hex())
    
    elif(tmpPktType == 1):
        print("received HT packet")
        print(tmpPkt.hex())

    elif(tmpPktType == 2):
        print("received VHT packet")
        print(tmpPkt.hex())
        if(mac80211.procCheckCrc32(tmpPkt[:-4], tmpPkt[-4:])):
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

        






    







