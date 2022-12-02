import socket
import mac80211
import phy80211header as p8h
# from matplotlib import pyplot as plt
import numpy as np
import time
import struct

# device info
staID = 0
print("cloud mac80211 example starts")

phyRxIp = "127.0.0.2"
phyRxPort = 9527
phyTxIp = "127.0.0.2"
phyTxPort = 9528
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
    print(len(rxPkt), rxAddr, packetSeq, tmpPktType, tmpPktLen)
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
        
        if(mac80211.procCheckCrc32(tmpPkt[:-4], tmpPkt[-4:])):
            pass
        else:
            print("cloud mac80211 crc32 fail")
    else:
        print("cloud mac80211 packet type error")

        






    







