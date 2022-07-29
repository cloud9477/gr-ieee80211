import socket
import mac80211
import phy80211header as p8h
from matplotlib import pyplot as plt
import numpy as np
import time
import struct

# mac layer variables
apMacIp = "127.0.0.1"
apMacPort = 9527
apPhyIp = "127.0.0.1"
apPhyPort = 9528
apMacSocket = socket.socket(family = socket.AF_INET, type = socket.SOCK_DGRAM)
apMacSocket.bind((apMacIp, apMacPort))

# figure variables
plt.ion()
plt.rcParams["figure.figsize"] = (12,8)
macNdpChanFig = plt.figure()

fig_ax = macNdpChanFig.add_subplot(221)
fig_ax.title.set_text('221 Channel user 0 ss 1 F')
fig_ax.set_ylim([-5, 5])
fig_ax_lineR, = fig_ax.plot(range(0,64), np.ones(64))
fig_ax_lineI, = fig_ax.plot(range(0,64), np.ones(64)*-1)

fig_bx = macNdpChanFig.add_subplot(222)
fig_bx.title.set_text('222 Channel user 0 ss 2 F')
fig_bx.set_ylim([-5, 5])
fig_bx_lineR, = fig_bx.plot(range(0,64), np.ones(64))
fig_bx_lineI, = fig_bx.plot(range(0,64), np.ones(64)*-1)

fig_cx = macNdpChanFig.add_subplot(223)
fig_cx.title.set_text('221 Channel user 1 ss 1 F')
fig_cx.set_ylim([-5, 5])
fig_cx_lineR, = fig_cx.plot(range(0,64), np.ones(64))
fig_cx_lineI, = fig_cx.plot(range(0,64), np.ones(64)*-1)

fig_dx = macNdpChanFig.add_subplot(224)
fig_dx.title.set_text('222 Channel user 1 ss 2 F')
fig_dx.set_ylim([-5, 5])
fig_dx_lineR, = fig_dx.plot(range(0,64), np.ones(64))
fig_dx_lineI, = fig_dx.plot(range(0,64), np.ones(64)*-1)

packetSeq = 0
while(True):
    rxMsg = apMacSocket.recvfrom(1500)
    rxPkt = rxMsg[0]
    rxAddr = rxMsg[1]
    tmpPktType = int(rxPkt[0])
    tmpPktLen = int(rxPkt[1]) + int(rxPkt[2]) * 256
    tmpPkt = rxPkt[3:(3+tmpPktLen)]
    print(len(rxPkt), rxAddr, packetSeq, tmpPktType, tmpPktLen)
    if(tmpPktType == 20):
        if(tmpPktLen == 1024):
            print("station NDP channel info recvd")
            tmpNdpChan2x1T = []
            for i in range(0, 128):
                tmpR, = struct.unpack('<f', rxPkt[3+i*8:3+i*8+4])
                tmpI, = struct.unpack('<f', rxPkt[3+i*8+4:3+i*8+8])
                # print(tmpR, tmpI)
                tmpNdpChan2x1T.append(tmpR + tmpI * 1j)
            fig_ax_lineR.set_ydata(np.real(tmpNdpChan2x1T[0:64]))
            fig_ax_lineI.set_ydata(np.imag(tmpNdpChan2x1T[0:64]))
            fig_bx_lineR.set_ydata(np.real(tmpNdpChan2x1T[64:128]))
            fig_bx_lineI.set_ydata(np.imag(tmpNdpChan2x1T[64:128]))
            macNdpChanFig.canvas.draw()
            macNdpChanFig.canvas.flush_events()
            print("compute beamforming matrix Q")


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
                elif(tmpMacPaylaod[0:5].decode("utf-8") == "CHAN0"):
                    print("channel info from station 0")
                elif(tmpMacPaylaod[0:5].decode("utf-8") == "CHAN1"):
                    print("channel info from station 1")
                else:
                    print("other packets")






    







