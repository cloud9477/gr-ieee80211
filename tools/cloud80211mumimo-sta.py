import socket
import mac80211
import phy80211header as p8h
from matplotlib import pyplot as plt
import numpy as np
import time
import struct

# device info
staNum = 0

# mac layer variables
staMacIp = "127.0.0.1"
staMacPort = 9527
staPhyIp = "127.0.0.1"
staPhyPort = 9528
staMacSocket = socket.socket(family = socket.AF_INET, type = socket.SOCK_DGRAM)
staMacSocket.bind((staMacIp, staMacPort))

# figure variables
plt.ion()
plt.rcParams["figure.figsize"] = (12,6)
macNdpChanFig = plt.figure()

fig_ax = macNdpChanFig.add_subplot(121)
fig_ax.set_ylim([-5, 5])
fig_ax_lineR, = fig_ax.plot(range(0,64), np.ones(64))
fig_ax_lineI, = fig_ax.plot(range(0,64), np.ones(64)*-1)

fig_bx = macNdpChanFig.add_subplot(122)
fig_bx.set_ylim([-5, 5])
fig_bx_lineR, = fig_bx.plot(range(0,64), np.ones(64))
fig_bx_lineI, = fig_bx.plot(range(0,64), np.ones(64)*-1)

""" figure update test """
# count = 0
# while(True):
#     count += 1
#     fig_ax_lineR.set_ydata(np.ones(64)*count)
#     fig_ax_lineI.set_ydata(np.ones(64)*count*-1)
#     fig_bx_lineR.set_ydata(np.ones(64)*count)
#     fig_bx_lineI.set_ydata(np.ones(64)*count*-1)
#     macNdpChanFig.canvas.draw()
#     macNdpChanFig.canvas.flush_events()

packetSeq = 0
while(True):
    rxMsg = staMacSocket.recvfrom(1500)
    rxPkt = rxMsg[0]
    rxAddr = rxMsg[1]
    tmpPktType = int(rxPkt[0])
    tmpPktLen = int(rxPkt[1]) + int(rxPkt[2]) * 256
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
            print("send channel info directly to AP")
            tmpHeaderInfo = "CHAN" + str(staNum)
            print(tmpHeaderInfo)
            tmpChanPkt = bytearray(tmpHeaderInfo,'utf-8') + rxPkt[3:1024+3]
            print("chan packet len", len(tmpChanPkt))
            tmpChanPktGrHeder = b'\x02\x00\x01' + struct.pack('<H',len(tmpChanPkt))
            print(tmpChanPktGrHeder.hex())

    elif(tmpPktType == 2):
        print("received VHT packet")





    







