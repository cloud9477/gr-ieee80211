import socket
import mac80211
import phy80211header as p8h
# from matplotlib import pyplot as plt
import numpy as np
import time
import struct

# device info
staID = 1
staChanRespTimeGap = 1

# mac layer variables
apMacAddress = '66:55:44:33:22:11'
staMacAddress = '66:55:44:33:22:2' + str(staID)
print("cloud80211 pyMac starts, station ID:" + str(staID) + ", mac address " + staMacAddress)

staMacIp = "127.0.0.1"
staMacPort = 9527
staPhyIp = "127.0.0.1"
staPhyPort = 9528
staMacSocket = socket.socket(family = socket.AF_INET, type = socket.SOCK_DGRAM)
staMacSocket.bind((staMacIp, staMacPort))

# figure variables
# plt.ion()
# plt.rcParams["figure.figsize"] = (12,6)
# macNdpChanFig = plt.figure()

# fig_ax = macNdpChanFig.add_subplot(121)
# fig_ax.title.set_text('Channel Sounding LTF 0 T')
# fig_ax.set_ylim([-5, 5])
# fig_ax_lineR, = fig_ax.plot(range(0,64), np.ones(64))
# fig_ax_lineI, = fig_ax.plot(range(0,64), np.ones(64)*-1)
#
# fig_bx = macNdpChanFig.add_subplot(122)
# fig_bx.title.set_text('Channel Sounding LTF 1 T')
# fig_bx.set_ylim([-5, 5])
# fig_bx_lineR, = fig_bx.plot(range(0,64), np.ones(64))
# fig_bx_lineI, = fig_bx.plot(range(0,64), np.ones(64)*-1)

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
            # tmpNdpChan2x1T = []
            # for i in range(0, 128):
            #     tmpR, = struct.unpack('<f', rxPkt[3+i*8:3+i*8+4])
            #     tmpI, = struct.unpack('<f', rxPkt[3+i*8+4:3+i*8+8])
            #     # print(tmpR, tmpI)
            #     tmpNdpChan2x1T.append(tmpR + tmpI * 1j)
            # fig_ax_lineR.set_ydata(np.real(tmpNdpChan2x1T[0:64]))
            # fig_ax_lineI.set_ydata(np.imag(tmpNdpChan2x1T[0:64]))
            # fig_bx_lineR.set_ydata(np.real(tmpNdpChan2x1T[64:128]))
            # fig_bx_lineI.set_ydata(np.imag(tmpNdpChan2x1T[64:128]))
            # macNdpChanFig.canvas.draw()
            # macNdpChanFig.canvas.flush_events()
            # print("send channel info directly to AP")
            # tmpHeaderInfo = "CHAN" + str(staID)
            # print("header", tmpHeaderInfo)
            # tmpChanPkt = bytearray(tmpHeaderInfo,'utf-8') + rxPkt[3:1024+3]
            # print("chan packet len", len(tmpChanPkt))
            tmpChanPkt = rxPkt[3:1024 + 3]
            print("write chan into file" + str(staID))
            fWaveBin = open("/home/cloud/sdr/cmu-chan"+str(staID)+".bin", 'wb')
            fWaveBin.write(tmpChanPkt)
            fWaveBin.close()
            print("write chan into file done!!!!!!!!")
            # mac80211nIns = mac80211.mac80211(2,  # type
            #                          8,  # sub type, 8 = QoS Data
            #                          1,  # to DS, station to AP
            #                          0,  # from DS
            #                          0,  # retry
            #                          0,  # protected
            #                          apMacAddress,  # dest add
            #                          staMacAddress,  # sour add
            #                          apMacAddress,  # recv add
            #                          2704,  # sequence
            #                          tmpChanPkt, True)
            # mac80211Packet = mac80211nIns.genPacket()
            #
            # tmpChanPktGrHeder = b'\x02\x00\x01' + struct.pack('<H',len(mac80211Packet))
            # # print(tmpChanPktGrHeder.hex())
            # tmpChanPktGr = tmpChanPktGrHeder + mac80211Packet
            # # print(tmpChanPktGr.hex())
            # print("channel pkt response time gap, user ", staID, ", gap time ", (staID*staChanRespTimeGap))
            # time.sleep(staID * staChanRespTimeGap)
            # staMacSocket.sendto(tmpChanPktGr, (staPhyIp, staPhyPort))
            # print("channel pkt passed to GR")

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
                else:
                    print("other packets")

        






    







