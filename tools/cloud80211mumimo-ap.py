import socket
import mac80211
import phy80211header as p8h
from matplotlib import pyplot as plt
import numpy as np
import time
import struct

def genMac80211UdpMSDU(udpPayloadStr):
    udpIns = mac80211.udp("10.10.0.6",  # sour ip
                          "10.10.0.1",  # dest ip
                          39379,  # sour port
                          8889,  # dest port
                          bytearray(udpPayloadStr, 'utf-8'))  # bytes payload
    udpPacket = udpIns.genPacket()
    ipv4Ins = mac80211.ipv4(43778,  # identification
                            64,  # TTL
                            "10.10.0.6",
                            "10.10.0.1",
                            udpPacket)
    ipv4Packet = ipv4Ins.genPacket()
    llcIns = mac80211.llc()
    llcPacket = llcIns.genPacket() + ipv4Packet
    print("llc packet")
    print(llcPacket.hex())
    return llcPacket

# mumimo related
apMacAddress = '66:55:44:33:22:11'
staMacAddress0 = '66:55:44:33:22:20'
staMacAddress1 = '66:55:44:33:22:21'

staChanRcvFlag0 = 0
staChanRcvFlag1 = 0
staChan0 = []
staChan1 = []
udpPayload0 = "This is packet for station 000"
udpPayload1 = "This is packet for station 001"

# mac layer variables
apMacIp = "127.0.0.1"
apMacPort = 9527
apPhyIp = "127.0.0.1"
apPhyPort = 9528
apMacSocket = socket.socket(family = socket.AF_INET, type = socket.SOCK_DGRAM)
apMacSocket.bind((apMacIp, apMacPort))

# figure variables
# plt.ion()
# plt.rcParams["figure.figsize"] = (12,8)
# macNdpChanFig = plt.figure()

# fig_ax = macNdpChanFig.add_subplot(221)
# fig_ax.title.set_text('221 Channel user 0 ss 1 F')
# fig_ax.set_ylim([-5, 5])
# fig_ax_lineR, = fig_ax.plot(range(0,52), np.ones(52))
# fig_ax_lineI, = fig_ax.plot(range(0,52), np.ones(52)*-1)

# fig_bx = macNdpChanFig.add_subplot(222)
# fig_bx.title.set_text('222 Channel user 0 ss 2 F')
# fig_bx.set_ylim([-5, 5])
# fig_bx_lineR, = fig_bx.plot(range(0,52), np.ones(52))
# fig_bx_lineI, = fig_bx.plot(range(0,52), np.ones(52)*-1)

# fig_cx = macNdpChanFig.add_subplot(223)
# fig_cx.title.set_text('221 Channel user 1 ss 1 F')
# fig_cx.set_ylim([-5, 5])
# fig_cx_lineR, = fig_cx.plot(range(0,52), np.ones(52))
# fig_cx_lineI, = fig_cx.plot(range(0,52), np.ones(52)*-1)

# fig_dx = macNdpChanFig.add_subplot(224)
# fig_dx.title.set_text('222 Channel user 1 ss 2 F')
# fig_dx.set_ylim([-5, 5])
# fig_dx_lineR, = fig_dx.plot(range(0,52), np.ones(52))
# fig_dx_lineI, = fig_dx.plot(range(0,52), np.ones(52)*-1)

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
            print("mac packet type", tmpType)
            if(tmpType == 136):
                # 0x88, QoS data
                print("qos data")
                tmpMacPaylaod = tmpPkt[26:-4]
                print(tmpMacPaylaod[0:5].decode("utf-8"))
                if(int(tmpMacPaylaod[0]) == 170 and int(tmpMacPaylaod[1]) == 170):
                    print("udp data packet")
                elif(tmpMacPaylaod[0:5].decode("utf-8") == "CHAN0"):
                    print("channel info from station 0")
                    staChanRcvFlag0 = 1
                    staChan0 = []
                    tmpMacPaylaod = tmpMacPaylaod[5:]
                    for i in range(0, 128):
                        tmpR, = struct.unpack('<f', tmpMacPaylaod[i*8:i*8+4])
                        tmpI, = struct.unpack('<f', tmpMacPaylaod[i*8+4:i*8+8])
                        staChan0.append(tmpR + tmpI * 1j)
                elif(tmpMacPaylaod[0:5].decode("utf-8") == "CHAN1"):
                    print("channel info from station 1")
                    staChanRcvFlag1 = 1
                    staChan1 = []
                    tmpMacPaylaod = tmpMacPaylaod[5:]
                    for i in range(0, 128):
                        tmpR, = struct.unpack('<f', tmpMacPaylaod[i*8:i*8+4])
                        tmpI, = struct.unpack('<f', tmpMacPaylaod[i*8+4:i*8+8])
                        staChan1.append(tmpR + tmpI * 1j)
                else:
                    print("other packets")

                if(staChanRcvFlag0 and staChanRcvFlag1):
                    # plt.figure(91)
                    # plt.plot(np.real(staChan0[0:64]))
                    # plt.plot(np.imag(staChan0[0:64]))
                    # plt.figure(92)
                    # plt.plot(np.real(staChan0[64:128]))
                    # plt.plot(np.imag(staChan0[64:128]))
                    # plt.figure(93)
                    # plt.plot(np.real(staChan1[0:64]))
                    # plt.plot(np.imag(staChan1[0:64]))
                    # plt.figure(94)
                    # plt.plot(np.real(staChan1[64:128]))
                    # plt.plot(np.imag(staChan1[64:128]))

                    print("got channel of both stations, computing the bfQ")
                    nScDataPilot = 56
                    nSts = 2
                    nRx = 1
                    print(staChan0)
                    ltfSym = []
                    ltfSym.append(p8h.procRemovePilots(p8h.procFftDemod(staChan0[0:64], nScDataPilot, nSts)))
                    ltfSym.append(p8h.procRemovePilots(p8h.procFftDemod(staChan0[64:128], nScDataPilot, nSts)))
                    print(ltfSym)
                    # update channel figure
                    # fig_ax_lineR.set_ydata(np.real(ltfSym[0]))
                    # fig_ax_lineI.set_ydata(np.imag(ltfSym[0]))
                    # fig_bx_lineR.set_ydata(np.real(ltfSym[1]))
                    # fig_bx_lineI.set_ydata(np.imag(ltfSym[1]))
                    # compute feedback
                    vFb1 = p8h.procVhtChannelFeedback(ltfSym, nSts, nRx)
                    print("feedback v 1")
                    for each in vFb1:
                        print(each)
                    ltfSym = []
                    ltfSym.append(p8h.procRemovePilots(p8h.procFftDemod(staChan1[0:64], nScDataPilot, nSts)))
                    ltfSym.append(p8h.procRemovePilots(p8h.procFftDemod(staChan1[64:128], nScDataPilot, nSts)))
                    # plt.figure(121)
                    # plt.plot(np.real(staChan1[0:64]))
                    # plt.plot(np.imag(staChan1[0:64]))
                    # plt.figure(122)
                    # plt.plot(np.real(staChan1[64:128]))
                    # plt.plot(np.imag(staChan1[64:128]))
                    # update channel figure
                    # fig_cx_lineR.set_ydata(np.real(ltfSym[0]))
                    # fig_cx_lineI.set_ydata(np.imag(ltfSym[0]))
                    # fig_dx_lineR.set_ydata(np.real(ltfSym[1]))
                    # fig_dx_lineI.set_ydata(np.imag(ltfSym[1]))
                    # refresh figure
                    # macNdpChanFig.canvas.draw()
                    # macNdpChanFig.canvas.flush_events()
                    # compute feedback
                    vFb2 = p8h.procVhtChannelFeedback(ltfSym, nSts, nRx)
                    print("feedback v 2")
                    for each in vFb2:
                        print(each)
                    # combine the channel together
                    bfH = []
                    for k in range(0, nScDataPilot):
                        print("bfH", k)
                        bfH.append(np.concatenate((vFb1[k], vFb2[k]), axis=1))
                        print(bfH[k])
                    # plt.figure(111)
                    # plt.plot(np.real([each[0][0] for each in vFb1]))
                    # plt.plot(np.imag([each[0][0] for each in vFb1]))
                    # plt.figure(112)
                    # plt.plot(np.real([each[1][0] for each in vFb1]))
                    # plt.plot(np.imag([each[1][0] for each in vFb1]))
                    # plt.figure(113)
                    # plt.plot(np.real([each[0][0] for each in vFb2]))
                    # plt.plot(np.imag([each[0][0] for each in vFb2]))
                    # plt.figure(114)
                    # plt.plot(np.real([each[1][0] for each in vFb2]))
                    # plt.plot(np.imag([each[1][0] for each in vFb2]))

                    # compute spatial matrix Q, ZF
                    bfQTmp = []
                    for k in range(0, nScDataPilot):
                        print("bfQ", k)
                        bfQTmp.append(np.matmul(bfH[k], np.linalg.inv(np.matmul(bfH[k].conjugate().T, bfH[k]))))
                        print(bfQTmp[k])
                    # normalize Q
                    bfQForFftNormd = []
                    for k in range(0, nScDataPilot):
                        bfQForFftNormd.append(bfQTmp[k] / np.linalg.norm(bfQTmp[k]) * np.sqrt(nSts))
                        print("bfQNormd", k)
                        print(bfQForFftNormd[k])
                    # map Q to FFT non-zero sub carriers
                    bfQForFftNormdForFft = [np.ones_like(bfQForFftNormd[0])] * 3 + bfQForFftNormd[0:28] + [np.ones_like(bfQForFftNormd[0])] + bfQForFftNormd[28:56] + [np.ones_like(bfQForFftNormd[0])] * 4
                    
                    plt.figure(101)
                    plt.plot(np.real([each[0][0] for each in bfQForFftNormdForFft]))
                    plt.plot(np.imag([each[0][0] for each in bfQForFftNormdForFft]))
                    plt.figure(102)
                    plt.plot(np.real([each[0][1] for each in bfQForFftNormdForFft]))
                    plt.plot(np.imag([each[0][1] for each in bfQForFftNormdForFft]))
                    plt.figure(103)
                    plt.plot(np.real([each[1][0] for each in bfQForFftNormdForFft]))
                    plt.plot(np.imag([each[1][0] for each in bfQForFftNormdForFft]))
                    plt.figure(104)
                    plt.plot(np.real([each[1][1] for each in bfQForFftNormdForFft]))
                    plt.plot(np.imag([each[1][1] for each in bfQForFftNormdForFft]))
                    

                    # divide Q into real and imag and pass to gr
                    [tmpBfQBytesReal, tmpBfQBytesImag] = p8h.procBfQMatrixToBytes(bfQForFftNormdForFft, 2)
                    tmpBfQPkt = b'\x0a' + tmpBfQBytesReal
                    apMacSocket.sendto(tmpBfQPkt, (apPhyIp, apPhyPort))
                    print("bfQ real pkt passed to GR")
                    time.sleep(0.1)
                    tmpBfQPkt = b'\x0b' + tmpBfQBytesImag
                    apMacSocket.sendto(tmpBfQPkt, (apPhyIp, apPhyPort))
                    print("bfQ imag pkt passed to GR")

                    time.sleep(0.1)
                    print("sending mu-mimo packet to GR")
                    mac80211nIns0 = mac80211.mac80211(2,  # type
                                     8,  # sub type, 8 = QoS Data
                                     0,  # to DS, station to AP
                                     1,  # from DS, AP to station
                                     0,  # retry
                                     0,  # protected
                                     staMacAddress0,  # dest add
                                     apMacAddress,  # sour add
                                     staMacAddress0,  # recv add
                                     2704,  # sequence
                                     genMac80211UdpMSDU(udpPayload0), True)
                    mac80211Packet0 = mac80211nIns0.genPacket()
                    mac80211nIns1 = mac80211.mac80211(2,  # type
                                     8,  # sub type, 8 = QoS Data
                                     0,  # to DS, station to AP
                                     1,  # from DS, AP to station
                                     0,  # retry
                                     0,  # protected
                                     staMacAddress1,  # dest add
                                     apMacAddress,  # sour add
                                     staMacAddress1,  # recv add
                                     2704,  # sequence
                                     genMac80211UdpMSDU(udpPayload1), True)
                    mac80211Packet1 = mac80211nIns1.genPacket()
                    print("packet 0 len:", len(mac80211Packet0), " packet 1 len:", len(mac80211Packet1))
                    # vht mu-mimo packet: format 1B, mcs0 1B, nss0 1B, len0 2B, mcs1 1B, nss1 1B, len1 2B, groupID 1B.
                    # 3, 0, 1, 100, 0, 0, 1, 100, 0, 2,
                    tmpMuMimoPktGr = b'\x03\x00\x01\x64\x00\x00\x01\x64\x00\x02' + mac80211Packet0 + mac80211Packet1
                    apMacSocket.sendto(tmpMuMimoPktGr, (apPhyIp, apPhyPort))
                    print("bfQ imag pkt passed to GR")
                    plt.show()










    







