import socket
import mac80211
import phy80211header as p8h
# from matplotlib import pyplot as plt
import numpy as np
import time
import struct

# some define
PKT_TYPE_L = 0
PKT_TYPE_HT = 1
PKT_TYPE_VHT = 2
PKT_TYPE_NDP = 20
DEF_FC_TYPE_MGMT = 0
DEF_FC_TYPE_CONTROL = 1
DEF_FC_TYPE_DATA = 2
DEF_FC_TYPE_EXTENSION = 3
DEF_FC_TYPE_STR = ["Management", "Control", "Data", "Extension"]
DEF_FC_SUBTYPE_MGMT_STR = ["Association Request", "Association Response", "Reassociation Request", "Reassociation Response", "Probe Request", "Probe Response", "Timing Advertisement", "Reserved", "Beacon", "ATIM", "Disassociation", "Authentication", "Deauthentication", "Action", "Action No Ack", "Reserved"]
DEF_FC_SUBTYPE_CONTROL_STR = ["Reserved", "Reserved", "Reserved", "Reserved", "Beamforming Report Poll", "VHT NDP Announcement", "Control Frame Extension", "Control Wrapper", "Block Ack Request (BlockAckReq)", "Block Ack (BlockAck)", "PS-Poll", "RTS", "CTS", "Ack", "CF-End", "CF-End +CF-Ack"]
DEF_FC_SUBTYPE_DATA_STR = ["Data", "Data +CF-Ack", "Data +CF-Poll", "Data +CF-Ack +CF-Poll", "Null (no data)", "CF-Ack (no data)", "CF-Poll (no data)", "CF-Ack +CF-Poll (no data)", "QoS Data", "QoS Data +CF-Ack", "QoS Data +CF-Poll", "QoS Data +CF-Ack +CF-Poll", "QoS Null (no data)", "Reserved", "QoS CF-Poll (no data)", "QoS CF-Ack +CF-Poll (no data)"]
DEF_FC_SUBTYPE_EXTENSION_STR = ["DMG Beacon"]
DEF_FC_SUBTYPE_MGMT_BEACON = 8

if __name__ == "__main__":
    # device info
    print("cloud80211, pyMacSniffer starts")
    snifferMacSocket = socket.socket(family = socket.AF_INET, type = socket.SOCK_DGRAM)
    snifferMacSocket.bind(("127.0.0.1", 9527))

    packetSeq = 0
    while(True):
        rxMsg = snifferMacSocket.recvfrom(1500)
        rxPkt = rxMsg[0]
        rxAddr = rxMsg[1]
        tmpPktType = int(rxPkt[0])
        tmpPktLen = int(rxPkt[1]) + int(rxPkt[2]) * 256
        tmpPkt = rxPkt[3:(3+tmpPktLen)]
        print(len(rxPkt), rxAddr, packetSeq, tmpPktType, tmpPktLen)
        packetSeq += 1
        if(tmpPktType == PKT_TYPE_NDP):
            if(tmpPktLen == 1024):
                print("cloud80211, sniffer: NDP channel info recvd, MU 2x2 station channel info")
                print("cloud80211, sniffer: nothing tbd")
        else:
            if(tmpPktType == PKT_TYPE_L):
                pass
                # print("cloud80211, sniffer: received legacy packet")
            elif(tmpPktType == PKT_TYPE_HT):
                pass
                # print("cloud80211, sniffer: received HT packet")
            elif(tmpPktType == PKT_TYPE_VHT):
                pass
                # print("cloud80211, sniffer: received VHT packet")
            else:
                print("cloud80211, sniffer: packet type error, return")
                continue

            # print("cloud80211, sniffer: packet bytes:")
            # print(tmpPkt.hex())

            if(mac80211.procCheckCrc32(tmpPkt[:-4], tmpPkt[-4:])):
                print("cloud80211, sniffer: CRC check pass")
            else:
                # print("cloud80211, sniffer: CRC check fail, return")
                continue
            
            # get the FC

            fc = struct.unpack('<H', tmpPkt[0:2])[0]
            print("cloud80211, sniffer: FC ", hex(fc))
            fc_protocalVer = fc & 3
            fc_type = (fc >> 2) & 3
            fc_subType = (fc >> 4) & 15
            fc_toDs = (fc >> 8) & 1
            fc_fromDs = (fc >> 9) & 1
            fc_moreFrag = (fc >> 10) & 1
            fc_retry = (fc >> 11) & 1
            fc_powerMgmt = (fc >> 12) & 1
            fc_moreData = (fc >> 13) & 1
            fc_protectFrame = (fc >> 14) & 1
            fc_htcOrder = (fc >> 15) & 1
            print("cloud80211, sniffer, FC Info protocol:%d, type:%d, sub type:%d, to DS:%d, from DS:%d, more frag:%d, retry:%d" % (fc_protocalVer, fc_type, fc_subType, fc_toDs, fc_fromDs, fc_moreFrag, fc_retry))
            if(fc_type == DEF_FC_TYPE_MGMT):
                print("cloud80211, sniffer, FC Info " + DEF_FC_TYPE_STR[fc_type] + ", " + DEF_FC_SUBTYPE_MGMT_STR[fc_subType])
            elif(fc_type == DEF_FC_TYPE_CONTROL):
                print("cloud80211, sniffer, FC Info " + DEF_FC_TYPE_STR[fc_type] + ", " + DEF_FC_SUBTYPE_CONTROL_STR[fc_subType])
            elif(fc_type == DEF_FC_TYPE_DATA):
                print("cloud80211, sniffer, FC Info " + DEF_FC_TYPE_STR[fc_type] + ", " + DEF_FC_SUBTYPE_DATA_STR[fc_subType])
            elif(fc_type == DEF_FC_TYPE_EXTENSION):
                print("cloud80211, sniffer, FC Info " + DEF_FC_TYPE_STR[fc_type] + ", " + DEF_FC_SUBTYPE_EXTENSION_STR[fc_subType])
            else:
                print("cloud80211, sniffer, FC type error")
            
            hdr_duration = struct.unpack('<H', tmpPkt[2:4])[0]
            hdr_add1 = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",tmpPkt[4:10])
            hdr_add2 = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",tmpPkt[10:16])
            hdr_add3 = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",tmpPkt[16:22])
            hdr_seq = struct.unpack('<H', tmpPkt[22:24])[0]
            hdr_qos = 0
            if(fc_type == DEF_FC_TYPE_DATA and fc_subType >= 8):
                hdr_qos = struct.unpack('<H', tmpPkt[24:26])[0]
            print("cloud80211, sniffer, header ")

            if(fc_type == DEF_FC_TYPE_MGMT):
                if(fc_subType == DEF_FC_SUBTYPE_MGMT_BEACON):
                    print("cloud80211, sniffer, beacon packet")
                    bc_ts = struct.unpack('<Q', tmpPkt[24:32])[0]


            
