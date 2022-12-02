import socket
import mac80211
import phy80211header as p8h
from matplotlib import pyplot as plt
import numpy as np
import time
import struct
from enum import Enum

class FC_TPYE(Enum):
    MGMT = 0
    CTRL = 1
    DATA = 2
    EXT = 3

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class FC_SUBTPYE_MGMT(Enum):
    ASSOREQ = 0
    ASSORES = 1
    REASSOREQ = 2
    REASSORES = 3
    PROBEREQ = 4
    PROBERES = 5
    TIMINGAD = 6
    RESERVED7 = 7
    BEACON = 8
    ATIM = 9
    DISASSO = 10
    AUTH = 11
    DEAUTH = 12
    ACT = 13
    ACTNOACK = 14
    RESERVED15 = 15

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class FC_SUBTPYE_CTRL(Enum):
    RESERVED0 = 0
    RESERVED1 = 1
    RESERVED2 = 2
    RESERVED3 = 3
    BFREPOPOLL = 4
    VHTNDPANNO = 5
    FRAMEEXT = 6
    WRAPPER = 7
    BLOCKACKREQ = 8
    BLOCKACK = 9
    PSPOLL = 10
    RTS = 11
    CTS = 12
    ACK = 13
    CFEND = 14
    CFENDCFACK = 15

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class FC_SUBTPYE_DATA(Enum):
    DATA = 0
    DATACFACK = 1
    DATACFPOLL = 2
    DATACFACKCFPOLL = 3
    NULL = 4
    CFACK = 5
    CFPOLL = 6
    CFACKCFPOLL = 7
    QOSDATA = 8
    QOSDATACFACK = 9
    QOSDATACFPOLL = 10
    QOSDATACFACKCFPOLL = 11
    QOSNULL = 12
    RESERVED13 = 13
    QOSCFPOLL = 14
    QOSCFACKCFPOLL = 15

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class FC_SUBTPYE_EXT(Enum):
    DMGBEACON = 0
    RESERVED1 = 1
    RESERVED2 = 2
    RESERVED3 = 3
    RESERVED4 = 4
    RESERVED5 = 5
    RESERVED6 = 6
    RESERVED7 = 7
    RESERVED8 = 8
    RESERVED9 = 9
    RESERVED10 = 10
    RESERVED11 = 11
    RESERVED12 = 12
    RESERVED13 = 13
    RESERVED14 = 14
    RESERVED15 = 15

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class MGMT_ELE(Enum):
    SSID = 0
    SUPOTRATE = 1
    DSSSPARAM = 3
    TIM = 5
    COUNTRY = 7
    BSSLOAD = 11
    HTCAP = 45
    RSN = 48
    HTOPS = 61
    ANTENNA = 64
    RMENABLED = 70
    EXTCAP = 127
    VHTCAP = 191
    VHTOPS = 192
    TXPOWER = 195
    VENDOR = 221

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

# some define
C_FC_SUBTYPE_MGMT_STR = ["Association Request", "Association Response", "Reassociation Request", "Reassociation Response", "Probe Request", "Probe Response", "Timing Advertisement", "Reserved", "Beacon", "ATIM", "Disassociation", "Authentication", "Deauthentication", "Action", "Action No Ack", "Reserved"]
C_FC_SUBTYPE_CTRL_STR = ["Reserved", "Reserved", "Reserved", "Reserved", "Beamforming Report Poll", "VHT NDP Announcement", "Control Frame Extension", "Control Wrapper", "Block Ack Request (BlockAckReq)", "Block Ack (BlockAck)", "PS-Poll", "RTS", "CTS", "Ack", "CF-End", "CF-End +CF-Ack"]
C_FC_SUBTYPE_DATA_STR = ["Data", "Data +CF-Ack", "Data +CF-Poll", "Data +CF-Ack +CF-Poll", "Null (no data)", "CF-Ack (no data)", "CF-Poll (no data)", "CF-Ack +CF-Poll (no data)", "QoS Data", "QoS Data +CF-Ack", "QoS Data +CF-Poll", "QoS Data +CF-Ack +CF-Poll", "QoS Null (no data)", "Reserved", "QoS CF-Poll (no data)", "QoS CF-Ack +CF-Poll (no data)"]
C_FC_SUBTYPE_EXT_STR = ["DMG Beacon", "Reserved", "Reserved", "Reserved", "Reserved","Reserved", "Reserved", "Reserved", "Reserved","Reserved", "Reserved", "Reserved", "Reserved", "Reserved", "Reserved", "Reserved"]

class frameControl:
    def __init__(self, fc):
        print("cloud mac80211header fc, input 16bit hex: %s", hex(fc))
        self.protocalVer = fc & 3
        self.type = FC_TPYE((fc >> 2) & 3)
        if(self.type == FC_TPYE.MGMT):
            self.subType = FC_SUBTPYE_MGMT((fc >> 4) & 15)
        elif(self.type == FC_TPYE.CTRL):
            self.subType = FC_SUBTPYE_CTRL((fc >> 4) & 15)
        elif(self.type == FC_TPYE.DATA):
            self.subType = FC_SUBTPYE_DATA((fc >> 4) & 15)
        elif(self.type == FC_TPYE.EXT):
            self.subType = FC_SUBTPYE_EXT((fc >> 4) & 15)
        else:
            self.subType = str((fc >> 4) & 15)
        self.toDs = (fc >> 8) & 1
        self.fromDs = (fc >> 9) & 1
        self.moreFrag = (fc >> 10) & 1
        self.retry = (fc >> 11) & 1
        self.powerMgmt = (fc >> 12) & 1
        self.moreData = (fc >> 13) & 1
        self.protectFrame = (fc >> 14) & 1
        self.htcOrder = (fc >> 15) & 1      # in non qos data set 1 for order, in qos data set 1 for htc
    
    def printInfo(self):
        print("cloud mac80211header, FC Info protocol:%d, type:%s, sub type:%s, to DS:%d, from DS:%d, more frag:%d, retry:%d" % (self.protocalVer, self.type, self.subType, self.toDs, self.fromDs, self.moreFrag, self.retry))
        if(self.type == FC_TPYE.MGMT):
            print("cloud mac80211header, FC Info %s, %s" % (self.type, C_FC_SUBTYPE_MGMT_STR[self.subType.value]))
        elif(self.type == FC_TPYE.CTRL):
            print("cloud mac80211header, FC Info %s, %s" % (self.type, C_FC_SUBTYPE_CTRL_STR[self.subType.value]))
        elif(self.type == FC_TPYE.DATA):
            print("cloud mac80211header, FC Info %s, %s" % (self.type, C_FC_SUBTYPE_DATA_STR[self.subType.value]))
        elif(self.type == FC_TPYE.EXT):
            print("cloud mac80211header, FC Info %s, %s" % (self.type, C_FC_SUBTYPE_EXT_STR[self.subType.value]))
        else:
            print("cloud mac80211header, FC type error")

def mgmtElementParser(inbytes):
    if(isinstance(inbytes, (bytes, bytearray)) and len(inbytes) > 0):
        elementIter = 0
        tmpMgmtElements = []
        while(elementIter < len(inbytes)):
            if(MGMT_ELE.has_value(inbytes[elementIter])):
                tmpElement = MGMT_ELE(inbytes[elementIter])
                elementIter += 1
                tmpLen = inbytes[elementIter]
                if(tmpElement == MGMT_ELE.SSID):
                    pass
                elementIte += tmpLen
            else:
                elementIter += 1
                tmpLen = inbytes[elementIter]
                elementIte += tmpLen
        return tmpMgmtElements
    print("cloud mac80211header, mgmtParser input type error")
    return []


def pktParser(pkt):
    if(isinstance(pkt, (bytes, bytearray))):
        pktLen = len(pkt)
        procdLen = 0
        procdLen += 2
        # fc
        if(procdLen <= pktLen):
            hdr_fc = frameControl(struct.unpack('<H', pkt[0:2])[0])
            hdr_fc.printInfo()
        # duration
        procdLen += 2
        if(procdLen <= pktLen):
            hdr_duration = struct.unpack('<H', pkt[2:4])[0]
            print("Packet duration %d us" % hdr_duration)
        # check type
        if(hdr_fc.type == FC_TPYE.MGMT):
            procdLen += 18
            if(procdLen <= pktLen):
                hdr_addRx = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen-18:procdLen-12])
                hdr_addTx = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen-12:procdLen-6])
                hdr_addDest = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen-6:procdLen])
                print("Management to %s from %s dest %s" % (hdr_addRx, hdr_addTx, hdr_addDest))
            procdLen += 2
            if(procdLen <= pktLen):
                hdr_seqCtrl= struct.unpack('<H', pkt[procdLen-2:procdLen])[0]
                print("Management sequence control %d" % hdr_seqCtrl)
            if(hdr_fc.subType == FC_SUBTPYE_MGMT.BEACON):
                # Timestamp, Beacon Interval, Cap
                procdLen += 12
                if(procdLen <= pktLen):
                    beacon_timestamp = struct.unpack('<Q', pkt[procdLen-12:procdLen-4])[0]
                    beacon_interval = struct.unpack('<Q', pkt[procdLen-4:procdLen-2])[0]
                    beacon_cap = struct.unpack('<Q', pkt[procdLen-2:procdLen])[0]
                # Elements
                if(procdLen <= pktLen):
                    beaconElements = mgmtElementParser(pkt[procdLen:])

            elif(hdr_fc.subType == FC_SUBTPYE_MGMT.PROBEREQ):
                pass
            elif(hdr_fc.subType == FC_SUBTPYE_MGMT.PROBERES):
                pass
            else:
                print("cloud mac80211header, not supported yet")
        elif(hdr_fc.type == FC_TPYE.CTRL):
            if(hdr_fc.subType == FC_SUBTPYE_CTRL.ACK):
                procdLen += 6
                if(procdLen <= pktLen):
                    hdr_addRx = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen-6:procdLen])
                    print("ACK to %s" % hdr_addRx)
            elif(hdr_fc.subType == FC_SUBTPYE_CTRL.BLOCKACK):
                procdLen += 12
                if(procdLen <= pktLen):
                    hdr_addRx = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen-12:procdLen-6])
                    hdr_addTx = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen-6:procdLen])
                    print("BLOCK ACK to %s from %s" % (hdr_addRx, hdr_addTx))
                    # details to be added
            elif(hdr_fc.subType == FC_SUBTPYE_CTRL.RTS):
                procdLen += 12
                if(procdLen <= pktLen):
                    hdr_addRx = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen-12:procdLen-6])
                    hdr_addTx = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen-6:procdLen])
                    print("RTS to %s from %s" % (hdr_addRx, hdr_addTx))
            elif(hdr_fc.subType == FC_SUBTPYE_CTRL.CTS):
                procdLen += 6
                if(procdLen <= pktLen):
                    hdr_addRx = "%x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB",pkt[procdLen-6:procdLen])
                    print("CTS to %s" % hdr_addRx)
            else:
                print("cloud mac80211header, not supported yet")
        elif(hdr_fc.type == FC_TPYE.DATA):
            if(hdr_fc.subType == FC_SUBTPYE_DATA.DATA):
                pass
            elif(hdr_fc.subType == FC_SUBTPYE_DATA.QOSDATA):
                pass
            elif(hdr_fc.subType == FC_SUBTPYE_DATA.QOSNULL):
                pass
            else:
                print("cloud mac80211header, not supported yet")
        else:
            print("cloud mac80211header, not supported yet")



            
