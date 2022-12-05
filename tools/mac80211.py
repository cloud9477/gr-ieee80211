"""
    GNU Radio IEEE 802.11a/g/n/ac 2x2
    Python tools
    Copyright (C) June 1, 2022  Zelin Yun

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import struct
import socket
import binascii
import zlib

"""
    the simulation generates the IEEE 802.11a/g/n/ac MAC data packet and PHY data signal from transportation layer (UDP)
    ISO layer       |   protocol
    Transport       |   UDP
    Network         |   IPv4
    MAC             |   IEEE80211
    PHY             |   IEEE80211n
"""

def procCheckCrc32(inBytes):
    if(len(inBytes) > 4):
        inBytesPayload = inBytes[:-4]
        inBytesCrc32 = inBytes[-4:]
        tmpPayloadCrc32 = zlib.crc32(inBytesPayload)
        tmpTailCrc32, =struct.unpack('<L',inBytesCrc32)
        if(tmpPayloadCrc32 == tmpTailCrc32):
            return True
        else:
            return False
    else:
        print("cloud mac80211: crc32 input len error")

def procGenBitCrc8(bitsIn):
    c = [1] * 8
    for b in bitsIn:
        next_c = [0] * 8
        next_c[0] = b ^ c[7]
        next_c[1] = b ^ c[7] ^ c[0]
        next_c[2] = b ^ c[7] ^ c[1]
        next_c[3] = c[2]
        next_c[4] = c[3]
        next_c[5] = c[4]
        next_c[6] = c[5]
        next_c[7] = c[6]
        c = next_c
    return [1 - b for b in c[::-1]]

# udp generator, input: ip, port and payload
class udp():
    def __init__(self, sIp, dIp, sPort, dPort):
        self.fakeSourIp = sIp   # ip of network layer, only to generate checksum
        self.fakeDestIp = dIp
        self.sourPort = sPort   # ports
        self.destPort = dPort
        self.payloadBytes = b''
        self.protocol = socket.IPPROTO_UDP
        self.len = 8
        self.checkSum = 0

    def __genCheckSum(self):
        self.checkSum = 0
        fakeSourIpBytes = socket.inet_aton(self.fakeSourIp)
        fakeDestIpBytes = socket.inet_aton(self.fakeDestIp)
        self.checkSum += fakeSourIpBytes[0] * 256 + fakeSourIpBytes[1]
        self.checkSum += fakeSourIpBytes[2] * 256 + fakeSourIpBytes[3]
        self.checkSum += fakeDestIpBytes[0] * 256 + fakeDestIpBytes[1]
        self.checkSum += fakeDestIpBytes[2] * 256 + fakeDestIpBytes[3]
        self.checkSum += self.protocol
        self.checkSum += self.len        # fake len = len
        self.checkSum += self.sourPort
        self.checkSum += self.destPort
        self.checkSum += self.len        # len
        for i in range(0, int(np.floor(len(self.payloadBytes) / 2))):
            self.checkSum += ((self.payloadBytes[i * 2]) * 256 + (self.payloadBytes[i * 2 + 1]))
        if (len(self.payloadBytes) > int(np.floor(len(self.payloadBytes) / 2)) * 2):
            self.checkSum += (self.payloadBytes[len(self.payloadBytes) - 1] * 256)
        while (self.checkSum > 65535):
            self.checkSum = self.checkSum % 65536 + int(np.floor(self.checkSum / 65536))
        self.checkSum = 65535 - self.checkSum

    def genPacket(self, payloadBytes):
        self.payloadBytes = payloadBytes
        self.len = len(payloadBytes) + 8
        self.__genCheckSum()
        return struct.pack('>HHHH',self.sourPort,self.destPort,self.len,self.checkSum)+self.payloadBytes

# ipv4 generator, takes UDP as payload, give id,
class ipv4():
    def __init__(self, id, ttl, sIp, dIp):
        self.ver = 4    # fixed, Version
        self.IHL = 5    # fixed, Internet Header Length, number of 32 bits
        self.DSCP = 0   # fixed
        self.ECN = 0    # fixed
        self.ID = id    # similar to sequence number
        self.flagReserved = 0   # fixed
        self.flagDF = 1      # fixed, do not frag
        self.flagMF = 0      # fixed, no more frag
        self.flag = (self.flagReserved << 2) + (self.flagDF << 1) + self.flagMF     # fixed
        self.fragOffset = 0     # no frag, so no frag offset
        self.TTL = ttl  # Time to live
        self.protocol = socket.IPPROTO_UDP  # fixed, protocol
        self.sourIp = sIp   # source ip
        self.destIp = dIp   # destination ip
        self.payload = b""
        self.len = 0
        self.checkSum = 0

    def __genCheckSum(self):
        self.checkSum = 0
        self.checkSum += (self.ver * (16 ** 3) + self.IHL * (16 ** 2) + self.DSCP * 4 + self.ECN)
        self.checkSum += self.len
        self.checkSum += self.ID
        self.checkSum += (self.flag * (2 ** 13) + self.fragOffset)
        self.checkSum += (self.TTL * 256 + self.protocol)
        sourIpBytes = socket.inet_aton(self.sourIp)
        destIpBytes = socket.inet_aton(self.destIp)
        self.checkSum += sourIpBytes[0] * 256 + sourIpBytes[1]
        self.checkSum += sourIpBytes[2] * 256 + sourIpBytes[3]
        self.checkSum += destIpBytes[0] * 256 + destIpBytes[1]
        self.checkSum += destIpBytes[2] * 256 + destIpBytes[3]
        while (self.checkSum > 65535):
            self.checkSum = self.checkSum % 65536 + int(np.floor(self.checkSum / 65536))
        self.checkSum = 65535 - self.checkSum

    def genPacket(self, payloadBytes):
        self.payload = payloadBytes     # payload, UDP packet
        self.len = self.IHL * 4 + len(payloadBytes)
        self.__genCheckSum()
        return struct.pack('>HHHHHH', (self.ver * (16 ** 3) + self.IHL * (16 ** 2) + self.DSCP * 4 + self.ECN), self.len, self.ID, (self.flag * (2 ** 13) + self.fragOffset), (self.TTL * 256 + self.protocol), self.checkSum) + socket.inet_aton(self.sourIp) + socket.inet_aton(self.destIp) + self.payload

# Logical Link Control, provide unified interface of data link layer
class llc():
    def __init__(self):
        self.SNAP_DSAP = 0xaa   # fixed, Source SAP
        self.SNAP_SSAP = 0xaa   # fixed, Destination SAP
        self.control = 0x03     # fixed
        self.RFC1024 = 0x000000 # fixed
        self.type = 0x0800  # 0x0800 for IP packet, 0x0806 for ARP

    def genPacket(self, payloadBytes):
        return struct.pack('>BBB', self.SNAP_DSAP, self.SNAP_SSAP, self.control) + struct.pack('>L', self.RFC1024)[:3] + struct.pack('>H', self.type) + payloadBytes

class mac80211():
    """
    802.11 a & n mac frame
    | FC 2 | Duration 2 | ADDR1 6 | ADDR2 6 | ADDR3 6 | seq 2 | payload | FCS 4 |
    | FC 2 | Duration 2 | ADDR1 6 | ADDR2 6 | ADDR3 6 | seq 2 | QoS 2 | HT Control 4 | payload | FCS 4 |
    QoS field only appears in QoS packet: block ack, QoS data and so on
    HT Control field only appears in control wrapper
    QoS is added after a/b/g/, only used when the packet is QoS packet
    QoS field: 0-3: Traffic ID, used to show the priority of the packet, the other parts usually are 0
    """
    def __init__(self, type, subType, toDs, fromDs, retry, protected, addr1, addr2, addr3, seq):
        self.fc_protocol = 0        # fixed, frame control - protocol, 2 bits
        self.fc_type = type         # frame control - type, 2 bits
        self.fc_subType = subType   # frame control - sub type, 4 bits
        if(self.fc_subType == 8):
            print("cloud mac80211: QoS Data")
        elif(self.fc_subType == 0):
            print("cloud mac80211: Data")
        self.QoS = 0
        self.fc_toDs = toDs         # frame control - station to ap, 1 bit
        self.fc_fromDs = fromDs     # frame control - ap to station, 1 bit
        self.fc_frag = 0            # fixed, frame control - if more frag? 0: no more frags, 1 bit
        self.fc_retry = retry       # frame control - if retry? 1: retry, 0: not, 1 bit
        self.fc_pwr = 0    # fixed, frame control - not entering power saving, 1 bit
        self.fc_more = 0   # fixed, frame control - no data buffered, 1 bit
        self.fc_protected = protected   # frame control - if encrypted, 1 bit
        self.fc_order = 0  # fixed, frame control - no frag, no order, 1 bit
        self.duration = 0    # 16 bits, to be computed
        self.addr1 = addr1
        self.addr2 = addr2
        self.addr3 = addr3
        self.sc_frag = 0    # fixed, sequence control - frag number, no frag so to be 0
        self.sc_seq = seq     # sequence control - seq number
        self.sc = self.sc_frag + self.sc_seq << 4
        self.payloadBytes = b""
        self.fc = self.fc_protocol + (self.fc_type << 2) + (self.fc_subType << 4) + (self.fc_toDs << 8) + (self.fc_fromDs << 9) + (self.fc_frag << 10) + (self.fc_retry << 11) + (self.fc_pwr << 12) + (self.fc_more << 13) + (self.fc_protected << 14) + (self.fc_order << 15)
        self.eofPaddingSf = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.eofPaddingSf = self.eofPaddingSf + self.__macBitCrc8(self.eofPaddingSf) + [0, 1, 1, 1, 0, 0, 1, 0] #0x4e

    def __macBitCrc8(self, bitsIn):
        c = [1] * 8
        for b in bitsIn:
            next_c = [0] * 8
            next_c[0] = b ^ c[7]
            next_c[1] = b ^ c[7] ^ c[0]
            next_c[2] = b ^ c[7] ^ c[1]
            next_c[3] = c[2]
            next_c[4] = c[3]
            next_c[5] = c[4]
            next_c[6] = c[5]
            next_c[7] = c[6]
            c = next_c
        return [1 - b for b in c[::-1]]

    def __genDuration(self):
        # manually set
        self.duration = 110  # used in sniffed packet

    def genPacket(self, payload):
        if(isinstance(payload, (bytes, bytearray))):
            self.payloadBytes = payload
            self.__genDuration()
            tmpPacket = struct.pack('<H', self.fc) + struct.pack('<H', self.duration)
            tmpPacket += binascii.unhexlify("".join(self.addr1.split(":")))
            tmpPacket += binascii.unhexlify("".join(self.addr2.split(":")))
            tmpPacket += binascii.unhexlify("".join(self.addr1.split(":")))
            tmpPacket += struct.pack('<H', self.sc)
            if(self.fc_subType == 8):
                tmpPacket += struct.pack('<H', self.QoS)   # only added when QoS packet
            tmpPacket += self.payloadBytes
            tmpPacket += struct.pack('<L',zlib.crc32(tmpPacket))
            print("cloud mac80211, gen pkt mac mpdu length: %d" % len(tmpPacket))
            return tmpPacket
        else:
            print("cloud mac80211, gen pkt input type error")
            return b""

def genAmpduHT(payloads):
    if(isinstance(payloads, list)):
        tmpAmpduPkt = b""
        for payloadIter in range(0, len(payloads)):
            delimiterMpduLen = len(payloads[payloadIter])
            if(delimiterMpduLen < 1 or delimiterMpduLen > 4095):
                print("cloud mac80211, gen ampdu ht: packet %d len %d error" % (payloadIter, delimiterMpduLen))
                return b""
            delimiterMpduLenBits = []
            for i in range(0, 12):      # HT only 12 bits for len
                delimiterMpduLenBits.append((delimiterMpduLen >> i) & (1))
            delimiterBits = [0, 0, 0, 0] + delimiterMpduLenBits
            delimiterBits = delimiterBits + procGenBitCrc8(delimiterBits)
            for i in range(0, 8):
                delimiterBits.append((0x4e >> i) & (1))
            tmpDelimiterBytes = b""
            for i in range(0, 4):
                tmpByte = 0
                for j in range(0, 8):
                    tmpByte = tmpByte + delimiterBits[i * 8 + j] * (2 ** j)
                tmpDelimiterBytes += bytearray([tmpByte])
            tmpPacket = tmpDelimiterBytes + payloads[payloadIter]
            if(payloadIter < (len(payloads)-1)):  # pad if not last one
                nBytePadding = int(np.ceil(len(tmpPacket) / 4) * 4 - len(tmpPacket))
                tmpPacket += b'\x00' * nBytePadding
            tmpAmpduPkt += tmpPacket
        return tmpAmpduPkt
    else:
        print("cloud mac80211, gen ampdu ht: input type error")
        return b""

def genAmpduVHT(payloads):
    if(isinstance(payloads, list)):
        tmpAmpduPkt = b""
        for eachPayload in payloads:
            delimiterMpduLen = len(eachPayload)
            delimiterMpduLenBits = []
            delimiterEof = 0
            if(len(payloads) == 1):
                delimiterEof = 1
            delimiterReserved = 0
            for i in range(0, 14):
                delimiterMpduLenBits.append((delimiterMpduLen >> i) & (1))
            delimiterBits = [delimiterEof] + [delimiterReserved] + delimiterMpduLenBits[12:14] + delimiterMpduLenBits[0:12]
            delimiterBits = delimiterBits + procGenBitCrc8(delimiterBits)
            for i in range(0, 8):
                delimiterBits.append((0x4e >> i) & (1))
            tmpDelimiterBytes = b""
            for i in range(0, 4):
                tmpByte = 0
                for j in range(0, 8):
                    tmpByte = tmpByte + delimiterBits[i * 8 + j] * (2 ** j)
                tmpDelimiterBytes += bytearray([tmpByte])
            tmpPacket = tmpDelimiterBytes + eachPayload
            # each packet is padded
            nBytePadding = int(np.ceil(len(tmpPacket) / 4) * 4 - len(tmpPacket))
            tmpPacket += b'\x00' * nBytePadding
            print(tmpPacket)
            tmpAmpduPkt += tmpPacket
        return tmpAmpduPkt
    else:
        print("cloud mac80211, gen ampdu vht: input type error")
        return b""

if __name__ == "__main__":
    udpPayload  = "123456789012345678901234567890"
    udpIns = udp("10.10.0.6",  # sour ip
                          "10.10.0.1",  # dest ip
                          39379,  # sour port
                          8889)  # dest port
    udpPacket = udpIns.genPacket(bytearray(udpPayload, 'utf-8'))
    print("udp packet")
    print(udpPacket.hex())
    ipv4Ins = ipv4(43778,  # identification
                            64,  # TTL
                            "10.10.0.6",
                            "10.10.0.1")
    ipv4Packet = ipv4Ins.genPacket(udpPacket)
    print("ipv4 packet")
    print(ipv4Packet.hex())
    llcIns = llc()
    llcPacket = llcIns.genPacket(ipv4Packet)
    print("llc packet")
    print(llcPacket.hex())
    
    mac80211Ins = mac80211(2,  # type
                                     8,  # sub type, 8 = QoS Data
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
    if(procCheckCrc32(mac80211Packet)):
        print("crc correct")
    else:
        print("crc wrong")

    mac80211Ampdu = genAmpduVHT([mac80211Packet, mac80211Packet])
    print("vht ampdu packet")
    print(mac80211Ampdu.hex())
    