import numpy as np
import struct
import socket
import binascii
import zlib
from matplotlib import pyplot as plt

"""
    the simulation generates the IEEE 802.11a PHY signal from transportation layer (UDP)
    ISO layer       |   protocol
    Transport       |   UDP
    Network         |   IPv4
    MAC             |   IEEE80211
    PHY             |   IEEE80211n
"""

def procCheckCrc32(inBytesPayload, inBytesCrc32):
    tmpRxCrc32 = zlib.crc32(inBytesPayload)
    if(tmpRxCrc32 == struct.unpack('<L',inBytesCrc32)):
        print("cloud80211 mac crc32 check pass")
    else:
        print("cloud80211 mac crc32 check fail")


# udp generator, input: ip, port and payload
class udp():
    def __init__(self, sIp, dIp, sPort, dPort, payloadBytes):
        self.fakeSourIp = sIp   # ip of network layer, only to generate checksum
        self.fakeDestIp = dIp
        self.sourPort = sPort   # ports
        self.destPort = dPort
        self.payloadBytes = payloadBytes
        self.protocol = socket.IPPROTO_UDP
        self.len = len(payloadBytes) + 8
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
            print("odd len")
            self.checkSum += (self.payloadBytes[len(self.payloadBytes) - 1] * 256)
        while (self.checkSum > 65535):
            self.checkSum = self.checkSum % 65536 + int(np.floor(self.checkSum / 65536))
        self.checkSum = 65535 - self.checkSum

    def genPacket(self):
        self.__genCheckSum()
        return struct.pack('>HHHH',self.sourPort,self.destPort,self.len,self.checkSum)+self.payloadBytes

# ipv4 generator, takes UDP as payload, give id,
class ipv4():
    def __init__(self, id, ttl, sIp, dIp, payloadBytes):
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
        self.payload = payloadBytes     # payload, UDP packet
        self.len = self.IHL * 4 + len(payloadBytes)
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

    def genPacket(self):
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

    def genPacket(self):
        return struct.pack('>BBB', self.SNAP_DSAP, self.SNAP_SSAP, self.control) + struct.pack('>L', self.RFC1024)[:3] + struct.pack('>H', self.type)

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
    def __init__(self, type, subType, toDs, fromDs, retry, protected, addr1, addr2, addr3, seq, payloadBytes, ifAmpdu):
        self.fc_protocol = 0        # fixed, frame control - protocol, 2 bits
        self.fc_type = type         # frame control - type, 2 bits
        self.fc_subType = subType   # frame control - sub type, 4 bits
        if(self.fc_subType == 8):
            print("mac80211: QoS Data")
        elif(self.fc_subType == 0):
            print("mac80211: Data")
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
        self.payloadBytes = payloadBytes
        self.lenBytes = len(payloadBytes) + 24 + 4  # input includes LLC, 8 bytes have been added
        self.fc = self.fc_protocol + (self.fc_type << 2) + (self.fc_subType << 4) + (self.fc_toDs << 8) + (self.fc_fromDs << 9) + (self.fc_frag << 10) + (self.fc_retry << 11) + (self.fc_pwr << 12) + (self.fc_more << 13) + (self.fc_protected << 14) + (self.fc_order << 15)
        self.ifAmpdu = ifAmpdu

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

    def __macByteCrc32(self, bytesIn):
        crc = 0xffffffff
        for eachByte in bytesIn:
            crc = crc ^ eachByte
            for i in range(0,8):
                mask = crc & 0x00000001
                mask = 0xffffffff - mask + 1
                crc = crc >> 1
                crc = crc ^ (0xedb88320 & mask)
        return 0xffffffff - crc

    def __genDuration(self):
        # manually set
        self.duration = 110  # used in sniffed packet

    def genPacket(self):
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
        print("mac mpdu length: %d" % len(tmpPacket))
        print(tmpPacket.hex())
        if(self.ifAmpdu):
            # ampdu for vht, single packet
            delimiterEof = 1
            delimiterReserved = 0
            delimiterMpduLen = len(tmpPacket)
            delimiterMpduLenBits = []
            for i in range(0, 14):
                delimiterMpduLenBits.append((delimiterMpduLen >> i) & (1))
            print("mac a-mpdu len to bits:")
            print(delimiterMpduLenBits)
            delimiterBits = [delimiterEof] + [delimiterReserved] + delimiterMpduLenBits[12:14] + delimiterMpduLenBits[0:12]
            delimiterBits = delimiterBits + self.__macBitCrc8(delimiterBits)
            for i in range(0, 8):
                delimiterBits.append((0x4e >> i) & (1))
            print("mac a-mpdu bits: %d" % len(delimiterBits))
            print(delimiterBits)
            tmpDelimiterBytes = b""
            for i in range(0, 4):
                tmpByte = 0
                for j in range(0, 8):
                    tmpByte = tmpByte + delimiterBits[i*8+j] * (2**j)
                tmpDelimiterBytes += bytearray([tmpByte])
            tmpPacket = tmpDelimiterBytes + tmpPacket
            print("mac a-mpdu padding")
            print("current byte number: %d" % len(tmpPacket))
            nBytePadding = int(np.ceil(len(tmpPacket)/4)*4 - len(tmpPacket))
            print("padding byte number: %d" % nBytePadding)
            tmpPacket += b'\x00'*nBytePadding
            print("mac a-mpdu length: %d" % len(tmpPacket))
            print(tmpPacket.hex())

        return tmpPacket