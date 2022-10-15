import struct
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

totalSamp = 20000000
index = 0

sigReal1 = []
fWaveComp1 = open("/home/cloud/sdr/debugSigByte.bin",'rb')
fWaveComp1.read(index * 1)
print("finish gap")
count = totalSamp
while(count > 0):
    count = count - 1
    try:
        tmpNum = int.from_bytes(fWaveComp1.read(1), "big")
        sigReal1.append(tmpNum * 10)
    except:
        break
fWaveComp1.close()
print("finish read 1")



sigReal2 = []
sigImag2 = []
fWaveComp2 = open("/home/cloud/sdr/debugSigComp0.bin",'rb')
fWaveComp2.read(index * 8)
print("finish gap")
count = totalSamp
while(count > 0):
    count = count - 1
    try:
        sigReal2.append(struct.unpack('f', fWaveComp2.read(1) + fWaveComp2.read(1) + fWaveComp2.read(1) + fWaveComp2.read(1))[0])
        sigImag2.append(struct.unpack('f', fWaveComp2.read(1) + fWaveComp2.read(1) + fWaveComp2.read(1) + fWaveComp2.read(1))[0])
    except:
        break
fWaveComp2.close()
print("finish read 2")




sigReal3 = []
sigImag3 = []
fWaveComp3 = open("/home/cloud/sdr/debugSigComp.bin",'rb')
fWaveComp3.read(index * 8)
print("finish gap")
count = totalSamp
while(count > 0):
    count = count - 1
    try:
        sigReal3.append(struct.unpack('f', fWaveComp3.read(1) + fWaveComp3.read(1) + fWaveComp3.read(1) + fWaveComp3.read(1))[0])
        sigImag3.append(struct.unpack('f', fWaveComp3.read(1) + fWaveComp3.read(1) + fWaveComp3.read(1) + fWaveComp3.read(1))[0])
    except:
        break
fWaveComp3.close()
print("finish read 3")



plt.plot(sigReal1)
#plt.plot(sigReal2)
#plt.plot(sigImag2)
plt.plot(sigReal3)

#plt.plot([sigImag2[i] - sigImag3[i] for i in range(0, 8000)])

plt.show()
