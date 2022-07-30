import struct
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

fWaveComp = open("/home/cloud/sdr/sig80211CfoGr.bin",'rb')

index = 0
fWaveComp.read(index * 8)
print("finish gap")
sigReal = []
sigImag = []
while(True):
    try:
        sigReal.append(struct.unpack('f', fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1))[0])
        sigImag.append(struct.unpack('f', fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1))[0])
    except:
        break
print("finish read")

fWaveComp.close()
plt.plot(sigReal)
plt.plot(sigImag)
plt.show()