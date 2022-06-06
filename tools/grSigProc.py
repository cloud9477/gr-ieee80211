import struct
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

fWaveComp = open("/home/cloud/sdr/stftrigger.bin",'rb')

index = 20000
fWaveComp.read(index * 8)
print("finish gap")
sig = []
for n in range(0,640):
    try:
        r = struct.unpack('f', fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1))[0]
        i = struct.unpack('f', fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1) + fWaveComp.read(1))[0]
        sig.append(r+i*1.0j)
    except:
        break
print("finish read")

ltf1 = sig[182:182+64]

fWaveComp.close()
plt.plot(np.real(ltf1))
plt.plot(np.imag(ltf1))
plt.show()