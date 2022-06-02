import struct
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

fWaveComp = open("/home/cloud/sdr/stftrigger.bin",'rb')

index = 0
fWaveComp.read(index * 8)
print("finish gap")
sigReal = []
count = 50000
while(count > 0):
    count = count - 1
    try:
        tmpNum = int.from_bytes(fWaveComp.read(1), "big")
        #print(tmpNum)
        sigReal.append(tmpNum)
    except:
        break
print("finish read")

fWaveComp.close()
plt.plot(sigReal)
plt.show()
