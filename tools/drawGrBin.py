import struct
from matplotlib import pyplot as plt
import os

figureNum = 1

def drawFloat(p):
    global figureNum
    sigReal = []
    fileName = p
    if(os.path.exists(fileName)):
        fileStats = os.stat(fileName)
        fileSize = fileStats.st_size
        sigSampNum = int(fileSize / 4)
        f = open(fileName,'rb')
        fileBytes = f.read(fileSize)
        f.close()
        print("finish read")
        for i in range(0, sigSampNum):
            sigReal.append(struct.unpack('f', fileBytes[i*4:i*4+4])[0])
        plt.figure(figureNum)
        figureNum += 1
        plt.plot(sigReal)

def drawComplex(p):
    global figureNum
    sigReal = []
    sigImag = []
    fileName = p
    if(os.path.exists(fileName)):
        fileStats = os.stat(fileName)
        fileSize = fileStats.st_size
        sigSampNum = int(fileSize / 8)
        f = open(fileName,'rb')
        fileBytes = f.read(fileSize)
        f.close()
        print("finish read")
        for i in range(0, sigSampNum):
            sigReal.append(struct.unpack('f', fileBytes[i*8:i*8+4])[0])
            sigImag.append(struct.unpack('f', fileBytes[i*8+4:i*8+8])[0])
        plt.figure(figureNum)
        figureNum += 1
        plt.plot(sigReal)
        plt.plot(sigImag)

if __name__ == "__main__":
    drawComplex("/home/cloud/sdr/sig80211GenMultipleMimo_2x2_0.bin")
    drawComplex("/home/cloud/sdr/sig80211GenMultipleMimo_2x2_1.bin")
    plt.show()