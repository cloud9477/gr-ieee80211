import numpy as np
import struct
from matplotlib import pyplot as plt
import sys
import os
sys.path.append(os.path.join(sys.path[0], '../'))
import mac80211
import mac80211header as m8h
import phy80211header as p8h
import phy80211
import time
# from presiso import presiso  # grc-generated hier_block


def testSnrPdrSuMimo(pktFormat, nMcs, listSnr, ampSig):
    pyToolPath = os.path.dirname(__file__)
    tmpPerfRes = []
    for snrIter in range(0, len(listSnr)):
        print("current snrIter %d of %d" % (snrIter, len(listSnr)))
        tmpNoiseAmp = np.sqrt((ampSig**2)/(10.0**(listSnr[snrIter]/10.0)))
        os.system("python3 " + os.path.join(pyToolPath, "./gr_sumimo.py ") + str(tmpNoiseAmp) + " > " + os.path.join(pyToolPath, "../../tmp/tmpSuMimoPerf.txt") + " &")
        tmpPreSize = 0
        tmpCurSize = 0
        while(True):
            time.sleep(1)
            tmpCurSize = os.path.getsize(os.path.join(pyToolPath, "../../tmp/tmpSuMimoPerf.txt"))
            if(tmpPreSize == tmpCurSize):
                break
            else:
                tmpPreSize = tmpCurSize
        os.system('pkill -f gr_sumimo.py')

        resFile = open(os.path.join(pyToolPath, "../../tmp/tmpSuMimoPerf.txt")).readlines()
        resFile.reverse()
        resLine = ""
        for each in resFile:
            if("crc32" in each and pktFormat in each):
                resLine = each
                break
        
        if(len(resLine)):
            resItems = resLine.split(",")
            tmpRes = []
            for i in range(3, 3+nMcs):
                tmpRes.append(int(resItems[i].split(":")[1]))
            tmpPerfRes.append(tmpRes)
        else:
            tmpPerfRes.append([0] * nMcs)
    return tmpPerfRes

if __name__ == "__main__":
    pyToolPath = os.path.dirname(__file__)
    perfSnrList = list(np.arange(0, 51, 1))
    perfSigAmp = 0.18750000

    vhtPerfRes = testSnrPdrSuMimo("vht", 10, perfSnrList, perfSigAmp)

    print(vhtPerfRes)

    widths = [8]
    heights = [4]
    pltFig = plt.figure(figsize=(8,4))
    spec = pltFig.add_gridspec(ncols=1, nrows=1, width_ratios=widths,
                            height_ratios=heights, wspace=0.3, hspace=0.4)
    cx = pltFig.add_subplot(spec[0, 0])
    for i in range(0, 9):
        cx.plot([each[i] for each in vhtPerfRes])
    plt.show()



    