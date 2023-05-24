from matplotlib import pyplot as plt
import numpy as np
import struct
import time
import json
import sys
import os

if __name__ == "__main__":
    f = open("perf_ax210.json")     # wireshark pcapng save to json
    data = json.load(f)
    f.close()

    perfRes = []
    for i in range(0, 30):
        perfRes.append([0] * 9)

    for each in data:
        mcs = int(each['_source']['layers']['radiotap']['radiotap.vht']['radiotap.vht.user']['radiotap.vht.mcs.0'])
        dataStrItems = each['_source']['layers']['data']['data.data'].split(":")
        snr = (int(dataStrItems[-2]) - 30) * 10 + (int(dataStrItems[-1]) - 30)      # last two bytes indicate snr
        perfRes[snr][mcs] += 1
    
    print(perfRes)

    widths = [8]
    heights = [4,4,4]
    pltFig = plt.figure(figsize=(8,12))
    spec = pltFig.add_gridspec(ncols=1, nrows=3, width_ratios=widths,
                            height_ratios=heights, wspace=0.3, hspace=0.4)
    ax = pltFig.add_subplot(spec[0, 0])
    for i in range(0, 9):
        ax.plot([each[i] for each in perfRes])
    plt.show()

    