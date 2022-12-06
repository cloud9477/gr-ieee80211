# Cloud Multi-User MIMO (CMU)

Introduction
------------
- The setup includes one AP (2x2) and two stations (1x1)
- The procedure is to 
    - 1. AP sends NDP
    - 2. Stations estimate the channel
    - 3. Stations send the channel info matrix V to AP
    - 4. AP generates beamforming steering matrix Q
    - 5. AP apply Q to the spatial streams to send multi-user packets
- However, we assume that the CPU is not powerful enough to support 2 TX and 2 RX processing at the same time. In that case, we only let the AP to do the 2x2 TX without RX. That makes the step 3 a little different.
- The procedure changes to
    - 1. AP sends NDP
    - 2. Stations estimate the channel
    - 3. AP gets the channel info matrix V from stations by Ethernet
    - 4. AP generates beamforming steering matrix Q
    - 5. AP apply Q to the spatial streams to send multi-user packets
- Since in the whole procedure, the AP RX only receives general data packets so it doesn't affect anything related to MU-MIMO.

Instructions
------------
- At the two station sides, run the GNU Radio SISO RX and the cmu_sta.py, it will save the channel info into a bin file.
- At the AP side, run the GNU Radio MIMO TX cmu_ap.py, it triggers the GNU Radio to send NDP first, and then fetches the channel info from the stations, next computes the Q and finally sends the MU-MIMO packet.
