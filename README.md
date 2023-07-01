# gr-ieee80211
- GR-WiFi: GNU Radio IEEE 802.11a/g/n/ac (WiFi 5) transceiver.
- Support SISO and upto 2x2 SU-MIMO and MU-MIMO.
- Ubuntu 22.04 and GNU Radio 3.10 (recommanded).
- V0.1 branch is the previous stable version.
- Main branch is working on faster TX and RX, not stable yet.
- For CUDA DLC please refer to [gr-ieee80211cu](https://github.com/cloud9477/gr-ieee80211cu), due to my recent schedule, CUDA DLC is only compatible to v0.1.
- For the CUDA DLC, it is only working now, from our test, not much performance benefit from it due to the time cost on data transfer between CPU and GPU.


TX
------
![alt text](https://github.com/cloud9477/gr-ieee80211/blob/main/figGrTx.png?raw=true)

RX
------
![alt text](https://github.com/cloud9477/gr-ieee80211/blob/main/figGrRx.png?raw=true)

Introduction
------
- This repo has actually two major components:
- 1 ```GNU Radio transceiver (GR-TRX)```
- 1.1 GNU Radio transceiver provides an IEEE 802.11 physical layer.
- 1.2 MAC layer for GNU Radio transceiver is under development.
- 1.3 Due to the non-real-time of GR and USRP, the MAC layer will not be compatible to IEEE 802.11 MAC.
- 2 ```Python Tool Box (PY-TB)```
- 2.1 Python Tool Box is in the **tools** folder.
- 2.2 PY-TB provides all kinds of scripts to generate WiFi signal (**./tools/pktGenExample.py**), test performance (**./tools/performance**) and so on.
- 2.3 We provide basic scripts to use GR-TRX in PY-TB, like a monitor (**./tools/macExampleGrRx.py**) and a transmitter (**./tools/macExampleGrTx.py**)
- 2.4 We provide some demo test in PY-TB for MU-MIMO transmissions (**./tools/cmu_vx/**).
- 2.5 **GR-TRX needs better processing performance so the C++ processing codes are optimized and if you are new to WiFi, you may feel difficult to understand. While PY-TB is used as reference so all the processing codes in PY-TB follow the exact standard documents and it is better for learning.**
- 2.6 For more details, please refer to the README in **tools** folder.

GR-IEEE80211 Receiver Design
------
- Pre-Processing: Auto-correlation of STF and the coarse CFO.
- Trigger: Detect auto-correlation plateau, give trigger of LTF with coarse starting point.
- Sync: Triggered by STF, use auto-correlation of LTF to find starting timing, re-estimate accurate CFO.
- Signal: Compensate CFO, estimate Legacy channel and passes channel and samples to Demod block.
- Demod: Further check HT signal and VHT signal A to get the correct packet format, demodulates OFDM and get soft bits.
- Decode: uses soft viterbi to decode, check FCS and assemble the packet.
- Demod and Decode: for NDP, also pass the channel info to MAC.

GR-IEEE80211 Transmitter Design
------
- Pkt Gen: generate packet byte stream with a queue.
- Encode: encode, including bcc, scrambling, stream parsing, interleaving, and assemble bits to chips (corressponding to QAM constellations).
- Modulation: map chips to QAM constellations, insert pilots.
- Modulation: also takes the beamforming spatial mapping matrix Q to transmit MU-MIMO packets.
- FFT and CP: OFDM freq to time with guard interval for 20MHz bandwidth.
- Pad: add Legacy preamble and scale signal, also add tags for USRP sink.

Installation
------
- Install GNU Radio 3.10 dev according to the GNU Radio offical website
- Install UHD
- Copy and install gr-ieee80211 module for GNU Radio
```console
sdr@sdr:~$ sudo apt-get update
sdr@sdr:~$ sudo apt-get install git gnuradio-dev uhd-host libuhd-dev cmake build-essential
sdr@sdr:~$ sudo cp /lib/uhd/utils/uhd-usrp.rules /etc/udev/rules.d/
sdr@sdr:~$ sudo udevadm control --reload-rules
sdr@sdr:~$ sudo udevadm trigger
sdr@sdr:~$ sudo uhd_images_downloader
sdr@sdr:~$ mkdir sdr
sdr@sdr:~$ cd sdr
sdr@sdr:~$ git clone https://github.com/cloud9477/gr-ieee80211.git
sdr@sdr:~$ cd gr-ieee80211/
sdr@sdr:~$ mkdir build
sdr@sdr:~$ cd build
sdr@sdr:~$ cmake ../
sdr@sdr:~$ make
sdr@sdr:~$ sudo make install
sdr@sdr:~$ sudo ldconfig
```
- Next, gr-ieee80211 uses a hierarchical block **preproc** for Legacy pre-processing. For the first time to use gr-ieee80211, please open the file **presiso.grc** in example folder in GNU Radio and then click **Generate the flow graph** button. You will see two new files in the example folder, **presiso.block.yml** and **presiso.py**. Next, please copy them to the **.grc_gnuradio** folder in your home folder. For example, here is the path of my GNU Radio GRC folder:
```
sdr@sdr:~/.grc_gnuradio$ pwd
/home/sdr/.grc_gnuradio
```
- After that, reopen the GNU Radio and then you will see the the **preproc** block in your tool box.


