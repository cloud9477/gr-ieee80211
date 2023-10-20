# gr-ieee80211
**<font color=#5050f0>GR-WiFi</font>** : GNU Radio IEEE 802.11a/g/n/ac (WiFi 5) Transceiver
------
- Support SISO and upto 2x2 **<font color=orange>SU-MIMO</font>** and **<font color=orange>MU-MIMO</font>**.
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
- This **<font color=#5050f0>GR-WiFi</font>** has actually two major components:
- **<font color=#f05050>[1] ------ GNU Radio transceiver (GR-TRX) ------</font>**
- 1.1 GNU Radio transceiver provides an IEEE 802.11 physical layer.
- 1.2 MAC layer for GNU Radio transceiver is under development.
- 1.3 Due to the non-real-time of GR and USRP, the MAC layer will not be compatible to IEEE 802.11 MAC.
- **<font color=#f05050>[2] ----------- Python Tool Box (PY-TB) ----------- </font>**
- 2.1 Python Tool Box is in the **gr-ieee80211/tools/** folder.
- 2.2 PY-TB provides all kinds of scripts to generate WiFi signal (**gr-ieee80211/tools/pktGenExample.py**), test performance (**gr-ieee80211/tools/performance**) and so on.
- 2.3 We provide basic scripts to use GR-TRX in PY-TB, like a monitor (**gr-ieee80211/tools/macExampleGrRx.py**) and a transmitter (**gr-ieee80211/tools/macExampleGrTx.py**)
- 2.4 We provide some demo test in PY-TB for MU-MIMO transmissions (**gr-ieee80211/tools/cmu_vx/**, x is 1 to 4).
- 2.5 **GR-TRX needs better processing performance so the C++ processing codes are optimized and if you are new to WiFi, you may feel difficult to understand. While PY-TB is used as reference so all the processing codes in PY-TB follow the exact standard documents and it is better for learning.**
- 2.6 For more details, please refer to the README in **gr-ieee80211/tools** folder.

GR-WiFi Receiver Design
------
- Pre-Processing: Auto-correlation of STF and the coarse CFO.
- Trigger: Detect auto-correlation plateau, give trigger of LTF with coarse starting point.
- Sync: Triggered by STF, use auto-correlation of LTF to find starting timing, re-estimate accurate CFO.
- Signal: Compensate CFO, estimate Legacy channel and passes channel and samples to Demod block.
- Demod: Further check HT signal and VHT signal A to get the correct packet format, demodulates OFDM and get soft bits.
- Decode: uses soft viterbi to decode, check FCS and assemble the packet.
- Demod and Decode: for NDP, also pass the channel info to MAC.

GR-WiFi Transmitter Design
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
- Next, gr-ieee80211 uses a hierarchical block **preproc** for Legacy pre-processing. For the first time to use gr-ieee80211, please open the file **presiso.grc** in example folder in GNU Radio and then click **Generate the flow graph** button. You will see two new files in the example folder, **presiso.block.yml** and **presiso.py**. Next, please copy them to the **.grc_gnuradio** folder in your home folder (if no **.grc_gnuradio** folder there, you can create one using command **$ mkdir .grc_gnuradio**). For example, here is the path of my GNU Radio GRC folder, my Ubuntu user name is sdr:
```
/home/sdr/.grc_gnuradio/
```
- After that, reopen the GNU Radio and then you will see the the **preproc** block in your tool box.

How To Use **<font color=#f05050>PY-TB</font>**
------
1. Follow the installation to install **<font color=#5050f0>GR-WiFi</font>** first. Then go to the **gr-ieee80211/tools** folder, run **pktGenExample.py**, you will see

![alt text](https://github.com/cloud9477/gr-ieee80211/blob/main/figExample1.png?raw=true)

2. This Python script **pktGenExample.py** is used to generate WiFi physical layer baseband signal with given MAC layer packet and given physical layer format. After generation, it shows the signal shape and also save it into a bin file which can be directly transmitted by GNU Radio File Source with USRP Sink. This script is written following exactly the IEEE 802.11 standard document 2020 version. For example, the binary convolutional coding and interleaving, they all follow the formulas in the standard. It helps you understand the standard better. And it provides a good reference for development. In the figure above, it shows the signal samples of Legacy, HT and VHT SISO formats.

![alt text](https://github.com/cloud9477/gr-ieee80211/blob/main/figExample2.png?raw=true)

3. In the second figure above, it shows the signal samples of HT and VHT 2x2 SU-MIMO formats.

How To Use **<font color=#f05050>GR-TRX</font>**
------
1. Follow the installation to install **<font color=#5050f0>GR-WiFi</font>** first. Then open GNU Radio, in GNU Radio, open the file **tx.grc** in **gr-ieee80211/example** folder, run the flow graph.
2. Open a terminal and then run the **macExampleGrTx.py** in **gr-ieee80211/tools** folder. You will see the following results:

![alt text](https://github.com/cloud9477/gr-ieee80211/blob/main/figExample3.png?raw=true)

3. The principle is that, first the **macExampleGrTx.py** of **<font color=#f05050>PY-TB</font>** generates a MAC layer packet and transfer it to the GNU Radio Socket PDU block using UDP. This is how we design the API of **<font color=#f05050>GR-TRX</font>**. Besides the MAC packet, the MCS and stream number are also passed to **<font color=#f05050>GR-TRX</font>**. Next the packet is coded and modulated with BCC and OFDM. Finally the signal is shown on the Time Sink. If you enable the USRP Sink, the signal will be sent out through the given channel and the devices nearby in the same channel should receive it, for example, a wireless adaptor working in the monitor mode. The transmitting time can be spicified with UHD API.

4. Next we see the receiver. In the previous "How To Use **<font color=#f05050>PY-TB</font>**", you will have one bin files in **gr-ieee80211/tmp** folder named **sig80211GenMultipleSiso_1x1_0.bin**. Open GNU Radio, in GNU Radio, open the file **rx.grc** in **gr-ieee80211/example** folder. Run the flow graph and you will see

![alt text](https://github.com/cloud9477/gr-ieee80211/blob/main/figExample4.png?raw=true)

5. The figure above shows that the signal generated by the **pktGenExample.py** is received in the **rx.grc**. The Time Sink shows the signal and the received packets are printed in the console. If you enable the USRP Source and disable the File Source and Time Sink, you can receive the packets transmitted in the real WiFi channel. For example, the beacons sent from the router in your home.

6. For the above examples, if you have a USRP B210 supporting 2x2 MIMO, you can try the **tx2.grc** and **rx2.grc**.