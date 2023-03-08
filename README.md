# gr-ieee80211
- GNU Radio IEEE 802.11a/g/n/ac transmitter and receiver.
- Supports upto 2x2 SU-MIMO and MU-MIMO.
- Ubuntu 22.04 and GNU Radio 3.10.
- For CUDA DLC please refer to [gr-ieee80211cu](https://github.com/cloud9477/gr-ieee80211cu).

![alt text](https://github.com/cloud9477/gr-ieee80211/blob/main/figGrTrx.png?raw=true)

Receiver Design
---------------
- Pre-Processing: Auto-correlation of STF and the coarse CFO.
- Trigger: Detect auto-correlation plateau, give trigger of LTF with coarse starting point.
- Sync: Triggered by STF, use auto-correlation of LTF to find starting timing, re-estimate accurate CFO.
- Signal: Compensate CFO, estimate Legacy channel and passes channel and samples to Demod block.
- Demod: Further check HT signal and VHT signal A to get the correct packet format, demodulates OFDM and get soft bits.
- Decode: uses soft viterbi to decode, check FCS and assemble the packet.
- Demod and Decode: for NDP, also pass the channel info to MAC.

Transmitter Design
---------------
- Transmitter encode: generate signal field bits
- Transmitter encode: encode psdu bits: scramble, bcc, stream parser, and interleave
- Transmitter modulation: generate training fields
- Transmitter modulation: modulate signal field with OFDM/QAM
- Transmitter modulation: modulate psdu with OFDM/QAM

How to use
---------------
- For basic TRX, you could use the python scripts in the **tools** or modify this module for your own purposes.
- The python scripts also have the details about how to generate an IEEE 802.11a/g/n/ac packet with non-optimized steps following the IEEE 802.11 standards. The C++ codes in the GNU Radio module could be difficult to understand before getting familiar with the standards.
- In MU-MIMO, the channel estimation, V and Q matrix are also processed in python. Q is then passed to GNU Radio for later MU-MIMO transmissions. For more details about MU-MIMO experiments, please go to the **tools** folder.

Installation
---------------
- Install GNU Radio 3.10 dev according to the GNU Radio offical website
- Install UHD
- Install liborc
- Copy and install gr-ieee80211 module for GNU Radio
- I usually put the GNU Radio modules in a folder named "sdr" in home folder, if you use a different path, please correct the paths in the codes. For example the python tools.
```console
sdr@sdr:~$ sudo apt-get install gnuradio-dev uhd-host libuhd-dev cmake build-essential
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


Some Tips
----------
- Use top to record CPU usage, the -1 is usage by core, -d is interval, -n is sample number, -b is logging.
```console
sdr@sdr:~$ top -1 -d 0.1 -n 200 -b > toplog.txt
```
