# gr-ieee80211
- GNU Radio IEEE 802.11a/g/n/ac transmitter and receiver.
- Supports upto 2x2 SU-MIMO and MU-MIMO.
- Main branch is GNU Radio 3.10 version with CUDA.
- For the CPU version with CUDA, please checkout to maint-3.10-cpu.

# Receiver Design
- STF gives trigger of LTF starting point and the CFO updating point by a byte stream.
- LTF if triggered by STF, it runs auto correlation to find fine start timing, give the starting index and accurate CFO
- LTF estimates channel and passes it to signal block.
- Signal block demod and decode signal field, further check HT signal and VHT signal A information and check correctness.
- Demod demodulates OFDM and get LLR of bits.
- Decode uses soft viterbi

# Transmitter Design
- Transmitter encode: generate signal field bits
- Transmitter encode: encode psdu bits: scramble, bcc, stream parser, and interleave
- Transmitter modulation: generate training fields
- Transmitter modulation: modulate signal field with OFDM/QAM
- Transmitter modulation: modulate psdu with OFDM/QAM

Issue in TX
------------
![alt text](https://github.com/cloud9477/gr-ieee80211/blob/main/figRmMisalign.png?raw=true)
- As shown in the figure, when use USRP B210 with 2 TX channels to tranmsmit an MIMO packet, it shows the misalignment between the two spatial streams.
- This is solved by adding the "tx_time" tag to the two streams in the block modulation
- To use the uhd lib, in CMake, find the library and link it.
- At the same time, the Sync select box in USRP Sink changes to PC Clock on Next PPS
# How to use
- MAC layer packets are generated by python.
- MU-MIMO: channel estimation, V and Q matrix are also processed in python. Q is then passed to GNU Radio for later MU-MIMO transmissions.

# Installation
- Install GNU Radio 3.10 dev according to the GNU Radio offical website
- Install UHD
- Install liborc
- Copy and install gr-ieee80211 module for GNU Radio
- I usually put the GNU Radio modules in a folder named "sdr" in home folder, if you use a different path, please correct the paths in the codes. For example the python tools.
```console
sdr@sdr-home:~$ sudo apt-get install gnuradio-dev uhd-host cmake build-essential
sdr@sdr-home:~$ sudo apt-get install liborc-0.4-dev libuhd-dev
sdr@sdr-home:~$ mkdir sdr
sdr@sdr-home:~$ cd sdr
sdr@sdr-home:~$ git clone https://github.com/cloud9477/gr-ieee80211.git
sdr@sdr-home:~$ cd gr-ieee80211/
sdr@sdr-home:~$ mkdir build
sdr@sdr-home:~$ cd build
sdr@sdr-home:~$ cmake ../
sdr@sdr-home:~$ make
sdr@sdr-home:~$ sudo make install
sdr@sdr-home:~$ sudo ldconfig
```


# CUDA Related Info

RTX 3070
--------
- Zotac 3070 Gaming
- Device name: NVIDIA GeForce RTX 3070
- SM Number: 46
- SM Max Thread Number: 1536
- Block Share Mem Size: 49152
- Block Max Thread Number: 1024
- Memory Clock Rate (KHz): 7001000
- Memory Bus Width (bits): 256
- Peak Memory Bandwidth (GB/s): 448.064000
