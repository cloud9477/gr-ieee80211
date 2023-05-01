#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# GNU Radio version: 3.10.1.1

import os
import sys
sys.path.append(os.environ.get('GRC_HIER_PATH', os.path.expanduser('~/.grc_gnuradio')))

from gnuradio import analog
from gnuradio import blocks
import pmt
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import ieee80211
from gnuradio import network
from presiso import presiso  # grc-generated hier_block
import time


terminalNoiseAmp = 0
if(len(sys.argv) > 1):
    terminalNoiseAmp = float(sys.argv[1])



class wifirx2(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 20e6
        self.noise_amp = noise_amp = 0.000
        self.freq_132 = freq_132 = 5660e6
        self.freq_100 = freq_100 = 5500e6

        ##################################################
        # Blocks
        ##################################################
        self.presiso_0 = presiso()
        self.network_socket_pdu_0 = network.socket_pdu('UDP_CLIENT', '127.0.0.1', '9527', 65535, False)
        self.ieee80211_trigger_0 = ieee80211.trigger()
        self.ieee80211_sync_0 = ieee80211.sync()
        self.ieee80211_signal2_0 = ieee80211.signal2()
        self.ieee80211_demod2_0 = ieee80211.demod2()
        self.ieee80211_decode_0 = ieee80211.decode()
        self.blocks_file_source_0_0 = blocks.file_source(gr.sizeof_gr_complex*1, '/home/cloud/sdr/sig80211GenMultipleMimo_2x2_0.bin', False, 0, 0)
        self.blocks_file_source_0_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, '/home/cloud/sdr/sig80211GenMultipleMimo_2x2_1.bin', False, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_add_xx_0_0 = blocks.add_vcc(1)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_fastnoise_source_x_0 = analog.fastnoise_source_c(analog.GR_GAUSSIAN, terminalNoiseAmp, 0, 8192)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.ieee80211_decode_0, 'out'), (self.network_socket_pdu_0, 'pdus'))
        self.connect((self.analog_fastnoise_source_x_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.analog_fastnoise_source_x_0, 0), (self.blocks_add_xx_0_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.ieee80211_signal2_0, 1))
        self.connect((self.blocks_add_xx_0, 0), (self.ieee80211_sync_0, 2))
        self.connect((self.blocks_add_xx_0, 0), (self.presiso_0, 0))
        self.connect((self.blocks_add_xx_0_0, 0), (self.ieee80211_signal2_0, 2))
        self.connect((self.blocks_file_source_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.blocks_file_source_0_0, 0), (self.blocks_add_xx_0_0, 1))
        self.connect((self.ieee80211_demod2_0, 0), (self.ieee80211_decode_0, 0))
        self.connect((self.ieee80211_signal2_0, 0), (self.ieee80211_demod2_0, 0))
        self.connect((self.ieee80211_signal2_0, 1), (self.ieee80211_demod2_0, 1))
        self.connect((self.ieee80211_sync_0, 0), (self.ieee80211_signal2_0, 0))
        self.connect((self.ieee80211_trigger_0, 0), (self.ieee80211_sync_0, 0))
        self.connect((self.presiso_0, 1), (self.ieee80211_sync_0, 1))
        self.connect((self.presiso_0, 0), (self.ieee80211_trigger_0, 0))


    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate

    def get_noise_amp(self):
        return self.noise_amp

    def set_noise_amp(self, noise_amp):
        self.noise_amp = noise_amp
        self.analog_fastnoise_source_x_0.set_amplitude(self.noise_amp)

    def get_freq_132(self):
        return self.freq_132

    def set_freq_132(self, freq_132):
        self.freq_132 = freq_132

    def get_freq_100(self):
        return self.freq_100

    def set_freq_100(self, freq_100):
        self.freq_100 = freq_100




def main(top_block_cls=wifirx2, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.wait()


if __name__ == '__main__':


    main()

