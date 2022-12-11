#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# GNU Radio version: 3.10.1.1

from gnuradio import analog
from gnuradio import blocks
import pmt
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import ieee80211
from gnuradio import network
import time




class wifirx(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 20e6
        self.freq_132 = freq_132 = 5660e6
        self.freq_100 = freq_100 = 5500e6

        ##################################################
        # Blocks
        ##################################################
        self.network_socket_pdu_0 = network.socket_pdu('UDP_CLIENT', '127.0.0.1', '9527', 65535, False)
        self.ieee80211_trigger_0 = ieee80211.trigger()
        self.ieee80211_sync_0 = ieee80211.sync()
        self.ieee80211_signal_0 = ieee80211.signal()
        self.ieee80211_preproc_0 = ieee80211.preproc()
        self.ieee80211_demod_0 = ieee80211.demod(0, 2)
        self.ieee80211_decode_0 = ieee80211.decode(1)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, '/home/cloud/sdr/sig80211GenMultipleSiso_1x1_200.bin', False, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, 'presiso.bin', False)
        self.blocks_file_sink_0.set_unbuffered(False)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_fastnoise_source_x_0 = analog.fastnoise_source_c(analog.GR_GAUSSIAN, 0.01, 6543, 8192)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.ieee80211_decode_0, 'out'), (self.network_socket_pdu_0, 'pdus'))
        self.connect((self.analog_fastnoise_source_x_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.ieee80211_preproc_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.ieee80211_signal_0, 1))
        self.connect((self.blocks_add_xx_0, 0), (self.ieee80211_sync_0, 2))
        self.connect((self.blocks_file_source_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.ieee80211_demod_0, 0), (self.ieee80211_decode_0, 0))
        self.connect((self.ieee80211_preproc_0, 1), (self.ieee80211_sync_0, 1))
        self.connect((self.ieee80211_preproc_0, 0), (self.ieee80211_trigger_0, 0))
        self.connect((self.ieee80211_signal_0, 0), (self.ieee80211_demod_0, 0))
        self.connect((self.ieee80211_sync_0, 0), (self.ieee80211_signal_0, 0))
        self.connect((self.ieee80211_trigger_0, 0), (self.ieee80211_sync_0, 0))


    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate

    def get_freq_132(self):
        return self.freq_132

    def set_freq_132(self, freq_132):
        self.freq_132 = freq_132

    def get_freq_100(self):
        return self.freq_100

    def set_freq_100(self, freq_100):
        self.freq_100 = freq_100




def main(top_block_cls=wifirx, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
