#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# GNU Radio version: 3.10.4.0

import os
import sys
sys.path.append(os.environ.get('GRC_HIER_PATH', os.path.expanduser('~/.grc_gnuradio')))

from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import ieee80211
from gnuradio import network
from gnuradio import uhd
import time
from presiso import presiso  # grc-generated hier_block




class wifirx(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 20e6
        self.freq = freq = 5500e6

        ##################################################
        # Blocks
        ##################################################
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("", '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_time_unknown_pps(uhd.time_spec(0))

        self.uhd_usrp_source_0.set_center_freq(freq, 0)
        self.uhd_usrp_source_0.set_antenna("RX2", 0)
        self.uhd_usrp_source_0.set_bandwidth(20e6, 0)
        self.uhd_usrp_source_0.set_normalized_gain(0.8, 0)
        self.presiso_0 = presiso()
        self.network_socket_pdu_0 = network.socket_pdu('UDP_CLIENT', '127.0.0.1', '9527', 65535, False)
        self.ieee80211_trigger_0 = ieee80211.trigger()
        self.ieee80211_sync_0 = ieee80211.sync()
        self.ieee80211_signal_0 = ieee80211.signal()
        self.ieee80211_demod_0 = ieee80211.demod(0, 2)
        self.ieee80211_decode_0 = ieee80211.decode(1)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.ieee80211_decode_0, 'out'), (self.network_socket_pdu_0, 'pdus'))
        self.connect((self.ieee80211_demod_0, 0), (self.ieee80211_decode_0, 0))
        self.connect((self.ieee80211_signal_0, 0), (self.ieee80211_demod_0, 0))
        self.connect((self.ieee80211_sync_0, 0), (self.ieee80211_signal_0, 0))
        self.connect((self.ieee80211_trigger_0, 0), (self.ieee80211_sync_0, 0))
        self.connect((self.presiso_0, 1), (self.ieee80211_sync_0, 1))
        self.connect((self.presiso_0, 0), (self.ieee80211_trigger_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.ieee80211_signal_0, 1))
        self.connect((self.uhd_usrp_source_0, 0), (self.ieee80211_sync_0, 2))
        self.connect((self.uhd_usrp_source_0, 0), (self.presiso_0, 0))


    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.uhd_usrp_source_0.set_center_freq(self.freq, 0)




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
