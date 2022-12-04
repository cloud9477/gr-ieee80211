#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# GNU Radio version: 3.10.4.0

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
from gnuradio import uhd
import time




class wifitx(gr.top_block):

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
        self.uhd_usrp_sink_0_0 = uhd.usrp_sink(
            ",".join(("", '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,2)),
            ),
            "packet_len",
        )
        self.uhd_usrp_sink_0_0.set_samp_rate(samp_rate)
        self.uhd_usrp_sink_0_0.set_time_unknown_pps(uhd.time_spec(0))

        self.uhd_usrp_sink_0_0.set_center_freq(freq, 0)
        self.uhd_usrp_sink_0_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0_0.set_bandwidth(20e6, 0)
        self.uhd_usrp_sink_0_0.set_normalized_gain(0.9, 0)

        self.uhd_usrp_sink_0_0.set_center_freq(freq, 1)
        self.uhd_usrp_sink_0_0.set_antenna("TX/RX", 1)
        self.uhd_usrp_sink_0_0.set_bandwidth(20e6, 1)
        self.uhd_usrp_sink_0_0.set_normalized_gain(0.9, 1)
        self.network_socket_pdu_0 = network.socket_pdu('UDP_SERVER', '127.0.0.1', '9528', 65535, False)
        self.ieee80211_modulation_0 = ieee80211.modulation()
        self.ieee80211_encode_0 = ieee80211.encode('packet_len')


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.network_socket_pdu_0, 'pdus'), (self.ieee80211_encode_0, 'pdus'))
        self.connect((self.ieee80211_encode_0, 1), (self.ieee80211_modulation_0, 1))
        self.connect((self.ieee80211_encode_0, 0), (self.ieee80211_modulation_0, 0))
        self.connect((self.ieee80211_modulation_0, 0), (self.uhd_usrp_sink_0_0, 0))
        self.connect((self.ieee80211_modulation_0, 1), (self.uhd_usrp_sink_0_0, 1))


    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_sink_0_0.set_samp_rate(self.samp_rate)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.uhd_usrp_sink_0_0.set_center_freq(self.freq, 0)
        self.uhd_usrp_sink_0_0.set_center_freq(self.freq, 1)




def main(top_block_cls=wifitx, options=None):
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
