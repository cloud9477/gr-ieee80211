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
        self.freq_132 = freq_132 = 5660e6
        self.freq_112 = freq_112 = 5560e6
        self.freq_100 = freq_100 = 5500e6

        ##################################################
        # Blocks
        ##################################################
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join(("", '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
            "len",
        )
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        _last_pps_time = self.uhd_usrp_sink_0.get_time_last_pps().get_real_secs()
        # Poll get_time_last_pps() every 50 ms until a change is seen
        while(self.uhd_usrp_sink_0.get_time_last_pps().get_real_secs() == _last_pps_time):
            time.sleep(0.05)
        # Set the time to PC time on next PPS
        self.uhd_usrp_sink_0.set_time_next_pps(uhd.time_spec(int(time.time()) + 1.0))
        # Sleep 1 second to ensure next PPS has come
        time.sleep(1)

        self.uhd_usrp_sink_0.set_center_freq(freq_112, 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0.set_bandwidth(20e6, 0)
        self.uhd_usrp_sink_0.set_normalized_gain(0.87, 0)
        self.network_socket_pdu_0 = network.socket_pdu('UDP_SERVER', '127.0.0.1', '9528', 65535, False)
        self.ieee80211_modulation_0 = ieee80211.modulation()
        self.ieee80211_encode_0 = ieee80211.encode('packet_len')
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_fastnoise_source_x_0 = analog.fastnoise_source_c(analog.GR_GAUSSIAN, noiseAmp, 13579, 8192)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.network_socket_pdu_0, 'pdus'), (self.ieee80211_encode_0, 'pdus'))
        self.connect((self.analog_fastnoise_source_x_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.blocks_add_xx_0, 0), (self.uhd_usrp_sink_0, 0))
        self.connect((self.ieee80211_encode_0, 1), (self.ieee80211_modulation_0, 1))
        self.connect((self.ieee80211_encode_0, 0), (self.ieee80211_modulation_0, 0))
        self.connect((self.ieee80211_modulation_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.ieee80211_modulation_0, 1), (self.blocks_null_sink_0, 0))


    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)

    def get_freq_132(self):
        return self.freq_132

    def set_freq_132(self, freq_132):
        self.freq_132 = freq_132

    def get_freq_112(self):
        return self.freq_112

    def set_freq_112(self, freq_112):
        self.freq_112 = freq_112
        self.uhd_usrp_sink_0.set_center_freq(self.freq_112, 0)

    def get_freq_100(self):
        return self.freq_100

    def set_freq_100(self, freq_100):
        self.freq_100 = freq_100




def main(top_block_cls=wifitx, options=None):
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
    noiseAmp = 0
    if(len(sys.argv) > 1):
        noiseAmp = float(sys.argv[1])
    main()
