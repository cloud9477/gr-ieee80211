#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# GNU Radio version: 3.9.5.0

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
from gnuradio import network
from gnuradio import uhd
import time
import ieee80211




class sta0(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 20e6
        self.rxp = rxp = 0.7
        self.freq = freq = 5420e6

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
        self.uhd_usrp_source_0.set_normalized_gain(rxp, 0)
        self.network_udp_sink_0 = network.udp_sink(gr.sizeof_char, 1, '127.0.0.1', 9527, 0, 1400, False)
        self.ieee80211_trigger_0 = ieee80211.trigger()
        self.ieee80211_sync_0 = ieee80211.sync()
        self.ieee80211_signal_0 = ieee80211.signal(1)
        self.ieee80211_demod_0 = ieee80211.demod(1, 0, 2)
        self.ieee80211_decode_0 = ieee80211.decode()
        self.blocks_pdu_to_tagged_stream_0 = blocks.pdu_to_tagged_stream(blocks.byte_t, 'packet_len')
        self.blocks_multiply_conjugate_cc_0 = blocks.multiply_conjugate_cc(1)
        self.blocks_moving_average_xx_1 = blocks.moving_average_ff(64, 1, 4000, 1)
        self.blocks_moving_average_xx_0 = blocks.moving_average_cc(48, 1, 4000, 1)
        self.blocks_divide_xx_0 = blocks.divide_ff(1)
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, 16)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(1)
        self.blocks_complex_to_mag_0 = blocks.complex_to_mag(1)
        self.analog_const_source_x_0 = analog.sig_source_c(0, analog.GR_CONST_WAVE, 0, 0, 0)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.ieee80211_decode_0, 'out'), (self.blocks_pdu_to_tagged_stream_0, 'pdus'))
        self.connect((self.analog_const_source_x_0, 0), (self.ieee80211_signal_0, 3))
        self.connect((self.blocks_complex_to_mag_0, 0), (self.blocks_divide_xx_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_moving_average_xx_1, 0))
        self.connect((self.blocks_delay_0, 0), (self.blocks_multiply_conjugate_cc_0, 0))
        self.connect((self.blocks_divide_xx_0, 0), (self.ieee80211_trigger_0, 0))
        self.connect((self.blocks_moving_average_xx_0, 0), (self.blocks_complex_to_mag_0, 0))
        self.connect((self.blocks_moving_average_xx_0, 0), (self.ieee80211_sync_0, 2))
        self.connect((self.blocks_moving_average_xx_1, 0), (self.blocks_divide_xx_0, 1))
        self.connect((self.blocks_multiply_conjugate_cc_0, 0), (self.blocks_moving_average_xx_0, 0))
        self.connect((self.blocks_pdu_to_tagged_stream_0, 0), (self.network_udp_sink_0, 0))
        self.connect((self.ieee80211_demod_0, 0), (self.ieee80211_decode_0, 0))
        self.connect((self.ieee80211_signal_0, 1), (self.ieee80211_demod_0, 2))
        self.connect((self.ieee80211_signal_0, 0), (self.ieee80211_demod_0, 1))
        self.connect((self.ieee80211_sync_0, 0), (self.ieee80211_demod_0, 0))
        self.connect((self.ieee80211_sync_0, 1), (self.ieee80211_signal_0, 1))
        self.connect((self.ieee80211_sync_0, 0), (self.ieee80211_signal_0, 0))
        self.connect((self.ieee80211_trigger_0, 0), (self.ieee80211_sync_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_delay_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_multiply_conjugate_cc_0, 1))
        self.connect((self.uhd_usrp_source_0, 0), (self.ieee80211_signal_0, 2))
        self.connect((self.uhd_usrp_source_0, 0), (self.ieee80211_sync_0, 1))


    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_rxp(self):
        return self.rxp

    def set_rxp(self, rxp):
        self.rxp = rxp
        self.uhd_usrp_source_0.set_normalized_gain(self.rxp, 0)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.uhd_usrp_source_0.set_center_freq(self.freq, 0)




def main(top_block_cls=sta0, options=None):
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
