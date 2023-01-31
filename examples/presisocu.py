# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: preproccu
# GNU Radio version: 3.10.1.1

from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from gnuradio import ieee80211







class presisocu(gr.hier_block2):
    def __init__(self):
        gr.hier_block2.__init__(
            self, "preproccu",
                gr.io_signature(1, 1, gr.sizeof_gr_complex*1),
                gr.io_signature.makev(2, 2, [gr.sizeof_float*1, gr.sizeof_gr_complex*1]),
        )

        ##################################################
        # Blocks
        ##################################################
        self.ieee80211_preproc_0 = ieee80211.preproc()
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, 63)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_delay_0, 0), (self.ieee80211_preproc_0, 0))
        self.connect((self.ieee80211_preproc_0, 0), (self, 0))
        self.connect((self.ieee80211_preproc_0, 1), (self, 1))
        self.connect((self, 0), (self.blocks_delay_0, 0))


