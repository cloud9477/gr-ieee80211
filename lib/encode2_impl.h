/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Encoder of 802.11a/g/n/ac 2x2 payload part
 *     Copyright (C) June 1, 2022  Zelin Yun
 *
 *     This program is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU Affero General Public License as published
 *     by the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU Affero General Public License for more details.
 *
 *     You should have received a copy of the GNU Affero General Public License
 *     along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef INCLUDED_IEEE80211_ENCODE2_IMPL_H
#define INCLUDED_IEEE80211_ENCODE2_IMPL_H

#include <gnuradio/ieee80211/encode2.h>
#include "cloud80211phy.h"

#define ENCODE_S_RDTAG 1
#define ENCODE_S_RDPKT 2
#define ENCODE_S_MOD 3
#define ENCODE_S_COPY 4
#define ENCODE_S_CLEAN 5

#define ENCODE_GR_PAD 160

namespace gr {
  namespace ieee80211 {

    class encode2_impl : public encode2
    {
    private:
      int d_sEncode;
      int d_nProc;
      int d_nGen;
      int d_nUsed;
      int d_nPassed;
      // input pkt
      std::vector<gr::tag_t> d_tags;
      int d_pktFormat;
      int d_pktSeq;
      int d_pktMcs0;
      int d_pktNss0;
      int d_pktLen0;
      int d_pktMcs1;
      int d_pktNss1;
      int d_pktLen1;
      int d_pktMuGroupId;
      int d_nPktTotal;
      int d_nPktRead;
      uint8_t d_sigBitsL[24];
      uint8_t d_sigBitsCodedL[48];
      uint8_t d_sigBitsNL[48];
      uint8_t d_sigBitsCodedNL[96];
      uint8_t d_sigBitsB[26];
      uint8_t d_sigBitsCodedB[52];
      std::vector<uint8_t> d_sigBitsIntedL;
      std::vector<uint8_t> d_sigBitsIntedNL;
      std::vector<uint8_t> d_sigBitsIntedB0;
      std::vector<uint8_t> d_sigBitsIntedB1;
      uint8_t d_pkt[4095];
      uint8_t d_bits0[32864];
      uint8_t d_bits1[65728];
      uint8_t d_bitsCoded[65728];
      uint8_t d_bitsPunct[65728];
      uint8_t d_bitsStream0[65728];
      uint8_t d_bitsStream1[65728];
      uint8_t d_bitsInted0[65728];
      uint8_t d_bitsInted1[65728];
      uint8_t d_chips0[65728];
      uint8_t d_chips1[65728];
      c8p_mod d_m;
      // copy samples out
      int d_nSampTotal;
      int d_nSampCopied;

     public:
      encode2_impl();
      ~encode2_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

    };

  } // namespace ieee80211
} // namespace gr

#endif /* INCLUDED_IEEE80211_ENCODE2_IMPL_H */
