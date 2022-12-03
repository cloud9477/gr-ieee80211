/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Viterbi Decode of CR 12, 23, 34, 56 Soft Ver.
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

#ifndef INCLUDED_IEEE80211_DECODE_IMPL_H
#define INCLUDED_IEEE80211_DECODE_IMPL_H

#include <gnuradio/ieee80211/decode.h>
#include "cloud80211phy.h"
#include <boost/crc.hpp>

#define dout d_debug&&std::cout

#define DECODE_S_IDLE 0
#define DECODE_S_DECODE 1
#define DECODE_S_CLEAN 2

/*
Legacy  max MPDU 4095
HT      max MPDU 7935     AMPDU 65535
VHT     max MPDU 11454    AMPDU 1048575

The max ampdu len can receive is indicated by different devices in HT Cap or VHT Cap
For VHT, max is 2^20-1=1048575, the minimal su packet apeplen/4 is 2^17=131072, 131072*4=524288
This phy only supports 20MHz, so we use 524288 here
However, for real usage, this may not necessary

#define DECODE_B_MAX 524288
#define DECODE_V_MAX 4196000    // max llr len
#define DECODE_D_MAX 11454      // max mpdu len

*/

#define DECODE_B_MAX 5000
#define DECODE_V_MAX 41000    // max llr len
#define DECODE_D_MAX 1600      // max mpdu len

namespace gr {
  namespace ieee80211 {

    class decode_impl : public decode
    {
    private:
      // block
      int d_nProc;
      int d_nGen;
      int d_sDecode;
      bool d_debug;
      uint64_t d_nPktCorrect;
      // tag
      std::vector<gr::tag_t> tags;
      int t_len;
      int t_format;
      int t_ampdu;
      int t_cr;
      int t_nUnCoded;
      int t_nTotal;
      int t_nProcd;
      int t_mcs;
      // NDP channel
      gr_complex d_mu2x1Chan[128];
      uint8_t d_mu2x1ChanFloatBytes[1027];
      std::vector<gr_complex> d_tagMu2x1Chan;
      // viterbi
      float v_accum_err0[64];
	    float v_accum_err1[64];
      float *v_ae_pTmp, *v_ae_pPre, *v_ae_pCur;
      int v_state_his[64][DECODE_V_MAX+1];
      int v_state_seq[DECODE_V_MAX+1];
      int v_op0, v_op1, v_next0, v_next1;
      float v_acc_tmp0, v_acc_tmp1, v_t0, v_t1;
      float v_tab_t[4];
      int v_t;
      int v_trellis;
      const int *v_cr_punc;
      int v_cr_p, v_cr_len;
      uint8_t v_scramBits[DECODE_V_MAX];
      uint8_t v_unCodedBits[DECODE_V_MAX];
      boost::crc_32_type d_crc32;
      uint8_t d_pktBytes[DECODE_D_MAX];
      int d_dataLen;
      // debug
      uint64_t d_legacyMcsCount[8];
      uint64_t d_vhtMcsCount[10];
      uint64_t d_htMcsCount[8];
      int d_inParam;


    public:
      decode_impl(int inpara);
      ~decode_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

      void vstb_init();
      int vstb_update(const float* llr, int len);
      void vstb_end();
      void descramble();
      void packetAssemble();
    };

  } // namespace ieee80211
} // namespace gr

#endif /* INCLUDED_IEEE80211_DECODE_IMPL_H */
