/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Demodulation of 802.11a/g/n/ac 1x1 and 2x2 formats
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

#ifndef INCLUDED_IEEE80211_DEMOD2_IMPL_H
#define INCLUDED_IEEE80211_DEMOD2_IMPL_H

#include <gnuradio/ieee80211/demod2.h>
#include <gnuradio/fft/fft.h>
#include "cloud80211phy.h"

#define dout d_debug&&std::cout

#define DEMOD_S_SYNC 0
#define DEMOD_S_RDTAG 1
#define DEMOD_S_FORMAT 2
#define DEMOD_S_VHT 3
#define DEMOD_S_HT 4
#define DEMOD_S_LEGACY 5
#define DEMOD_S_WRTAG 6
#define DEMOD_S_DEMOD 7
#define DEMOD_S_CLEAN 8


namespace gr {
  namespace ieee80211 {

    class demod2_impl : public demod2
    {
     private:
      int d_testCount;
      // block
      bool d_debug;
      int d_nProc;
      int d_nGen;
      int d_sDemod;
      // parameters
      int d_nRxAnt;
      int d_muPos;
      int d_muGroupId;
      // received info from tag
      std::vector<gr::tag_t> tags;
      int d_nSigLMcs;
      int d_nSigLLen;
      gr_complex d_H[64];
      int d_nSigLSamp;
      int d_nSampConsumed;
      // check format
      gr_complex d_sig1[64];
      gr_complex d_sig2[64];
      float d_sigHtIntedLlr[96];
      float d_sigHtCodedLlr[96];
      float d_sigVhtAIntedLlr[96];
      float d_sigVhtACodedLlr[96];
      float d_sigVhtB20IntedLlr[52];
      float d_sigVhtB20CodedLlr[52];
      uint8_t d_sigHtBits[48];
      uint8_t d_sigVhtABits[48];
      uint8_t d_sigVhtB20Bits[26];
      // fft
      fft::fft_complex_fwd d_ofdm_fft;
      gr_complex d_fftLtfOut1[64];
      gr_complex d_fftLtfOut2[64];
      gr_complex d_fftLtfOut12[64];
      gr_complex d_fftLtfOut22[64];
      // packet info
      c8p_mod d_m;
      c8p_sigHt d_sigHt;
      c8p_sigVhtA d_sigVhtA;
      
      gr_complex d_mu2x1Chan[128];
      std::vector<gr_complex> d_tagMu2x1Chan;
      int d_nSymProcd;
      int d_unCoded;
      int d_nTrellis;
      // pilot
      int d_pilotP;
      float d_pilot[4];
      float d_pilot2[4];
      // non-legacy channel
      /*      P_LTF 4x4
      | 1 -1  1  1 |
      | 1  1 -1  1 |
      | 1  1  1 -1 |
      |-1  1  1  1 |
      for 2x2 in the array
      | 0  1 |
      | 2  3 |      */
      gr_complex d_H_NL[64][C8P_MAX_N_LTF];
      gr_complex d_H_NL_INV[64][C8P_MAX_N_LTF];
      gr_complex d_qam[C8P_MAX_N_SS][52];
      float d_llrInted[C8P_MAX_N_SS][C8P_MAX_N_CBPSS];     // interleaved LLR
      float d_llrSpasd[C8P_MAX_N_SS][C8P_MAX_N_CBPSS];     // stream parsered LLR

     public:
      demod2_impl(int mupos, int mugid);
      ~demod2_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
      void nonLegacyChanEstimate(const gr_complex* sig1, const gr_complex* sig2);
      void vhtChanUpdate(const gr_complex* sig1, const gr_complex* sig2);
      void htChanUpdate(const gr_complex* sig1, const gr_complex* sig2);
      void legacyChanUpdate(const gr_complex* sig1);
      void vhtSigBDemod(const gr_complex* sig1, const gr_complex* sig2);
      void fft(const gr_complex* sig, gr_complex* res);
      void pilotShift(float* pilots);

    };

  } // namespace ieee80211
} // namespace gr

#endif /* INCLUDED_IEEE80211_DEMOD2_IMPL_H */
