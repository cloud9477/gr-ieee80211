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

#define DEMOD_S_RDTAG 0
#define DEMOD_S_FORMAT 1
#define DEMOD_S_VHT 2
#define DEMOD_S_HT 3
#define DEMOD_S_LEGACY 4
#define DEMOD_S_COPY 5
#define DEMOD_S_CLEAN 6


namespace gr {
  namespace ieee80211 {

    class demod2_impl : public demod2
    {
     private:
      // block
      int d_nProc;
      int d_nGen;
      int d_nProced;
      int d_nGened;
      int d_sDemod;
      // received info from tag
      std::vector<gr::tag_t> d_tags;
      float d_cfo;
      float d_snr;
      float d_rssi;
      int d_pktSeq;
      int d_pktMcsL;
      int d_pktLenL;
      int d_nSampTotal;
      int d_nSampConsumed;
      int d_nSymProcd;

      fft::fft_complex_fwd d_ofdm_fft1;
      fft::fft_complex_fwd d_ofdm_fft2;

      std::vector<gr_complex> d_HL;
      std::vector<gr_complex> d_HNL;
      std::vector<gr_complex> d_HNL2;
      std::vector<gr_complex> d_HNL2INV;
      // check format
      svSigDecoder d_decoder;
      float d_sigHtIntedLlr[96];
      float d_sigHtCodedLlr[96];
      float d_sigVhtAIntedLlr[96];
      float d_sigVhtACodedLlr[96];
      float d_sigVhtB20IntedLlr[52];
      float d_sigVhtB20CodedLlr[52];
      uint8_t d_sigHtBits[48];
      uint8_t d_sigVhtABits[48];
      uint8_t d_sigVhtB20Bits[26];
      // fft buffer
      gr_complex d_fftLtfOut1[64];
      gr_complex d_fftLtfOut2[64];
      // packet info
      c8p_mod d_m;
      c8p_sigHt d_sigHt;
      c8p_sigVhtA d_sigVhtA;

     public:
      demod2_impl();
      ~demod2_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
      void chanEstNL(const gr_complex* sig1, const gr_complex* sig2);
      void vhtSigBDemod(const gr_complex* sig1, const gr_complex* sig2);
      void addTag();
    };

  } // namespace ieee80211
} // namespace gr

#endif /* INCLUDED_IEEE80211_DEMOD2_IMPL_H */
