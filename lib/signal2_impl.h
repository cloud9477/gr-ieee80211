/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Legacy Signal Field Information
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

#ifndef INCLUDED_IEEE80211_SIGNAL2_IMPL_H
#define INCLUDED_IEEE80211_SIGNAL2_IMPL_H

#include <gnuradio/ieee80211/signal2.h>
#include <gnuradio/fft/fft.h>
#include <volk/volk.h>
#include "cloud80211phy.h"

#define S_TRIGGER 0
#define S_DEMOD 1
#define S_COPY 2
#define S_PAD 3

namespace gr {
  namespace ieee80211 {

    class signal2_impl : public signal2
    {
     private:
      // for block
      int d_sSignal;
      int d_nProc;
      int d_nGen;
      int d_nUsed;
      int d_nPassed;
      // signal soft viterbi ver
      svSigDecoder d_decoder;
      float d_cfoRad;
      float d_snr;
      std::vector<gr_complex> d_h;
      float d_sigLegacyCodedLlr[48];
      uint8_t d_sigLegacyBits[24];
      int d_nSigPktSeq;
      int d_nSigMcs;
      int d_nSigLen;
      int d_nSigDBPS;
      int d_nSymbol;
      int d_nSample;
      int d_nSampleCopied;
      // fft
      fft::fft_complex_fwd d_ofdm_fft1;
      fft::fft_complex_fwd d_ofdm_fft2;
      fft::fft_complex_fwd d_ofdm_ffts;
      gr_complex *d_fftin1;
      gr_complex *d_fftin2;
      gr_complex *d_fftins;
      const gr_complex *d_sampin1;
      const gr_complex *d_sampin2;
      const gr_complex *d_sampins;

     public:
      signal2_impl();
      ~signal2_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
    };

  } // namespace ieee80211
} // namespace gr

#endif /* INCLUDED_IEEE80211_SIGNAL2_IMPL_H */
