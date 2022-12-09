/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2, for SISO
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

#ifndef INCLUDED_IEEE80211_SIGNAL_IMPL_H
#define INCLUDED_IEEE80211_SIGNAL_IMPL_H

#include <gnuradio/ieee80211/signal.h>
#include <gnuradio/fft/fft.h>
#include <volk/volk.h>
#include <chrono>
#include "cloud80211phy.h"
#include "cloud80211phycu.cuh"

#define S_TRIGGER 0
#define S_DEMOD 1
#define S_COPY 2
#define S_PAD 3

#define F_LEGACY 0
#define F_HT 1
#define F_VHT 2

#define dout d_debug&&std::cout

namespace gr {
  namespace ieee80211 {

    class signal_impl : public signal
    {
    private:
      // for block
      int d_nProc;
      int d_nGen;
      int d_sSignal;
      bool d_debug;
      // tag
      float t_rad;
      float t_snr;
      // signal soft viterbi ver
      float d_cfoRad;
      gr_complex d_sigAfterCfoComp[240];
      gr_complex d_H[64];
      gr_complex d_sig[64];
      float d_sigLegacyIntedLlr[48];
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
      fft::fft_complex_fwd d_ofdm_fft;
      gr_complex d_fftLtfOut1[64];
      gr_complex d_fftLtfOut2[64];
      gr_complex d_fftSigOut[64];
      size_t d_fftSize;
      // performance
      std::chrono::_V2::system_clock::time_point d_ts;
      std::chrono::_V2::system_clock::time_point d_te;
      uint64_t d_usUsed;
      uint64_t d_sampCount;

     public:
      signal_impl();
      ~signal_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
      void fftDemod(gr_complex* in, gr_complex* out);
    };

  } // namespace ieee80211
} // namespace gr

#endif /* INCLUDED_IEEE80211_SIGNAL_IMPL_H */
