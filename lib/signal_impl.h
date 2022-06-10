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

#ifndef INCLUDED_IEEE80211_SIGNAL_IMPL_H
#define INCLUDED_IEEE80211_SIGNAL_IMPL_H

#include <ieee80211/signal.h>
#include <fftw3.h>
#include "cloud80211phy.h"

#define S_TRIGGER 0
#define S_DEMOD 1
#define S_NONLEGACY 2
#define S_COPY 3

namespace gr {
  namespace ieee80211 {

    class signal_impl : public signal
    {
     private:
      // for block
      int d_nProc;
      // for process
      int d_sSignal;    //  block state
      int d_nSym; 
      int d_nSample;
      gr_complex d_H[64];
      gr_complex d_sig[64];
      // hard viterbi ver
      uint8_t d_sigIntedBits[48];
      uint8_t d_sigCodedBits[48];
      // soft viterbi ver
      float d_sigIntedLlr[48];
      float d_sigCodedLlr[48];
      uint8_t d_sigBits[24];
      // fft
      fftw_complex* d_fftLtfIn1;
      fftw_complex* d_fftLtfIn2;
      fftw_complex* d_fftLtfOut1;
      fftw_complex* d_fftLtfOut2;
      fftw_complex* d_fftSigIn;
      fftw_complex* d_fftSigOut;
      fftw_plan d_fftP;
      // for debug
      int d_fTest;

     public:
      signal_impl();
      ~signal_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

    };

  } // namespace ieee80211
} // namespace gr

#endif /* INCLUDED_IEEE80211_SIGNAL_IMPL_H */
