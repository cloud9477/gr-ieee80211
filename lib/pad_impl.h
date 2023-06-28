/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 20M bw and upto 2x2
 *     TX Legacy preamble and padding
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

#ifndef INCLUDED_IEEE80211_PAD_IMPL_H
#define INCLUDED_IEEE80211_PAD_IMPL_H

#include <gnuradio/ieee80211/pad.h>
#include <gnuradio/fft/fft.h>
#include "cloud80211phy.h"

#define PAD_S_TAG 0
#define PAD_S_PRE 1
#define PAD_S_SIG 2
#define PAD_S_DATA 3

#define PAD_SCALE 5.333333f

namespace gr {
  namespace ieee80211 {

    class pad_impl : public pad
    {
    private:
      int d_sPad;
      int d_nProc;
      int d_nGen;
      int d_nProced;
      int d_nGened;
      std::vector<gr::tag_t> d_tags;
      int d_pktFormat;
      int d_pktNss;
      int d_pktLen;
      float d_scaler;
      gr_complex d_preamblel0[400];
      fft::fft_complex_rev d_ofdm_fft;
      int d_nSampCopied;
      int d_nSampTotal;
      float d_scaleMask[320];
      int d_scaleTotal;
    public:
      pad_impl();
      ~pad_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

    };

  } // namespace ieee80211
} // namespace gr

#endif /* INCLUDED_IEEE80211_PAD_IMPL_H */
