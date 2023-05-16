/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Long Training Field Sync
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

#ifndef INCLUDED_IEEE80211_SYNC_IMPL_H
#define INCLUDED_IEEE80211_SYNC_IMPL_H

#include <gnuradio/ieee80211/sync.h>
#include <chrono>

#define SYNC_S_IDLE 0
#define SYNC_S_SYNC 1

#define SYNC_MAX_BUF_LEN 240
#define SYNC_MAX_RES_LEN SYNC_MAX_BUF_LEN - 128 - 1

namespace gr {
  namespace ieee80211 {

    class sync_impl : public sync
    {
     private:
      // for block
      int d_sSync;
      int d_nProc;
      int d_nUsed;
      // for processing
      float *d_maxAcP;
      float d_maxAc;
      double d_maxAcD;
      int d_maxIndex;
      int d_lIndex;
      int d_rIndex;
      int d_mIndex;
      gr_complex d_conjMultiAvg;
      float d_tmpAc[SYNC_MAX_RES_LEN];
      float d_tmpPwr[SYNC_MAX_RES_LEN];
      gr_complex d_tmpConjSamp[128];

     public:
      sync_impl();
      ~sync_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
      void ltf_autoCorrelation(const gr_complex* sig);
      float ltf_cfo(const gr_complex* sig);    // STF CFO with LTF residual CFO
    };

  } // namespace ieee80211
} // namespace gr

#endif /* INCLUDED_IEEE80211_SYNC_IMPL_H */
