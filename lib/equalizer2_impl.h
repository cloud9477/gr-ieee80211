/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 20M bw and upto 2x2
 *     QAM modulation
 *     Copyright (C) June 1, 2023  Zelin Yun
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

#ifndef INCLUDED_IEEE80211_EQUALIZER2_IMPL_H
#define INCLUDED_IEEE80211_EQUALIZER2_IMPL_H

#include <gnuradio/ieee80211/equalizer2.h>

#define EQ_S_RDTAG 0
#define EQ_S_DATA 1
#define EQ_S_CLEAN 2

namespace gr {
  namespace ieee80211 {

    class equalizer2_impl : public equalizer2
    {
     private:
      int d_sEq;
      int d_nProc;
      int d_nGen;
      int d_nProced;
      int d_nGened;
      std::vector<gr::tag_t> d_tags;
      int d_pktFormat;
      int d_pktMcs;
      int d_pktNss;
      int d_pktLen;
      std::vector<gr_complex> d_HL;
      std::vector<gr_complex> d_HNL;
      std::vector<gr_complex> d_HNL2;
      std::vector<gr_complex> d_HNL2INV;

     public:
      equalizer2_impl();
      ~equalizer2_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
           /*
      void vhtChanUpdate(const gr_complex* sig1, const gr_complex* sig2);
      void htChanUpdate(const gr_complex* sig1, const gr_complex* sig2);
      void legacyChanUpdate(const gr_complex* sig1);
      void fftDemod(const gr_complex* sig, gr_complex* res);*/
    };

  } // namespace ieee80211
} // namespace gr

#endif /* INCLUDED_IEEE80211_EQUALIZER2_IMPL_H */
