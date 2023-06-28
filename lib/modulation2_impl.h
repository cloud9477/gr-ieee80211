/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 20M bw and upto 2x2
 *     QAM modulation
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

#ifndef INCLUDED_IEEE80211_MODULATION2_IMPL_H
#define INCLUDED_IEEE80211_MODULATION2_IMPL_H

#include <gnuradio/ieee80211/modulation2.h>
#include <gnuradio/pdu.h>
#include "cloud80211phy.h"

using namespace boost::placeholders;

#define MODUL_S_RD_TAG 0
#define MODUL_S_SIG 1
#define MODUL_S_DATA 2
#define MODUL_S_CLEAN 3

#define MODUL_GR_GAP 160

#define MODUL_N_PADSYM 2

namespace gr {
  namespace ieee80211 {

    class modulation2_impl : public modulation2
    {
    private:
      // block
      int d_sModul;
      int d_nProc;
      int d_nGen;
      int d_nProced;
      int d_nGened;
      bool d_debug;
      // tags
      std::vector<gr::tag_t> d_tags;
      int d_pktFormat;
      int d_pktSeq;
      int d_pktMcs0;
      int d_pktNss0;
      int d_pktLen0;
      int d_pktMcs1;
      int d_pktNss1;
      int d_pktLen1;
      // modulation
      c8p_mod d_m;
      std::vector<uint8_t> d_sigBitsIntedL;
      std::vector<uint8_t> d_sigBitsIntedNL;
      std::vector<uint8_t> d_sigBitsIntedB0;
      std::vector<uint8_t> d_sigBitsIntedB1;
      gr_complex d_vhtMuBfQ[256];
      gr_complex d_sigl[64];     // legacy
      gr_complex d_signl[384];    // nl siso
      gr_complex d_signl0[448];   // nl 2x2
      gr_complex d_signl1[448];   // nl 2x2
      gr_complex d_signl1vht[448];   // nl 2x2
      gr_complex *d_sigP0, *d_sigP1;
      int d_nSampSigTotal;
      int d_nSampSigCopied;
      int d_nSymCopied;
      gr_complex d_pilotsL[1408][4];
      gr_complex d_pilotsVHT[1408][4];
      gr_complex d_pilotsHT[1408][4];
      gr_complex d_pilotsHT20[1408][4];
      gr_complex d_pilotsHT21[1408][4];
      void msgRead(pmt::pmt_t msg);

     public:
      modulation2_impl();
      ~modulation2_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

    };

  } // namespace ieee80211
} // namespace gr

#endif /* INCLUDED_IEEE80211_MODULATION2_IMPL_H */
