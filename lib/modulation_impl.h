/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 20M bw and upto 2x2
 *     QAM modulation and OFDM
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

#ifndef INCLUDED_IEEE80211_MODULATION_IMPL_H
#define INCLUDED_IEEE80211_MODULATION_IMPL_H

#include <ieee80211/modulation.h>
#include "cloud80211phy.h"

#define MODUL_S_IDLE 0
#define MODUL_S_RD_TAG 1
#define MODUL_S_MOD 2
#define MODUL_S_COPY 3

namespace gr {
  namespace ieee80211 {

    class modulation_impl : public modulation
    {
    private:
      // block
      int d_sModul;
      // tag
      std::vector<gr::tag_t> d_tags;
      std::vector<uint8_t> d_tagLegacyBits;
      std::vector<uint8_t> d_tagVhtABits;
      std::vector<uint8_t> d_tagVhtB20Bits;
      std::vector<uint8_t> d_tagHtBits;
      // modulation
      c8p_mod d_m;
      uint8_t d_legacySigInted[48];
      uint8_t d_htSigInted[96];
      uint8_t d_vhtSigAInted[96];
      uint8_t d_vhtSigB20Inted[52];
      fft::fft_complex_rev d_ofdm_ifft;
      // sig, ind 2 is with CSD
      gr_complex d_stf_l[160];
      gr_complex d_stf_l2[160];
      gr_complex d_stf_nl[80];
      gr_complex d_stf_nl2[80];
      gr_complex d_ltf_l[160];
      gr_complex d_ltf_l2[160];
      gr_complex d_ltf_nl[80];
      gr_complex d_ltf_nl2[80];
      gr_complex d_ltf_nl_n[80];

      
      

      float d_sig1[53000];
      float d_sig2[53000];

     public:
      modulation_impl();
      ~modulation_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
      
      void ifft(const gr_complex* sig, gr_complex* res);
    };

  } // namespace ieee80211
} // namespace gr

#endif /* INCLUDED_IEEE80211_MODULATION_IMPL_H */
