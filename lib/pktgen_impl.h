/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 20M bw and upto 2x2
 *     tx pkt stream generator
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

#ifndef INCLUDED_IEEE80211_PKTGEN_IMPL_H
#define INCLUDED_IEEE80211_PKTGEN_IMPL_H

#include <gnuradio/ieee80211/pktgen.h>
#include <gnuradio/pdu.h>
#include <vector>
#include <queue>
#include "cloud80211phy.h"

using namespace boost::placeholders;

#define PKTGEN_S_IDLE 0
#define PKTGEN_S_SCEDULE 1
#define PKTGEN_S_COPY 2
#define PKTGEN_S_PAD 3

#define PKTGEN_GR_PAD 160

namespace gr {
  namespace ieee80211 {

    class pktgen_impl : public pktgen
    {
    private:
      int d_sPktgen;
      int d_nGen;
      int d_pktSeq;
      std::queue< std::vector<uint8_t> > d_pktQ;
      std::vector<uint8_t> d_pktV;
      void msgRead(pmt::pmt_t msg);
      bool pktPop();

      int d_pktFormat;
      int d_pktMcs0;
      int d_pktNss0;
      int d_pktLen0;
      int d_pktMcs1;
      int d_pktNss1;
      int d_pktLen1;
      int d_pktMuGroupId;
      
      int d_headerShift;
      int d_nTotal;
      int d_nCopied;

    protected:
      int calculate_output_stream_length(const gr_vector_int &ninput_items);

    public:
      pktgen_impl(const std::string& lengthtagname = "packet_len");
      ~pktgen_impl();

      int work(
              int noutput_items,
              gr_vector_int &ninput_items,
              gr_vector_const_void_star &input_items,
              gr_vector_void_star &output_items
      );
    };

  } // namespace ieee80211
} // namespace gr

#endif /* INCLUDED_IEEE80211_PKTGEN_IMPL_H */
