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

#include <gnuradio/io_signature.h>
#include "pktgen_impl.h"

namespace gr {
  namespace ieee80211 {

    pktgen::sptr
    pktgen::make(const std::string& tsb_tag_key)
    {
      return gnuradio::make_block_sptr<pktgen_impl>(tsb_tag_key
        );
    }


    /*
     * The private constructor
     */
    pktgen_impl::pktgen_impl(const std::string& tsb_tag_key)
      : gr::tagged_stream_block("genpkt",
              gr::io_signature::make(0, 0, 0),
              gr::io_signature::make(1, 1, sizeof(uint8_t)), tsb_tag_key)
    {
      d_sPktgen = PKTGEN_S_IDLE;
      d_pktSeq = 0;

      message_port_register_in(pmt::mp("pdus"));
      set_msg_handler(pmt::mp("pdus"), boost::bind(&pktgen_impl::msgRead, this, _1));
    }

    /*
     * Our virtual destructor.
     */
    pktgen_impl::~pktgen_impl()
    {
    }

    void
    pktgen_impl::msgRead(pmt::pmt_t msg)
    {
      /* 1B format, 1B mcs, 1B nss, 2B len, total 5B, len is 0 then NDP*/
      pmt::pmt_t msgVec = pmt::cdr(msg);
      int pktLen = pmt::blob_length(msgVec);
      size_t tmpOffset(0);
      const uint8_t *tmpPkt = (const uint8_t *)pmt::uniform_vector_elements(msgVec, tmpOffset);
      if(pktLen < 5){
        return;
      }
      std::vector<uint8_t> pktVec(tmpPkt, tmpPkt + pktLen);
      d_pktQ.push(pktVec);
    }

    int
    pktgen_impl::calculate_output_stream_length(const gr_vector_int &ninput_items)
    {
      if(d_sPktgen == PKTGEN_S_SCEDULE)
      {
        d_sPktgen = PKTGEN_S_COPY;
        return d_nTotal + PKTGEN_GR_PAD;
      }
      else if(d_sPktgen == PKTGEN_S_IDLE && pktPop())
      {
        d_sPktgen = PKTGEN_S_COPY;
        return d_nTotal + PKTGEN_GR_PAD;
      }
      return 0;
    }

    bool
    pktgen_impl::pktPop()
    {
      if(d_pktQ.size())
      {
        d_pktV = d_pktQ.front();
        d_pktQ.pop();
        d_pktFormat = (int)d_pktV[0];
        if(d_pktFormat == C8P_F_VHT_MU)
        {
          d_pktMcs0 = (int)d_pktV[1];
          d_pktNss0 = (int)d_pktV[2];
          d_pktLen0 = ((int)d_pktV[4] * 256  + (int)d_pktV[3]);
          d_pktMcs1 = (int)d_pktV[5];
          d_pktNss1 = (int)d_pktV[6];
          d_pktLen1 = ((int)d_pktV[8] * 256  + (int)d_pktV[7]);
          d_pktMuGroupId = (int)d_pktV[9];
          d_headerShift = 10;
          d_nTotal = d_pktLen0 + d_pktLen1;
        }
        else
        {
          d_pktMcs0 = (int)d_pktV[1];
          d_pktNss0 = (int)d_pktV[2];
          d_pktLen0 = ((int)d_pktV[4] * 256  + (int)d_pktV[3]);
          d_headerShift = 5;
          d_nTotal = d_pktLen0;
        }
        d_nCopied = 0;

        if(d_pktV.size() < (uint64_t)(d_nTotal + d_headerShift) || d_nTotal > 4095)
        {
          return false;
        }

        // write tag
        pmt::pmt_t dict = pmt::make_dict();
        dict = pmt::dict_add(dict, pmt::mp("format"), pmt::from_long(d_pktFormat));
        dict = pmt::dict_add(dict, pmt::mp("mcs0"), pmt::from_long(d_pktMcs0));
        dict = pmt::dict_add(dict, pmt::mp("nss0"), pmt::from_long(d_pktNss0));
        dict = pmt::dict_add(dict, pmt::mp("len0"), pmt::from_long(d_pktLen0));
        dict = pmt::dict_add(dict, pmt::mp("seq"), pmt::from_long(d_pktSeq));
        if(d_pktFormat == C8P_F_VHT_MU)
        {
          dict = pmt::dict_add(dict, pmt::mp("mcs1"), pmt::from_long(d_pktMcs1));
          dict = pmt::dict_add(dict, pmt::mp("nss1"), pmt::from_long(d_pktNss1));
          dict = pmt::dict_add(dict, pmt::mp("len1"), pmt::from_long(d_pktLen1));
          dict = pmt::dict_add(dict, pmt::mp("gid"), pmt::from_long(d_pktMuGroupId));
        }
        pmt::pmt_t pairs = pmt::dict_items(dict);
        for (size_t i = 0; i < pmt::length(pairs); i++) {
            pmt::pmt_t pair = pmt::nth(i, pairs);
            add_item_tag(0,                   // output port index
                          nitems_written(0),  // output sample index
                          pmt::car(pair),     
                          pmt::cdr(pair),
                          alias_pmt());
        }
        return true;
      }
      return false;
    }

    int
    pktgen_impl::work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      uint8_t* outPkt = static_cast<uint8_t*>(output_items[0]);
      d_nGen = noutput_items;
      if(d_sPktgen == PKTGEN_S_IDLE)
      {
        if(pktPop())
        {
          d_sPktgen = PKTGEN_S_SCEDULE;
        }
        return 0;
      }

      else if(d_sPktgen == PKTGEN_S_SCEDULE)
      {
        return 0;
      }

      else if(d_sPktgen == PKTGEN_S_COPY)
      {
        if(d_nGen >= (d_nTotal - d_nCopied))
        {
          int tmpCopied = d_nTotal - d_nCopied;
          memcpy(outPkt, d_pktV.data() + d_headerShift + d_nCopied, (d_nTotal - d_nCopied));
          d_sPktgen = PKTGEN_S_PAD;
          std::cout<<"ieee80211 pktgen write packet done "<<d_pktSeq<<std::endl;
          d_pktSeq++;
          d_nTotal = PKTGEN_GR_PAD;
          d_nCopied = 0;
          return tmpCopied;
        }
        else
        {
          memcpy(outPkt, d_pktV.data() + d_headerShift + d_nCopied, d_nGen);
          d_nCopied += d_nGen;
          return d_nGen;
        }
      }

      else
      {
        if(d_nGen >= (d_nTotal - d_nCopied))
        {
          d_sPktgen = PKTGEN_S_IDLE;
          return (d_nTotal - d_nCopied);
        }
        else
        {
          d_nCopied += d_nGen;
          return d_nGen;
        }
      }

      return 0;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
