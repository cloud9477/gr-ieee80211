/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Encoder of 802.11a/g/n/ac 1x1 and 2x2 payload part
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
#include "encode_impl.h"

namespace gr {
  namespace ieee80211 {

    encode::sptr
    encode::make(const std::string& tsb_tag_key)
    {
      return gnuradio::make_block_sptr<encode_impl>(tsb_tag_key
        );
    }


    /*
     * The private constructor
     */
    encode_impl::encode_impl(const std::string& tsb_tag_key)
      : gr::tagged_stream_block("encode",
              gr::io_signature::make(0, 0, 0),
              gr::io_signature::make(1, 1, sizeof(uint8_t)), tsb_tag_key)
    {
      //message_port_register_in(pdu::pdu_port_id());
      d_sEncode = ENCODE_S_IDLE;

      message_port_register_in(pmt::mp("pdus"));
      set_msg_handler(pmt::mp("pdus"), boost::bind(&encode_impl::msgRead, this, _1));
    }

    /*
     * Our virtual destructor.
     */
    encode_impl::~encode_impl()
    {
    }

    void
    encode_impl::msgRead(pmt::pmt_t msg)
    {
      std::cout<<"ieee80211 encode, new msg";
      pmt::pmt_t vector = pmt::cdr(msg);
      int tmpMsgLen = pmt::blob_length(vector);
      size_t tmpOffset(0);
      const uint8_t *tmpPkt = (const uint8_t *)pmt::uniform_vector_elements(vector, tmpOffset);
      uint16_t tmpLen;
      // 1B format, 1B mcs, 1B nss, 2B len, total 5B, len is 0 then NDP
      if((tmpMsgLen < 5) || (tmpMsgLen > DECODE_D_MAX)){
        return;
      }
      memcpy(d_msg, tmpPkt, tmpMsgLen);
      tmpLen = (((uint16_t)d_msg[3])<<8  + (uint16_t)d_msg[4]);
      formatToModSu(&d_m, (int)d_msg[0], (int)d_msg[1], (int)d_msg[2], (int)tmpLen);
      d_sEncode = ENCODE_S_SCEDULE;
    }

    int
    encode_impl::calculate_output_stream_length(const gr_vector_int &ninput_items)
    {
      if(d_sEncode == ENCODE_S_SCEDULE)
      {
        std::cout<<"schedule in calculate"<<std::endl;
        d_nBitsGen = d_m.nSym * d_m.nCBPS;
        d_nBitsGenProcd = 0;
        d_sEncode = ENCODE_S_ENCODE;
      }
      return d_nBitsGen;
    }

    int
    encode_impl::work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      //auto in = static_cast<const input_type*>(input_items[0]);
      uint8_t* out = static_cast<uint8_t*>(output_items[0]);
      d_nGen = noutput_items;

      switch(d_sEncode)
      {
        case ENCODE_S_IDLE:
        {
          std::cout<<"idle"<<std::endl;
          return 0;
        }

        case ENCODE_S_SCEDULE:
        {
          std::cout<<"schedule in work"<<std::endl;
          return 0;
        }

        case ENCODE_S_ENCODE:
        {
          std::cout<<"encode and gen tag"<<std::endl;
          d_sEncode = ENCODE_S_COPY;
          return 0;
        }

        case ENCODE_S_COPY:
        {
          std::cout<<"copy"<<std::endl;
          int o1 = 0;
          int nCBPS = 1000; 
          while((o1 + nCBPS) < d_nGen)
          {
            o1 += nCBPS;
            d_nBitsGenProcd += nCBPS;
            if(d_nBitsGenProcd >= d_nBitsGen)
            {
              std::cout<<"copy done"<<std::endl;
              d_sEncode = ENCODE_S_IDLE;
              break;
            }
          }
          return o1;
        }
      }

      // Tell runtime system how many output items we produced.
      d_sEncode = ENCODE_S_IDLE;
      return 0;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
