/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Viterbi Decode of CR 12, 23, 34, 56 Soft Ver.
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
#include "decode_impl.h"

namespace gr {
  namespace ieee80211 {
    decode::sptr
    decode::make()
    {
      return gnuradio::make_block_sptr<decode_impl>(
        );
    }

    decode_impl::decode_impl()
      : gr::block("decode",
              gr::io_signature::make(1, 1, sizeof(float)),
              gr::io_signature::make(0, 0, 0)),
              d_debug(0)
    {
      d_sDecode = DECODE_S_IDLE;
    }

    decode_impl::~decode_impl()
    {
    }

    void
    decode_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      gr_vector_int::size_type ninputs = ninput_items_required.size();
      for(int i=0; i < ninputs; i++)
      {
	      ninput_items_required[i] = noutput_items + 160;
      }
    }

    int
    decode_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const float* inSig = static_cast<const float*>(input_items[0]);
      d_nProc = ninput_items[0];
      if(d_sDecode == DECODE_S_IDLE)
      {
        get_tags_in_range(tags, 0, nitems_read(0) , nitems_read(0) + 1);
        if(tags.size())
        {
          pmt::pmt_t d_meta = pmt::make_dict();
          for (auto tag : tags){
            d_meta = pmt::dict_add(d_meta, tag.key, tag.value);
          }
          t_format = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("format"), pmt::from_long(99999)));
          t_ampdu = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("ampdu"), pmt::from_long(99999)));
          t_len = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len"), pmt::from_long(99999)));
          t_nCoded = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("coded"), pmt::from_long(99999)));
          t_nTotal = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("total"), pmt::from_long(99999)));
          t_cr = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("cr"), pmt::from_long(99999)));
          d_sDecode = DECODE_S_DECODE;
        }
        consume_each(0);
        return 0;
      }
      else if(d_sDecode == DECODE_S_DECODE)
      {

      }

      consume_each (noutput_items);
      return noutput_items;
    }

    void
    decode_impl::vstb_init()
    {

    }
    void
    decode_impl::vstb_update()
    {

    }
    void
    decode_impl::vstb_end()
    {
      
    }

  } /* namespace ieee80211 */
} /* namespace gr */
