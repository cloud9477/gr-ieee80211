/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Demodulation and decoding of 802.11a/g/n/ac 1x1 and 2x2 formats
 *     Copyright (C) Dec 1, 2022  Zelin Yun
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
#include "demodcu_impl.h"

namespace gr {
  namespace ieee80211 {

    demodcu::sptr
    demodcu::make()
    {
      return gnuradio::make_block_sptr<demodcu_impl>(
        );
    }


    /*
     * The private constructor
     */
    demodcu_impl::demodcu_impl()
      : gr::block("demodcu",
              gr::io_signature::make(1, 1, sizeof(gr_complex)),
              gr::io_signature::make(0, 0, 0))
    {}

    /*
     * Our virtual destructor.
     */
    demodcu_impl::~demodcu_impl()
    {
    }

    void
    demodcu_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = noutput_items + 160;
    }

    int
    demodcu_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const gr_complex* inSig = static_cast<const gr_complex*>(input_items[0]);


      consume_each (noutput_items);
      return noutput_items;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
