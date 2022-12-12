/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2, for SISO
 *     Legacy Signal Field Information
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
#include "preproc_impl.h"

namespace gr {
  namespace ieee80211 {

    preproc::sptr
    preproc::make()
    {
      return gnuradio::make_block_sptr<preproc_impl>(
        );
    }


    /*
     * The private constructor
     */
    preproc_impl::preproc_impl()
      : gr::block("preproc",
              gr::io_signature::make(1, 1, sizeof(gr_complex)),
              gr::io_signature::makev(2, 2, std::vector<int>{sizeof(float), sizeof(gr_complex)}))
    {
      preprocMall();
    }

    /*
     * Our virtual destructor.
     */
    preproc_impl::~preproc_impl()
    {
      preprocFree();
    }

    void
    preproc_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = noutput_items + PREPROC_MIN;
    }

    int
    preproc_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const gr_complex* inSig = static_cast<const gr_complex*>(input_items[0]);
      float* outAc = static_cast<float*>(output_items[0]);
      gr_complex* outConj = static_cast<gr_complex*>(output_items[1]);

      int nGen = std::min(noutput_items, std::min(ninput_items[0], ninput_items[1]));

      if(nGen > 64 && nGen < PREPROC_MAX)
      {
        cuPreProc(nGen, (const cuFloatComplex*)inSig, outAc, (cuFloatComplex*)outConj);
        consume_each(nGen - 64);
        return (nGen - 64);
      }
      
      consume_each (0);
      return 0;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
