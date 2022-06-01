/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Short Training Field Trigger
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
#include "trigger_impl.h"

namespace gr {
  namespace ieee80211 {

    trigger::sptr
    trigger::make()
    {
      return gnuradio::make_block_sptr<trigger_impl>(
        );
    }

    trigger_impl::trigger_impl()
      : gr::block("trigger",
              gr::io_signature::makev(3, 3, std::vector<int>{sizeof(gr_complex), sizeof(float), sizeof(gr_complex)}),
              gr::io_signature::makev(2, 2, std::vector<int>{sizeof(gr_complex), sizeof(unsigned char)}))
    {
      d_nBuf = 80;
      d_nProc = 0;
      d_tAutoCorre = 0.3f;
      d_nPlateau = 0;
      d_conjAc = 0.0f;

      sampCount = 0;
      triggerCount = 0;
    }

    trigger_impl::~trigger_impl()
    {}

    void
    trigger_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      gr_vector_int::size_type ninputs = ninput_items_required.size();
      for(int i=0; i < ninputs; i++)
      {
	      ninput_items_required[i] = noutput_items + d_nBuf;
      }
    }

    int
    trigger_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const gr_complex* inSig = static_cast<const gr_complex*>(input_items[0]);
      const float* inAc = static_cast<const float*>(input_items[1]);
      const gr_complex* inConj = static_cast<const gr_complex*>(input_items[2]);
      gr_complex* outSig = static_cast<gr_complex*>(output_items[0]);
      char* outTrigger = static_cast<char*>(output_items[1]);

      if(noutput_items < d_nBuf)
      {
        consume_each (0);
        return 0;
      }

      d_nProc = noutput_items - d_nBuf;
      for(int i=0;i<d_nProc;i++){
        sampCount++;
        outSig[i] = inSig[i];
        outTrigger[i] = 0;
        if(inAc[i] > d_tAutoCorre){
          d_nPlateau++;
          if(inAc[i] > d_conjAc)
          {
            d_conjAc = inAc[i];
            d_conjMulti = inConj[i];
          }
        }
        else
        {
          if(d_nPlateau > 20)
          {
            if(d_nPlateau < 180)
            {
              d_radStep = atan2f(d_conjMulti.imag(), d_conjMulti.real()) / 16.0f;
              float tmpCfo = d_radStep * 20000000.0f / 2.0f / M_PI;
              std::cout<<"ieee80211 stf: trigger "<<sampCount<<" "<<tmpCfo<<std::endl;
            }
          }
          d_nPlateau = 0;
          d_conjAc = 0.0f;
        }
      }

      consume_each (d_nProc);
      return d_nProc;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
