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
              gr::io_signature::make(1, 1, sizeof(float)),
              gr::io_signature::make(1, 1, sizeof(unsigned char)))
    {
      d_nBuf = 0;
      d_nProc = 0;
      d_nPlateau = 0;
      d_fPlateau = 0;
      d_fPlateauEnd = 0;
      d_conjAc = 0.0f;

      sampCount = 0;
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
      const float* inAc = static_cast<const float*>(input_items[0]);
      char* outTrigger = static_cast<char*>(output_items[0]);

      if(noutput_items < d_nBuf)
      {
        consume_each (0);
        return 0;
      }

      d_nProc = noutput_items - d_nBuf;
      for(int i=0;i<d_nProc;i++){
        sampCount++;
        // outTrigger[i] = 0;
        if(inAc[i] > 0.3f){
          d_nPlateau++;
          if(inAc[i] > d_conjAc)
          {
            d_conjAc = inAc[i];
            // indicate to update conjugate
            outTrigger[i] |= 0x02;
          }
          if(d_nPlateau > 20 && (d_fPlateau+d_fPlateauEnd)==0)
          {
            d_fPlateau = 1;
            d_fPlateauEnd = 1;
            d_countDown = 80;
          }
        }
        else
        {
          //     // d_radStep = atan2f(d_conjMulti.imag(), d_conjMulti.real()) / 16.0f;
          //     // float tmpCfo = d_radStep * 20000000.0f / 2.0f / M_PI;
          d_nPlateau = 0;
          d_fPlateauEnd = 0;
          d_conjAc = 0.0f;
        }
        if(d_fPlateau)
        {
          d_countDown--;
          if(d_countDown==0)
          {
            d_fPlateau = 0;
            outTrigger[i] |= 0x01;
            std::cout<<"ieee80211 stf: trigger "<<sampCount<<std::endl;
          }
        }
      }

      consume_each (d_nProc);
      return d_nProc;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
