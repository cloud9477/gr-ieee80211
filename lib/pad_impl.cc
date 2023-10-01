/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 20M bw and upto 2x2
 *     TX Legacy preamble and padding
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
#include "pad_impl.h"

namespace gr {
  namespace ieee80211 {

    pad::sptr
    pad::make()
    {
      return gnuradio::make_block_sptr<pad_impl>(
        );
    }


    /*
     * The private constructor
     */
    pad_impl::pad_impl()
      : gr::block("pad",
              gr::io_signature::make(1, 1, sizeof(gr_complex)),
              gr::io_signature::make(1, 1, sizeof(gr_complex))),
              d_ofdm_fft(64,1)
    {
      d_sPad = PAD_S_TAG;
      for(int i=0;i<240;i++)
      {
        d_scaleMask[i] = 1.0f / sqrtf(52.0f) / PAD_SCALE;
        d_scaleMask[i] = 1.0f / sqrtf(52.0f) / PAD_SCALE;
        d_scaleMask[i] = 1.0f / sqrtf(52.0f) / PAD_SCALE;
      }
      for(int i=240;i<320;i++)
      {
        d_scaleMask[i] = 1.0f / sqrtf(12.0f) / PAD_SCALE;
      }
      memset((uint8_t*)d_preamblel0, 0, sizeof(gr_complex)*80);
      memcpy(d_ofdm_fft.get_inbuf(), &C8P_STF_F[32], sizeof(gr_complex)*32);
      memcpy(d_ofdm_fft.get_inbuf()+32, &C8P_STF_F[0], sizeof(gr_complex)*32);
      d_ofdm_fft.execute();
      memcpy(d_preamblel0+80, d_ofdm_fft.get_outbuf()+32, sizeof(gr_complex)*32);
      memcpy(d_preamblel0+112, d_ofdm_fft.get_outbuf(), sizeof(gr_complex)*64);
      memcpy(d_preamblel0+176, d_ofdm_fft.get_outbuf(), sizeof(gr_complex)*64);
      memcpy(d_ofdm_fft.get_inbuf(), &C8P_LTF_L_F[32], sizeof(gr_complex)*32);
      memcpy(d_ofdm_fft.get_inbuf()+32, &C8P_LTF_L_F[0], sizeof(gr_complex)*32);
      d_ofdm_fft.execute();
      memcpy(d_preamblel0+240, d_ofdm_fft.get_outbuf()+32, sizeof(gr_complex)*32);
      memcpy(d_preamblel0+272, d_ofdm_fft.get_outbuf(), sizeof(gr_complex)*64);
      memcpy(d_preamblel0+336, d_ofdm_fft.get_outbuf(), sizeof(gr_complex)*64);
      d_preamblel0[80] *= 0.5f;
      d_preamblel0[239] *= 0.5f;
      d_preamblel0[240] *= 0.5f;
      d_preamblel0[399] *= 0.5f;
      for(int i=80;i<240;i++)
      {
        d_preamblel0[i] = d_preamblel0[i] / sqrtf(12.0f) / PAD_SCALE;
      }
      for(int i=240;i<400;i++)
      {
        d_preamblel0[i] = d_preamblel0[i] / sqrtf(52.0f) / PAD_SCALE;
      }
    }

    /*
     * Our virtual destructor.
     */
    pad_impl::~pad_impl()
    {
    }

    void
    pad_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = noutput_items;
    }

    int
    pad_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const gr_complex *inSig0 = static_cast<const gr_complex*>(input_items[0]);
      gr_complex *outSig0 = static_cast<gr_complex*>(output_items[0]);
      d_nProc = ninput_items[0];
      d_nGen = noutput_items;
      d_nProced = 0;
      d_nGened = 0;

      if(d_sPad == PAD_S_TAG)
      {
        get_tags_in_range(d_tags, 0, nitems_read(0) , nitems_read(0) + 1);
        if (d_tags.size())
        {
          pmt::pmt_t d_meta = pmt::make_dict();
          for (auto tag : d_tags){
            d_meta = pmt::dict_add(d_meta, tag.key, tag.value);
          }
          d_pktFormat = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("format"), pmt::from_long(-1)));
          d_pktNss = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("nss"), pmt::from_long(-1)));
          d_pktLen = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("packet_len"), pmt::from_long(-1)));
          std::cout<<"ieee80211 pad, get tag format:"<<d_pktFormat<<", nss:"<<d_pktNss<<", len:"<<d_pktLen<<std::endl;
          d_nSampCopied = 0;
          if(d_pktFormat == C8P_F_L)
          {
            d_scaleTotal = 80;
            d_scaler = 1.0f / sqrt(52.0f) / PAD_SCALE;
          }
          else
          {
            d_scaleTotal = 320;
            d_scaler = 1.0f / sqrt(56.0f) / PAD_SCALE;
          }
          d_nSampTotal = (d_pktLen - d_scaleTotal);

          pmt::pmt_t dict = pmt::make_dict();
          dict = pmt::dict_add(dict, pmt::mp("len"), pmt::from_long(d_pktLen+400));
          pmt::pmt_t pairs = pmt::dict_items(dict);
          for (size_t i = 0; i < pmt::length(pairs); i++) {
              pmt::pmt_t pair = pmt::nth(i, pairs);
              add_item_tag(0, nitems_written(0), pmt::car(pair), pmt::cdr(pair), alias_pmt());
          }
          
          d_sPad = PAD_S_PRE;
        }
      }

      if(d_sPad == PAD_S_PRE)
      {
        if(d_nGen < (400 - d_nSampCopied))
        {
          memcpy(outSig0, d_preamblel0 + d_nSampCopied, d_nGen * sizeof(gr_complex));
          d_nGened += d_nGen;
          d_nSampCopied += d_nGen;
        }
        else
        {
          memcpy(outSig0, d_preamblel0 + d_nSampCopied, (400 - d_nSampCopied) * sizeof(gr_complex));
          d_nGened += (400 - d_nSampCopied);
          d_nSampCopied = 0;
          d_sPad = PAD_S_SIG;
        }
      }

      if(d_sPad == PAD_S_SIG)
      {
        int tmpMin = std::min((d_nGen - d_nGened), d_nProc);
        if(tmpMin < (d_scaleTotal - d_nSampCopied))
        {
          for(int i=0;i<tmpMin;i++)
          {
            outSig0[d_nGened+i] = inSig0[i] * d_scaleMask[d_nSampCopied];
            d_nSampCopied++;
          }
          d_nGened += tmpMin;
          d_nProced += tmpMin;
        }
        else
        {
          tmpMin = (d_scaleTotal - d_nSampCopied);
          for(int i=0;i<tmpMin;i++)
          {
            outSig0[d_nGened+i] = inSig0[i] * d_scaleMask[d_nSampCopied];
            d_nSampCopied++;
          }
          d_nGened += tmpMin;
          d_nProced += tmpMin;
          d_nSampCopied = 0;
          d_sPad = PAD_S_DATA;
        }
      }

      if(d_sPad == PAD_S_DATA)
      {
        int tmpMin = std::min((d_nGen - d_nGened), (d_nProc - d_nProced));
        if(tmpMin < (d_nSampTotal - d_nSampCopied))
        {
          for(int i=0;i<tmpMin;i++)
          {
            outSig0[d_nGened+i] = inSig0[d_nProced+i] * d_scaler;
          }
          d_nSampCopied += tmpMin;
          d_nGened += tmpMin;
          d_nProced += tmpMin;
        }
        else
        {
          for(int i=0;i<(d_nSampTotal - d_nSampCopied);i++)
          {
            outSig0[d_nGened+i] = inSig0[d_nProced+i] * d_scaler;
          }
          d_nGened += (d_nSampTotal - d_nSampCopied);
          d_nProced += (d_nSampTotal - d_nSampCopied);
          std::cout<<"ieee80211 pad, data done"<<std::endl;
          d_sPad = PAD_S_TAG;
        }
      }

      consume_each (d_nProced);
      return d_nGened;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
