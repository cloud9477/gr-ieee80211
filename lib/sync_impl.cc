/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Long Training Field Sync
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
#include "sync_impl.h"

namespace gr {
  namespace ieee80211 {

    sync::sptr
    sync::make()
    {
      return gnuradio::make_block_sptr<sync_impl>(
        );
    }


    /*
     * The private constructor
     */
    sync_impl::sync_impl()
      : gr::block("sync",
              gr::io_signature::makev(3, 3, std::vector<int>{sizeof(uint8_t), sizeof(gr_complex), sizeof(gr_complex)}),
              gr::io_signature::make(1, 1, sizeof(uint8_t)))
    {
      d_sSync = SYNC_S_IDLE;
    }

    /*
     * Our virtual destructor.
     */
    sync_impl::~sync_impl()
    {
    }

    void
    sync_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = noutput_items;
      ninput_items_required[1] = noutput_items;
      ninput_items_required[2] = noutput_items;
    }

    int
    sync_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const uint8_t* trigger = static_cast<const uint8_t*>(input_items[0]);
      const gr_complex* inConj = static_cast<const gr_complex*>(input_items[1]);
      const gr_complex* inSig = static_cast<const gr_complex*>(input_items[2]);
      uint8_t* sync = static_cast<uint8_t*>(output_items[0]);
      d_nProc = noutput_items;

      if(d_sSync == SYNC_S_IDLE)
      {
        int i=0;
        for(i=0;i<d_nProc;i++)
        {
          sync[i] = 0x00;
          if(trigger[i] & 0x01)
          {
            d_sSync = SYNC_S_SYNC;
            break;
          }
          else if(trigger[i] & 0x02)
          {
            d_conjMultiAvg = inConj[i];
          }
        }
        consume_each(i);
        return i;
      }
      else
      {
        if(d_nProc >= SYNC_MAX_BUF_LEN)
        {
          ltf_autoCorrelation(inSig);
          d_maxAcP = std::max_element(d_tmpAc, d_tmpAc + SYNC_MAX_RES_LEN);
          memset(sync, 0, SYNC_MAX_RES_LEN);
          if(*d_maxAcP > 0.5)  // some miss trigger not higher than 0.5
          {
            d_maxAc = *d_maxAcP * 0.8;
            d_maxIndex = std::distance(d_tmpAc, d_maxAcP);
            d_lIndex = d_maxIndex;
            d_rIndex = d_maxIndex;
            for(int j=d_maxIndex; j>=0; j--)
            {
              if(d_tmpAc[j] < d_maxAc)
              {
                d_lIndex = j;
                break;
              }
            }
            for(int j=d_maxIndex; j<SYNC_MAX_RES_LEN; j++)
            {
              if(d_tmpAc[j] < d_maxAc)
              {
                d_rIndex = j;
                break;
              }
            }
            d_mIndex = (d_lIndex+d_rIndex)/2;
            sync[d_mIndex] = 0x01;  // sync index is LTF starting index + 16
            pmt::pmt_t dict = pmt::make_dict();   // add tag to pass cfo and snr
            dict = pmt::dict_add(dict, pmt::mp("rad"), pmt::from_float(ltf_cfo(&inSig[d_mIndex])));
            dict = pmt::dict_add(dict, pmt::mp("snr"), pmt::from_float(10.0f * log10f((*d_maxAcP) / (1 - (*d_maxAcP)))));
            dict = pmt::dict_add(dict, pmt::mp("rssi"), pmt::from_float(d_tmpPwr[d_maxIndex] / 64.0f));
            pmt::pmt_t pairs = pmt::dict_items(dict);
            for (size_t i = 0; i < pmt::length(pairs); i++) {
                pmt::pmt_t pair = pmt::nth(i, pairs);
                add_item_tag(0,                   // output port index
                              nitems_written(0) + d_mIndex,  // output sample index
                              pmt::car(pair),
                              pmt::cdr(pair),
                              alias_pmt());
            }
          }
          d_sSync = SYNC_S_IDLE;
          consume_each(SYNC_MAX_RES_LEN);
          return SYNC_MAX_RES_LEN;
        }
        else
        {
          consume_each(0);
          return 0;
        }
      }

      std::cout<<"ieee80211 sync, state error."<<std::endl;
      d_sSync = SYNC_S_IDLE;
      consume_each(d_nProc);
      return d_nProc;
    }

    void
    sync_impl::ltf_autoCorrelation(const gr_complex* sig)
    {
      gr_complex tmpMultiSum = gr_complex(0.0f, 0.0f);
      float tmpSig1Sum = 0.0f;
      float tmpSig2Sum = 0.0f;
      // for 20MHz, init part 64 samples
      for(int i=0;i<64;i++)
      {
        tmpMultiSum += sig[i] * std::conj(sig[i+64]);
        tmpSig1Sum += std::abs(sig[i])*std::abs(sig[i]);
        tmpSig2Sum += std::abs(sig[i+64])*std::abs(sig[i+64]);
      }
      for(int i=0;i<SYNC_MAX_RES_LEN;i++)   // sliding window to compute auto correlation
      {
        d_tmpAc[i] = std::abs(tmpMultiSum)/std::sqrt(tmpSig1Sum)/std::sqrt(tmpSig2Sum);
        d_tmpPwr[i] = tmpSig1Sum;
        tmpMultiSum -= sig[i] * std::conj(sig[i+64]);
        tmpSig1Sum -= std::abs(sig[i])*std::abs(sig[i]);
        tmpSig2Sum -= std::abs(sig[i+64])*std::abs(sig[i+64]);
        tmpMultiSum += sig[i+64] * std::conj(sig[i+64+64]);
        tmpSig1Sum += std::abs(sig[i+64])*std::abs(sig[i+64]);
        tmpSig2Sum += std::abs(sig[i+64+64])*std::abs(sig[i+64+64]);
      }
    }

    float
    sync_impl::ltf_cfo(const gr_complex* sig)
    {
      gr_complex tmpConjSum = gr_complex(0.0f, 0.0f);
      float tmpRadStepStf = atan2f(d_conjMultiAvg.imag(), d_conjMultiAvg.real()) / 16.0f;
      for(int i=0;i<128;i++)
      {
        d_tmpConjSamp[i] = sig[i] * gr_complex(cosf(i * tmpRadStepStf), sinf(i * tmpRadStepStf));
      }
      for(int i=0;i<64;i++)
      {
        tmpConjSum += d_tmpConjSamp[i] * std::conj(d_tmpConjSamp[i+64]);
      }
      float tmpRadStepLtf = atan2f((tmpConjSum/64.0f).imag(), (tmpConjSum/64.0f).real()) / 64.0f;
      return (tmpRadStepStf + tmpRadStepLtf);
    }

  } /* namespace ieee80211 */
} /* namespace gr */
