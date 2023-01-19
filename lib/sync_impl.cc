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
              gr::io_signature::make(1, 1, sizeof(uint8_t)))//gr::io_signature::makev(2, 2, std::vector<int>{sizeof(uint8_t), sizeof(float)}))
    {
      d_nProc = 0;
      d_debug = false;
      d_sSync = SYNC_S_IDLE;
      d_sampCount = 0;
      d_usUsed = 0;
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
      const gr_complex* inSig = static_cast<const gr_complex*>(input_items[2]);
      const gr_complex* inConj = static_cast<const gr_complex*>(input_items[1]);
      uint8_t* sync = static_cast<uint8_t*>(output_items[0]);

      d_ts = std::chrono::high_resolution_clock::now();
      d_nProc = noutput_items;
      if(d_sampCount > 57864000)
      {
        dout<<"sync procd samp: "<<d_sampCount<<", used time: "<<d_usUsed<<"us, avg "<<((double)d_sampCount / (double)d_usUsed)<<" samp/us"<<std::endl;
      }

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
        d_sampCount += i;
        consume_each (i);
        d_te = std::chrono::high_resolution_clock::now();
        d_usUsed += std::chrono::duration_cast<std::chrono::microseconds>(d_te - d_ts).count();
        return i;
      }
      else
      {
        if(d_nProc >= SYNC_MAX_BUF_LEN)
        {
          ltf_autoCorrelation(inSig);
          float* tmpMaxAcP = std::max_element(d_tmpAc, d_tmpAc + SYNC_MAX_RES_LEN);
          memset(sync, 0, SYNC_MAX_RES_LEN);
          if(*tmpMaxAcP > 0.5)  // some miss trigger not higher than 0.5
          {
            float tmpMaxAc = *tmpMaxAcP * 0.8;
            int tmpMaxIndex = std::distance(d_tmpAc, tmpMaxAcP);
            int tmpL=tmpMaxIndex;
            int tmpR=tmpMaxIndex;
            for(int j=tmpMaxIndex; j>=0; j--)
            {
              if(d_tmpAc[j] < tmpMaxAc)
              {
                tmpL = j;
                break;
              }
            }
            for(int j=tmpMaxIndex; j<SYNC_MAX_RES_LEN; j++)
            {
              if(d_tmpAc[j] < tmpMaxAc)
              {
                tmpR = j;
                break;
              }
            }
            int tmpM = (tmpL+tmpR)/2;
            d_snr = 5.0f * log10f((*tmpMaxAcP) / (1 - (*tmpMaxAcP)));
            // sync index is LTF starting index + 16
            // dout<<"ieee80211 sync, ac max value: "<<*tmpMaxAcP<<", snr: "<<d_snr<<std::endl;
            float tmpTotalRadStep = ltf_cfo(&inSig[tmpM]);
            // dout<<"ieee80211 sync, total cfo:"<<(tmpTotalRadStep) * 20000000.0f / 2.0f / M_PI<<std::endl;
            sync[tmpM] = 0x01;

            // add tag to pass cfo and snr
            pmt::pmt_t dict = pmt::make_dict();
            dict = pmt::dict_add(dict, pmt::mp("rad"), pmt::from_double((double)tmpTotalRadStep));
            dict = pmt::dict_add(dict, pmt::mp("snr"), pmt::from_double((double)d_snr));
            pmt::pmt_t pairs = pmt::dict_items(dict);
            for (size_t i = 0; i < pmt::length(pairs); i++) {
                pmt::pmt_t pair = pmt::nth(i, pairs);
                add_item_tag(0,                   // output port index
                              nitems_written(0) + tmpM,  // output sample index
                              pmt::car(pair),
                              pmt::cdr(pair),
                              alias_pmt());
            }
          }
          d_sSync = SYNC_S_IDLE;
          d_sampCount += SYNC_MAX_RES_LEN;
          consume_each(SYNC_MAX_RES_LEN);
          d_te = std::chrono::high_resolution_clock::now();
          d_usUsed += std::chrono::duration_cast<std::chrono::microseconds>(d_te - d_ts).count();
          return SYNC_MAX_RES_LEN;
        }
        else
        {
          consume_each(0);
          d_te = std::chrono::high_resolution_clock::now();
          d_usUsed += std::chrono::duration_cast<std::chrono::microseconds>(d_te - d_ts).count();
          return 0;
        }
      }
      // error but return to IDLE
      d_sSync = SYNC_S_IDLE;
      consume_each (0);
      d_te = std::chrono::high_resolution_clock::now();
      d_usUsed += std::chrono::duration_cast<std::chrono::microseconds>(d_te - d_ts).count();
      return 0;
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
      // this rad step must be from pre sample * conj (next sample)
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
      // dout<<"ieee80211 sync, stf cfo:"<<(tmpRadStepStf) * 20000000.0f / 2.0f / M_PI<<", ltf cfo:"<<(tmpRadStepLtf) * 20000000.0f / 2.0f / M_PI<<std::endl;
      return (tmpRadStepStf + tmpRadStepLtf);
    }

  } /* namespace ieee80211 */
} /* namespace gr */
